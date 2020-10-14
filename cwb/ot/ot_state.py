from cwb.common.config_parser import \
        construct_single_distribution, \
        construct_regularizer, \
        construct_regularizer_derivative, \
        construct_optimizer
from cwb.common.architecture import PotentialModel
from cwb.common.utils import calc_distances
from cwb.common.utils import \
        tf_ckpt_register, \
        tf_set_float_dtype

import tensorflow as tf
import numpy as np
import math
import pickle

class OTState:
    def __init__(self, conf):
        print('Initializing OT state...')
        self.conf = conf
        float_dtype = tf_set_float_dtype(conf.get('nn_dtype', 'float32'))
        self.float_dtype = float_dtype
        self.load_source_target()
        reg_eps = conf['regularizer_desc']['eps']
        self.reg_fn = construct_regularizer(conf['regularizer_desc'], reg_eps, float_dtype)
        self.reg_d_fn = construct_regularizer_derivative(conf['regularizer_desc'], reg_eps, float_dtype)
        self.start_time = tf.timestamp()
        self.summary_writer = tf.summary.create_file_writer(self.conf['log_dir'])
        self.setup_potential_architecture()

    def load_source_target(self):
        conf = self.conf
        self.mu_source = construct_single_distribution(conf['source_distribution'], self.float_dtype)
        self.mu_target = construct_single_distribution(conf['target_distribution'], self.float_dtype)

    def setup_potential_architecture(self):
        self.potential_vars = []
        ckpt_kwargs = {}
        f = PotentialModel(self.conf)
        g = PotentialModel(self.conf)
        f.build(input_shape=(None, self.conf['point_dim']))
        g.build(input_shape=(None, self.conf['point_dim']))
        ckpt_kwargs['f'] = f
        ckpt_kwargs['g'] = g
        self.potential_vars.extend(f.trainable_variables)
        self.potential_vars.extend(g.trainable_variables)
        self.potential_f = f
        self.potential_g = g

        self.potential_step = tf.Variable(0, dtype=tf.int64)
        ckpt_kwargs['potential_step'] = self.potential_step
        self.potential_optimizer = construct_optimizer(self.conf['potential_optimizer_desc'])
        ckpt_kwargs['potential_optimizer'] = self.potential_optimizer

        self.potential_ckpt_manager = tf_ckpt_register(
                ckpt_kwargs,
                self.conf['potential_ckpt_dir'],
                max_to_keep=self.conf['ckpt_max_to_keep'])

    @tf.function
    def train_potential_step(self):
        batch_size = self.conf['batch_size']

        with tf.GradientTape() as tape:
            f = self.potential_f
            g = self.potential_g
            s_f = self.mu_source.sample(batch_size)
            s_g = self.mu_target.sample(batch_size)
            val_f = f(s_f)
            val_g = g(s_g)
            cs = calc_distances(s_f, s_g)
            reg = self.reg_fn(val_f + val_g - cs)
            obj = val_f + val_g - reg

            obj = tf.reduce_mean(obj)
            reg = tf.reduce_mean(reg)
            neg_obj = -obj

        grads = tape.gradient(neg_obj, self.potential_vars)
        self.potential_optimizer.apply_gradients(zip(grads, self.potential_vars))
        self.potential_step.assign_add(1)

        return obj, reg


    @tf.function
    def train_multiple_steps(self, multi_steps):
        potential_obj = tf.constant(0.0)
        reg_total = tf.constant(0.0)
        for i in tf.range(multi_steps):
            potential_obj, reg_total = self.train_potential_step()
        return potential_obj, reg_total


    def train_potentials(self):
        conf = self.conf
        while int(self.potential_step) < conf['potential_total_epochs']:
            potential_obj, reg_total = self.train_multiple_steps(self.conf['train_multi_steps'])
            elapsed_time = tf.timestamp() - self.start_time
            step_int = int(self.potential_step)

            if step_int % conf['print_frequency'] == 0:
                print('Potential Step: {} | Obj: {:4E} | Reg: {:4E} | Elapsed: {:2f}s'.format(step_int, float(potential_obj), float(reg_total), float(elapsed_time)))

            if step_int % conf['log_frequency'] == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar('potential_obj', potential_obj, step=self.potential_step)

            if step_int % conf['ckpt_save_period'] == 0:
                save_path = self.potential_ckpt_manager.save()
                print('Saved checkpoint for potential step {} at {}'.format(step_int, save_path))

    def eval(self):
        # evaluate OT distance over a large batch
        batch_size = self.conf['batch_size']
        num_batches = self.conf['eval_num_batches']

        f = self.potential_f
        g = self.potential_g

        obj_total = 0
        reg_total = 0
        for i in range(num_batches):
            s_f = self.mu_source.sample(batch_size)
            s_g = self.mu_target.sample(batch_size)
            val_f = f(s_f)
            val_g = g(s_g)
            cs = calc_distances(s_f, s_g)
            reg = self.reg_fn(val_f + val_g - cs)
            obj = val_f + val_g - reg
            obj_total += tf.reduce_sum(obj)
            reg_total += tf.reduce_sum(reg)

        total_count = tf.convert_to_tensor(num_batches * batch_size)
        obj_avg = obj_total / tf.cast(total_count, self.float_dtype)
        reg_avg = reg_total / tf.cast(total_count, self.float_dtype)

        pickle.dump({
            'obj_avg': obj_avg,
            'reg_avg': reg_avg
            }, open(self.conf['eval_file'], 'wb'))

        return obj_avg

