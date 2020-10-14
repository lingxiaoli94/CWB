from cwb.common.config_parser import \
        construct_single_distribution, \
        construct_distribution_list, \
        construct_regularizer, \
        construct_regularizer_derivative, \
        construct_optimizer, \
        load_empirical_npz
from cwb.common.distributions import \
        uniform_nd_tf, \
        uniform_bbox_tf, \
        BoundedDistributionAdapter, \
        TranslatedDistributionAdapter
from cwb.common.moving_averages import MovingAverages
from cwb.common.architecture import RFFLayer, PotentialModel, TransportMapModel
from cwb.common.utils import compute_plan_pdf, calc_distances, calc_cross_distances
from cwb.common.utils import \
        tf_ckpt_register, \
        tf_set_float_dtype
from .validator import Validator


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import os
from sklearn import decomposition

class BarycenterState:
    def __init__(self, conf):
        print('Initializing Barycenter state...')
        self.conf = conf
        float_dtype = tf_set_float_dtype(conf.get('nn_dtype', 'float32'))
        self.float_dtype = float_dtype

        self.build_distribution_list()
        self.use_zero_mean_reduction = conf.get('use_zero_mean_reduction', True)
        if self.use_zero_mean_reduction:
            # Zero mean all input distributions
            # This only works if there is a single placeholder
            self.apply_zero_mean_reduction()

        self.infer_support()
        self.setup_regularizers()
        self.start_time = tf.timestamp()
        self.summary_writer = tf.summary.create_file_writer(self.conf['log_dir'])

        self.setup_potential_architecture()
        if conf['estimate_map']:
            self.setup_map_architecture()
        self.validator = Validator(self)

    def use_soft_zero_mean(self):
        return self.conf.get('soft_zero_mean', False)

    def setup_regularizers(self):
        desc = self.conf['regularizer_desc']
        eps = desc['eps']
        if desc['scale_eps'] and self.conf['supp_desc']['shape'] == 'bbox_inferred':
            base_diameter_sqr = desc['base_diameter_sqr']
            eps = eps * self.diameter_sqr / base_diameter_sqr
        self.reg_fn = construct_regularizer(desc, eps, self.float_dtype)
        self.reg_d_fn = construct_regularizer_derivative(desc, eps, self.float_dtype)

    def infer_support(self):
        print('Initializing the support measure...')
        conf = self.conf
        if conf['supp_desc']['shape'] == 'bbox_inferred':
            self.supp = self.infer_bbox_support()
        else:
            if self.use_zero_mean_reduction:
                raise Exception('Only allow prescribed support without zero-mean reduction.')
            self.supp = construct_single_distribution(conf['supp_desc'], self.float_dtype)

    def build_distribution_list(self):
        conf = self.conf
        self.mu_list, self.name_list, self.weight_list = \
                construct_distribution_list(conf['distribution_list'], self.float_dtype)
        self.source_list = self.mu_list.copy() # make a copy of untransformed distributions
        self.num_sources = len(self.source_list)

    def get_representative_samples_from_source(self, i):
        mu_desc = self.conf['distribution_list'][i]
        if mu_desc['shape'] == 'empirical_npz':
            # If empirical, simply use all the samples.
            ps = load_empirical_npz(mu_desc)
        elif mu_desc['shape'] == 'empirical_npy':
            ps = np.load(mu_desc['npy_path'])
        elif mu_desc['shape'] == 'empirical':
            ps = np.array(mu_desc['points'])
        else:
            sample_count = self.conf['infer_sample_count']
            ps = self.source_list[i].sample(sample_count).numpy()
        return ps


    def apply_zero_mean_reduction(self):
        print('Applying zero-mean reduction...')
        list_desc = self.conf['distribution_list']
        self.mean_list = []
        for i, mu_desc in enumerate(list_desc):
            name = mu_desc['name']
            if mu_desc['shape'] == 'gaussian':
                cur_mean = np.array(mu_desc['mean'])
            else:
                ps = self.get_representative_samples_from_source(i)
                cur_mean = ps.mean(axis=0)

            self.mean_list.append(tf.convert_to_tensor(cur_mean, dtype=self.float_dtype))

        cur_mean = 0
        w_sum = 0
        for w in self.weight_list:
            w_sum += w
        for i in range(self.num_sources):
            w = self.weight_list[i]
            cur_mean += (w / w_sum) * self.mean_list[i]
        self.target_mean = cur_mean

        # Adapt source distributions to have zero mean.
        for i in range(self.num_sources):
            new_distribution = TranslatedDistributionAdapter(self.mu_list[i], -self.mean_list[i])
            self.mu_list[i] = new_distribution

    def infer_bbox_support(self):
        print('Inferring support to be the bounding box of samples from sources...')
        percentile = self.conf['supp_desc'].get('percentile', 1.0)
        point_dim = self.conf['point_dim']
        all_points = []
        for i, mu in enumerate(self.mu_list):
            ps = self.get_representative_samples_from_source(i)
            if self.use_zero_mean_reduction:
                ps -= self.mean_list[i]
            all_points.append(ps)
        all_points = np.concatenate(all_points, axis=0)
        all_count = all_points.shape[0]

        pca = decomposition.PCA(n_components=point_dim)
        pca.fit(all_points)
        pca_components = pca.components_ # DxD
        a_min = np.zeros([point_dim])
        a_max = np.zeros([point_dim])
        diameter_sqr = 0
        for i in range(point_dim):
            proj = np.sum(all_points * np.expand_dims(pca_components[i, :], 0), -1)
            proj = np.sort(proj)
            lb = proj[int(math.floor((1 - percentile) * all_count))]
            rb = proj[int(math.ceil(percentile * all_count)) - 1]
            a_min[i] = lb
            a_max[i] = rb
            diameter_sqr += (rb - lb) ** 2
        print('Diagonal length squared of the inferred bounding box: {:.4f}.'.format(diameter_sqr))
        self.diameter_sqr = diameter_sqr

        supp = uniform_bbox_tf(pca_components, a_min, a_max, self.float_dtype)

        if percentile < 1.0:
            # To prevent sampling outside of the uniform support, we need to modify the distributions.
            for i in range(self.num_sources):
                new_distribution = BoundedDistributionAdapter(self.mu_list[i], supp)
                self.mu_list[i] = new_distribution

        return supp

    def create_nn_model(self):
        model_type = self.conf.get('potential_model_type', 'nn')
        if  model_type == 'nn':
            return PotentialModel(self.conf)
        elif model_type == 'rff':
            return RFFLayer(self.conf)
        else:
            raise 'Unknown model type: {}'.format(model_type)

    def setup_potential_architecture(self):
        self.potential_f_list = []
        self.potential_g_list = []
        self.potential_vars = []
        ckpt_kwargs = {}
        for i in range(self.num_sources):
            f = self.create_nn_model()
            g = self.create_nn_model()
            f.build(input_shape=(None, self.conf['point_dim']))
            g.build(input_shape=(None, self.conf['point_dim']))
            ckpt_kwargs['f_{}'.format(i)] = f
            ckpt_kwargs['g_{}'.format(i)] = g
            self.potential_vars.extend(f.trainable_variables)
            self.potential_vars.extend(g.trainable_variables)
            self.potential_f_list.append(f)
            self.potential_g_list.append(g)

        if self.conf['moving_averages']['potential_enabled']:
            self.potential_MA = MovingAverages(
                    'potential',
                    self.potential_vars,
                    ckpt_kwargs,
                    self.conf['moving_averages']['decay'],
                    self.float_dtype)

        self.potential_step = tf.Variable(0, dtype=tf.int64)
        ckpt_kwargs['potential_step'] = self.potential_step
        self.potential_optimizer = construct_optimizer(self.conf['potential_optimizer_desc'])
        ckpt_kwargs['potential_optimizer'] = self.potential_optimizer

        self.potential_ckpt_manager = tf_ckpt_register(
                ckpt_kwargs,
                self.conf['potential_ckpt_dir'],
                max_to_keep=self.conf['ckpt_max_to_keep'])

    def setup_map_architecture(self):
        self.T_list = []
        self.map_vars = []
        ckpt_kwargs = {}
        for i in range(self.num_sources):
            T = TransportMapModel(self.conf)
            T.build(input_shape=(None, self.conf['point_dim']))
            ckpt_kwargs['T_{}'.format(i)] = T
            self.T_list.append(T)
            self.map_vars.extend(T.trainable_variables)

        self.map_step = tf.Variable(0, dtype=tf.int64)
        ckpt_kwargs['map_step'] = self.map_step
        self.map_optimizer = construct_optimizer(self.conf['map_optimizer_desc'])
        ckpt_kwargs['map_optimizer'] = self.map_optimizer

        self.map_ckpt_manager = tf_ckpt_register(
                ckpt_kwargs,
                self.conf['map_ckpt_dir'],
                max_to_keep=self.conf['ckpt_max_to_keep'])


    def calc_g_mean(self, x):
        return tf.reduce_sum(
                tf.stack(
                    [self.weight_list[i] * self.potential_g_list[i](x) for i in range(self.num_sources)], axis=0), axis=0)


    def calc_potentials(self, i):
        # Prepare dual potentials. This is only used in testing and also in training T's.
        # Note the potentials are defined with respect to translated input distributions,
        # if use_zero_mean_reduction is enabled.
        return (self.potential_f_list[i], lambda x: self.potential_g_list[i](x) - self.calc_g_mean(x))


    def adapt_zero_mean_to_original(self, i, T):
        if self.use_zero_mean_reduction:
            # In an ideal scenario, T will map a centered distribution to another centered distribution.
            # However the image of T is not guaranteed to zero mean in practice.
            # We correct the offset here.
            ps = tf.convert_to_tensor(self.get_representative_samples_from_source(i), dtype=self.float_dtype)
            ps -= self.mean_list[i]
            num_remain = ps.shape[0]
            cur = 0
            T_sum = 0
            while num_remain > 0:
                count = min(num_remain, self.conf['batch_size'])
                ps_sub = ps[cur:cur + count, :]
                T_sum += tf.reduce_sum(T(ps_sub), axis=0)
                cur += count
                num_remain -= count
            # T_mean = tf.reduce_mean(T(ps), axis=0, keepdims=True) # TODO: fix memory usage
            T_mean = T_sum / ps.shape[0]
            return lambda x: T(x - self.mean_list[i]) - T_mean + self.target_mean
        else:
            return T

    def prepare_transport_map(self, i):
        return self.adapt_zero_mean_to_original(i, self.T_list[i])

    def prepare_barycentric_projection(self, i):
        batch_size = self.conf['batch_size']
        def projection_at(f, g, xs):
            # |X| cannot be too large here as we are computing cross distances
            ys = self.supp.sample(batch_size)
            cs = calc_cross_distances(xs, ys) # |X| x |Y|
            fxs = f(xs) # |X|
            gys = g(ys) # |Y|
            tmp = tf.expand_dims(fxs, 1) + tf.expand_dims(gys, 0) - cs # |X| x |Y|
            tmp = self.reg_d_fn(tmp) # |X| x |Y|
            tmp = tf.expand_dims(ys, 0) * tf.expand_dims(tmp, 2) # |X| x |Y| x D
            tmp = tf.reduce_mean(tmp, 1) # |X| x D
            return tmp

        def projection_fn(f, g):
            return lambda xs: projection_at(f, g, xs)

        f, g = self.calc_potentials(i)
        return self.adapt_zero_mean_to_original(i, projection_fn(f, g))


    def prepare_potential_gradient_map(self, i):
        def gradient_at(f, x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = f(x)
            grad = tape.gradient(y, x)
            return grad
        def gradient_fn(f):
            return lambda x: gradient_at(f, x)
        f = self.potential_f_list[i]
        grad_f = gradient_fn(f)
        return self.adapt_zero_mean_to_original(i, lambda x: x - grad_f(x) / 2)


    def prepare_potentials(self, i):
        f, g = self.calc_potentials(i)

        if self.use_zero_mean_reduction:
            m0 = self.mean_list[i]
            m1 = self.target_mean
        else:
            m0 = 0
            m1 = 0

        return (lambda x: f(x - m0), lambda y: g(y - m1))


    def prepare_plan_pdf_fn(self, i, use_batch):
        # Prepare the transport plan. This is only used in testing.
        # As in all prepare_* functions, zero-mean shift is adjusted here.
        # Returned plan_pdfs can work either with batching (use_batch=True) or not.
        f, g = self.calc_potentials(i)

        if self.use_zero_mean_reduction:
            m0 = self.mean_list[i]
            m1 = self.target_mean
        else:
            m0 = 0
            m1 = 0

        return lambda x, y: compute_plan_pdf(f, g, self.mu_list[i], self.supp, self.reg_d_fn, x - m0, y - m1, use_batch)

    def get_source(self, i):
        return self.source_list[i]

    def get_name(self, i):
        return self.name_list[i]


    @tf.function
    def train_potential_step(self):
        batch_size = self.conf['batch_size']
        supp_samples = self.supp.sample(batch_size)

        with tf.GradientTape() as tape:
            obj = 0
            reg_tot = 0
            for i in range(self.num_sources):
                w = self.weight_list[i]
                f = self.potential_f_list[i]
                g = self.potential_g_list[i]
                s_f = self.mu_list[i].sample(batch_size)
                s_g = supp_samples
                val_f = f(s_f)
                val_g = g(s_g) - self.calc_g_mean(s_g) # TODO: optimize here
                cs = calc_distances(s_f, s_g)
                reg_i = self.reg_fn(val_f + val_g - cs)
                obj += w * (val_f - reg_i)
                reg_tot += w * reg_i

            obj = tf.reduce_mean(obj)
            reg_tot = tf.reduce_mean(reg_tot)
            neg_obj = -obj

        grads = tape.gradient(neg_obj, self.potential_vars)
        self.potential_optimizer.apply_gradients(zip(grads, self.potential_vars))

        if self.conf['moving_averages']['potential_enabled']:
            self.potential_MA.update_step()

        self.potential_step.assign_add(1)
        return obj, reg_tot

    def train_potentials(self):
        conf = self.conf
        while int(self.potential_step) < conf['potential_total_epochs']:
            potential_obj, reg_total = self.train_potential_step()
            elapsed_time = tf.timestamp() - self.start_time
            step_int = int(self.potential_step)

            if step_int % conf['print_frequency'] == 0:
                print('Potential Step: {} | Obj: {:4E} | Reg: {:4E} | Elapsed: {:.2f}s'.format(step_int, float(potential_obj), float(reg_total), float(elapsed_time)))

            if step_int % conf['log_frequency'] == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar('potential_obj', potential_obj, step=self.potential_step)

            if step_int % conf['ckpt_save_period'] == 0:
                save_path = self.potential_ckpt_manager.save()
                print('Saved checkpoint for potential step {} at {}'.format(step_int, save_path))

            if step_int % conf['val_frequency'] == 0:
                self.validate_potential_training()


    @staticmethod
    def compute_single_transport_map_loss(batch_size, T, mu_x, mu_y, f, g, reg_d_fn):
        # train the transport T from mu_x to mu_y
        xs = mu_x.sample(batch_size)
        ys = mu_y.sample(batch_size)
        C = calc_distances(xs, ys)
        fxs = f(xs)
        gys = g(ys)
        obj = reg_d_fn(fxs + gys - C)
        obj = calc_distances(T(xs), ys) * obj
        obj = tf.reduce_mean(obj)

        return obj


    @tf.function
    def train_all_transport_maps_step(self):
        batch_size = self.conf['batch_size']

        with tf.GradientTape() as tape:
            total_obj = 0
            for i in range(self.num_sources):
                f, g = self.calc_potentials(i)
                total_obj += self.compute_single_transport_map_loss(
                        batch_size=batch_size,
                        T=self.T_list[i],
                        mu_x=self.mu_list[i],
                        mu_y=self.supp,
                        f=f,
                        g=g,
                        reg_d_fn=self.reg_d_fn)

        grads = tape.gradient(total_obj, self.map_vars)
        self.map_optimizer.apply_gradients(zip(grads, self.map_vars))

        return total_obj

    def train_transport_maps(self):
        if self.conf['moving_averages']['potential_enabled']:
            self.potential_MA.swap_in_averages()
        conf = self.conf
        while int(self.map_step) < conf['map_total_epochs']:
            map_obj = self.train_all_transport_maps_step()
            self.map_step.assign_add(1)
            elapsed_time = tf.timestamp() - self.start_time
            step_int = int(self.map_step)

            if step_int % conf['print_frequency'] == 0:
                print('Map Step: {} | Obj: {:4E} | Elapsed: {:.2f}s'.format(step_int, float(map_obj), float(elapsed_time)))

            if step_int % conf['log_frequency'] == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar('map_obj', map_obj, step=self.map_step)

            if step_int % conf['ckpt_save_period'] == 0:
                save_path = self.map_ckpt_manager.save()
                print('Saved checkpoint for map step {} at {}'.format(step_int, save_path))

            if step_int % conf['val_frequency'] == 0:
                self.validate_map_training()
        if self.conf['moving_averages']['potential_enabled']:
            self.potential_MA.swap_out_averages()

    def validate_potential_training(self):
        self.validator.validate_potential_training()

    def validate_map_training(self):
        self.validator.validate_map_training()

    def sample_barycenter(self, num_samples):
        print('Sampling from barycenter using pushforwards...')
        if self.conf['moving_averages']['potential_enabled']:
            self.potential_MA.swap_in_averages()
        num_remain = num_samples
        samples_list = []
        while num_remain > 0:
            batch_size = self.conf.get('max_batch_size', 50000)
            result = {}
            for i, n in enumerate(self.name_list):
                result[n] = self.source_list[i].sample(batch_size).numpy()
            monge_map_kind = self.conf.get('preferred_monge_map_kind', 'potential_gradient')
            print('Sampling barycenter using Monge map kind {}...'.format(monge_map_kind))
            pushforward_list = []
            for i in range(self.num_sources):
                pi = self.source_list[i].sample(batch_size)
                if monge_map_kind == 'seguy':
                    Ti = self.prepare_transport_map(i)
                elif monge_map_kind == 'potential_gradient':
                    Ti = self.prepare_potential_gradient_map(i)
                elif monge_map_kind == 'barycentric_projection':
                    Ti = self.prepare_barycentric_projection(i)
                else:
                    raise Exception('Unknown Monge map kind: {}'.format(monge_map_kind))
                pushforward_list.append(Ti(pi))
            pushforwards = tf.concat(pushforward_list, axis=0)

            # Trimming: assume equal weights here for now.
            pushforwards = pushforwards.numpy()
            num_take = min(num_remain, pushforwards.shape[0])
            pushforwards = pushforwards[np.random.choice(pushforwards.shape[0], num_take, replace=False), :]
            num_remain -= num_take
            samples_list.append(pushforwards)
        result = np.concatenate(samples_list, axis=0)

        if self.conf['moving_averages']['potential_enabled']:
            self.potential_MA.swap_out_averages()
        return result

    def sample_plans_mcmc(self, mcmc_desc):
        conf = self.conf
        num_sources = self.num_sources
        step = int(self.potential_step)
        print('Running MCMC on plans at step {}...'.format(step))

        sample_plan_tf = self.sample_plan # = tf.function(self.sample_plan, autograph=True)

        all_samples = []
        for k in range(num_sources):
            plan_pdf = self.prepare_plan_pdf_fn(k, use_batch=False) # sample a single chain
            p = self.source_list[k].sample(1)
            q = self.prepare_transport_map(k)(p)
            p = tf.squeeze(p, 0)
            q = tf.squeeze(q, 0)
            init_state = tf.concat([p, q], axis=0) # tf.reshape(mcmc_desc['init_state'], [-1])
            samples = sample_plan_tf(plan_pdf, dims=conf['point_dim'], desc=mcmc_desc, init_state=init_state)
            # samples is Nx2xD
            samples = tf.reshape(samples, [-1, 2, conf['point_dim']])
            all_samples.append(samples)

        all_samples = np.swapaxes(np.array(all_samples), 1, 2) # Ex2xNxD

        samples_dir = mcmc_desc['npy_dir']
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        npy_name = mcmc_desc['npy_file']
        save_filename = os.path.join(samples_dir, npy_name)
        np.save(save_filename, all_samples)

    @staticmethod
    def sample_plan(plan, dims, desc, init_state):
        def log_prob_fn(w):
            # first D components of w is x, last D is y
            p = plan(w[:dims], w[dims:]) + desc['eps']
            return tf.math.log(p)

        if desc['kind'] == 'HMC':
            kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob_fn,
                step_size=desc['step_size'],
                num_leapfrog_steps=desc['num_leapfrog_steps'],
                state_gradients_are_stopped=True)
        elif desc['kind'] == 'Metropolis':
            kernel=tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=log_prob_fn,
                new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=desc['proposal_scale']))
        else:
            raise Exception('Unrecognized MCMC kind: {}'.format(desc['kind']))

        states = tfp.mcmc.sample_chain(
                num_results=desc['num_results'],
                num_burnin_steps=desc['num_burnin_steps'],
                num_steps_between_results=desc['thinning'],
                current_state=init_state,
                kernel=kernel,
                trace_fn=None)
        return states

