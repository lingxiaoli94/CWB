import tensorflow as tf
import numpy as np
import os

class Validator:
    def __init__(self, state):
        self.state = state

    def validate_marginals(self, P, discrete_num, da, is_3d):
        print('Validating marginals of transport plans...')
        state = self.state
        conf = state.conf
        step_int = int(state.potential_step)
        N = P.shape[0]
        batch_size = conf['marginalization_batch_size']
        total_batches = N // batch_size

        num_sources = state.num_sources
        results = []

        for k in range(num_sources):
            marginal_list_0 = []
            marginal_list_1 = []
            plan_pdf = state.prepare_plan_pdf_fn(k, use_batch=True)
            for i in range(total_batches):
                P_seg = P[i * batch_size : (i+1) * batch_size, :]
                plan_0 = plan_pdf(P_seg, P)
                plan_1 = plan_pdf(P, P_seg)
                marginal_0 = tf.reduce_sum(plan_0, 1) * da
                marginal_1 = tf.reduce_sum(plan_1, 0) * da
                marginal_list_0.append(marginal_0)
                marginal_list_1.append(marginal_1)
            result = (tf.concat(marginal_list_0, axis=0), tf.concat(marginal_list_1, axis=0))
            results.append(result)

        # save results
        if not is_3d:
            all_marginals = np.zeros(shape=[num_sources, 2, discrete_num, discrete_num])
            for k in range(num_sources):
                marginal_0, marginal_1 = results[k]
                marginal_0 = tf.reshape(marginal_0, [discrete_num, discrete_num])
                marginal_1 = tf.reshape(marginal_1, [discrete_num, discrete_num])
                all_marginals[k, 0, :, :] = marginal_0.numpy()
                all_marginals[k, 1, :, :] = marginal_1.numpy()
        else:
            all_marginals = np.zeros(shape=[num_sources, 2, discrete_num, discrete_num, discrete_num])
            for k in range(num_sources):
                marginal_0, marginal_1 = results[k]
                marginal_0 = tf.reshape(marginal_0, [discrete_num, discrete_num, discrete_num])
                marginal_1 = tf.reshape(marginal_1, [discrete_num, discrete_num, discrete_num])
                all_marginals[k, 0, :, :, :] = marginal_0.numpy()
                all_marginals[k, 1, :, :, :] = marginal_1.numpy()

        marginal_dir = conf['val_entries']['marginal']['npy_dir']
        if not os.path.exists(marginal_dir):
            os.makedirs(marginal_dir)
        prefix = conf['val_entries']['marginal']['prefix']
        if is_3d:
            prefix = prefix + '-3d'
        save_filename = '{}-{}.npy'.format(os.path.join(marginal_dir, prefix), step_int)
        np.save(save_filename, all_marginals)


    def validate_potentials(self, P, discrete_num):
        print('Validating potentials...')
        state = self.state
        conf = state.conf
        step_int = int(state.potential_step)
        num_sources = state.num_sources
        all_potentials = np.zeros(shape=[num_sources, 2, discrete_num, discrete_num])
        for i in range(num_sources):
            f0, f1 = state.prepare_potentials(i)
            all_potentials[i, 0, :, :] = np.reshape(f0(P), [discrete_num, discrete_num])
            all_potentials[i, 1, :, :] = np.reshape(f1(P), [discrete_num, discrete_num])
        potential_dir = conf['val_entries']['potential']['npy_dir']
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)
        prefix = conf['val_entries']['potential']['prefix']
        save_filename = '{}-{}.npy'.format(os.path.join(potential_dir, prefix), step_int)
        np.save(save_filename, all_potentials)


    def validate_sources(self, P, discrete_num):
        print('Validating source distributions...')
        state = self.state
        conf = state.conf
        num_sources = state.num_sources
        source_dir = conf['val_entries']['source']['npy_dir']
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        for i in range(num_sources):
            mu = state.get_source(i)
            name = state.get_name(i)
            pdfs = mu.pdf(P)
            path = '{}.npy'.format(name)
            path = os.path.join(source_dir, path)
            np.save(path, np.reshape(pdfs, [discrete_num, discrete_num]))


    def validate_potential_training(self):
        state = self.state
        conf = state.conf
        if state.conf['moving_averages']['potential_enabled']:
            state.potential_MA.swap_in_averages()

        if conf['point_dim'] == 2 and 'discrete_num' in conf:
            print('Perform validation in 2D...')
            discrete_num = conf['discrete_num']
            x0, x1, y0, y1 = conf['discrete_extent']
            x0 = tf.cast(x0, state.float_dtype)
            x1 = tf.cast(x1, state.float_dtype)
            y0 = tf.cast(y0, state.float_dtype)
            y1 = tf.cast(y1, state.float_dtype)
            x_discrete = tf.linspace(x0, x1, discrete_num)
            y_discrete = tf.linspace(y0, y1, discrete_num)
            h_x = (x1 - x0) / discrete_num
            h_y = (y1 - y0) / discrete_num
            X, Y = tf.meshgrid(x_discrete, y_discrete)
            P = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], -1) # Nx2, N = discrete_num^2


            if conf['val_entries']['source']['enabled']:
                self.validate_sources(P, discrete_num)
            if conf['val_entries']['marginal']['enabled']:
                self.validate_marginals(P, discrete_num, h_x * h_y, is_3d=False)
            if conf['val_entries']['potential']['enabled']:
                self.validate_potentials(P, discrete_num)

        if conf['point_dim'] == 3 and 'discrete_num_3d' in conf:
            discrete_num = conf['discrete_num_3d']
            x0, x1, y0, y1, z0, z1 = conf['discrete_extent_3d']
            x0 = tf.cast(x0, state.float_dtype)
            x1 = tf.cast(x1, state.float_dtype)
            y0 = tf.cast(y0, state.float_dtype)
            y1 = tf.cast(y1, state.float_dtype)
            z0 = tf.cast(z0, state.float_dtype)
            z1 = tf.cast(z1, state.float_dtype)
            x_discrete = tf.linspace(x0, x1, discrete_num)
            y_discrete = tf.linspace(y0, y1, discrete_num)
            z_discrete = tf.linspace(z0, z1, discrete_num)
            h_x = (x1 - x0) / discrete_num
            h_y = (y1 - y0) / discrete_num
            h_z = (z1 - z0) / discrete_num
            X, Y, Z = tf.meshgrid(x_discrete, y_discrete, z_discrete)
            P = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1]), tf.reshape(Z, [-1])], -1) # Nx3
            if conf['val_entries']['marginal']['enabled']:
                self.validate_marginals(P, discrete_num, h_x * h_y * h_z, is_3d=True)

        if conf['val_entries']['potential_gradient']['enabled']:
            self.validate_pushforwards('potential_gradient')
        if conf['val_entries']['barycentric_projection']['enabled']:
            self.validate_pushforwards('barycentric_projection')

        if state.conf['moving_averages']['potential_enabled']:
            state.potential_MA.swap_out_averages()

    def batch_validate_pushforward(self, T, ps, batch_size):
        pushforward_list = []
        num_remain = int(ps.shape[0])
        cur = 0
        while num_remain > 0:
            count = min(num_remain, batch_size)
            qs = T(ps[cur:cur + count, :])
            pushforward_list.append(qs.numpy())
            cur += count
            num_remain -= count
        return np.concatenate(pushforward_list, axis=0)


    def validate_pushforwards(self, kind):
        # This funciton is generic for all kinds of pushforward maps.
        assert (kind in ['map', 'potential_gradient', 'barycentric_projection'])
        state = self.state
        conf = state.conf
        if not conf['val_entries'][kind]['enabled']:
            return
        print('Validating pushforwards of kind {}...'.format(kind))
        step_int = int(state.map_step if kind == 'map' else state.potential_step)
        point_dim = conf['point_dim']
        num_samples = conf['val_entries'][kind]['num_samples']
        num_sources = state.num_sources
        results = np.zeros(shape=[num_sources, 2, num_samples, point_dim])
        for i in range(num_sources):
            ps = state.get_source(i).sample(num_samples)
            if kind == 'map':
                T = state.prepare_transport_map(i)
            elif kind == 'potential_gradient':
                T = state.prepare_potential_gradient_map(i)
            else:
                T = state.prepare_barycentric_projection(i)
            qs = self.batch_validate_pushforward(T, ps, conf['batch_size'])
            results[i, 0, :, :] = ps
            results[i, 1, :, :] = qs

        result_dir = conf['val_entries'][kind]['npy_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        prefix = conf['val_entries'][kind]['prefix']
        save_filename = '{}-{}.npy'.format(os.path.join(result_dir, prefix), step_int)
        np.save(save_filename, results)


    def validate_map_training(self):
        self.validate_pushforwards('map')

