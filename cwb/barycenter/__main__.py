from .barycenter_state import BarycenterState

import numpy as np
import tensorflow as tf
import os
import yaml
import random
import argparse
import time

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--reseed', action='store_true')
    parser.add_argument('--mcmc', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('yaml', type=str)
    args = parser.parse_args()

    conf = yaml.safe_load(open(args.yaml).read())

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu_devices) > 0:
        gpu_id = conf.get('GPU', 0)
        if args.gpu is not None:
            gpu_id = args.gpu
        try:
            tf.config.experimental.set_visible_devices(gpu_devices[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpu_devices[gpu_id], True)
            print('Using GPU #{}'.format(gpu_id))
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)
    else:
        print('No GPU detected - fallback to CPU.')

    if args.reseed:
        tf.random.set_seed(time.time())
        np.random.seed(int(time.time()))
        random.seed(time.time())


    if args.train or args.test:
        state = BarycenterState(conf)

        if args.train:
            state.train_potentials()
            if conf['estimate_map']:
                state.train_transport_maps()

        if args.test:
            test_conf = conf['test']
            if 'sample_barycenter' in test_conf:
                sample_wb_desc = test_conf['sample_barycenter']
                result = state.sample_barycenter(sample_wb_desc['num_samples'])
                if not os.path.exists(sample_wb_desc['npy_dir']):
                    os.makedirs(sample_wb_desc['npy_dir'])
                np.save(os.path.join(sample_wb_desc['npy_dir'], sample_wb_desc['npy_file']), result)
            if 'sample_plans_mcmc' in test_conf:
                ent = conf['test']['sample_plans_mcmc']
                if args.mcmc:
                    state.sample_plans_mcmc(ent)

    if args.visualize:
        assert conf['point_dim'] == 2, 'can only visualize in 2D!'

        from .visualize import \
                visualize_sources, \
                visualize_barycenter_samples, \
                visualize_mcmc_samples, \
                create_vis_fn_from_name, \
                visualize_folder
        if 'discrete_extent' in conf:
            for name, ent in conf['val_entries'].items():
                if ent['enabled']:
                    if not os.path.exists(ent['npy_dir']):
                        continue
                    if name == 'source':
                        visualize_sources(ent['npy_dir'], ent['vis_dir'], ent['vis_name'], conf)
                    else:
                        vis_fn = create_vis_fn_from_name(name)
                        visualize_folder(ent['npy_dir'], ent['vis_dir'], vis_fn, conf)

        test_conf = conf['test']

        if 'sample_barycenter' in test_conf:
            ent = test_conf['sample_barycenter']
            visualize_barycenter_samples(ent['npy_dir'], ent['npy_file'], ent['vis_dir'], ent['vis_name'], conf)
        if 'sample_plans_mcmc' in test_conf:
            ent = test_conf['sample_plans_mcmc']
            if args.mcmc:
                visualize_mcmc_samples(np.load(os.path.join(ent['npy_dir'], ent['npy_file'])), ent['vis_dir'], ent['vis_name'], conf)



