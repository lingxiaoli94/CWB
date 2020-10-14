from .common import *
from . import validate

import argparse
import tensorflow as tf
import os
import subprocess
import numpy as np
import time
import pickle

def batch_run_exp(exp, method, repeat_range):
    for rep in repeat_range:
        for dim in dim_range:
            data_dir = get_data_nd_dir(dim)
            result_dir = get_result_nd_dir(dim)
            result_filename = get_result_filename(method, rep)
            start_time = time.time()
            print('Repeat #{}: running experiment {} dimension {} with method {}...'.format(rep, exp, dim, method))
            if method == 'cuturi':
                from . import cuturi_solver
                cuturi_solver.run(
                        exp,
                        dim,
                        sample_count=5000,
                        support_size=5000,
                        use_sinkhorn=True,
                        data_dir=get_data_nd_dir(dim),
                        result_dir=result_dir,
                        result_filename=result_filename)
            elif method == 'cwb':
                from . import cwb_solver
                cwb_solver.run(
                        exp,
                        dim,
                        sample_count=1000000,
                        working_dir=get_working_nd_dir('cwb', dim, rep),
                        data_dir=get_data_nd_dir(dim),
                        result_dir=result_dir,
                        result_filename=result_filename)
            elif method in ['bregman', 'exact_lp', 'conv']:
                discrete_num = get_discrete_num(dim, method)
                if not discrete_num:
                    print('Cannot use method {} on dimension {}, skipping...'.format(method, dim))
                    continue
                from . import grid_solver
                grid_solver.run(
                        exp,
                        dim,
                        method,
                        discrete_num,
                        data_dir=data_dir,
                        result_dir=result_dir,
                        result_filename=result_filename)
            elif method == 'claici':
                from . import claici_solver
                claici_solver.run(
                        exp,
                        dim,
                        data_dir=data_dir,
                        result_dir=result_dir,
                        result_filename=result_filename,
                        support_size=100,
                        internal_num_samples=5000,
                        max_iters=20)
            elif method == 'gaussian_iterative':
                if exp != 'gaussian':
                    continue
                from .gaussian import iterative_solver
                iterative_solver.run(
                        dim,
                        data_dir=get_data_nd_dir(dim),
                        result_dir=result_dir,
                        result_filename=result_filename)

            end_time = time.time()
            msg = 'Experiment {} in dimension {} with method {} takes {:4f}\n'.format(exp, dim, method, end_time - start_time)
            print(msg)

def batch_validate_exp(exp, method, repeat_range):
    for rep in repeat_range:
        for dim in dim_range:
            exp_result_file = get_result_file_path(dim, method, rep)
            print('Repeat #{}: validating experiment {} dimension {} with method {}...'.format(rep, exp, dim, method))
            validate.run(
                    dim,
                    exp,
                    method,
                    exp_result_file,
                    stats_dir=get_stats_nd_dir(dim),
                    stats_filename=get_stats_filename(method, rep))

def batch_evolve_exp(exp, method, repeat_range):
    for rep in repeat_range:
        for dim in dim_range:
            exp_result_file = get_result_file_path(dim, method, rep)
            print('Repeat #{}: evolving result of experiment {} dimension {} with method {}...'.format(rep, exp, dim, method))
            validate.evolve(
                    dim,
                    exp,
                    method,
                    exp_result_file,
                    evolve_dir=get_evolve_nd_dir(dim),
                    evolve_filename=get_evolve_filename(method, rep))

def batch_gen_data(exp):
    for dim in dim_range:
        if exp == 'gaussian':
            from .gaussian import data_generator
            data_generator.gen_data(
                    dim,
                    g_num_gaussians,
                    get_data_nd_dir(dim),
                    exp)
        elif exp == 'mixture':
            from .mixture import data_generator
            data_generator.gen_data(
                    dim,
                    g_num_mixtures,
                    g_num_components,
                    get_data_nd_dir(dim),
                    exp)
        elif exp == 'cube':
            from .cube import data_generator
            data_generator.gen_data(
                    dim,
                    g_num_cubes,
                    get_data_nd_dir(dim),
                    exp)
        elif exp == 'poisson':
            from .poisson import data_generator
            data_generator.gen_data(
                    dim,
                    get_data_nd_dir(dim),
                    exp)
        else:
            raise Exception('Unknown experiment: {}'.format(exp))

def batch_adapt_data_to_h5(exp):
    for dim in dim_range:
        if exp == 'gaussian':
            from .gaussian import data_generator
            data_generator.adapt_to_h5(get_data_nd_dir(dim))
        elif exp == 'poisson':
            from .poisson import data_generator
            data_generator.adapt_to_h5(get_data_nd_dir(dim))


if __name__ == '__main__':
    np.random.seed(44)
    tf.random.set_seed(44)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu_devices) > 0:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--dims', nargs='+', type=int, required=True)
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--adapt_h5', action='store_true')
    parser.add_argument('--run', type=str)
    parser.add_argument('--validate', type=str)
    parser.add_argument('--evolve', type=str)
    parser.add_argument('--repeat_start', type=int, default=0)
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--reseed', action='store_true')

    args = parser.parse_args()

    if args.reseed:
        t = int(time.time() * 1000.0) & 0xffffffff
        np.random.seed(t)
        tf.random.set_seed(t)

    exp = args.exp
    repeat_start = args.repeat_start
    repeat_times = args.repeat_times
    repeat_range = range(repeat_start, repeat_start + repeat_times)

    if args.dims:
        dim_range = args.dims

    if args.gen_data:
        batch_gen_data(exp)

    if args.adapt_h5:
        batch_adapt_data_to_h5(exp)

    if args.run is not None:
        method = args.run
        batch_run_exp(exp, method, repeat_range)

    if args.validate is not None:
        method = args.validate
        batch_validate_exp(exp, method, repeat_range)

    if args.evolve is not None:
        method = args.evolve
        batch_evolve_exp(exp, method, repeat_range)
