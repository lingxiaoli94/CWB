from .common import \
        load_result_by_method, \
        load_source_list, \
        make_uniform_samples, \
        get_result_file_path, \
        get_vis_nd_dir, \
        get_vis_file_path, \
        get_data_nd_dir

import numpy as np
import ot
import os
import pickle
import tensorflow as tf
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def run(exp, dim, method, num_samples):
    if dim != 2:
        print('Visualization only works in 2D!')
        return
    exp_result_file = get_result_file_path(dim, method)
    data_dir = get_data_nd_dir(dim)
    source_list = load_source_list(data_dir)
    num_sources = len(source_list)
    exp_hist, exp_samples, exp_metadata = load_result_by_method(exp_result_file, method)
    print('Experiment {} has {} samples.'.format(exp, exp_samples.shape[0]))
    is_grid = 'width' in exp_metadata
    if is_grid:
        grid_width = exp_metadata['width']
        grid_height = exp_metadata['height']
    fig, axes = plt.subplots(nrows=num_sources + 1 + (1 if is_grid else 0), ncols=1, figsize=(15, 15))
    for r, ax in enumerate(axes):
        if r < num_sources:
            samples = source_list[r].sample(num_samples)
        elif r == num_sources:
            samples = make_uniform_samples(exp_hist, exp_samples, num_samples)

        if r <= num_sources:
            ax.scatter(samples[:, 0], samples[:, 1], s=0.5)
        else:
            assert(is_grid)
            XX = np.reshape(exp_samples[:, 0], [grid_width, grid_height])
            YY = np.reshape(exp_samples[:, 1], [grid_width, grid_height])
            ZZ = np.reshape(exp_hist, [grid_width, grid_height])
            cs = ax.contourf(XX, YY, ZZ)
            fig.colorbar(cs, ax=ax)
        ax.set_aspect('equal')
        ax.grid(True)

    vis_dir = get_vis_nd_dir(dim)
    vis_file = get_vis_file_path(dim, method)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    fig.savefig(vis_file)
    plt.close()


if __name__ == '__main__':
    np.random.seed(44)
    tf.random.set_seed(44)
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('dim', type=int)
    parser.add_argument('method', type=str)
    parser.add_argument('num_samples', type=int)
    args = parser.parse_args()

    exp = args.exp
    dim = args.dim
    method = args.method
    num_samples = args.num_samples

    run(exp, dim, method, num_samples)
