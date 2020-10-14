from .common import load_source_list

import math
import numpy as np
import ot
import os
import pickle
import tensorflow as tf
import argparse

g_conv_reg = 1e-2
g_bregman_reg = 1e-2
g_hist_eps = 1e-6
g_infer_sample_count = 100000

def recur_gen_points(i, cur, discrete_axes, discrete_points):
    if i == len(discrete_axes):
        discrete_points.append(np.array(cur))
        return
    for j in range(discrete_axes[i].shape[0]):
        cur.append(discrete_axes[i][j])
        recur_gen_points(i+1, cur, discrete_axes, discrete_points)
        del cur[-1]


def run(exp, dim, method, discrete_num, data_dir, result_dir, result_filename):
    # discrete_num is roughly the total number of grid points
    nu_list = load_source_list(data_dir)
    num_sources = len(nu_list)

    discrete_axes = []
    p_min = np.ones([dim]) * 1e100
    p_max = np.ones([dim]) * -1e100
    for nu in nu_list:
        samples = nu.sample(g_infer_sample_count).numpy()
        for i in range(samples.shape[0]):
            p_min = np.minimum(p_min, samples[i])
            p_max = np.maximum(p_max, samples[i])

    vol = np.prod(p_max - p_min)
    cell_size = np.power(vol / discrete_num, 1 / dim)
    for i in range(dim):
        count = int(math.ceil((p_max[i] - p_min[i]) / cell_size))
        discrete_axes.append(np.linspace(p_min[i], p_max[i], count))

    discrete_points = []
    recur_gen_points(0, [], discrete_axes, discrete_points)
    n_points = len(discrete_points)
    print('Total number of grid points: {} '.format(n_points))
    discrete_points = np.array(discrete_points)
    discrete_points_tf = tf.cast(discrete_points, tf.float32)

    A = []
    for i in range(num_sources):
        nu = nu_list[i]
        density = nu.pdf(discrete_points_tf).numpy()
        A.append(density / density.sum())
    if method == 'conv':
        if dim != 2:
            print('Method conv can only run in 2D!')
            return
        A = np.stack(A, axis=0)
        A = np.reshape(A, [-1, discrete_axes[0].shape[0], discrete_axes[1].shape[0]])
        hist = ot.bregman.convolutional_barycenter2d(A, g_conv_reg, verbose=True, numItermax=500)
        hist = np.reshape(hist, [-1])
    else:
        A = np.stack(A, axis=1)
        M = ot.dist(discrete_points, discrete_points)

        if method == 'bregman':
            hist = ot.bregman.barycenter(A, M, g_bregman_reg, method='sinkhorn_stabilized', verbose=True)
        elif method == 'exact_lp':
            hist = ot.lp.barycenter(A, M, verbose=True)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    kwargs = {
            'discrete_points': discrete_points,
            'hist': hist}

    # width and height for 2D image visualization
    if dim == 2:
        kwargs['width'] = discrete_axes[0].shape[0]
        kwargs['height'] = discrete_axes[1].shape[0]

    np.savez(open(os.path.join(result_dir, result_filename), 'wb'), **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('dim', type=int)
    parser.add_argument('method', type=str)
    parser.add_argument('discrete_num', type=int)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('result_dir', type=str)
    parser.add_argument('result_filename', type=str)
    args = parser.parse_args()

    exp = args.exp
    dim = args.dim
    method = args.method
    discrete_num = args.discrete_num
    data_dir = args.data_dir
    result_dir = args.result_dir
    result_filename = args.result_filename

    run(exp, dim, method, discrete_num, data_dir, result_dir, result_filename)
