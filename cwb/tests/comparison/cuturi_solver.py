from .common import load_source_list

import tensorflow as tf
import numpy as np
import ot
import os
import pickle
import argparse

g_centering_trick = False # doesn't change results much
g_sinkhorn_reg = 0.1

def free_support_barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, numItermax=100, stopThr=1e-7, use_sinkhorn=False):
    iter_count = 0
    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = np.ones((k,)) / k
    if weights is None:
        weights = np.ones((N,)) / N

    X = X_init

    log_dict = {}
    displacement_square_norm = stopThr + 1.
    while (displacement_square_norm > stopThr and iter_count < numItermax):
        T_sum = np.zeros((k, d))
        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights.tolist()):
            M_i = ot.dist(X, measure_locations_i)
            if use_sinkhorn:
                T_i = ot.bregman.sinkhorn(b, measure_weights_i, M_i, g_sinkhorn_reg)
            else:
                T_i = ot.emd(b, measure_weights_i, M_i)
            T_sum = T_sum + weight_i * np.reshape(1. / b, (-1, 1)) * np.matmul(T_i, measure_locations_i)

        displacement_square_norm = np.sum(np.square(T_sum - X))

        X = T_sum
        print('iteration %d, displacement_square_norm=%f\n', iter_count, displacement_square_norm)

        iter_count += 1

    return X

def run(exp, dim, sample_count, support_size, use_sinkhorn, data_dir, result_dir, result_filename):
    source_list = load_source_list(data_dir)
    measure_samples = []
    measure_weights = []
    mean_list = []
    barycenter_mean = 0
    num_sources = len(source_list)
    for i, nu in enumerate(source_list):
        samples = nu.sample(sample_count).numpy()
        if g_centering_trick:
            cur_mean = np.mean(samples)
            mean_list.append(cur_mean)
            samples -= cur_mean
            barycenter_mean += cur_mean / num_sources
        measure_samples.append(samples)
        measure_weights.append(np.ones([sample_count]) / sample_count)
    n = len(measure_samples)

    # X_init = np.zeros([support_size, dim])
    X_init = []
    for i in range(num_sources):
        X_init.append(measure_samples[i][:support_size // num_sources])
    X_init.append(np.zeros([support_size % num_sources, dim]))
    X_init = np.concatenate(X_init, axis=0)

    # assume all input measures are equally weighted
    wb = free_support_barycenter(
            measure_samples,
            measure_weights,
            X_init,
            use_sinkhorn=use_sinkhorn)

    if g_centering_trick:
        wb += barycenter_mean

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    np.save(open(os.path.join(result_dir, result_filename), 'wb'), wb)


