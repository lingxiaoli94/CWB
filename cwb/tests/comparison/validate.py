from .common import \
        make_uniform_hist, \
        get_result_nd_dir, \
        get_data_nd_dir, \
        get_stats_nd_dir, \
        get_result_file_path, \
        load_result_by_method, \
        get_result_filename, \
        make_uniform_samples
from cwb.common.config_parser import construct_single_distribution
from cwb.ot import emd2 as cwb_emd2

import tensorflow as tf
import numpy as np
import ot
import math
import pickle
import argparse
import os

g_lp_emd_limit = 5000
g_stochastic_limit = 1000000
g_enable_stochastic_objective = False
g_enable_lp_objective = False

def fit_gaussian(samples, hist):
    # samples - NxD, hist: N
    n, d = samples.shape
    mean = np.sum(samples * np.expand_dims(hist, 1), axis=0)
    p = samples - np.expand_dims(mean, 0)
    cov = np.zeros([d, d])
    for i in range(n):
        cov += hist[i] * np.outer(p[i, :], p[i, :])
    return mean, cov

def compare_gaussian_with_empirical(gt_mean, gt_cov, exp_hist, exp_samples):
    exp_mean, exp_cov = fit_gaussian(exp_samples, exp_hist)
    mean_diff = np.linalg.norm(gt_mean - exp_mean)
    cov_diff = np.linalg.norm(gt_cov - exp_cov)
    return {'mean_diff': mean_diff, 'cov_diff': cov_diff}

def compare_emd(gt_hist, gt_samples, exp_hist, exp_samples):
    dist = ot.dist(gt_samples, exp_samples)
    W2 = ot.emd2(gt_hist, exp_hist, dist, numItermax=int(10e6))
    W2 = math.sqrt(W2)
    return W2

def sparsify_samples(hist, samples):
    if samples.shape[0] <= g_lp_emd_limit:
        return (hist, samples)

    # if requires sparsify, then hist is usually uniform anyway
    uniform_samples = make_uniform_samples(hist, samples, g_lp_emd_limit)
    uniform_hist = np.full([uniform_samples.shape[0]], 1 / uniform_samples.shape[0])

    return uniform_hist, uniform_samples

def evolve(dim, exp, method, exp_result_file, evolve_dir, evolve_filename, num_evolve=15, upper_lim=15000):
    # two types of evolvement comparison:
    # a) W2_lp b) cov

    params = load_gt_as_params(dim, exp)
    exp_hist, exp_samples, _ = load_result_by_method(exp_result_file, method)
    gt_samples = params['gt_samples']
    gt_hist = params['gt_hist']
    assert(np.all(exp_hist == exp_hist[0])) # uniform
    assert(np.all(gt_hist == gt_hist[0])) # uniform

    if 'gt_gaussian' in params:
        gt_gaussian = params['gt_gaussian']
        gt_mean = gt_gaussian['mean']
        gt_normal_A = gt_gaussian['normal_A']
        gt_cov = np.matmul(gt_normal_A, np.transpose(gt_normal_A))
    else:
        gt_mean, gt_cov = fit_gaussian(gt_samples, gt_hist)

    num_samples = exp_samples.shape[0]
    inds = np.random.permutation(num_samples)
    exp_samples = exp_samples[inds, :]
    print('num samples: {} to evolve for {} times with upper limit {}'.format(num_samples, num_evolve, upper_lim))
    # gt_hist, gt_samples = sparsify_samples(gt_hist, gt_samples)
    gt_inds = np.random.permutation(gt_samples.shape[0])
    gt_hist = gt_hist[gt_inds]
    gt_samples = gt_samples[gt_inds, :]

    batch = upper_lim // num_evolve
    result_W2 = []
    result_cov = []
    for i in range(num_evolve):
        print('evolve itr {}'.format(i))
        count = (i+1) * batch
        samples = exp_samples[:count]
        hist = make_uniform_hist(count)
        W2 = compare_emd(hist, gt_samples[:count, :], hist, samples)
        result_W2.append(W2)
        print('evolve itr {} W2: {:6e}'.format(i, W2))
        diff = compare_gaussian_with_empirical(gt_mean, gt_cov, hist, samples)
        result_cov.append(diff)
    result = {'W2_lp': result_W2, 'param_diff': result_cov}
    result['batch'] = batch

    if not os.path.exists(evolve_dir):
        os.makedirs(evolve_dir)
    evolve_file_path = os.path.join(evolve_dir, evolve_filename)
    pickle.dump(result, open(evolve_file_path, 'wb'))


def compute_validation_stats(dim, exp_hist, exp_samples, params):
    print('exp_samples: {}'.format(exp_samples[:5, :]))
    data_dir = params['data_dir']
    exp_hist_sparse, exp_samples_sparse = sparsify_samples(exp_hist, exp_samples)
    stats = {}
    if 'gt_gaussian' in params:
        gt_gaussian = params['gt_gaussian']
        gt_normal_A = gt_gaussian['normal_A']
        gt_cov = np.matmul(gt_normal_A, np.transpose(gt_normal_A))
        tmp = compare_gaussian_with_empirical(
                gt_gaussian['mean'], gt_cov, exp_hist, exp_samples)
        stats['fit_gaussian_mean_loss'] = tmp['mean_diff']
        stats['fit_gaussian_cov_loss'] = tmp['cov_diff']

    if 'gt_hist' in params and 'gt_samples' in params:
        gt_hist = params['gt_hist']
        gt_samples = params['gt_samples']
        # print('gt_samples count: {}'.format(gt_samples.shape[0]))
        # print('exp_hist_sparse: {}'.format(exp_hist_sparse))
        # print('exp_samples_sparse: {}'.format(exp_samples_sparse))
        W2_lp = compare_emd(
                *sparsify_samples(gt_hist, gt_samples),
                exp_hist_sparse, exp_samples_sparse)
        stats['W2_lp'] = W2_lp

        # moment matching
        gt_mean, gt_cov = fit_gaussian(gt_samples, gt_hist)
        tmp = compare_gaussian_with_empirical(gt_mean, gt_cov, exp_hist, exp_samples)
        stats['mm_mean_loss'] = tmp['mean_diff']
        stats['mm_cov_loss'] = tmp['cov_diff']

    # compute barycenter objective using source distributions
    source_list = []
    source_samples = [] # samples are only for computing lp barycenter objective
    for p in os.listdir(data_dir):
        if p.endswith('.pkl'):
            pkl_path = os.path.join(data_dir, p)
            source_desc = {'pkl_path': os.path.abspath(pkl_path)}
            source_list.append(source_desc)
            nu = construct_single_distribution(source_desc, tf.float32)
            source_samples.append(nu.sample(g_lp_emd_limit))
    weight = 1 / len(source_list) # assuming equal weights here


    if g_enable_stochastic_objective:
        barycenter_obj_stochastic = 0
        exp_samples_uniform = make_uniform_samples(exp_hist, exp_samples, g_stochastic_limit)
        for i, source in enumerate(source_list):
           barycenter_obj_stochastic += weight * cwb_emd2(dim, source, exp_samples_uniform)
        stats['barycenter_obj_stochastic'] = barycenter_obj_stochastic

    if g_enable_lp_objective:
        barycenter_obj_lp = 0
        for i, source in enumerate(source_list):
            lp_dist = ot.dist(source_samples[i], exp_samples_sparse)
            source_samples_count = source_samples[i].shape[0]
            source_hist = np.full([source_samples_count], 1 / source_samples_count)
            barycenter_obj_lp += weight * ot.emd2(source_hist, exp_hist_sparse, lp_dist, numItermax=int(10e7))
        stats['barycenter_obj_lp'] = barycenter_obj_lp


    return stats


def load_gt_as_params(dim, exp):
    result_dir = get_result_nd_dir(dim)
    data_dir = get_data_nd_dir(dim)
    params = {'data_dir': data_dir}

    if exp == 'gaussian':
        # we have ground truth information only for Gaussian case
        # assume the rep is 0
        gt_result_file = os.path.join(result_dir, get_result_filename('gaussian_iterative', 0))
        gt_gaussian = pickle.load(open(gt_result_file, 'rb'))
        gt_mean = gt_gaussian['mean']
        gt_normal_A = gt_gaussian['normal_A']
        gt_cov = np.matmul(gt_normal_A, np.transpose(gt_normal_A))
        # gt_samples is only for lp emd
        gt_samples = np.random.multivariate_normal(
                gt_mean, gt_cov, size=g_lp_emd_limit)
        gt_hist = make_uniform_hist(g_lp_emd_limit)
        params['gt_hist'] = gt_hist
        params['gt_samples'] = gt_samples
        params['gt_gaussian'] = gt_gaussian
    elif exp == 'poisson':
        from .poisson.data_generator import fetch_full_samples
        gt_samples = fetch_full_samples(dim)
        gt_hist = make_uniform_hist(gt_samples.shape[0])
        params['gt_samples'] = gt_samples
        params['gt_hist'] = gt_hist

    return params


def run(dim, exp, method, exp_result_file, stats_dir, stats_filename):

    if not os.path.exists(exp_result_file):
        print('Cannot validate method {} in dimension {} for experiment {}: result file does not exist.'.format(method, dim, exp))
        return

    params = load_gt_as_params(dim, exp)
    hist, samples, _ = load_result_by_method(exp_result_file, method)
    stats = compute_validation_stats(dim, hist, samples, params=params)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    stats_file_path = os.path.join(stats_dir, stats_filename)
    pickle.dump(stats, open(stats_file_path, 'wb'))

if __name__ == '__main__':
    # only used to compute evolving statistics
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim', type=int)
    parser.add_argument('exp', type=str)
    parser.add_argument('method', type=str)
    parser.add_argument('result_file', type=str)
    parser.add_argument('evolve_dir', type=str)
    parser.add_argument('evolve_filename', type=str)
    args = parser.parse_args()
    evolve(args.dim, args.exp, args.method, args.result_file, args.evolve_dir, args.evolve_filename)
