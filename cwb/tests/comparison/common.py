import os
import numpy as np
import tensorflow as tf
from cwb.common.config_parser import construct_single_distribution

g_num_gaussians = 5

g_num_mixtures = 3
g_num_components = 3

g_num_cubes = 3

g_data_dir = 'data/'
g_result_dir = 'result/'
g_stats_dir = 'stats/'
g_evolve_dir = 'evolve/'
g_vis_dir = 'vis/'
g_cwb_working_dir = 'cwb_tmp/'
g_claici_working_dir = 'claici_tmp/'

g_result_filename_dict = {
        'cwb': 'cwb.npz',
        'cuturi': 'cuturi.pkl',
        'claici': 'claici.npy',
        'bregman': 'bregman.npz',
        'exact_lp': 'exact_lp.npz',
        'conv': 'conv.npz',
        'staib': 'staib.npz',
        'gaussian_iterative': 'gaussian_iterative.pkl'}

def get_discrete_num(dim, method):
    if method == 'conv':
        return 100000
    else:
        return 1000
    return None


def get_result_filename(method, rep):
    return '{:02}-{}'.format(rep, g_result_filename_dict[method])

def get_stats_filename(method, rep):
    return '{:02}-{}'.format(rep, method + '.pkl')

def get_evolve_filename(method, rep):
    return '{:02}-{}'.format(rep, method + '.pkl')

def get_data_nd_dir(dim):
    return os.path.join(g_data_dir, '{}d'.format(dim))

def get_result_nd_dir(dim):
    return os.path.join(g_result_dir, '{}d'.format(dim))

def get_stats_nd_dir(dim):
    return os.path.join(g_stats_dir, '{}d'.format(dim))

def get_evolve_nd_dir(dim):
    return os.path.join(g_evolve_dir, '{}d'.format(dim))

def get_working_nd_dir(kind, dim, repeat):
    if kind == 'cwb':
        return os.path.join(g_cwb_working_dir, '{}d-{:04}'.format(dim, repeat))
    elif kind == 'claici':
        return os.path.join(g_claici_working_dir, '{}d-{:04}'.format(dim, repeat))
    return None

def get_vis_nd_dir(dim):
    return os.path.join(g_vis_dir, '{}d'.format(dim))

def get_result_file_path(dim, method, rep):
    return os.path.join(get_result_nd_dir(dim), get_result_filename(method, rep))

def get_stats_file_path(dim, method, rep):
    return os.path.join(get_stats_nd_dir(dim), get_stats_filename(method, rep))

def get_vis_file_path(dim, method):
    return os.path.join(get_vis_nd_dir(dim), method + '.png')

def make_uniform_hist(n):
    return np.ones([n]) / n

def load_result_by_method(exp_result_file, method):
    metadata = {}
    if method == 'cwb':
        result = np.load(exp_result_file)
        samples = result
        samples_count = samples.shape[0]
        hist = make_uniform_hist(samples_count)
    elif method in ['cuturi', 'claici']:
        samples = np.load(exp_result_file)
        samples_count = samples.shape[0]
        hist = make_uniform_hist(samples_count)
    elif method in ['bregman', 'exact_lp', 'conv', 'staib']:
        result = np.load(exp_result_file)
        samples = result['discrete_points']
        hist = result['hist']
        if 'width' in result:
            metadata['width'] = result['width']
            metadata['height'] = result['height']
    else:
        raise Exception('Unknown method: {}'.format(method))

    return (hist, samples, metadata)

def load_source_list(data_dir):
    source_list = []
    file_list = sorted(os.listdir(data_dir)) # consistent order
    for p in file_list:
        if p.endswith('.pkl'):
            pkl_path = os.path.join(data_dir, p)
            nu = construct_single_distribution({'pkl_path': pkl_path}, tf.float32)
            source_list.append(nu)
    return source_list

def check_is_uniform(hist):
    return np.all(hist == hist[0])


def make_uniform_samples(hist, supp, sample_count):
    n = supp.shape[0]
    if check_is_uniform(hist):
        if n < sample_count:
            return supp
        inds = np.random.choice(n, size=[sample_count], replace=False)
        return supp[inds, :]
    replace = False if sample_count <= n else True
    inds = np.random.choice(n, size=[sample_count], replace=replace, p=hist)
    return supp[inds, :]

