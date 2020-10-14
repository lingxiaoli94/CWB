import cwb.data.poisson

import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import argparse
import os
import h5py
import re
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

g_subset_count = 5
g_subset_sample_filename_fmt = 'samples_0_{}.npy'
g_full_sample_filename = 'samples_0_all.npy'
g_data_package = cwb.data.poisson

def extract_from_posterior(dim, posterior_samples):
    # skip intercept, and just extract coefficients
    return posterior_samples[:, 1:dim+1]

def fetch_full_samples(dim):
    fp = pkg_resources.open_binary(g_data_package, g_full_sample_filename)
    posterior_samples = np.load(fp)
    return extract_from_posterior(dim, posterior_samples)


def gen_data(dim, out_dir, out_prefix):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(g_subset_count):
        fp = pkg_resources.open_binary(g_data_package, g_subset_sample_filename_fmt.format(i))
        posterior_samples = np.load(fp)
        posterior_samples = extract_from_posterior(dim, posterior_samples)
        npy_path = '{}_{}.npy'.format(os.path.join(out_dir, out_prefix), i)
        pkl_path = '{}_{}.pkl'.format(os.path.join(out_dir, out_prefix), i)
        np.save(npy_path, posterior_samples)
        dict_pkl = {
                'dim': dim,
                'shape':
                'empirical_npy',
                'npy_path': os.path.abspath(npy_path)}
        fp = open(pkl_path, 'wb')
        pickle.dump(dict_pkl, fp)

def adapt_to_h5(data_dir):
    pkls = sorted([p for p in os.listdir(data_dir) if p.endswith('.pkl')])
    for p in pkls:
        pickle_file = os.path.join(data_dir, p)
        dict_pkl = pickle.load(open(pickle_file, 'rb'))
        h5_file = os.path.splitext(pickle_file)[0] + '.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('data', data=np.transpose(np.load(dict_pkl['npy_path'])))

if __name__ == '__main__':
    np.random.seed(44)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim', type=int)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('out_prefix', type=str)
    args = parser.parse_args()

    dim = args.dim
    out_dir = args.out_dir
    out_prefix = args.out_prefix

    gen_data(dim, out_dir, out_prefix)
