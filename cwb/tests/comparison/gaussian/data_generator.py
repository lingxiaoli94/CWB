import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import argparse
import os
import h5py

normal_scale = 0.3
normal_min = 0.2
bbox = [-1,1]
L = bbox[1] - bbox[0]
max_cond = 80
min_cond = 2

def check(mean, normal_A):
    cov = np.matmul(normal_A, np.transpose(normal_A))
    cond = np.linalg.cond(cov)
    return min_cond <= cond <= max_cond

def gen_gaussian(dim):
    while True:
        mean = np.random.rand(dim) * L + bbox[0]
        normal_A = (np.random.rand(dim, dim) - 0.5) * L * normal_scale
        if check(mean, normal_A):
            break
    return (mean, normal_A)

def gen_data(dim, num_gaussians, out_dir, out_prefix):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(num_gaussians):
        mean, normal_A = gen_gaussian(dim)
        dict_pkl = {'dim': dim, 'shape': 'gaussian', 'mean': mean, 'normal_A': normal_A}
        fp = open('{}_{}.pkl'.format(os.path.join(out_dir, out_prefix), i), 'wb')
        pickle.dump(dict_pkl, fp)

def adapt_to_h5(data_dir):
    pkls = sorted([p for p in os.listdir(data_dir) if p.endswith('.pkl')])
    for p in pkls:
        pickle_file = os.path.join(data_dir, p)
        dict_pkl = pickle.load(open(pickle_file, 'rb'))
        h5_file = os.path.splitext(pickle_file)[0] + '.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('dim', data=dict_pkl['dim'])
            f.create_dataset('mean', data=dict_pkl['mean'])
            f.create_dataset('normal_A', data=dict_pkl['normal_A'])


if __name__ == '__main__':
    np.random.seed(44)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim', type=int)
    parser.add_argument('num_gaussians', type=int)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('out_prefix', type=str)
    args = parser.parse_args()

    dim = args.dim
    out_dir = args.out_dir
    out_prefix = args.out_prefix
    num_gaussians = args.num_gaussians

    gen_data(dim, num_gaussians, out_dir, out_prefix)
