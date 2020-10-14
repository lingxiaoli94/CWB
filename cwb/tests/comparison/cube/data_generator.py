import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import argparse
import os
import h5py

normal_scale = 0.3
normal_min = 0.2
bbox = [-2,2]
L = bbox[1] - bbox[0]

def gen_cube(dim):
    p_min = np.random.rand(dim) * L + bbox[0]
    p_max = p_min + np.random.rand(dim) * L / 2
    return (p_min, p_max)

def gen_data(dim, num_cubes, out_dir, out_prefix):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(num_cubes):
        p_min, p_max = gen_cube(dim)
        dict_pkl = {'dim': dim, 'shape': 'cube', 'p_min': p_min, 'p_max': p_max}
        fp = open('{}_{}.pkl'.format(os.path.join(out_dir, out_prefix), i), 'wb')
        pickle.dump(dict_pkl, fp)

if __name__ == '__main__':
    np.random.seed(44)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim', type=int)
    parser.add_argument('num_cubes', type=int)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('out_prefix', type=str)
    args = parser.parse_args()

    dim = args.dim
    out_dir = args.out_dir
    out_prefix = args.out_prefix
    num_cubes = args.num_cubes

    gen_data(dim, num_cubes, out_dir, out_prefix)
