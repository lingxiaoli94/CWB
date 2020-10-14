import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import argparse
import os

g_use_anchor = True

normal_scale = 0.16
bbox = [-1,1]
L = bbox[1] - bbox[0]
max_cond = 50
min_cond = 1
min_dist_tolerance = 0.7
max_dist_tolerance = 10.0
anchor_offset_lim = 1.5
mixture_offset_lim = 2.0

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

def gen_arbitrary_mixture(num_components, dim):
    while True:
        mixture = []
        for i in range(num_components):
            mean, normal_A = gen_gaussian(dim)
            mixture.append({'mean': mean, 'normal_A': normal_A})
        min_dist = 1e100
        max_dist = 0
        for i in range(num_components):
            for j in range(i+1, num_components):
                tmp = np.linalg.norm(mixture[i]['mean'] - mixture[j]['mean'])
                min_dist = min(min_dist, tmp)
                max_dist = max(max_dist, tmp)
        if min_dist_tolerance < min_dist and max_dist < max_dist_tolerance:
            break
    return mixture

def gen_anchors(num_components, dim):
    while True:
        anchors = []
        for i in range(num_components):
            anchors.append(np.random.rand(dim) * L + bbox[0])

        min_dist = 1e100
        for i in range(num_components):
            for j in range(i+1, num_components):
                tmp = np.linalg.norm(anchors[i] - anchors[j])
                min_dist = min(min_dist, tmp)
        if min_dist_tolerance < min_dist:
            break
    return anchors


def gen_anchored_mixture(num_components, dim, anchors):
    mixture_offset = (np.random.rand(dim) - 0.5) * mixture_offset_lim
    mixture = []
    for i in range(num_components):
        anchor = anchors[i]
        _, normal_A = gen_gaussian(dim)
        offset = (np.random.rand(dim) - 0.5) * anchor_offset_lim
        mixture.append({'mean': anchor + offset + mixture_offset, 'normal_A': normal_A})
    return mixture


def gen_data(dim, num_mixtures, num_components, out_dir, out_prefix):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    mixture_weights = np.full([num_components], 1.0 / num_components)
    if g_use_anchor:
        anchors = gen_anchors(num_components, dim)
    for i in range(num_mixtures):
        if g_use_anchor:
            mixture = gen_anchored_mixture(num_components, dim, anchors)
        else:
            mixture = gen_mixture(num_components, dim)
        for j in range(len(mixture)):
            mixture[j]['dim'] = dim
            mixture[j]['shape'] = 'gaussian'
        dict_pkl = {'dim': dim, 'shape': 'mixture', 'nu_list': mixture, 'weight_list': mixture_weights}
        fp = open('{}_{}.pkl'.format(os.path.join(out_dir, out_prefix), i), 'wb')
        pickle.dump(dict_pkl, fp)


if __name__ == '__main__':
    np.random.seed(44)

    parser = argparse.ArgumentParser()
    parser.add_argument('dim', type=int)
    parser.add_argument('num_mixtures', type=int)
    parser.add_argument('num_components', type=int)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('out_prefix', type=str)
    args = parser.parse_args()

    dim = args.dim
    out_dir = args.out_dir
    out_prefix = args.out_prefix
    num_mixtures = args.num_mixtures
    num_components = args.num_components

    gen_data(dim, num_mixtures, num_components, out_dir, out_prefix)
