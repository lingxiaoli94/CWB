import cwb.templates

import numpy as np
import yaml
import os
import scipy.linalg
import pickle
import argparse
import subprocess

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

g_mixture_template_name = 'mixture_template.yaml'
g_gaussian_template_name = 'gaussian_template.yaml'
g_cube_template_name = 'cube_template.yaml'
g_poisson_template_name = 'poisson_template.yaml'

g_sample_dir = 'barycenter_dir'
g_sample_npy = 'barycenter.npy'
g_conf_filename = 'config.yaml'

def get_template_name(exp):
    if exp == 'gaussian':
        return g_gaussian_template_name
    elif exp == 'mixture':
        return g_mixture_template_name
    elif exp == 'cube':
        return g_cube_template_name
    elif exp == 'poisson':
        return g_poisson_template_name
    raise Exception('No template for experiment {}!'.format(exp))

def run(exp, dim, sample_count, working_dir, data_dir, result_dir, result_filename):
    original_working_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    with pkg_resources.path(cwb.templates, get_template_name(exp)) as template_file:
        conf = yaml.safe_load(open(template_file))
    conf['point_dim'] = dim

    source_list = []
    data_files = sorted([p for p in os.listdir(data_dir) if p.endswith('.pkl')])
    num_sources = len(data_files)
    weight = 1 / num_sources
    for i, p in enumerate(data_files):
        name = 'source{}'.format(i)
        source_list.append({
            'name': name,
            'pkl_path': os.path.abspath(os.path.join(data_dir, p)),
            'weight': weight})

    conf['distribution_list'] = source_list
    conf['test'] = {
            'sample_barycenter': {
                    'num_samples': sample_count,
                    'npy_dir': g_sample_dir,
                    'npy_file': g_sample_npy,
                    'vis_name': 'barycenter.png',
                    'vis_dir': 'barycenter_vis_dir'
                }
            }

    open(os.path.join(working_dir, g_conf_filename), 'w').write(yaml.dump(conf, indent=4))

    os.chdir(working_dir)
    subprocess.run(['python -m cwb.barycenter --train --test --reseed {}'.format(g_conf_filename)], shell=True)
    os.chdir(original_working_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    subprocess.run(['cp {} {}'.format(
        os.path.join(working_dir, g_sample_dir, g_sample_npy),
        os.path.join(result_dir, result_filename))], shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('dim', type=int)
    parser.add_argument('sample_count', type=int)
    parser.add_argument('working_dir', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('result_dir', type=str)
    parser.add_argument('result_filename', type=str)
    args = parser.parse_args()

    exp = args.exp
    dim = args.dim
    sample_count = args.sample_count
    working_dir = args.working_dir
    data_dir = args.data_dir
    result_dir = args.result_dir
    result_filename = args.result_filename

    run(exp, dim, sample_count, working_dir, data_dir, result_dir, result_filename)
