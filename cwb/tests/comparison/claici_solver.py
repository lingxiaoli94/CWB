import numpy as np
import yaml
import os
import scipy.linalg
import pickle
import argparse
import subprocess
import re
import h5py

g_claici_program = 'claici_barycenter'

def run(exp, dim, data_dir, result_dir, result_filename, support_size, internal_num_samples, max_iters):
    h5_file_path = os.path.join(result_dir, os.path.splitext(result_filename)[0]) + '.h5'
    if exp in ['poisson']:
        converted_exp = 'empirical'
    elif exp == 'gaussian':
        converted_exp = 'gaussian'

    subprocess.run(['{} --exp={} --dim={} --data_dir={} --result_file={} --num_points={} --num_samples={} --max_iters={}'.format(
        g_claici_program,
        converted_exp,
        dim,
        data_dir,
        h5_file_path,
        support_size,
        internal_num_samples,
        max_iters)], shell=True)

    with h5py.File(h5_file_path, 'r') as f:
        result = f['points'][:]
        result = np.transpose(result)

    result = np.array(result)
    np.save(os.path.join(result_dir, result_filename), result)

