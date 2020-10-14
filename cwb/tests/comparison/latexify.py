from .common import *

import pickle
import numpy
import os
import argparse

dim_range = []

g_method_to_text = {
        'cwb': 'Ours',
        'cuturi': 'CD14',
        'claici': 'Claici',
        'staib': 'Staib'
        }
g_exp_to_text = {
        'mixture': 'Gaussian mixtures',
        'gaussian': 'Gaussian',
        'cube': 'Cube',
        'poisson': 'Poisson regression'}
g_loss_key_to_name = {
        'fit_gaussian_mean_loss': 'Gaussian mean diff',
        'fit_gaussian_cov_loss': 'Gaussian cov diff',
        'mm_mean_loss': 'mm mean loss',
        'mm_cov_loss': 'mm cov loss',
        'W2_lp': 'W2_lp',
        'barycenter_obj_lp': 'barycenter_obj_lp',
        'barycenter_obj_stochastic': 'barycenter_obj_stochastic'}
g_loss_keys = {'fit_gaussian_mean_loss', 'fit_gaussian_cov_loss', 'mm_mean_loss', 'mm_cov_loss', 'W2_lp'}

def numerical_to_text(x, sig_fig=2, math_mode=False, compact=False):
    s = "{0:.{1:d}e}".format(x, sig_fig)
    if 'e' not in s:
        return s
    a,b = s.split("e")
    b = int(b)
    if compact:
        res = ' ^{\\times 10^{' + str(b) + '}}'
    else:
        res = ' \\times 10^{' + str(b) + '}'
    res = a + res
    if not math_mode:
        res = '$' + res + '$'
    return res


def generate_stats_table(exp, method, repeat_range):
    loss_keys = g_loss_keys
    stats_mean_list = []
    stats_std_list = []
    for dim in dim_range:
        rep_list = []
        for rep in repeat_range:
            stats_file = get_stats_file_path(dim, method, rep)
            stats = pickle.load(open(stats_file, 'rb'))
            rep_list.append(stats)
        stats_mean = {}
        stats_std = {}
        for k in loss_keys:
            for j, rep in enumerate(rep_list):
                if k in rep:
                    print('dim {} {:04} {} {:6e}'.format(dim, j, k, rep[k]))
            arr = np.array([rep[k] for rep in rep_list if k in rep])
            stats_mean[k] = np.mean(arr)
            stats_std[k] = np.std(arr)
        stats_mean_list.append(stats_mean)
        stats_std_list.append(stats_std)

    num_dims = len(stats_mean_list)

    latex = ''
    latex += '\\begin{table}\n'
    latex += '\\centering\n'
    latex += '\\caption{' + 'Loss table for {} experiment with method {}'.format(g_exp_to_text[exp], g_method_to_text[method])  + '}\n'
    latex += '\\begin{tabular}{' + ''.join(['l'] * (num_dims + 1)) + '}\n'
    latex += '\\toprule\n'
    latex += 'Loss & ' + ' & '.join(['$d={}$'.format(dim) for dim in dim_range]) + '\\\\\n'
    latex += '\\midrule\n'
    loss_keys_set = set(loss_keys)
    for loss_key in g_loss_keys:
        if not (loss_key in loss_keys_set):
            continue
        row = g_loss_key_to_name[loss_key] + ' & '
        stats_arr = []
        for i in range(num_dims):
            stats_arr.append((stats_mean_list[i][loss_key], stats_std_list[i][loss_key]))
        row += ' & '.join(['{}({})'.format(numerical_to_text(s[0]), numerical_to_text(s[1])) for s in stats_arr])
        row += '\\\\\n'
        latex += row

    latex += '\\bottomrule\n'

    latex += '\\end{tabular}\n'
    latex += '\\end{table}\n'
    return latex


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('method', type=str)
    parser.add_argument('--dims', nargs='+', type=int, required=True)
    parser.add_argument('--repeat_start', type=int, default=0)
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--losses', type=str, nargs='+', required=True)
    args = parser.parse_args()
    dim_range = args.dims
    exp = args.exp
    method = args.method
    g_loss_keys = args.losses
    repeat_start = args.repeat_start
    repeat_times = args.repeat_times
    repeat_range = range(repeat_start, repeat_start + repeat_times)
    latex = generate_stats_table(exp, method, repeat_range)
    print(latex)
