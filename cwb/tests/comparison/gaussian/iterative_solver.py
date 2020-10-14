import numpy as np
import os
import scipy.linalg
import pickle
import argparse

def compute_obj_zero_mean(wb_cov, input_covs, weights):
    wb_cov_rt = scipy.linalg.sqrtm(wb_cov)
    res = np.trace(wb_cov)
    n = len(input_covs)
    for j in range(n):
        cov_j = input_covs[j]
        res += weights[j] * np.trace(cov_j)
        tmp = np.matmul(wb_cov_rt, np.matmul(cov_j, wb_cov_rt))
        tmp = scipy.linalg.sqrtm(tmp)
        res -= 2 * weights[j] * np.trace(tmp)

    return res

def run(dim, data_dir, result_dir, result_filename):
    gaussians = []
    covs = []
    for p in os.listdir(data_dir):
        if p.endswith('.pkl'):
            d = pickle.load(open(os.path.join(data_dir, p), 'rb'))
            gaussians.append(d)
            normal_A = d['normal_A']
            covs.append(np.matmul(normal_A, np.transpose(normal_A)))
    n = len(gaussians)
    # all weights are equal
    weights = [1 / n for _ in range(n)]
    wb_mean = np.zeros([dim])
    for j, d in enumerate(gaussians):
        wb_mean += weights[j] * d['mean']

    # objective for just transporting the means
    obj_mean = 0
    for j, d in enumerate(gaussians):
        obj_mean += weights[j] * np.sum(np.square(wb_mean - d['mean']))

    pre_obj = 1e100
    wb_cov = np.eye(dim)
    itr_count = 0
    while True:
        itr_count += 1
        cur_obj = compute_obj_zero_mean(wb_cov, covs, weights)
        cur_obj = cur_obj + obj_mean

        if abs(pre_obj - cur_obj) < 1e-10 and itr_count >= 1000:
            break

        cov_rt = scipy.linalg.sqrtm(wb_cov) # S_n
        inv_cov_rt = np.linalg.inv(cov_rt)

        Q = 0
        for j in range(n):
            tmp = np.matmul(cov_rt, np.matmul(covs[j], cov_rt))
            tmp = scipy.linalg.sqrtm(tmp)
            tmp = weights[j] * tmp
            Q += tmp
        Q = np.matmul(Q, Q)
        wb_cov = np.matmul(inv_cov_rt, np.matmul(Q, inv_cov_rt)) # S_{n+1}
        pre_obj = cur_obj

    print('Iterations: {}'.format(itr_count))
    print('Obj: {:6f}'.format(cur_obj))

    wb_normal_A = np.linalg.cholesky(wb_cov)
    result = {
            'mean': wb_mean,
            'normal_A': wb_normal_A,
            'dim': dim,
            'obj': cur_obj,
            'iterations': itr_count
            }

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pickle.dump(result, open(os.path.join(result_dir, result_filename), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dim', type=int)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('result_dir', type=str)
    parser.add_argument('result_filename', type=str)
    args = parser.parse_args()

    dim = args.dim
    data_dir = args.data_dir
    result_dir = args.result_dir
    result_filename = args.result_filename

    run(dim, data_dir, result_dir, result_filename)
