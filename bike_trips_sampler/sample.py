import numpy as np
from optparse import OptionParser
import pystan
import pickle
import time

MODEL = 'pois' # 'nb'

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  assert(np.isfinite(X).all())
  print('X max: ', np.max(X))
  data.close()
  return X[:, :-1], Y

def parse_args():

    parser = OptionParser()

    parser.add_option("--n_splits", type="int", dest="n_splits")
    parser.add_option("--split_idx", type="int", dest="split_idx")
    parser.add_option("--seed", type="int", dest="seed", default=0)

    (options, args) = parser.parse_args()

    return options

def main():

    options = parse_args()
    print(options)

    n_splits = options.n_splits
    split_idx = options.split_idx
    seed = options.seed

    stan_mdl = pickle.load(open('stan_{}.pkl'.format(MODEL), 'rb'))
    dnm = 'biketrips_large.npz'
    X, Y = load_data(dnm)
    n_max = X.shape[0] - X.shape[0] % n_splits

    if split_idx < 0:
        ll_mult = 1
        save_name = 'samples_all'
        data_idx = np.arange(n_max)
    else:
        ll_mult = n_splits
        save_name = 'samples_' + str(split_idx)
        np.random.seed(seed)
        idx_order = np.random.permutation(n_max)
        partition = np.split(idx_order, n_splits)
        data_idx = partition[split_idx]

    X, Y = X[data_idx], Y[data_idx]
    sampler_data = {'x': X, 'y': Y.astype(int), 'd': X.shape[1], 'n': X.shape[0], 'n_rep': ll_mult}
    g_thin = 2
    g_iter = 220000
    warmup = 20000
    # g_iter = 220
    # warmup = 20
    t0 = time.time()
    thd = sampler_data['d']+1

    fit = stan_mdl.sampling(data=sampler_data, iter=g_iter, chains=1, thin=g_thin, warmup=warmup, verbose=True)

    # samples = fit.extract(permuted=False)
    # print(samples.shape) # [:, 0, :thd]
    pars = ['theta'] if MODEL == 'pois' else ['beta']
    samples = fit.extract(pars=pars,permuted=False)
    if MODEL == 'pois':
        samples = samples['theta'][:, 0, :]
    else:
        samples = samples['beta'][:, 0, :]
    np.save('./samples/' + save_name, samples)
    tf = time.time()
    print('Took', tf - t0)
    print(fit.stansummary(pars=pars))

    return

if __name__ == '__main__':
    main()
