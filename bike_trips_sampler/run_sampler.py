from optparse import OptionParser
import os
import pickle
import pystan

MODEL = 'pois'

def parse_args():

    parser = OptionParser()
    parser.set_defaults()

    parser.add_option("--n_splits", type="int", dest="n_splits", default=5)

    (options, args) = parser.parse_args()

    return options

def main():

    options = parse_args()
    print(options)

    n_splits = options.n_splits

    if not os.path.exists('stan_{}.pkl'.format(MODEL)):
        print('Compiling model')
        stan_mdl = pystan.StanModel(file='./{}.stan'.format(MODEL), verbose=True)
        with open('stan_{}.pkl'.format(MODEL), 'wb') as f:
            pickle.dump(stan_mdl, f)
    else:
        print('Loading model')
        stan_mdl = pickle.load(open('stan_{}.pkl'.format(MODEL), 'rb'))

    try:
        os.makedirs('samples')
    except:
        pass

    for split_idx in range(-1, n_splits):
        if split_idx < 0:
            exp_name = 'posterior_all'
        else:
            exp_name = 'posterior_' + str(split_idx)

        print(exp_name)

        job_cmd = 'python ' +\
                  'sample.py ' +\
                  ' --n_splits ' + str(n_splits) +\
                  ' --split_idx ' + str(split_idx)

        os.system(job_cmd)

    return

if __name__ == '__main__':
    main()
