from .ot_state import OTState

import numpy as np
import tensorflow as tf
import os
import yaml
import random
import argparse

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # tf.autograph.set_verbosity(10, alsologtostdout=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu_devices) > 0:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('yaml', type=str)
    args = parser.parse_args()

    conf = yaml.safe_load(open(args.yaml).read())
    state = OTState(conf)

    if args.train:
        state.train_potentials()

    if args.test:
        state.eval()
