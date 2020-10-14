import cwb.templates

import shutil
import tempfile
import yaml
import numpy
import numpy as np
import os
import math
import tensorflow as tf
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

def emd2(dim, source, target):
    # compute the 2-Wasserstein distance squared between two empirical distributions
    original_wd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        conf = yaml.safe_load(pkg_resources.open_text(cwb.templates, 'ot_default_config.yaml').read())
        conf['point_dim'] = dim

        if isinstance(source, np.ndarray):
            np.save('source.npy', source)
            conf['source_distribution'] = {'shape': 'empirical_npy', 'npy_path': 'source.npy'}
        else:
            # otherwise a yaml description
            conf['source_distribution'] = source

        if isinstance(target, np.ndarray):
            np.save('target.npy', target)
            conf['target_distribution'] = {'shape': 'empirical_npy', 'npy_path': 'target.npy'}
        else:
            # otherwise a yaml description
            conf['target_distribution'] = target

        from .ot_state import OTState
        state = OTState(conf)
        state.train_potentials()
        result = state.eval()
        os.chdir(original_wd)

    return result
