from cwb.common.distributions import \
        uniform_rectangle_tf, \
        uniform_disk_tf, \
        uniform_annulus_tf, \
        gaussian_tf, \
        monochromatic_image_tf, \
        point_set_2d_tf, \
        ellipse_np, \
        composite_tf, \
        empirical_tf, \
        uniform_nd_tf, \
        mixture_tf, \
        uniform_bbox_tf
from cwb.common.marching_cubes import march, super_resolution, sample_segs
from cwb.common.utils import safe_exp_tf


import tensorflow as tf
import numpy as np
import pickle

def load_empirical_npz(nu_desc):
    npz = np.load(nu_desc['npz_path'], allow_pickle=True)
    key = nu_desc['npz_key']
    return npz[key]


def construct_single_distribution(nu_desc, float_dtype):
    # unwrap if pkl_path is in nu_desc
    if 'pkl_path' in nu_desc:
        dict_pkl = pickle.load(open(nu_desc['pkl_path'], 'rb'))
        for k, v in dict_pkl.items():
            nu_desc[k] = v

    if nu_desc['shape'] == 'rectangle':
        nu_tf = uniform_rectangle_tf(nu_desc['extent'], float_dtype=float_dtype)
    elif nu_desc['shape'] == 'disk':
        nu_tf = uniform_disk_tf(nu_desc['center'], nu_desc['radius'], float_dtype=float_dtype)
    elif nu_desc['shape'] == 'annulus':
        nu_tf = uniform_annulus_tf(nu_desc['center'], nu_desc['inner_radius'], nu_desc['outer_radius'], float_dtype=float_dtype)
    elif nu_desc['shape'] == 'gaussian':
        nu_tf = gaussian_tf(nu_desc['dim'], nu_desc['mean'], nu_desc['normal_A'], float_dtype=float_dtype)
    elif nu_desc['shape'] == 'ellipse':
        ellipse_dist = ellipse_np(nu_desc['center'], abs(nu_desc['a']), abs(nu_desc['b']), nu_desc['T'])
        ps = np.array([ellipse_dist.sample() for _ in range(nu_desc['point_set_size'])])
        gaussian_noise = nu_desc['gaussian_noise'] if 'gaussian_noise' in nu_desc else 0.0
        nu_tf = point_set_2d_tf(ps, gaussian_noise, fast_pdf=nu_desc.get('fast_pdf', True), float_dtype=float_dtype)
    elif nu_desc['shape'] == 'uniform':
        nu_tf = uniform_nd_tf(nu_desc['extent'], float_dtype=float_dtype)
    elif nu_desc['shape'] == 'cube':
        p_min = np.array(nu_desc['p_min'])
        p_max = np.array(nu_desc['p_max'])
        dim = p_min.shape[0]
        components = []
        for i in range(dim):
            axis = np.zeros([dim])
            axis[i] = 1.0
            components.append(axis)
        nu_tf = uniform_bbox_tf(np.array(components), p_min, p_max)
    elif nu_desc['shape'] == 'mnist':
        # TODO: avoid reloading data
        (img_train, _), _ = tf.keras.datasets.mnist.load_data()
        img = img_train[nu_desc['index'], :, :] / 255.0

        img = tf.transpose(img, [1, 0]) # make image have the right orientation
        img = tf.reverse(img, [1]) # flip upside-down

        if 'super_res' in nu_desc and nu_desc['super_res']:
            img = super_resolution(img)
        extent = [0, 1, 0, 1]
        if nu_desc.get('bd_only', False):
            # only sample on the boundary of the digit
            img_mask = np.where(img > nu_desc['march_threshold'], 1, 0)
            segs = march(img_mask, 1.0 / img.shape[0])
            ps = sample_segs(segs, nu_desc['bd_point_set_size'])
            gaussian_noise = nu_desc['gaussian_noise'] if 'gaussian_noise' in nu_desc else 0.0
            nu_tf = point_set_2d_tf(ps, gaussian_noise, fast_pdf=nu_desc.get('fast_pdf', True), float_dtype=float_dtype)
        else:
            uniform_noise = nu_desc.get('uniform_noise', True)
            nu_tf = monochromatic_image_tf(
                    img,
                    extent=extent,
                    uniform_noise=uniform_noise,
                    float_dtype=float_dtype)
    elif nu_desc['shape'] == 'composite':
        nu1 = construct_single_distribution(nu_desc['nu1'], float_dtype)
        nu2 = construct_single_distribution(nu_desc['nu2'], float_dtype)
        nu_tf = composite_tf(nu1, nu2, nu_desc['w1'], nu_desc['w2'])
    elif nu_desc['shape'] == 'mixture':
        nu_list = [construct_single_distribution(child_desc, float_dtype) for child_desc in nu_desc['nu_list']]
        w_list = nu_desc['weight_list']
        nu_tf = mixture_tf(nu_desc['dim'], nu_list, w_list, float_dtype)
    elif nu_desc['shape'] == 'empirical_npz':
        nu_tf = empirical_tf(load_empirical_npz(nu_desc), nu_desc.get('gaussian_noise', 0.0), float_dtype)
    elif nu_desc['shape'] == 'empirical_npy':
        nu_tf = empirical_tf(np.load(nu_desc['npy_path']), nu_desc.get('gaussian_noise', 0.0), float_dtype)
    elif nu_desc['shape'] == 'empirical':
        nu_tf = empirical_tf(nu_desc['points'], nu_desc.get('gaussian_noise', 0.0), float_dtype)
    elif nu_desc['shape'] == 'placeholder':
        return None # None indicates a placeholder
    else:
        raise Exception('Unrecognized shape: {}'.format(nu_desc['shape']))

    # nu_tf is a dict with two functions {'sample': sample_fn, 'pdf': pdf_fn}
    return nu_tf


def construct_distribution_list(list_desc, float_dtype):
    vertex_list = []
    weight_list = []
    name_list = []
    for vertex_desc in list_desc:
        vertex_list.append(construct_single_distribution(vertex_desc, float_dtype))
        weight_list.append(vertex_desc['weight'])
        name_list.append(vertex_desc['name'])
    return vertex_list, name_list, weight_list


def construct_regularizer(regularizer_desc, eps, float_dtype):
    eps = tf.cast(eps, float_dtype)
    zero = tf.cast(0.0, float_dtype)
    if regularizer_desc['name'] == 'entropy':
        if regularizer_desc.get('safe_version', True):
            return lambda t: eps * safe_exp_tf(t / eps, regularizer_desc.get('safe_threshold', 10.0))
        else:
            return lambda t: eps * tf.math.exp(t / eps)
    elif regularizer_desc['name'] == 'l2':
        return lambda t: tf.square(tf.maximum(zero, t)) / (2 * eps)
    else:
        raise Exception('Unknown regularizer: {}'.format(regularizer_desc['name']))

def construct_regularizer_derivative(regularizer_desc, eps, float_dtype):
    eps = tf.cast(eps, float_dtype)
    zero = tf.cast(0.0, float_dtype)
    if regularizer_desc['name'] == 'entropy':
        if regularizer_desc.get('safe_version', True):
            return lambda t: safe_exp_tf(t / eps, regularizer_desc.get('safe_threshold', 10.0))
        else:
            return lambda t: tf.math.exp(t / eps)
    elif regularizer_desc['name'] == 'l2':
        return lambda t: tf.maximum(zero, t) / eps
    else:
        raise Exception('Unknown regularizer: {}'.format(regularizer_desc['name']))

def construct_optimizer(desc):
    kind = desc['kind']
    lr = desc['learning_rate']
    mm = desc['momentum'] if 'momentum' in desc else 0.0
    kwargs = {}
    if 'decay' in desc:
        kwargs['decay'] = desc['decay']
    if kind == 'SGD':
        opt = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=mm,
                **kwargs)
    elif kind == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr, **kwargs)
    else:
        raise Exception('Unknown optimizer kind: {}'.format(kind))

    return opt
