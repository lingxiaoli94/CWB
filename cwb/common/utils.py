import tensorflow as tf
import math
import contextlib

@tf.function
def calc_distances(ps, qs):
    return tf.reduce_sum(tf.square(ps - qs), axis=1)

@tf.function
def calc_cross_distances(ps, qs):
    ps_sqr = tf.reduce_sum(ps * ps, axis=1)
    qs_sqr = tf.reduce_sum(qs * qs, axis=1)
    dot = tf.matmul(ps, qs, transpose_b=True)
    return tf.expand_dims(ps_sqr, 1) + tf.expand_dims(qs_sqr, 0) - 2 * dot

@tf.function
def compute_plan_pdf(f, g, nu_f, nu_g, reg_d_fn, xs, ys, use_batch):
    # xs - NxD, ys - N'xD, or xs, ys are 1d arrays
    # output: plan - NxN', assuming cost is Euclidean distance, or a scalar if xs, ys are 1d arrays
    # plan(x,y) = reg_d_fn(f(x)+g(y)-c(x,y))*nu_f.pdf(x)*nu_g.pdf(y)
    if not use_batch:
        xs = tf.expand_dims(tf.convert_to_tensor(xs), 0)
        ys = tf.expand_dims(tf.convert_to_tensor(ys), 0)

    fx = f(xs)
    pdf_x = nu_f.pdf(xs)
    gy = g(ys)
    pdf_y = nu_g.pdf(ys)
    C = calc_cross_distances(xs, ys)
    plan = reg_d_fn(tf.expand_dims(fx, 1) + tf.expand_dims(gy, 0) - C) \
            * tf.expand_dims(pdf_x, 1) * tf.expand_dims(pdf_y, 0)

    if not use_batch:
        plan = plan[0, 0] # just a single number
    return plan

@tf.function
def compute_kth_marginal_moment(ps, k):
    # ps - N x D
    mean = tf.reduce_mean(ps, 0) # D
    ps = ps - tf.expand_dims(mean, 0) # N x D
    tmp = tf.math.pow(ps, k) # N x D
    return tf.reduce_mean(tmp, 0) # D

@tf.function
def compute_CMD(ps, qs, L, K = 5):
    p_mean = tf.reduce_mean(ps, 0) # D
    q_mean = tf.reduce_mean(qs, 0) # D
    result = tf.norm(p_mean - q_mean) / L
    for k in range(2, K + 1):
        p_ck = compute_kth_marginal_moment(ps, k)
        q_ck = compute_kth_marginal_moment(qs, k)
        tmp = tf.norm(p_ck - q_ck) / (L ** k)
        result += tmp
    return result


@tf.function
def safe_exp_tf(x, t):
    # x is 1D array, t is the threshold
    res = tf.where(x >= t, tf.square(x - t + 2) * tf.exp(t) / 4, tf.exp(tf.minimum(x, t)))
    return res

@contextlib.contextmanager
def tf_optimizer_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)

def tf_ckpt_register(ckpt_kwargs, ckpt_dir, max_to_keep):
    ckpt = tf.train.Checkpoint(**ckpt_kwargs)
    manager = tf.train.CheckpointManager(
            ckpt,
            ckpt_dir,
            max_to_keep=max_to_keep)
    restore_status = ckpt.restore(manager.latest_checkpoint)
    restore_status.expect_partial()

    if manager.latest_checkpoint:
        print('Restored trained model from {}'.format(manager.latest_checkpoint))
    else:
        print('Training model from scratch.')

    return manager

def tf_set_float_dtype(dtype_str):
    if dtype_str == 'float64':
        print('Using float64 instead of float32 in neural networks...')
        float_dtype = tf.float64
        tf.keras.backend.set_floatx('float64')
    else:
        assert(dtype_str == 'float32')
        float_dtype = tf.float32

    return float_dtype
