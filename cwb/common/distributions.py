import numpy as np
import random
import scipy.stats
import math
import tensorflow as tf

from .utils import calc_cross_distances

class DistributionWrapper:
    def __init__(self, sample_fn, pdf_fn):
        self.sample_fn = sample_fn
        self.pdf_fn = pdf_fn

    def sample(self, *args):
        return self.sample_fn(*args)

    def pdf(self, *args):
        return self.pdf_fn(*args)


class BoundedDistributionAdapter:
    def __init__(self, base_distribution, supp):
        self.base_distribution = base_distribution
        self.supp = supp

    def sample(self, batch_size):
        ps = self.base_distribution.sample(batch_size)
        pdfs = self.supp.pdf(ps)
        qs = self.supp.sample(batch_size)
        return tf.where(tf.expand_dims(pdfs > 0, 1), ps, qs)

    def pdf(self, ps):
        base_pdfs = self.base_distribution.pdf(ps)
        supp_pdfs = self.supp.pdf(ps)
        return tf.where(supp_pdfs > 0, base_pdfs, 0)


class TranslatedDistributionAdapter:
    def __init__(self, base_distribution, delta):
        self.base_distribution = base_distribution
        self.delta = delta

    def sample(self, batch_size):
        ps = self.base_distribution.sample(batch_size)
        ps = ps + tf.expand_dims(self.delta, 0)
        return ps

    def pdf(self, ps):
        return self.base_distribution.pdf(ps - tf.expand_dims(self.delta, 0))


# numpy versions, without batching

def uniform_1d_np(start, end):
    return DistributionWrapper(
        sample_fn=lambda: np.random.random() * (end - start) + start,
        pdf_fn=lambda x: 1 / (end - start) if (start <= x and x <= end) else 0.0)

def gaussian_1d_np(mean, sdv):
    N = scipy.stats.norm(loc=mean, scale=sdv)
    return DistributionWrapper(
        sample_fn=N.rvs,
        pdf_fn=N.pdf)

def uniform_rectangle_np(x_start, x_end, y_start, y_end):
    nu_x = uniform_1d_np(x_start, x_end)
    nu_y = uniform_1d_np(y_start, y_end)
    def sample():
        return np.array([nu_x.sample(), nu_y.sample()])
    return DistributionWrapper(
        sample_fn=sample,
        pdf_fn=lambda p: nu_x.pdf(p[0]) * nu_y.pdf(p[1]))

def uniform_annulus_np(center, r1, r2):
    area = math.pi * (r2 ** 2 - r1 ** 2)
    def sample():
        angle = random.uniform(0, 2 * math.pi)
        l = random.uniform(r1 ** 2, r2 ** 2)
        l = math.sqrt(l)
        return center + l * np.array([math.cos(angle), math.sin(angle)])
    def inside(p):
        d = np.dot(p - center, p - center)
        return r1 ** 2 <= d and d <= r2 ** 2
    return DistributionWrapper(
        sample_fn=sample,
        pdf_fn=lambda p: 1 / area if inside(p) else 0.0)

def uniform_disk_np(center, radius):
    return uniform_annulus_np(center, 0, radius)

def gaussian_np(mean, cov):
    N = scipy.stats.multivariate_normal(mean, cov)
    return DistributionWrapper(sample_fn=N.rvs, pdf_fn=N.pdf)

def ellipse_np(center, a, b, T):
    center = np.array(center, dtype=float)
    T = np.array(T, dtype=float)
    def sample():
        succeed = False
        speed_bound = math.sqrt(a ** 2 + b ** 2)
        while not succeed:
            t = random.uniform(0, 2 * math.pi)
            p = np.array([a * math.cos(t), b * math.sin(t)])
            speed = math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
            if random.uniform(0, speed_bound) < speed:
                succeed = True
                break
        return center + np.dot(T, p)

    return DistributionWrapper(sample_fn=sample, pdf_fn=None)

def rectangle_frame_np(extent):
    # a rectangular frame, with given extent
    x0, x1, y0, y1 = extent
    lx = x1 - x0
    ly = y1 - y0
    circumference = 2 * (lx + ly)
    base = np.array([x0, y0])
    def sample():
        t = random.uniform(0, circumference)
        ox = 0
        oy = 0
        if t < lx:
            ox = t
        elif t < 2 * lx:
            ox = t - lx
            oy = ly
        elif t < 2 * lx + ly:
            oy = t - 2 * lx
        else:
            oy = t - 2 * lx - ly
            ox = lx
        return base + np.array([ox, oy])

    return DistributionWrapper(sample_fn=sample, pdf_fn=None)


# tensorflow versions, with batching

def uniform_rectangle_tf(extent, float_dtype=tf.float32):
    x0, x1, y0, y1 = extent
    density = tf.cast(1 / ((x1 - x0) * (y1 - y0)), float_dtype)
    def sample(batch_size):
        xs = tf.random.uniform(shape=(batch_size,), minval=x0, maxval=x1, dtype=float_dtype)
        ys = tf.random.uniform(shape=(batch_size,), minval=y0, maxval=y1, dtype=float_dtype)
        return tf.stack([xs, ys], axis=1)
    def pdf(ps):
        ps = tf.cast(ps, float_dtype)
        mask_x = (x0 <= ps[:, 0]) & (ps[:, 0] <= x1)
        mask_y = (y0 <= ps[:, 1]) & (ps[:, 1] <= y1)
        mask = mask_x & mask_y
        return tf.where(mask, density, 0)

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def uniform_annulus_tf(center, inner_radius, outer_radius, float_dtype=tf.float32):
    area = tf.cast(math.pi * (outer_radius ** 2 - inner_radius ** 2), float_dtype)
    density = tf.cast(1 / area, float_dtype)
    def sample(batch_size):
        angles = tf.random.uniform(shape=(batch_size,), minval=0, maxval=2*math.pi, dtype=float_dtype)
        rs = tf.random.uniform(shape=(batch_size,), minval=inner_radius**2, maxval=outer_radius**2, dtype=float_dtype)
        rs = tf.sqrt(rs)
        ps = tf.expand_dims(rs, 1) * tf.stack([tf.math.cos(angles), tf.math.sin(angles)], axis=1)
        ps += tf.expand_dims(tf.convert_to_tensor(center), axis=0)
        return ps
    def pdf(ps):
        ps = tf.cast(ps, float_dtype)
        d = tf.reduce_sum((ps - tf.expand_dims(center, 0)) ** 2, -1)
        mask = (inner_radius ** 2 <= d) & (d <= outer_radius ** 2)
        return tf.where(mask, density, 0)

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def uniform_disk_tf(center, radius, float_dtype=tf.float32):
    return uniform_annulus_tf(center, 0, radius, float_dtype)

def gaussian_tf(point_dim, mean, normal_A, float_dtype=tf.float32):
    mean = tf.cast(mean, float_dtype)
    normal_A = tf.cast(normal_A, float_dtype)
    cov = tf.matmul(normal_A, normal_A, transpose_b=True)
    cov_inv = tf.linalg.inv(cov)
    det_cov = tf.linalg.det(cov)
    coeff = 1 / ((math.sqrt(2 * math.pi)) ** point_dim)
    coeff *= 1 / math.sqrt(det_cov)

    def sample(batch_size):
        zs = tf.random.normal(shape=[batch_size, point_dim], dtype=float_dtype)
        zs = tf.linalg.matvec(
                tf.tile(tf.expand_dims(normal_A, 0), [batch_size, 1, 1]),
                zs)
        zs = tf.expand_dims(mean, 0) + zs
        return zs

    def pdf(ps):
        ps = tf.cast(ps, float_dtype)
        ps_centered = ps - tf.expand_dims(mean, 0)
        batch_size = tf.shape(ps)[0]
        tmp = tf.linalg.matvec(
                tf.tile(tf.expand_dims(cov_inv, 0), [batch_size, 1, 1]),
                ps_centered)
        tmp = tf.reduce_sum(ps_centered * tmp, -1)
        tmp = -0.5 * tmp
        result = coeff * tf.exp(tmp)
        return result

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def monochromatic_image_tf(img, extent, uniform_noise, float_dtype=tf.float32):
    x0, x1, y0, y1 = extent
    N, M = img.shape # img is N x M
    hx = tf.cast((x1 - x0) / N, float_dtype)
    hy = tf.cast((y1 - y0) / M, float_dtype)
    x_flat = tf.reshape(tf.tile(tf.expand_dims(hx * (tf.range(N, dtype=float_dtype) + 0.5), 1), [1, M]), [-1])
    y_flat = tf.reshape(tf.tile(tf.expand_dims(hy * (tf.range(M, dtype=float_dtype) + 0.5), 0), [N, 1]), [-1])
    img_flat = tf.cast(tf.reshape(img, [-1]), float_dtype)
    bar_end = tf.cumsum(img_flat)
    bar_start = tf.cumsum(img_flat, exclusive=True)
    total_p = bar_end[-1]

    noise_uniform = uniform_rectangle_tf([-hx/2, hx/2, -hy/2, hy/2], float_dtype)

    def sample(batch_size):
        ds = tf.random.uniform(shape=(batch_size,), minval=0, maxval=total_p, dtype=float_dtype)
        mask = tf.expand_dims(ds, 1) < tf.expand_dims(bar_end, 0)
        mask &= tf.expand_dims(ds, 1) >= tf.expand_dims(bar_start, 0)
        # mask - [batch_size, NM]
        xs = tf.linalg.matvec(tf.cast(mask, float_dtype), x_flat) # [batch_size]
        ys = tf.linalg.matvec(tf.cast(mask, float_dtype), y_flat) # [batch_size]
        ps = tf.stack([xs, ys], 1)

        if uniform_noise:
            # uniform noise
            ps += noise_uniform.sample(batch_size)

        return ps

    def pdf(ps):
        # compute only uniform density
        ps = tf.cast(ps, float_dtype)
        xis = tf.clip_by_value(tf.cast(tf.math.round(ps[:, 0] / hx - 0.5), tf.int32), 0, N - 1)
        yjs = tf.clip_by_value(tf.cast(tf.math.round(ps[:, 1] / hy - 0.5), tf.int32), 0, M - 1)
        inds = N * xis + yjs
        ws = tf.gather(img_flat, inds, axis=0) / total_p
        result = ws / (hx * hy)
        return result

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def point_set_2d_tf(ps, gaussian_noise=0.01, fast_pdf=True, float_dtype=tf.float32):
    ps = tf.cast(ps, float_dtype)
    noise_gaussian = gaussian_tf(2, [0, 0], gaussian_noise * np.identity(2), float_dtype)
    noise_gaussian_1d = gaussian_tf(1, [0], gaussian_noise * np.identity(1), float_dtype)
    def sample(batch_size):
        inds = tf.random.uniform(shape=[batch_size], minval=0, maxval=ps.shape[0], dtype=tf.int32)
        qs = tf.stack([tf.gather(ps[:, 0], inds), tf.gather(ps[:, 1], inds)], axis=1)
        qs += noise_gaussian.sample(batch_size)
        return qs

    def pdf(qs):
        qs = tf.cast(qs, float_dtype)
        if fast_pdf:
            dist = calc_cross_distances(qs, ps) # |Q| x |P|
            # unnormalized pdf, as this is only used for visualization
            min_dist = tf.reduce_min(dist, axis=1) # |Q|
            min_dist = tf.sqrt(tf.maximum(0.0, min_dist)) # numerical issue
            noise_pdfs = noise_gaussian_1d.pdf(tf.expand_dims(min_dist, 1)) # |Q|
            return noise_pdfs
        else:
            q_minus_p = tf.expand_dims(qs, 0) - tf.expand_dims(ps, 1) # |P| x |Q| x 2
            noise_pdfs = noise_gaussian.pdf(tf.reshape(q_minus_p, [-1, 2])) # |PQ|
            noise_pdfs = tf.reshape(noise_pdfs, [tf.shape(ps)[0], tf.shape(qs)[0]]) # |P| x |Q|
            return tf.reduce_mean(noise_pdfs, 0)

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

# TODO: remove this, and use mixture_tf instead
def composite_tf(nu1, nu2, w1, w2, float_dtype=tf.float32):
    def sample(batch_size):
        ps1 = nu1.sample(batch_size)
        ps2 = nu2.sample(batch_size)
        darts = tf.random.uniform(shape=[batch_size], minval=0, maxval=w1+w2, dtype=float_dtype)
        ps = tf.where(tf.expand_dims(darts < w1, 1), ps1, ps2)
        return ps

    def pdf(ps):
        ps = tf.cast(ps, float_dtype)
        return w1 * nu1.pdf(ps) + w2 * nu2.pdf(ps)

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def mixture_tf(dim, nu_list, w_list, float_dtype=tf.float32):
    n_components = len(nu_list)
    w_list = tf.convert_to_tensor(w_list)
    w_list = tf.cast(w_list, float_dtype)
    w_sum = tf.reduce_sum(w_list)
    w_log = tf.math.log(w_list)
    def sample(batch_size):
        samples_list = tf.stack([nu_list[i].sample(batch_size) for i in range(n_components)], axis=1) # BxCxD
        component_id = tf.random.categorical(tf.expand_dims(w_log, 0), batch_size) # 1xB
        component_id = tf.squeeze(component_id, 0) # B

        inds_0 = tf.tile(tf.expand_dims(component_id, 1), [1, dim]) # BxD
        inds_1 = tf.tile(tf.expand_dims(tf.range(dim), 0), [batch_size, 1]) # BxD
        inds = tf.stack([tf.cast(inds_0, tf.int32), inds_1], axis=2) # BxDx2
        ps = tf.gather_nd(samples_list, inds, batch_dims=1) # BxD
        return ps

    def pdf(ps):
        ps = tf.cast(ps, float_dtype)
        pdf_list = tf.stack([nu_list[i].pdf(ps) for i in range(n_components)], axis=1) # BxC
        pdfs = tf.reduce_sum(pdf_list * tf.expand_dims(w_list, 0), 1) # B
        return pdfs

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

# TODO: merge this and point_set_2d_tf
def empirical_tf(ps, gaussian_noise=0.0, float_dtype=tf.float32, max_sub=5000):
    dim = ps.shape[1]
    print('Create empirical distribution with {} points'.format(ps.shape[0]))
    ps = tf.cast(ps, float_dtype)
    num_points = ps.shape[0]
    if gaussian_noise > 0:
        noise_gaussian_1d = gaussian_tf(1, [0], gaussian_noise * np.identity(1), float_dtype)
    else:
        # this is only in computing pdf
        noise_gaussian_1d = gaussian_tf(1, [0], 0.01 * np.identity(1), float_dtype)

    # ps can be very large! so just take a subset for computing pdf
    if num_points > max_sub:
        ps_sub_inds = tf.random.shuffle(tf.range(num_points))
        ps_sub_inds = ps_sub_inds[:max_sub]
        ps_sub = tf.gather(ps, ps_sub_inds, axis=0)
    else:
        ps_sub = ps

    # In autograph, this will retrace for different batch_size
    def sample(batch_size):
        if batch_size <= num_points:
            inds = tf.random.shuffle(tf.range(num_points))
            inds = inds[:batch_size]
        else:
            inds = tf.random.shuffle(tf.range(num_points))
            additional_inds = tf.random.uniform(shape=[batch_size - num_points], minval=0, maxval=ps.shape[0], dtype=tf.int32)
            inds = tf.concat([inds, additional_inds], axis=0)

        qs = tf.gather(ps, inds, axis=0)
        if gaussian_noise > 0: # ok to trace once
            qs += tf.reshape(noise_gaussian_1d.sample(batch_size * dim), [batch_size, dim])
        return qs

    def pdf(qs):
        dist = calc_cross_distances(ps_sub, qs) # |P| x |Q|
        min_dist = tf.reduce_min(dist, axis=0) # |Q|
        min_dist = tf.sqrt(tf.maximum(0.0, min_dist))
        noise_pdfs = noise_gaussian_1d.pdf(tf.reshape(min_dist, [-1, 1])) # |Q|
        return noise_pdfs


    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def uniform_nd_tf(extent, float_dtype=tf.float32):
    dim = len(extent)
    volume = 1.0
    for e in extent:
        volume *= e[1] - e[0]
    density = tf.cast(1 / volume, float_dtype) if volume > 1e-6 else 0.0
    def sample(batch_size):
        coords = [tf.random.uniform(shape=[batch_size], minval=e[0], maxval=e[1], dtype=float_dtype) for e in extent]
        coords = tf.stack(coords, axis=1)
        return coords

    def pdf(qs):
        masks = [tf.logical_and(qs[:, i] >= extent[i][0], qs[:, i] <= extent[i][1]) for i in range(dim)]
        masks = tf.stack(masks, axis=1) # NxD
        masks = tf.reduce_all(masks, 1) # N
        return tf.where(masks, volume, 0)

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

def uniform_bbox_tf(components, a_min, a_max, float_dtype=tf.float32):
    point_dim = components.shape[1]
    components = tf.convert_to_tensor(components, float_dtype)
    a_min = tf.convert_to_tensor(a_min, float_dtype)
    a_max = tf.convert_to_tensor(a_max, float_dtype)
    volume = tf.reduce_prod(a_max - a_min)
    density = 1 / volume

    def sample(batch_size):
        rs = tf.random.uniform(shape=[batch_size, point_dim], dtype=float_dtype)
        rs = tf.expand_dims(a_min, 0) + rs * tf.expand_dims(a_max - a_min, 0)
        ps = tf.matmul(rs, components)
        return ps

    def pdf(qs):
        # qs - NxD
        ws = tf.matmul(qs, components, transpose_b=False) # NxD
        masks_min = tf.reduce_all(tf.expand_dims(a_min, 0) <= ws, axis=1) # N
        masks_max = tf.reduce_all(tf.expand_dims(a_max, 0) >= ws, axis=1) # N
        masks = tf.logical_and(masks_min, masks_max) # N
        return tf.where(masks, density, 0)

    return DistributionWrapper(sample_fn=sample, pdf_fn=pdf)

