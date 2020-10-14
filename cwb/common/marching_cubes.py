import numpy as np
import math

'''
corner bit indices
    1------0
    |      |
    |      |
    3------2
'''

g_rules = [
        [],                 # 0000
        ['UR'],             # 0001
        ['LU'],             # 0010
        ['LR'],             # 0011
        ['RD'],             # 0100
        ['UD'],             # 0101
        ['RD', 'LU'],       # 0110
        ['LD'],             # 0111
        ['DL'],             # 1000
        ['DL', 'UR'],       # 1001
        ['DU'],             # 1010
        ['DR'],             # 1011
        ['RL'],             # 1100
        ['UL'],             # 1101
        ['RU'],             # 1110
        [],                 # 1111
    ]

g_offsets = {
        'L': np.array([-1, 0]),
        'R': np.array([1, 0]),
        'U': np.array([0, 1]),
        'D': np.array([0, -1]),
}

def march(A, cell_size):
    '''
        Input:
            0/1 matrix A, cell size, one grid corresponding to one entry in the matrix,
            assuming x-axis corresponds to rows in A.
        Output:
            a list of (u, v) where u,v are 2d points, representing an edge
    '''
    results = []
    n, m = A.shape
    for i in range(n-1):
        for j in range(m-1):
            c = np.array([(i+1) * cell_size, (j+1) * cell_size]) # center coordinate
            mask = (A[i,j] << 3) | (A[i+1, j] << 2) | (A[i, j+1] << 1) | A[i+1, j+1]
            rule = g_rules[mask]
            for r in rule:
                u = c + g_offsets[r[0]] * cell_size / 2
                v = c + g_offsets[r[1]] * cell_size / 2
                results.append((u,v))
    return results

def super_resolution(img):
    # output a image with dimensions doubled
    n, m = img.shape
    res = np.zeros(shape=[n * 2, m * 2])
    def get_loc_old(i, j):
        return np.array([i + 0.5, j + 0.5])
    def calc_weight(oi, oj, loc):
        oloc = get_loc_old(oi, oj)
        d_sqr = np.dot(oloc - loc, oloc - loc)
        return 1 / (0.5 + d_sqr)
    for i in range(2 * n):
        for j in range(2 * m):
            loc = np.array([i * 0.5 + 0.25, j * 0.5 + 0.25])
            oi = i // 2
            oj = j // 2
            queue = [(calc_weight(oi, oj, loc), img[oi, oj])]
            for d in g_offsets.values():
                ni = oi + d[0]
                nj = oj + d[1]
                if 0 <= ni < n and 0 <= nj < m:
                    queue.append((calc_weight(ni, nj, loc), img[ni, nj]))
            w_sum = 0
            p_sum = 0
            for pr in queue:
                w_sum += pr[0]
                p_sum += pr[0] * pr[1]
            res[i, j] = p_sum / w_sum

    return res


def sample_segs(segs, n):
    # sample n points uniformly on segs
    def calc_len(seg):
        return math.sqrt(np.dot(seg[0] - seg[1], seg[0] - seg[1]))
    segs = sorted(segs, key=calc_len)
    lens = []
    for s in segs:
        lens.append(calc_len(s))
    lens = np.array(lens)
    bar_end = np.cumsum(lens)
    ds = np.random.uniform(0, bar_end[-1], size=(n,))
    res = np.zeros(shape=[n, 2])
    for i, d in enumerate(ds):
        lb = -1
        rb = len(segs) - 1
        while lb < rb - 1:
            m = (lb + rb) // 2
            if bar_end[m] >= d:
                rb = m
            else:
                lb = m
        m = rb
        # sample a point uniformly on segment m
        l = np.random.uniform(0, 1)
        res[i, :] = l * (segs[m][1] - segs[m][0]) + segs[m][0]
    return res







