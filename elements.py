import numpy as np
import numba


@numba.njit(nogil=True)
def TEST(t, M, o, func_name, s):
    num, dim = t.shape
    n = numba.prange(num)
    y = np.zeros((num, 2))
    for i in n:
        y[i, 0] = yourbenchamrkhere.func(func_name, t[i,:], M, o, s)
        y[i, 1] = i
    return y


@numba.njit(cache=True)
def my_min(t):
    num = t.shape[0]
    n = numba.prange(1,num)
    min_fit = t[0, 0]
    min_pos = 0
    max_fit = t[0, 0]
    max_pos = 0
    for i in n:
        if t[i, 0] < min_fit:
            min_fit = t[i, 0]
            min_pos = i
        if t[i, 0] > max_fit:
            max_fit = t[i, 0]
            max_pos = i
    return  min_pos, min_fit, max_pos, max_fit


@numba.njit(cache=True)
def get_rand_0_1():
    r = np.random.random()
    while r == 0:
        r = np.random.random()
    return r


@numba.njit(cache=True)
def my_clip(w, lb, ub):
    n = numba.prange(w.shape[0])
    m = numba.prange(w.shape[1])
    for i in n:
        for j in m:
            if w[i][j] > ub:
                w[i][j] = ub
            elif w[i][j] < lb:
                w[i][j] = lb
    return w