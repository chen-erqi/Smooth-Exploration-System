import numba
import numpy as np
from elements import *


@numba.njit(nogil=True, fastmath=True)
def SWOA(max_iter, search_num, lb, ub, dim, M, o, func_name, s):

    whales = np.random.random((search_num, dim)) * (ub - lb) + lb
    best_fit = np.zeros(max_iter + 1)
    cur_iter = 1
    n = numba.prange(search_num)
    m = numba.prange(dim)

    fit = TEST(whales, M, o, func_name, s)
    cur_best_pos, cur_best_fit, cur_bad_pos, cur_bad_fit = my_min(fit)
    all_best_pop = whales[cur_best_pos, :].copy()
    all_best_fit = cur_best_fit
    best_fit[0] = cur_best_fit

    while cur_iter <= max_iter:

        rate = cur_iter / max_iter

        a = 2 - 2 * rate
        b = 1

        for i in n:

            A = 2 * a * get_rand_0_1() - a

            r_dim = np.random.choice(dim, size=(int(np.ceil(max(rate, np.e * 0.1) * dim))), replace=False) # Eq1

            trial = whales[i, :].copy()

            if np.random.random() <= 0.5:
                if abs(A) < 1:
                    for j in r_dim:
                        C = 2 * get_rand_0_1()
                        D = np.abs(C * all_best_pop[j] - whales[i, j])
                        trial[j] = all_best_pop[j] - A * D
                else: # Eq 14
                    if np.random.random() < 0.5:
                        a = np.random.randint(low=0, high=search_num)
                        while a == i:
                            a = np.random.randint(low=0, high=search_num)
                        b = np.random.randint(low=0, high=search_num)
                        while b == i or a == b:
                            b = np.random.randint(low=0, high=search_num)
                        c = np.random.randint(low=0, high=search_num)
                        while c == i or c == a or c == b:
                            c = np.random.randint(low=0, high=search_num)
                        trial = whales[c, :] + (whales[a, :] - whales[b, :]) 
                    else:
                         for j in m:
                             trial[j] = harmonic_mean(whales[i, j], whales[i-1, j]) 
            else:
                for j in r_dim:
                    l = np.random.random() * 2 - 1
                    D = np.abs(all_best_pop[j] - whales[i, j])
                    trial[j] = all_best_pop[j] + D * math.cos(2 * math.pi * l) * math.exp(b * l)

            tem_fit = TEST(my_clip(trial.reshape((1, dim)), lb=lb, ub=ub), M, o, func_name, s)[0, 0]
            if tem_fit < fit[i, 0]:
                whales[i, :] = trial.copy()
                fit[i, 0] = tem_fit
                if tem_fit < all_best_fit:
                    all_best_fit = tem_fit
                    all_best_pop = trial.copy()

        best_fit[cur_iter] = all_best_fit

        cur_iter += 1

    return best_fit