import numpy as np
from numba import jit
from itertools import pairwise

@jit(nopython=True)
def calc_cost_seg(x, m, n, d, window):
    x_mn = x[m:n]
    mean = np.mean(x_mn)
    diff = x_mn - mean
    cost = np.real(np.vdot(diff, diff))
    return 2 * cost if d != 0 and d != window / 2 else cost  

@jit(nopython=True)
def calc_cost(F_d, d, cp_d_list, window): 
    cost = 0
    for i in range(len(cp_d_list) - 1):
        cost += calc_cost_seg(F_d, cp_d_list[i], cp_d_list[i + 1], d, window)
    return cost

@jit(nopython=True) 
def calc_mat(m, n, d, window):
    mat_size = n - m
    others = -1 / mat_size
    subject = 1 + others
    diagonal = subject ** 2 + (others ** 2) * (mat_size - 1)
    non_diagonal = diagonal - 1 
    mat = np.ones((mat_size, mat_size)) * non_diagonal
    np.fill_diagonal(mat, diagonal)
    if d != 0 and d != window / 2:
        return 2 * mat
    else:
        return mat

@jit(nopython=True) 
def calc_cost_seg_mat(m, n, d, window, T): 
    cost_mat = np.zeros((T, T))
    cost_mat[m:n, m:n] = calc_mat(m, n, d, window)
    return cost_mat

def calc_cost_seg_mat_series(cp_d_list, d, window, T): 
    cost_mat = np.zeros((T, T))
    for m, n in pairwise(cp_d_list):
        cost_mat[m:n, m:n] = calc_mat(m, n, d, window)
    return cost_mat 
