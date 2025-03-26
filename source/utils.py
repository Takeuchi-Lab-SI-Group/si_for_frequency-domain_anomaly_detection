import numpy as np
from numba import jit


@jit(nopython=True)
def mat_sum(A, B):
    return A + B


def mat_diff(A, B):
    return A - B


@jit(nopython=True)
def Roth_column_lemma(A, b):
    return b @ (b.T @ A)


@jit(nopython=True)
def Roth_column_lemma_sum(A, b, c):
    return mat_sum(Roth_column_lemma(A, b), Roth_column_lemma(A, c))


@jit(nopython=True)
def mat_prod_to_vec(A, B):
    return ((A @ B).T).ravel()


@jit(nopython=True)
def vec_prod(a, b):
    return np.dot(a, b)
