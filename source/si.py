import numpy as np
import itertools
from source import utils


class HypothesisTesting:
    def __init__(
        self,
        tau,
        freq_tau_obs,
        F_freq,
        sigma,
        window,
        FFT_mat,
        cp_list_sa,
    ):
        self.tau = tau
        self.freq_tau_obs = freq_tau_obs
        self.F, self.freq = F_freq
        self.T = self.F.shape[1]
        self.N = window * self.T
        self.sigma = sigma
        self.window = window
        self.FFT_mat_r, self.FFT_mat_i = FFT_mat
        self.cp_list_sa = cp_list_sa

    def calc_test_statistic(self, X):
        P_X = self.calc_projection_X(X)
        stat = (1 / self.sigma) * np.linalg.norm(P_X, ord=2)
        self.a, self.b = X - P_X, P_X / stat
        self.calc_dof()
        return stat

    def calc_projection_X(self, X):
        self.calc_projection_mat_coef()
        P_X = np.zeros((self.N,))

        for i, d in enumerate(self.freq_tau_obs):
            FFT_d_r = self.FFT_mat_r[d].reshape(-1, 1)
            FFT_d_i = self.FFT_mat_i[d].reshape(-1, 1)
            X_vec_inverse = X.reshape(self.T, self.window).T

            if d == 0 or d == self.window / 2:
                P_X += utils.mat_prod_to_vec(
                    utils.Roth_column_lemma(X_vec_inverse, FFT_d_r),
                    self.P_coef_list[i].T,
                )
            else:
                P_X += utils.mat_prod_to_vec(
                    utils.Roth_column_lemma_sum(X_vec_inverse, FFT_d_r, FFT_d_i),
                    self.P_coef_list[i].T,
                )

        return P_X

    def calc_projection_mat_coef(self):
        self.P_coef_list = []
        for d in self.freq_tau_obs:
            d_index = np.argwhere(self.freq == d)[0][0]
            cp_d_list = self.cp_list_sa[d_index]
            tau_index = np.argwhere(np.array(cp_d_list) == self.tau)[0][0]
            left = self.tau - cp_d_list[tau_index - 1]
            right = cp_d_list[tau_index + 1] - self.tau
            self.P_coef_list.append(self.projection_mat_coef_d(d, left, right))

    def projection_mat_coef_d(self, d, left, right):
        P_coef_d = np.zeros((self.T, self.T))

        if d == 0 or d == self.window / 2:
            chi_coef = (left * right) / ((left + right) * self.window)
        else:
            chi_coef = (2 * left * right) / ((left + right) * self.window)

        P_coef_d[self.tau - left : self.tau, self.tau - left : self.tau] = chi_coef / (
            left**2
        )
        P_coef_d[self.tau - left : self.tau, self.tau : self.tau + right] = (
            -chi_coef / (left * right)
        )
        P_coef_d[self.tau : self.tau + right, self.tau - left : self.tau] = (
            -chi_coef / (left * right)
        )
        P_coef_d[self.tau : self.tau + right, self.tau : self.tau + right] = (
            chi_coef / (right**2)
        )

        return P_coef_d

    def calc_dof(self):
        self.dof = 2 * len(self.freq_tau_obs)
        if 0 in self.freq_tau_obs:
            self.dof -= 1
        if self.window / 2 in self.freq_tau_obs:
            self.dof -= 1

    def calc_FFT_mat_ab(self):
        a_vec_inverse = (self.a.reshape(self.T, self.window)).T
        b_vec_inverse = (self.b.reshape(self.T, self.window)).T

        freq_all = int(self.window / 2) + 1
        FFT_mat_a, FFT_mat_b = [], []
        for d in range(freq_all):
            FFT_d_r = self.FFT_mat_r[d].reshape(-1, 1)
            FFT_d_i = self.FFT_mat_i[d].reshape(-1, 1)

            if d == 0 or d == self.window / 2:
                FFT_mat_a_d = utils.Roth_column_lemma(a_vec_inverse, FFT_d_r)
                FFT_mat_b_d = utils.Roth_column_lemma(b_vec_inverse, FFT_d_r)
            else:
                FFT_mat_a_d = utils.Roth_column_lemma_sum(
                    a_vec_inverse, FFT_d_r, FFT_d_i
                )
                FFT_mat_b_d = utils.Roth_column_lemma_sum(
                    b_vec_inverse, FFT_d_r, FFT_d_i
                )

            FFT_mat_a.append(FFT_mat_a_d)
            FFT_mat_b.append(self.threshold_matrix(FFT_mat_b_d))

        return FFT_mat_a, FFT_mat_b

    def threshold_matrix(self, mat, threshold=1e-10):
        mat[np.abs(mat) < threshold] = 0.0
        return mat


def polytope_to_interval(
    interval_ts,
    d_list,
    cost_mat_list,
    freq_tau_obs,
    regularization,
    a,
    b,
    FFT_mat_a,
    FFT_mat_b,
    metropolis_threshold=0,
):
    alpha = beta = gamma = 0
    for d, cost_mat in zip(d_list, cost_mat_list):
        if d in freq_tau_obs:
            FFT_mat_b_d = FFT_mat_b[d]
            Ab = utils.mat_prod_to_vec(FFT_mat_b_d, cost_mat)
            alpha += utils.vec_prod(b, Ab)
            beta += 2 * utils.vec_prod(a, Ab)

        FFT_mat_a_d = FFT_mat_a[d]
        Aa = utils.mat_prod_to_vec(FFT_mat_a_d, cost_mat)
        gamma += utils.vec_prod(a, Aa)

    gamma += regularization + metropolis_threshold

    return intersection(interval_ts, poly_lt_zero(alpha, beta, gamma))


def intersection(intervals1, intervals2, verify=False):  # from sicore's interval.py
    if verify:
        _verify_and_raise(intervals1)
        _verify_and_raise(intervals2)

    result_intervals = []
    for i1, i2 in itertools.product(intervals1, intervals2):
        lower = max(i1[0], i2[0])
        upper = min(i1[1], i2[1])
        if lower < upper:
            result_intervals.append([lower, upper])

    return result_intervals


def _verify_and_raise(intervals):  # from sicore's interval.py
    for s, e in intervals:
        if s >= e:
            raise ValueError(f"Inverted or no-range interval found in {[s, e]}")


def poly_lt_zero(alpha, beta, gamma, tol=1e-10):
    alpha, beta, gamma = [0 if -tol < i < tol else i for i in [alpha, beta, gamma]]
    interval = [[0.0, np.inf]]

    if alpha == 0:
        if beta == 0:
            if gamma > 0:
                raise ValueError(
                    "Test direction of interest does not intersect with the inequality."
                )
        elif beta < 0:
            interval = [[-gamma / beta, np.inf]]
        elif beta > 0:
            interval = [[np.NINF, -gamma / beta]]

    elif alpha > 0:
        disc = beta**2 - 4 * alpha * gamma
        if disc <= 0:
            raise ValueError(
                "Test direction of interest does not intersect with the inequality."
            )
        else:
            interval = [
                [
                    (-beta - np.sqrt(disc)) / (2 * alpha),
                    (-beta + np.sqrt(disc)) / (2 * alpha),
                ]
            ]

    else:
        disc = beta**2 - 4 * alpha * gamma
        if disc > 0:
            interval = [
                [np.NINF, (-beta + np.sqrt(disc)) / (2 * alpha)],
                [(-beta - np.sqrt(disc)) / (2 * alpha), np.inf],
            ]

    return intersection(interval, [[0.0, np.inf]])
