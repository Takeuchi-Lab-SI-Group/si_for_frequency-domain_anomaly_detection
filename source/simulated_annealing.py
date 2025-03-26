import numpy as np

from source import parameter, cost_func, utils
from source.si import polytope_to_interval


class SimulatedAnnealing:
    def __init__(
        self,
        F_freq,
        window,
        beta_list,
        gamma,
        cp_list,
        cost_list,
        K_d_list,
        cp_set,
        loss_dp,
        d_index_dp_list,
        seed,
        a,
        b,
        FFT_mat_a,
        FFT_mat_b,
        freq_tau_obs,
        SI_mode,
    ):
        self.F, self.freq = F_freq
        self.T = self.F.shape[1]
        self.window = window
        self.beta_list = beta_list
        self.gamma = gamma
        self.cp_list = cp_list
        self.cost_list = cost_list
        self.K_d_list = K_d_list
        self.cp_set = cp_set
        self.seed = seed
        self.loss = loss_dp
        self.d_index_dp_list = d_index_dp_list
        self.d_index_dp_len = len(d_index_dp_list)
        self.a = a
        self.b = b
        self.FFT_mat_a = FFT_mat_a
        self.FFT_mat_b = FFT_mat_b
        self.freq_tau_obs = freq_tau_obs
        self.SI_mode = SI_mode
        self.interval_ts = [[0.0, np.inf]]

    def perform_sa(
        self,
        tempinit_init=1000,
        temp_factor=1.5,
        loop_param=1,
        accepatance_rate_init=0.5,
        cooling_rate=0.8,
    ):
        rng = np.random.default_rng(self.seed)

        self.temp = tempinit_init
        iteration = int(loop_param * self.d_index_dp_len * self.T)
        while True:
            self.accept = self.reject = 0
            self.update_univariate(
                list(self.cp_list),
                list(self.cost_list),
                list(self.K_d_list),
                self.cp_set,
                rng,
                iteration,
                calc_init_temp=True,
            )
            if (self.accept) / (self.accept + self.reject) >= accepatance_rate_init:
                break
            self.temp *= temp_factor

        while True:
            self.accept = self.reject = 0
            self.update_univariate(
                list(self.cp_list),
                list(self.cost_list),
                list(self.K_d_list),
                self.cp_set,
                rng,
                iteration,
            )
            self.update_multivariate(
                list(self.cp_list),
                list(self.cost_list),
                list(self.K_d_list),
                self.cp_set,
                rng,
            )
            if self.accept == 0:
                break
            self.cool(cooling_rate)

        return self.cp_list, self.interval_ts

    def update_univariate(
        self, cp_list, cost_list, K_d_list, cp_set, rng, iteration, calc_init_temp=False
    ):
        step = 0
        while step < iteration:
            d_index_cand = self.d_index_dp_list[rng.integers(self.d_index_dp_len)]
            cp_d_list_cand = self.neighborhood_solution(cp_list[d_index_cand], rng)
            if cp_d_list_cand is None:
                continue

            cost_cand = cost_func.calc_cost(
                self.F[d_index_cand],
                self.freq[d_index_cand],
                cp_d_list_cand,
                self.window,
            )
            K_d_cand = len(cp_d_list_cand) - 2
            cp_list_cand = list(cp_list)
            cp_list_cand[d_index_cand] = cp_d_list_cand
            cp_set_cand = (
                len(set(sum([cp_list_cand[i] for i in self.d_index_dp_list], []))) - 2
            )

            penalty = (K_d_cand - K_d_list[d_index_cand]) * parameter.beta(
                self.beta_list, self.freq[d_index_cand], self.window
            ) + (cp_set_cand - cp_set) * self.gamma
            delta = (cost_cand - cost_list[d_index_cand]) + penalty
            theta = rng.random()

            cp_d_list_update, delta_status = self.update_solution(
                cp_d_list_cand, delta, theta
            )

            if self.SI_mode:
                self.selectionevent_sa(
                    [d_index_cand],
                    cp_list,
                    cp_list_cand,
                    penalty,
                    theta,
                    delta_status,
                    self.a,
                    self.b,
                    self.FFT_mat_a,
                    self.FFT_mat_b,
                )

            if cp_d_list_update is not None:
                cp_list[d_index_cand] = cp_d_list_update
                cost_list[d_index_cand] = cost_cand
                K_d_list[d_index_cand] = K_d_cand
                cp_set = cp_set_cand

            if not calc_init_temp:
                if delta_status == "accept":
                    self.loss += delta

            step += 1

        if not calc_init_temp:
            self.cp_list = cp_list
            self.cost_list = cost_list
            self.K_d_list = K_d_list
            self.cp_set = cp_set

    def neighborhood_solution(self, cp_d_list, rng):
        neighbor = rng.integers(1, 4)
        if neighbor == 1:
            cp_d_list_cand = self.add_changepoint(cp_d_list, rng)
        elif neighbor == 2:
            cp_d_list_cand = self.remove_changepoint(cp_d_list, rng)
        else:
            cp_d_list_cand = self.move_changepoint(cp_d_list, rng)
        return cp_d_list_cand

    def add_changepoint(self, cp_d_list, rng):
        cp_d_list_cand = list(cp_d_list)
        K_d = len(cp_d_list_cand) - 2

        if K_d == self.T - 1:
            return None

        cp_cand = list(set(range(1, self.T)) - set(cp_d_list_cand))
        cp = cp_cand[rng.integers(len(cp_cand))]
        return sorted(cp_d_list_cand + [cp])

    def remove_changepoint(self, cp_d_list, rng):
        cp_d_list_cand = list(cp_d_list)
        K_d = len(cp_d_list_cand) - 2

        if K_d == 0:
            return None

        candidate = cp_d_list_cand[1:-1]
        cp_d_list_cand.remove(candidate[rng.integers(len(candidate))])
        return cp_d_list_cand

    def move_changepoint(self, cp_d_list, rng):
        cp_d_list_cand = list(cp_d_list)
        K_d = len(cp_d_list_cand) - 2

        if K_d == 0:
            return None

        ordinal = rng.integers(K_d) + 1
        cp_move_prev = cp_d_list_cand[ordinal - 1]
        cp_move = cp_d_list_cand[ordinal]
        cp_move_next = cp_d_list_cand[ordinal + 1]
        move_range = list(range(cp_move_prev + 1, cp_move_next))
        move_range.remove(cp_move)
        if len(move_range) == 0:
            return None

        cp_move_new = move_range[rng.integers(len(move_range))]
        cp_d_list_cand.remove(cp_move)
        return sorted(cp_d_list_cand + [cp_move_new])

    def update_solution(self, cp_list_cand, delta, theta):
        if delta <= -self.temp * np.log(theta):
            status = "accept"
            self.accept += 1
            cp_list_update = list(cp_list_cand)
        else:
            status = "reject"
            self.reject += 1
            cp_list_update = None
        return cp_list_update, status

    def selectionevent_sa(
        self,
        d_index_list,
        cp_list,
        cp_list_cand,
        penalty,
        theta,
        delta_status,
        a,
        b,
        FFT_mat_a,
        FFT_mat_b,
    ):
        d_list = [self.freq[i] for i in d_index_list]
        if len(set(d_list) & set(self.freq_tau_obs)) != 0:
            cost_mat_sa = []

            for i, d in zip(d_index_list, d_list):
                cost_mat_series = cost_func.calc_cost_seg_mat_series(
                    cp_list[i], d, self.window, self.T
                )
                cost_mat_series_cand = cost_func.calc_cost_seg_mat_series(
                    cp_list_cand[i], d, self.window, self.T
                )
                cost_mat_sa.append(
                    utils.mat_diff(cost_mat_series_cand, cost_mat_series)
                )

            metropolis_threshold = self.temp * np.log(theta)

            if delta_status == "accept":
                self.interval_ts = polytope_to_interval(
                    self.interval_ts,
                    d_list,
                    cost_mat_sa,
                    self.freq_tau_obs,
                    penalty,
                    a,
                    b,
                    FFT_mat_a,
                    FFT_mat_b,
                    metropolis_threshold,
                )
            if delta_status == "reject":
                self.interval_ts = polytope_to_interval(
                    self.interval_ts,
                    d_list,
                    [mat * -1 for mat in cost_mat_sa],
                    self.freq_tau_obs,
                    -penalty,
                    a,
                    b,
                    FFT_mat_a,
                    FFT_mat_b,
                    -metropolis_threshold,
                )

    def update_multivariate(self, cp_list, cost_list, K_d_list, cp_set, rng):
        cp_list_cand, d_index_list = self.merge_changepoint(cp_list, rng)

        if cp_list_cand is None:
            return

        delta_cost = 0
        cost_cand_list = []
        delta_K_d = 0
        K_d_cand_list = []
        for i in d_index_list:
            cost_cand = cost_func.calc_cost(
                self.F[i], self.freq[i], cp_list_cand[i], self.window
            )
            cost_cand_list.append(cost_cand)
            delta_cost += cost_cand - cost_list[i]
            K_d_cand = len(cp_list_cand[i]) - 2
            K_d_cand_list.append(K_d_cand)
            delta_K_d += (K_d_cand - K_d_list[i]) * parameter.beta(
                self.beta_list, self.freq[i], self.window
            )
        cp_set_cand = (
            len(set(sum([cp_list_cand[i] for i in self.d_index_dp_list], []))) - 2
        )

        penalty = delta_K_d + (cp_set_cand - cp_set) * self.gamma
        delta = delta_cost + penalty
        theta = rng.random()
        cp_list_update, delta_status = self.update_solution(cp_list_cand, delta, theta)

        if self.SI_mode:
            self.selectionevent_sa(
                d_index_list,
                cp_list,
                cp_list_cand,
                penalty,
                theta,
                delta_status,
                self.a,
                self.b,
                self.FFT_mat_a,
                self.FFT_mat_b,
            )

        if cp_list_update is None:
            return

        cp_list = cp_list_update
        for num, i in enumerate(d_index_list):
            cost_list[i] = cost_cand_list[num]
            K_d_list[i] = K_d_cand_list[num]
        cp_set = cp_set_cand
        self.loss += delta
        self.cp_list = cp_list
        self.cost_list = cost_list
        self.K_d_list = K_d_list
        self.cp_set = cp_set

    def merge_changepoint(self, cp_list, rng):
        cp_list_cand = [list(cp_d_list) for cp_d_list in cp_list]
        cp_set_list = sorted(
            list(set(sum([cp_list_cand[i] for i in self.d_index_dp_list], [])))
        )
        T_K = len(cp_set_list) - 2

        if T_K <= 1:
            return None, None

        ordinal = rng.integers(T_K - 1) + 1
        cp_merge_l, cp_merge_r = cp_set_list[ordinal], cp_set_list[ordinal + 1]
        move_range = list(range(cp_merge_l, cp_merge_r + 1))
        cp_merge = move_range[rng.integers(len(move_range))]

        d_index_list = []
        for i, cp_d_list in enumerate(cp_list_cand):
            if len(cp_d_list) == 2:
                continue

            flag = False
            if cp_merge_l in cp_d_list:
                cp_d_list.remove(cp_merge_l)
                flag = True

            if cp_merge_r in cp_d_list:
                cp_d_list.remove(cp_merge_r)
                flag = True

            if flag:
                cp_list_cand[i] = sorted(cp_d_list + [cp_merge])
                d_index_list.append(i)

        return cp_list_cand, d_index_list

    def cool(self, cooling_rate=0.8):  # 0.8-0.99
        self.temp *= cooling_rate
