import numpy as np
from source import parameter, cost_func, utils
from source.si import polytope_to_interval


class DynamicProgramming:
    def __init__(
        self,
        F_freq,
        window,
        beta_list,
        gamma,
        a=None,
        b=None,
        FFT_mat_a=None,
        FFT_mat_b=None,
        freq_tau_obs=None,
        SI_mode=False,
    ):
        self.F, self.freq = F_freq
        self.T = self.F.shape[1]
        self.window = window
        self.beta_list = beta_list
        self.gamma = gamma
        self.a = a
        self.b = b
        self.FFT_mat_a = FFT_mat_a
        self.FFT_mat_b = FFT_mat_b
        self.freq_tau_obs = freq_tau_obs
        self.SI_mode = SI_mode
        self.interval_ts = [[0.0, np.inf]]

    def perform_dp(self):
        cp_list = []
        K_d_list = []
        cost_list = []
        loss_dp = 0
        d_index_dp_list = []

        for i, (F_d, d) in enumerate(zip(self.F, self.freq)):
            self.beta_d = parameter.beta(self.beta_list, d, self.window)
            self.loss_min_list = [-self.beta_d, 0.0]
            self.cp_t_list = [[], [1]]
            self.selected_event_list = [[0, -self.beta_d], [0, 0]]

            for t in range(2, self.T + 1):
                self.optimal_partitioning(F_d, t, d)
                self.loss_min_list.append(self.loss_min)
                self.cp_t_list.append(self.cp_t)

            cp_list.append([0] + self.cp_t + [self.T])
            K_d = int(len(self.cp_t))
            K_d_list.append(K_d)
            cost_list.append(self.loss_min - self.beta_d * K_d)
            loss_dp += self.loss_min
            if K_d != 0:
                d_index_dp_list.append(i)

        cp_set = len(set(sum(cp_list, []))) - 2
        loss_dp += cp_set * self.gamma

        return (
            cp_list,
            cost_list,
            K_d_list,
            cp_set,
            loss_dp,
            d_index_dp_list,
            self.interval_ts,
        )

    # Dynamic Programming
    def optimal_partitioning(self, F_d, t, d):
        loss_candidate_list = [
            self.loss_min_list[i]
            + cost_func.calc_cost_seg(F_d, i, t, d, self.window)
            + self.beta_d
            for i in range(t)
        ]
        selected_index = np.argmin(loss_candidate_list)

        if self.SI_mode:
            if d in self.freq_tau_obs:
                self.selectionevent_dp(selected_index, t, d)

        self.loss_min = loss_candidate_list[selected_index]
        self.cp_t = list(self.cp_t_list[selected_index])

        if selected_index >= 2:
            self.cp_t.append(selected_index)

    def selectionevent_dp(self, selected_index, t, d):
        selected_event = [
            utils.mat_sum(
                self.selected_event_list[selected_index][0],
                cost_func.calc_cost_seg_mat(selected_index, t, d, self.window, self.T),
            ),
            self.selected_event_list[selected_index][1] + self.beta_d,
        ]
        self.selected_event_list.append(selected_event)

        for i in range(t):
            if i == selected_index:
                continue
            event = [
                utils.mat_sum(
                    self.selected_event_list[i][0],
                    cost_func.calc_cost_seg_mat(i, t, d, self.window, self.T),
                ),
                self.selected_event_list[i][1] + self.beta_d,
            ]
            cost_mat = utils.mat_diff(selected_event[0], event[0])
            penalty = selected_event[1] - event[1]
            self.interval_ts = polytope_to_interval(
                self.interval_ts,
                [d],
                [cost_mat],
                self.freq_tau_obs,
                penalty,
                self.a,
                self.b,
                self.FFT_mat_a,
                self.FFT_mat_b,
            )
