import numpy as np
from itertools import pairwise
from source import fourier
from source.parameter import Penalty
from source.dynamic_programming import DynamicProgramming
from source.simulated_annealing import SimulatedAnnealing
from source.si import intersection


class ChangepointDetection:
    def __init__(
        self, 
        x, 
        sigma, 
        window, 
        freq_l=0, 
        freq_h=None, 
        seed=0, 
        a=None, 
        b=None, 
        FFT_mat_a=None, 
        FFT_mat_b=None, 
        freq_tau_obs=None, 
        SI_mode=False
    ):
        self.x = x[:window * int(len(x) / window)]
        self.sigma = sigma
        self.window = window
        self.freq_l = freq_l
        if freq_h is None:
            self.freq_h = int(window / 2)
        else:
            self.freq_h = freq_h
        self.seed = seed
        self.a = a
        self.b = b
        self.FFT_mat_a = FFT_mat_a
        self.FFT_mat_b = FFT_mat_b
        self.freq_tau_obs = freq_tau_obs
        self.SI_mode = SI_mode
        
    def detect_changepoints(self):
        self.F_freq = fourier.time_freq_analysis(self.x, self.window, self.freq_l, self.freq_h)

        penalty_cp = Penalty(self.F_freq, self.sigma, self.window)
        penalty_cp.penalty_parameter()
        self.beta_list, self.gamma = penalty_cp.beta_list, penalty_cp.gamma

        dynamic_programming_cp = DynamicProgramming(
            self.F_freq, self.window, self.beta_list, self.gamma, self.a, self.b, 
            self.FFT_mat_a, self.FFT_mat_b, self.freq_tau_obs, self.SI_mode
        )
        
        (
            self.cp_list_dp,
            cost_list_dp,
            K_d_list_dp,
            cp_set_dp,
            loss_dp,
            d_index_dp_list,
            interval_dp,
        ) = dynamic_programming_cp.perform_dp()

        if len(d_index_dp_list) <= 1:
            self.cp_list_sa = list(self.cp_list_dp)

            if self.SI_mode:
                self.interval_ts = interval_dp
                return
            else:
                sigma_est = self.estimate_variance()
                return self.cp_list_dp, self.cp_list_sa, sigma_est

        simulated_annealing_cp = SimulatedAnnealing(
            self.F_freq, self.window, self.beta_list, self.gamma, self.cp_list_dp, cost_list_dp, 
            K_d_list_dp, cp_set_dp, loss_dp, d_index_dp_list, self.seed, self.a, self.b, 
            self.FFT_mat_a, self.FFT_mat_b, self.freq_tau_obs, self.SI_mode
        ) 
        self.cp_list_sa, interval_sa = simulated_annealing_cp.perform_sa()

        if self.SI_mode:
            self.interval_ts = intersection(interval_dp, interval_sa)
            return
        else:
            sigma_est = self.estimate_variance()
            return self.cp_list_dp, self.cp_list_sa, sigma_est
    
    def estimate_variance(self): 
        var_est_list = []
        for cp_d_list, F_d in zip(self.cp_list_sa, self.F_freq[0]):
            var_est = 0

            for m, n in pairwise(cp_d_list):
                F_d_mn = F_d[m:n]
                if len(F_d_mn) == 1:
                    continue

                var = (np.var(np.real(F_d_mn), ddof=1) + np.var(np.imag(F_d_mn), ddof=1)) / (self.window)
                if var > var_est:
                    var_est = var

            var_est_list.append(var_est)
        
        sigma_est = np.average(np.sqrt(var_est_list))
        
        return sigma_est
