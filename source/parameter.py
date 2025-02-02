import numpy as np


class Penalty:
    def __init__(self, F_freq, sigma, window):
        self.F, self.freq = F_freq
        self.sigma = sigma
        self.window = window
    
    def penalty_parameter(self):
        T = self.F.shape[1]
        self.beta_list = np.array([3 * self.window * (self.sigma ** 2) * np.log(T), 
                                   2 * self.window * (self.sigma ** 2) * np.log(T)]) 
        self.gamma = 0.5 * self.window * (self.sigma ** 2) * np.log(T)

def beta(beta_list, d, window):
    if d != 0 and d != window / 2:
        return beta_list[0]
    else:
        return beta_list[1]
