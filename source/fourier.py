import numpy as np
from numba import jit

def time_freq_analysis(x, window, freq_l, freq_h):
    if not (0 <= freq_l <= freq_h <= int(window / 2)):
        raise ValueError(
            "freq_l must be between 0 and freq_h and freq_h must be between freq_l and int(window / 2)"
        )
    
    T = int(len(x) / window) 
    F = np.zeros((window, T), dtype="complex128")
    for i in range(T):
        F[:,i] = np.fft.fft(x[window * i:window * (i + 1)]) 
    return  [F[freq_l:freq_h + 1], np.arange(freq_l, freq_h + 1)]

@jit(nopython=True)
def calc_FFT_mat(window): 
    freq_all = int(window / 2) + 1
    FFT_mat_r = np.empty((freq_all, window))
    FFT_mat_i = np.empty((freq_all, window))
    const = 2 * np.pi / window
    for d in range(freq_all):
        const_d = const * d
        FFT_mat_r[d] = np.array([np.cos(const_d * i) for i in range(window)])
        FFT_mat_i[d] = np.array([-np.sin(const_d * i) for i in range(window)])
    return [FFT_mat_r, FFT_mat_i]
