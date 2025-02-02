import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import pickle
import time
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor  
from source.statistical_inference import ChangepointInference

def get_p_value(parameters):
    num, window, T = parameters
    sampling = 20480 
    sigma = 1.0
    N = window * T
    number_of_freq = 3
    omegas = 2 * np.pi * np.arange(0, sampling, sampling/window)[:int(window / 2) + 1]

    rng = np.random.default_rng(seed=num)

    amplitude = rng.random(number_of_freq) 
    freq_index = rng.choice(int(window / 2) + 1, number_of_freq, replace=False)
    omega = [omegas[i] for i in freq_index] 

    t = np.arange(0, N / sampling, 1 / sampling) 
    mu = np.zeros(N)
    for f in range(number_of_freq):
        mu += amplitude[f] * np.sin(omega[f] * t)
    x = mu + rng.normal(loc=0.0, scale=sigma, size=N)

    CP_detection = ChangepointInference(x, sigma, window, seed=0) 
    cp_list_dp, cp_list_sa, sigma_est = CP_detection.detect_changepoints() 

    if len([cp for cp in cp_list_dp if len(cp) != 2]) <= 1:
        return None
    
    tau_cand = sorted(list(set(sum([cp[1:-1] for cp in cp_list_sa if len(cp) != 2], []))))

    if len(tau_cand) == 0:
        return None
    
    tau = rng.choice(tau_cand)

    try:
        start = time.time()
        naive_p_value, OC_p_value, para_p_value = CP_detection.inference(tau, sigma, parametric=True) 
        end = time.time()
        computation_time = end - start
        return naive_p_value, OC_p_value, para_p_value, computation_time
    except Exception as e:
        print(e)
        return None

window = int(sys.argv[1])
T = int(sys.argv[2])
if window == 1024:
    init = 1
    itr = 1029 # Iteration required to obtain 1000 p-values for all settings
elif window == 512:
    init = 10001
    itr = 1223 # Iteration required to obtain 1000 p-values for all settings

parameters = [(i, window, T) for i in range(init, init+itr)]

with ProcessPoolExecutor(max_workers=1) as pool:  
    result = list(
        tqdm(pool.map(  
            get_p_value,
            parameters,    
        ),
        total=itr)
    )

with open(f'fpr_{window}_{T}.pickle', mode='wb') as fo:
  pickle.dump(result, fo)
