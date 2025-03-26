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
    num, window, delta = parameters
    sampling = 20480
    T = 60
    sigma = 1.0
    N = window * T
    number_of_freq = 3
    omegas = (
        2 * np.pi * np.arange(0, sampling, sampling / window)[: int(window / 2) + 1]
    )

    rng = np.random.default_rng(seed=num)

    amplitude = rng.random(number_of_freq)
    freq_index = rng.choice(int(window / 2) + 1, number_of_freq, replace=False)
    omega = [omegas[i] for i in freq_index]

    step = 2
    t = np.arange(0, N / sampling, 1 / sampling)
    mu = np.zeros((N))
    for f in range(number_of_freq):
        boundary_1 = int(N / 3) + window * (f - 1) * step
        boundary_2 = int(2 * N / 3) + window * (f - 1) * step
        mu[:boundary_1] += amplitude[f] * np.sin(omega[f] * t[:boundary_1])
        mu[boundary_1:boundary_2] += (amplitude[f] + delta) * np.sin(
            omega[f] * t[boundary_1:boundary_2]
        )
        mu[boundary_2:] += (amplitude[f] + 2 * delta) * np.sin(
            omega[f] * t[boundary_2:]
        )
    x = mu + rng.normal(loc=0.0, scale=sigma, size=N)

    kappa = 0.5
    CP_detection = ChangepointInference(x, sigma, window, kappa, seed=0)
    cp_list_dp, cp_list_sa, sigma_est = CP_detection.detect_changepoints()

    if len([cp for cp in cp_list_dp if len(cp) != 2]) <= 1:
        return None

    # Correctly detetcted CP is tested.
    tau_1 = int(T / 3)
    tau_2 = int(2 * T / 3)
    tau_cand_all = [
        [tau_1 - 2, tau_1 - 1, tau_1, tau_1 + 1, tau_1 + 2],
        [tau_2 - 2, tau_2 - 1, tau_2, tau_2 + 1, tau_2 + 2],
    ]
    tau_cand_index = rng.permutation([0, 1])

    for index in tau_cand_index:
        tau_cand = tau_cand_all[index]
        tau_cand_perm = rng.permutation(tau_cand)

        for tau in tau_cand_perm:
            freq_tau = [
                d for i, d in enumerate(CP_detection.F_freq[1]) if tau in cp_list_sa[i]
            ]

            if set(freq_tau).issubset(set(freq_index)) and len(freq_tau) >= 1:
                tau_range = [
                    tau_cand[num * step]
                    for d in freq_tau
                    for num, f in enumerate(freq_index)
                    if f == d
                ]
                tau_min, tau_max = min(tau_range), max(tau_range)

                if tau_min <= tau <= tau_max:  # correctly detetcted
                    try:
                        start = time.time()
                        naive_p_value, OC_p_value, para_p_value = (
                            CP_detection.inference(tau, sigma, parametric=True)
                        )
                        end = time.time()
                        computation_time = end - start
                        return naive_p_value, OC_p_value, para_p_value, computation_time
                    except Exception as e:
                        print(e)
                        return None

    return None


window = int(sys.argv[1])
delta = float(sys.argv[2])
if window == 1024:
    init = 1
    itr = 2744  # Iteration required to obtain 1000 p-values for all settings
elif window == 512:
    init = 10001
    itr = 3745  # Iteration required to obtain 1000 p-values for all settings

parameters = [(i, window, delta) for i in range(init, init + itr)]

with ProcessPoolExecutor(max_workers=1) as pool:
    result = list(
        tqdm(
            pool.map(
                get_p_value,
                parameters,
            ),
            total=itr,
        )
    )

with open(f"tpr_{window}_{delta}.pickle", mode="wb") as fo:
    pickle.dump(result, fo)
