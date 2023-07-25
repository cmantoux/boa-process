import numpy as np
import pandas as pd
from tqdm.auto import *
import pickle
import os
import xarray as xr

import sys
sys.path.append("../..")

import src.postprocessing



############ ARGUMENTS AND MODEL PARAMETERS ############


EXP = 1 # experiment number (used for the seed)
K = 30 # number of experiments per parameter values
H_max = 10
n_iter = 50000 # number of MCMC steps

N_range     = [10, 30, 50, 100]
T_range     = [3, 5, 10, 20]
p_ext_range = [0.1, 0.35, 0.55, 0.75]
H_range     = [0, 1, 2, 5, 10]
s_range     = [0.2, 0.8]
eps_range   = [0] # Non-noisy BOA

p_c = np.array([0.461, 0.652, 0.743, 0.796, 0.831, 0.856, 0.874, 0.888, 0.899, 0.909, 0.916])

results = ["p_ext_mean", "p_ext_std", "rmse_p_ext", "H_recovery", "H_inf",
           "extinction_test_1", "extinction_test_2"]
results += [f"p(H={h})" for h in range(H_max+1)]
results += [f"s_{h}_mean" for h in range(H_max+1)]
results += [f"s_{h}_rmse" for h in range(H_max+1)]

sim_indices = []
for a in range(len(N_range)):
    for b in range(len(T_range)):
        for c in range(len(p_ext_range)):
            for d in range(len(H_range)):
                for e in range(len(s_range)):
                    for f in range(len(eps_range)):
                        for g in range(K):
                            sim_indices.append([a, b, c, d, e, f, g])

result_array = xr.DataArray(np.zeros((
                        len(N_range),
                        len(T_range),
                        len(p_ext_range),
                        len(H_range),
                        len(s_range),
                        len(eps_range),
                        K,
                        40
                    )),
                     dims=("N", "T", "p_ext", "H", "s", "eps", "k", "result"),
                     coords={
                        "N": N_range,
                        "T": T_range,
                        "p_ext": p_ext_range,
                        "H": H_range,
                        "s": s_range,
                        "eps": eps_range,
                        "k": range(K),
                        "result": results
                     }
                    )


for sim_index in tqdm(sim_indices):
    sim_name = os.environ["B"] + f"/E{EXP}/E{EXP}_" + src.postprocessing.sim_index_to_string(sim_index) + ".pkl"
    os.system(f"gunzip {sim_name}")
    f = open(sim_name, "rb")
    hist = pickle.load(f)
    f.close()
    os.system(f"gzip {sim_name}")

    N     = N_range[sim_index[0]]
    T     = T_range[sim_index[1]]
    p_ext = p_ext_range[sim_index[2]]
    H     = H_range[sim_index[3]]
    s     = s_range[sim_index[4]]
    eps   = eps_range[sim_index[5]]
    
    rmse_p_ext = np.sqrt(((hist["p_ext"] - p_ext)**2).mean())
    
    H_values, H_counts = np.unique(hist["H"], return_counts=True)
    cdf = (H_counts / H_counts.sum()).cumsum()
    i = 0
    while cdf[i] < 0.05: i+=1
    H_inf = H_values[i]
    H_recovery = int(H_inf == H)
   
    H_probs = [(hist["H"]==h).mean() for h in range(H_max+1)]

    test_1 = (hist["p_ext"] < p_c[hist["H"]]).mean()
    idx = hist["H"]==H_inf
    test_2 = (hist["p_ext"][idx] < p_c[H_inf]).mean()

    a, b, c, d, e, f, g = sim_index
    result_array.data[a, b, c, d, e, f, g, results.index("rmse_p_ext")] = rmse_p_ext
    result_array.data[a, b, c, d, e, f, g, results.index("p_ext_mean")] = hist["p_ext"].mean()
    result_array.data[a, b, c, d, e, f, g, results.index("p_ext_std")] = hist["p_ext"].std()
    result_array.data[a, b, c, d, e, f, g, results.index("H_recovery")] = H_recovery
    result_array.data[a, b, c, d, e, f, g, results.index("H_inf")] = H_inf
    result_array.data[a, b, c, d, e, f, g, results.index("extinction_test_1")] = test_1
    result_array.data[a, b, c, d, e, f, g, results.index("extinction_test_2")] = test_2
    for h in range(H_max+1):
        result_array.data[a, b, c, d, e, f, g, results.index(f"p(H={h})")] = H_probs[h]
        result_array.data[a, b, c, d, e, f, g, results.index(f"s_{h}_mean")] = hist["s_mean"][h]
        result_array.data[a, b, c, d, e, f, g, results.index(f"s_{h}_rmse")] = hist["s_rmse"][h]

    del hist

f = open(os.environ["B"]+f"/E{EXP}_results.pkl", "wb")
pickle.dump(result_array, f)
f.close()
