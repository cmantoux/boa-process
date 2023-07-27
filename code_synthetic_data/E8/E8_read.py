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


EXP = 8 # experiment number (used for the seed)
K = 30 # number of experiments per parameter values
H_max = 10
n_iter = 50000 # number of MCMC steps

M_range     = [1, 2, 5, 10]
N_range     = [30, 50, 100]
T_range     = [5, 10]
p_ext_range = [0.1, 0.35, 0.55, 0.75]
H_range     = [0, 1, 2, 5]
s_range     = [0.2, 0.8]
eps_range   = [0., 0.01, 0.02, 0.05] # Noisy BOA

p_c = np.array([0.461, 0.652, 0.743, 0.796, 0.831, 0.856, 0.874, 0.888, 0.899, 0.909, 0.916])

results = ["p_ext_mean", "p_ext_std", "rmse_p_ext", "H_recovery", "H_inf",
           "GER", "MaxGER", "eps_mean", "eps_std", "rmse_eps"]
results += [f"p(H={h})" for h in range(H_max+1)]
results += [f"s_{h}_mean" for h in range(H_max+1)]
results += [f"s_{h}_rmse" for h in range(H_max+1)]

sim_indices = []
for z in range(len(M_range)):
    for a in range(len(N_range)):
        for b in range(len(T_range)):
            for c in range(len(p_ext_range)):
                for d in range(len(H_range)):
                    for e in range(len(s_range)):
                        for f in range(len(eps_range)):
                            for g in range(K):
                                sim_indices.append([z, a, b, c, d, e, f, g])

result_array = xr.DataArray(np.zeros((
                        len(M_range),
                        len(N_range),
                        len(T_range),
                        len(p_ext_range),
                        len(H_range),
                        len(s_range),
                        len(eps_range),
                        K,
                        43
                    )),
                     dims=("M", "N", "T", "p_ext", "H", "s", "eps", "k", "result"),
                     coords={
                        "M": M_range,
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
   
    M     = M_range[sim_index[0]] 
    N     = N_range[sim_index[1]]
    T     = T_range[sim_index[2]]
    p_ext = p_ext_range[sim_index[3]]
    H     = H_range[sim_index[4]]
    s     = s_range[sim_index[5]]
    eps   = eps_range[sim_index[6]]
    
    rmse_p_ext = np.sqrt(((hist["p_ext"] - p_ext)**2).mean())
    rmse_eps = np.sqrt(((hist["eps"] - eps)**2).mean())

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

    z, a, b, c, d, e, f, g = sim_index
    result_array.data[z, a, b, c, d, e, f, g, results.index("rmse_p_ext")] = rmse_p_ext
    result_array.data[z, a, b, c, d, e, f, g, results.index("p_ext_mean")] = hist["p_ext"].mean()
    result_array.data[z, a, b, c, d, e, f, g, results.index("p_ext_std")] = hist["p_ext"].std()
    result_array.data[z, a, b, c, d, e, f, g, results.index("H_recovery")] = H_recovery
    result_array.data[z, a, b, c, d, e, f, g, results.index("H_inf")] = H_inf
    result_array.data[z, a, b, c, d, e, f, g, results.index("GER")] = test_1
    result_array.data[z, a, b, c, d, e, f, g, results.index("MaxGER")] = test_2
    result_array.data[z, a, b, c, d, e, f, g, results.index("eps_mean")] = hist["eps"].mean()
    result_array.data[z, a, b, c, d, e, f, g, results.index("eps_std")] = hist["eps"].std()
    result_array.data[z, a, b, c, d, e, f, g, results.index("rmse_eps")] = rmse_eps
    for h in range(H_max+1):
        result_array.data[z, a, b, c, d, e, f, g, results.index(f"p(H={h})")] = H_probs[h]
        result_array.data[z, a, b, c, d, e, f, g, results.index(f"s_{h}_mean")] = hist["s_mean"][:,h].mean()
        result_array.data[z, a, b, c, d, e, f, g, results.index(f"s_{h}_rmse")] = hist["s_rmse"][:,h].mean()

    del hist

f = open(os.environ["B"]+"/E8_results.pkl", "wb")
pickle.dump(result_array, f)
f.close()
