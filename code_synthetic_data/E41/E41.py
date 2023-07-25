import numpy as np
import pandas as pd
from tqdm.auto import *
import pickle
import os
import argparse

import sys
sys.path.append("../..")

import src.estimation
import src.initialization
import src.simulation
import src.postprocessing


############ ARGUMENTS AND MODEL PARAMETERS ############


EXP = 41 # experiment number (used for the seed)
K = 30 # number of experiments per parameter values
H_max = 10
n_iter = 50000 # number of MCMC steps

result_dir = f"{os.environ['B']}/E{EXP}"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

parser = argparse.ArgumentParser(description=f"Run experiment {EXP} for the BOA process estimation.")
parser.add_argument("-s", "--sim-index", type=int, nargs='+', dest="sim_index", default=[0]*7, help="List of features to use (ID between 0 and 5)")
args = parser.parse_args()

# sim_index has 7 components
# - 6 to indicate the value of the parameters
# - 1 to indicate the experiment replication number
# CAUTION : EXCEPTIONALLY, IN NOISY BOA EXPERIMENTS, EPS IS NOT GIVEN IN THE ARGUMENT: EPS IS LOOPED OVER IN THE SCRIPT
sim_index = args.sim_index
print(sim_index)
assert(len(sim_index)==7)

N_range     = [30, 50, 100]
T_range     = [5, 10]
p_ext_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H_range     = [0, 1, 2, 5]
s_range     = [0.2, 0.8]
eps_range   = [0., 0.01, 0.02, 0.05] # Noisy BOA

N     = N_range[sim_index[0]]
T     = T_range[sim_index[1]]
p_ext = p_ext_range[sim_index[2]]
H     = H_range[sim_index[3]]
s     = s_range[sim_index[4]]

for e, eps in enumerate(eps_range):
    sim_index[5] = e
    sim_name = os.environ["B"] + f"/E{EXP}/E{EXP}_" + src.postprocessing.sim_index_to_string(sim_index) + ".pkl"
    if os.path.exists(sim_name+".gz"):
        print(f"Skip eps {eps}")
        continue

    print("N:     ", N)
    print("T:     ", T)
    print("p_ext: ", p_ext)
    print("H:     ", H)
    print("s:     ", s)
    print("eps:   ", eps)

    seed = EXP + 10 * int("".join(map(str, sim_index)))


    ############ DATA GENERATION ############


    def generate_data(N, T, p_ext, H, s, eps, seed):
        np.random.seed(seed)
        init = src.initialization.init_random(N, H, s)
        return src.simulation.simulate(p_ext, H, N, T, init, eps)


    observations, seed_ages, extinctions = generate_data(
        N=N, T=T, p_ext=p_ext, H=H, s=s, eps=eps, seed=seed)

    print(observations.shape)


    ############ ESTIMATION ############


    hist = src.estimation.estimate_single_street(observations, H_max=H_max, niter=n_iter, prop=1, noisy_BOA=True)


    ############ SAVING RESULTS ############


    hist = src.postprocessing.simplify_history(hist, keep_length=n_iter//2, true_s=s)

    f = open(sim_name, "wb")
    pickle.dump(hist, f)
    f.close()

    del hist

    os.system(f"gzip {sim_name}")
