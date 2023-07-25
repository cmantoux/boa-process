import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from tqdm.auto import *
from numba import njit


############################################################
############### SIMULATION OF THE (NOISY) BOA PROCESS
############################################################


def simulate(p_ext, H, N, T, init, eps=0):
    """
    Simulation of the noisy BOA process
    
    p: extinction probability between 0 and 1
    H: maximum seed viability age (integer)
    N: number of plant patches
    T: number of time steps
    init: size (N) array giving the age of seeds at times T=0
    eps: probability of external colonization
    
    NB: when eps=0 (default case), the function simulates the non-noisy BOA model
    """
    O = np.zeros((T, N)) # 1 if seeds are observed in the patch, 0 otherwise
    C = np.zeros((T, N)) # age of the youngest seeds in each patch
    C[0] = init
    E = (np.random.rand(T, N) < (1-p_ext)) * 1 # extinction events (0 = extinction)
    O[0,:] = (C[0,:] <= H) * E[0,:]
    for t in range(1,T):
        C[t] = C[t-1] + 1
        for i in range(N):
            neighbors_i = list({max(0,i-1), i, min(i+1,N-1)})
            if O[t-1,i]==1: # Seed production
                C[t,neighbors_i] = 0
        O[t] = (C[t] <= H) * E[t] # Germination + Extinction
        colonization = 1*(np.random.rand(N) < eps)
        O[t] = np.maximum(O[t], colonization) # Colonization
        # O[t] = observation at time t
    return O, C, E

