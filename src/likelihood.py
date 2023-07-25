import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from tqdm.auto import *
from numba import njit


############################################################
############### FUNCTIONS TO COMPUTE THE LIKELIHOOD
############################################################



@njit
def sigmoid(logit):
    """Base function used in computations"""
    return 1/(1+np.exp(-logit))


@njit
def logit(p):
    """Inverse of sigmoid"""
    return np.log(p/(1-p))


@njit
def compute_C(O, L, H):
    """
    Computes the age of the seeds for each time step.
    O: (T, N) numpy array of 0 and 1 containing plant observations (T=time, N=patches)
    L: (N) numpy array giving the age of the seeds at time T=0
    H: maximum seed viability age (integer)
    
    Returns a (T, N) array
    """
    T, N = O.shape
    C = np.zeros((T, N)) + H + 1
    C[0] = L
    for t in range(1,T):
        for i in range(N):
            neighbors_i = np.arange(max(0,i-1), min(i+1,N-1)+1)
            if O[t-1,i]==0:
                C[t,i] = min(C[t,i], C[t-1,i] + 1)
            else:
                C[t][neighbors_i] = 0
    return C


@njit
def log_prob(O, L, p_ext, H, s, eps=0, eps_m=0.5, eps_M=1, noisy_BOA=True):
    """
    Computes log p(O, L, p_ext, H, s, eps) for the noisy (and non-noisy) BOA model
    
    O: (T, N) numpy array of 0 and 1 containing plant observations
    L: (N, H_max) numpy matrix of 0 and 1 containing the age of seeds at time T=0,
       for each patch (<=N) and each possible maximum seed viability (<=H_max)
    p_ext: extinction probability between 0 and 1
    H: maximum seed viability age (integer)
    s: (H_max) vector givins, for each possible maximal seed age, the probability of
       patches containing a viable seed at T=0
    eps: probability of external colonization
    eps_m, eps_M: parameters for the a priori distribution of the variable
                  eps (eps ~ 1/2 U(0,eps_m) + 1/2 U(0,eps_M))
    noisy_BOA: if True, the probability will use the eps parameter to include the
               colonization probability. If False, no colonization is allowed.

    NB: If noisy_BOA=False, the parameters eps, eps_m, eps_M are not used.
    """
    
    T, N = O.shape
    H_max = L.shape[1]
    
    res = 0
    if noisy_BOA:
        # p(eps) (prior probability)
        if eps > eps_M:
            res -= np.inf
        elif eps > eps_m:
            res -= np.log(1/(2*eps_M))
        else:
            res -= np.log(1/(2*eps_M) + 1/(2*eps_m))
    # p(L)
    for h in range(H_max):
        nb_young_seeds = (L[:,h] <= h).sum()
        res += nb_young_seeds * np.log(s[h]/(h+1)) + (N - nb_young_seeds) * np.log(1-s[h]) # log p(L | p_ext, H, s)
    C = compute_C(O, L[:,H], H)
    # p(O)
    for t in range(T):
        for i in range(N):
            # Here, prob = p(O[t,i]=1 | L, ext, H, s)
            if (C[t,i] <= H):        # If there is a seed:
                prob = (1-p_ext)     # -> survival iff. no extinction
            elif noisy_BOA:          # If there is no seed (and model is noisy BOA):
                prob = eps*(1-p_ext) # -> survival iff. colonization
            else:                    # If no seed and non-noisy model:
                prob = 0             # -> 100% death
            if O[t,i]==1:
                res += np.log(prob)
            else:
                res += np.log(1-prob)
    return res
