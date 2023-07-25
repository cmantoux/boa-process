import numpy as np
from tqdm.auto import *

from src.likelihood import *


############################################################
############### PARAMETER ESTIMATION WITH MCMC
############################################################



def estimate_single_street(O, H_max=10, niter=100, prop=0.05, eps_m=0.005, eps_M=0.05, noisy_BOA=True, fix_H=None, verbose=True):
    """
    See the documentation of estimate_multiple_streets.
    """
    
    hist = estimate_multiple_streets([O], H_max=H_max, niter=niter, prop=prop, eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA, fix_H=fix_H, verbose=verbose)
    hist["p_ext"] = hist["p_ext"][:,0]
    hist["s"] = hist["s"][:,0,:]
    hist["log_lk"] = hist["log_lk"][:,0]
    hist["L"] = np.array([
        hist["L"][it][0] for it in range(niter)
    ])
    
    return hist


def estimate_multiple_streets(Os, H_max=10, niter=100, prop=0.05, eps_m=0.005, eps_M=0.05, noisy_BOA=True, fix_H=None, verbose=True):
    """
    We refer the reader to NOTATIONS.md for the interpretation of the variable names.
    
    Os: list of arrays containing data observations
    H_max: maximal value of H used in the estimation
    niter: number of MCMC steps
    prop: initial value of the Metropolis step size (tuned adaptively in the MCMC)
    eps_m, eps_M: lower and upper values for the spike and slab prior of eps
    A priori, eps has 50% chances to be chosen uniformly in [0, eps_m] and 50% uniformly in [0, eps_M]
    noisy_BOA: if False, the MCMC will fix eps=0, otherwise it will try to estimate eps
    fix_H: if no argument is given, the function will estimate H jointly with the other variables.
           Else, H will be fixed at the value fix_H.
    
    This function implements a Metropolis within Gibbs sampler on the BOA and noisy BOA processes.
    It returns the list of values taken by the model parameters along the convergence.
    
    The algorithm consists in a main loop sampling alternatively on p_ext, H, s, eps and L.
    As an important remark, the variable L has H_max+1 separate versions (and hence has shape
    (N, H_max+1) instead of (N)), one for each possible value of H.
    This allows avoiding changes in the viability of the seeds when the value of H is updated.
    We also estimate H_max+1 separate versions of s, one for each value of H: s[h] represents the
    proportion of viable seeds in L[h]. At each step of the loop, all the H_max+1 values are updated,
    regardless of the value of H. This trick avoids a single seed configuration L[H] of a given H
    getting so optimized that other configurations L[H_new] (and thus new values of H) have no chance
    to be selected by a Metropolis-Hastings transition.
    """
    
    #### INITIALIZATION ####
    
    M = len(Os) # Number of streets
    
    logit_p_ext = np.zeros(M)
    logit_s = np.zeros((M, H_max+1))
    logit_eps = logit(eps_m)
    H = 0
    L = np.empty(M, dtype=object)
    H_min = 0
    log_lk = np.zeros(M) # log p(O, L, eps, H=h) for each h
    
    for k in range(M):
        T, N = Os[k].shape
        L[k] = np.zeros((N, H_max+1))
    
    # Find the minimal value of H with non-zero likelihood
    if fix_H is None:
        H = H_min
        while log_lk.sum()==-np.inf:
            H += 1
            log_lk = np.array([
                log_prob(Os[k], L[k], p_ext=sigmoid(logit_p_ext[k]), H=h,
                         s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps),
                         eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)
                for k in range(M)
            ])
        H_min = max(H, H_min)
    else:
        H = fix_H

    hist = {
        "p_ext":  np.zeros((niter, M)),
        "H":      np.zeros(niter, dtype=int),
        "s":      np.zeros((niter, M, H_max+1)),
        "eps":    np.zeros(niter),
        "L":      np.empty(niter, dtype=object),
        "log_lk": np.zeros((niter, M))
    }
    
    
    # Parameters for the adaptive adjustment of the proposition variances
    target_mcmc_rate = 0.3 # Asymptotically, ~30% transitions will be accepted
    block_size = 50        # Proposition variances are updated every 50 iterations
    accepts_p, accepts_H, accepts_s, accepts_eps, accepts_L = 0, 0, 0, 0, 0
    block_accepts_p, block_accepts_s, block_accepts_eps = 0, 0, 0
    prop_p, prop_s, prop_eps = prop, prop, prop
    
    
    
    #### MAIN LOOP ####
    
    iterator = trange(niter) if verbose else range(niter)
    for it in iterator:
        
        # Sample H
        if fix_H is None:
            H_new = np.random.randint(H_min, H_max+1)
            new_log_lk = np.array([
                log_prob(Os[k], L[k], p_ext=sigmoid(logit_p_ext[k]), H=H_new,
                         s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps), eps_m=eps_m,
                         eps_M=eps_M, noisy_BOA=noisy_BOA)
                for k in range(M)
            ])
            if np.log(np.random.rand()) < new_log_lk.sum() - log_lk.sum():
                H = H_new
                log_lk = new_log_lk
                accepts_H += 1
        
        if noisy_BOA:
            # Sample eps
            logit_eps_new = logit_eps + prop_eps * np.random.randn()
            new_log_lk = np.array([
                log_prob(Os[k], L[k], p_ext=sigmoid(logit_p_ext[k]), H=H,
                         s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps_new),
                         eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)
                for k in range(M)
            ])
            if np.log(np.random.rand()) < new_log_lk.sum() - log_lk.sum():
                logit_eps = logit_eps_new
                log_lk = new_log_lk
                accepts_eps += 1/M
                block_accepts_eps += 1/M

        for k in range(M):
            T, N = Os[k].shape

            # Sample p_ext
            logit_p_ext_new = logit_p_ext[k] + prop_p * np.random.randn()
            new_log_lk = log_prob(Os[k], L[k], p_ext=sigmoid(logit_p_ext_new), H=H,
                                  s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps),
                                  eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)
            if np.log(np.random.rand()) < new_log_lk - log_lk[k]:
                logit_p_ext[k] = logit_p_ext_new
                log_lk[k] = new_log_lk
                accepts_p += 1/M
                block_accepts_p += 1/M
            
            # Sample s
            logit_s_new = logit_s[k] + prop_s * np.random.randn(H_max+1)
            new_log_lk = log_prob(Os[k], L[k], p_ext=sigmoid(logit_p_ext[k]), H=H,
                                  s=sigmoid(logit_s_new), eps=sigmoid(logit_eps),
                                  eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)
            if np.log(np.random.rand()) < new_log_lk - log_lk[k]:
                logit_s[k] = logit_s_new
                log_lk[k] = new_log_lk
                accepts_s += 1/M
                block_accepts_s += 1/M

            # Sample L
            idx = np.random.randint(N) # At each MCMC step, we only update one tree base
            for h in range(H_max+1):
                L_new = L[k].copy()
                old_log_lk = log_prob(Os[k], L[k], p_ext=sigmoid(logit_p_ext[k]), H=h,
                                      s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps),
                                      eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)
                
                # Generate a random proposal:
                L_new[idx,h] = np.random.randint(0, h+2)
                new_log_lk = log_prob(Os[k], L_new, p_ext=sigmoid(logit_p_ext[k]), H=h,
                                      s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps),
                                      eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)

                if np.log(np.random.rand()) < (new_log_lk - old_log_lk):
                    L[k] = L_new
                    accepts_L += 1/((H_max+1)*M)
                    if h==H:
                        log_lk[k] = new_log_lk
                    else:
                        log_lk[k] = log_prob(Os[k], L_new, p_ext=sigmoid(logit_p_ext[k]), H=H,
                                             s=sigmoid(logit_s[k]), eps=sigmoid(logit_eps),
                                             eps_m=eps_m, eps_M=eps_M, noisy_BOA=noisy_BOA)
        
        hist["p_ext"][it] = sigmoid(logit_p_ext)
        hist["H"][it] = H
        hist["s"][it] = sigmoid(logit_s)
        hist["eps"][it] = sigmoid(logit_eps)
        hist["L"][it] = L.copy()
        hist["log_lk"][it] = log_lk
        
        if it%block_size==0:
            #### Adaptive step ####
            # Change the proposal variances to get closer to the optimal acceptance rate
            g = min(1, 3/(it+1)**0.55)

            rate_p   = block_accepts_p / block_size
            rate_s   = block_accepts_s / block_size
            if noisy_BOA:
                rate_eps = block_accepts_eps / block_size
            
            D_p   = 2 * (rate_p > target_mcmc_rate) - 1
            D_s   = 2 * (rate_s > target_mcmc_rate) - 1
            if noisy_BOA:
                D_eps = 2 * (rate_eps > target_mcmc_rate) - 1
            
            prop_p   = np.exp(np.log(prop_p) + g * D_p)
            prop_s   = np.exp(np.log(prop_s) + g * D_s)
            if noisy_BOA:
                prop_eps = np.exp(np.log(prop_eps) + g * D_eps)
            
            block_accepts_p   = 0
            block_accepts_s   = 0
            if noisy_BOA:
                block_accepts_eps = 0
            
    if verbose:
        if noisy_BOA:
            print("Acceptance rates on (p_ext, H, s, eps, L):")
            print(round(accepts_p/niter, 2),
                  round(accepts_H/niter, 2),
                  round(accepts_s/niter, 2),
                  round(accepts_eps/niter, 2),
                  round(accepts_L/niter, 2))
        else:
            del hist["eps"]
            print("Acceptance rates on (p_ext, H, s, L):")
            print(round(accepts_p/niter, 2),
                  round(accepts_H/niter, 2),
                  round(accepts_s/niter, 2),
                  round(accepts_L/niter, 2))
    
    return hist