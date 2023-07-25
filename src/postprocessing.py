import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
from tqdm.auto import *
from numba import njit


############################################################
############### PROCESSING THE MCMC RESULTS
############################################################


def simplify_history(hist, keep_length, true_s=None):
    """
    Shortens the MCMC chains and drop unused variables.
    """
    hist["p_ext"] = hist["p_ext"][-keep_length:]
    hist["H"] = hist["H"][-keep_length:]
    hist["log_lk"] = hist["log_lk"][-keep_length:]
    if "eps" in hist.keys():
        hist["eps"] = hist["eps"][-keep_length:]
    if "L" in hist.keys():
        hist["last_L"] = hist["L"][-1]
        del hist["L"]
    if "s" in hist.keys():
        hist["s_mean"] = hist["s"][-keep_length:].mean(axis=0)
        if true_s is not None:
            hist["s_rmse"] = np.sqrt(((hist["s"][-keep_length:] - true_s)**2).mean(axis=0))
        del hist["s"]
    return hist



def compare_p_ext_with_p_c(hist, p_c):
    """
    hist: object given by estimate_single_street or estimate_multiple_streets
    p_c: array with size H_max+1, giving the critical probability for each value of h <= H_max.
    
    Returns an estimation of P(p_ext < p_c(H) | O) (where both p_ext and H are random).
    Only the second half of the MCMC values is used in the computation.
    
    If there are multiple streets, the function returns the list P(p_ext[k] < p_c(H) | O), for each street k.
    """
    
    p_c = np.array(p_c) # convert p_c to a numpy array (in case it is a python list)
    niter = hist["p_ext"].shape[0]
    
    if len(hist["p_ext"].shape)==1:
        # If single street
        return (hist["p_ext"][niter//2:] < p_c[hist["H"][niter//2:]]).mean()
    else:
        # If multiple street
        return (hist["p_ext"][niter//2:] < p_c[hist["H"][niter//2:]][:,None]).mean(axis=0)

    
def compute_summary(simplified_history, p_c=None, H_true=None, p_ext_true=None, eps_true=None, street_names=None):
    """
    simplified_history: object given by simplify_history
    p_c: array with size H_max+1, giving the critical probability for each value of h <= H_max. If None, use pre-computed values.
    p_ext_true: optional true value for p_ext
    
    Given the output of estimate_multiple_streets or estimate_single_streets, computes a series of indicators
    summarizing the output of the MCMC.
    
    
    p_ext_mean, p_ext_std: posterior mean and standard deviation of p_ext
    eps_mean, eps_std: posterior mean and standard deviation of eps
    H_inf: smallest value of H st. p(H<=H_inf) >= 5%
    GER: probability p(p_ext < p_c(H) | H=H_inf, data) --> big probability = no extinction
    MaxGER: probability p(p_ext < p_c(H) | data) --> big probability = no extinction
    log_lk: model complete log-likelihood log p(observation, latent variables, parameters)
    p(H=...): posterior probability of H=...
    s_..._mean: posterior mean of s given H=...
    s_..._rmse: if simplified_history was provided a s_true, returns the posterior rmse of s given H=...
    
    Additionally, if H_true, p_ext_true and eps_true are given:
    rmse_p_ext: posterior RMSE for p_ext
    rmse_eps: posterior RMSE for eps
    H_recovery: 1 if H_inf=H_true, 0 otherwisee
    """
    
    if p_c is None:
        p_c = np.loadtxt("src/critical_prob.csv", skiprows=1, delimiter=",")[:,1]
    
    hist = simplified_history
    H_max = hist["s_mean"].shape[-1] - 1
    if street_names is not None:
        M = len(street_names)
    
    H_values, H_counts = np.unique(hist["H"], return_counts=True)
    cdf = (H_counts / H_counts.sum()).cumsum()
    i = 0
    while cdf[i] < 0.05: i+=1
    H_inf = H_values[i]

    H_probs = [(hist["H"]==h).mean() for h in range(H_max+1)]

    if len(hist["p_ext"].shape)==1:
        test_1 = (hist["p_ext"] < p_c[hist["H"]]).mean()
        idx = hist["H"]==H_inf
        test_2 = (hist["p_ext"][idx] < p_c[H_inf]).mean()
    else:
        test_1 = (hist["p_ext"] < p_c[hist["H"]][:,None]).mean(axis=0)
        idx = hist["H"]==H_inf
        test_2 = (hist["p_ext"][idx] < p_c[H_inf]).mean(axis=0)
    
    result = {}
    result["H_inf"] = H_inf
    if H_true is not None:
        H_recovery = int(H_inf == H_true)
        result["H_recovery"] = H_recovery
    if p_ext_true is not None:
        if street_names is not None:
            rmse_p_ext = {street_name[k]: np.sqrt(((hist["p_ext"] - p_ext_true)**2)[:,k].mean()) for k in range(M)}
        else:
            rmse_p_ext = np.sqrt(((hist["p_ext"] - p_ext_true)**2).mean(axis=0))
        result["rmse_p_ext"] = rmse_p_ext
    if "eps" in hist:
        result["eps_mean"] = hist["eps"].mean()
        result["eps_std"]  = hist["eps"].std()
        if eps_true is not None:
            rmse_eps = np.sqrt(((hist["eps"] - eps_true)**2).mean())
            result["rmse_eps"] = rmse_eps
    for h in range(H_max+1):
        result[f"p(H={h})"] = H_probs[h]
    for h in range(H_max+1):
        result[f"s_{h}_mean"] = hist["s_mean"][...,h].mean()
        if "s_rmse" in hist.keys():
            result[f"s_{h}_rmse"] = hist["s_rmse"][...,h].mean()

    if street_names is not None:
        result["p_ext_mean"] = {street_names[k]: hist["p_ext"][:,k].mean() for k in range(M)}
        result["p_ext_std"]  = {street_names[k]: hist["p_ext"][:,k].std() for k in range(M)}
        result["GER"] = {street_names[k]: 1 - test_1[k] for k in range(M)}
        result["MaxGER"]  = {street_names[k]: 1 - test_2[k] for k in range(M)}
        result["log_lk"] = {street_names[k]: hist["log_lk"][-1,k] for k in range(M)}
    else:
        result["p_ext_mean"] = hist["p_ext"].mean(axis=0)
        result["p_ext_std"]  = hist["p_ext"].std(axis=0)
        result["GER"] = 1 - test_1
        result["MaxGER"] = 1 - test_2
        result["log_lk"] = hist["log_lk"][-1]
    return result


def sim_index_to_string(sim_index):
    """
    Function used to name the results of the experiments on synthetic data
    """
    
    tab = [
       "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
       "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
       "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
       "U", "V", "W", "X", "Y", "Z"]
    s = ""
    for i in sim_index:
        s += tab[i]
    return s


############################################################
############### PLOTTING THE RESULTS
############################################################


def plot_convergence(hist, p_ext_true=None, H_true=None, eps_true=None):
    """
    hist: object given by estimate_single_street or estimate_multiple_streets
    other parameters (optional): true hidden value of the parameters
    """
    niter = len(hist["s"])
    font = 14

    if "eps" in hist:
        cols = 3
    else:
        cols = 2
        
    H_min, H_max = hist["H"][niter//2:].min(), hist["H"][niter//2:].max()
        
    figure(figsize=(cols*4,8), dpi=70)
    k = 1
    
    subplot(2,cols,k); k+=1
    plot(hist["p_ext"])
    if p_ext_true is not None:
        axhline(p_ext_true, color="red")
    title("p_ext", fontsize=font)
    
    subplot(2,cols,k); k+=1
    plot(hist["H"])
    yticks(np.arange(min(hist["H"]), max(hist["H"])+1))
    if H_true is not None:
        axhline(H_true, color="red")
    ylim(0)
    title("H", fontsize=font)
    
    if "eps" in hist:
        subplot(2,cols,k); k+=1
        plot(hist["eps"])
        if eps_true is not None:
            axhline(eps_true, color="red")
        title("eps", fontsize=font)

    subplot(2,cols,k); k+=1
    if len(hist["p_ext"].shape)==1:
        hist["p_ext"] = hist["p_ext"].reshape(-1,1)
    for m in range(hist["p_ext"].shape[1]):
        sns.kdeplot(hist["p_ext"][niter//2:,m])
    a, b = xlim()
    xlim(max(0,a),min(1,b))
    if p_ext_true is not None:
        axvline(p_ext_true, color="red")
    title("p_ext", fontsize=font)
    
    subplot(2,cols,k); k+=1
    plt.hist(hist["H"][niter//2:],
            bins=np.arange(H_min, H_max+2)-0.5)
    xticks(np.arange(H_min, H_max+1))
    if H_true is not None:
        axvline(H_true, color="red")
    title("H", fontsize=font)
    
    if "eps" in hist:
        subplot(2,cols,k); k+=1
        plt.hist(hist["eps"][niter//2:])
        if eps_true is not None:
            axvline(eps_true, color="red")
        title("eps", fontsize=font)

    subplots_adjust()
    fig_title = "MCMC convergence result"
    if hist["p_ext"].shape[1] != 1:
        fig_title += " (one color per street)"
    suptitle(fig_title, fontsize=font+3)
    tight_layout()


def plot_posterior(simplified_history, summary, street_names=None, street_design=None, H_output=None, p_ext_output=None, plot_size=5):
    """
    simplified_history: object given by simplify_history
    summary: object given by compute_summary
    street_names: list of street names
    
    Plots the posterior distribution of p_ext (for each street) and of H given the observations.
    """
    
    plt.figure(figsize=(plot_size,plot_size))
    probas_H = [summary[f"p(H={h})"] for h in range(11)]
    H_inf = summary["H_inf"]
    colors = ["tab:orange" if h==H_inf else "tab:blue" for h in range(11)]
    plt.bar(x=range(11), height=probas_H, color=colors)
    plt.xticks(range(11), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r"Value of $H$", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.tight_layout()
    if H_output is not None:
        plt.savefig(H_output)
    plt.show()
    
    plt.figure(figsize=(plot_size,plot_size))
    M = simplified_history["p_ext"].shape[1]
    if street_names is None:
        street_names = [None]*M
    cmap = matplotlib.cm.get_cmap("tab20")
    color_order = [0,2,4,6,8,10,12,14,16,18,20]
    linestyles = ["-", "--", "-."]
    for m in range(M):
        street = street_names[m]
        if street_design is not None:
            color = color_order[street_design.loc[street]["color"]]
            ls = street_design.loc[street]["linestyle"]
        else:
            color, ls = color_order[m], 0
        sns.kdeplot(simplified_history["p_ext"][:,m], color=cmap(color), ls=linestyles[ls], label=street)
    plt.xlim()
    plt.xlabel(r"Value of $p_{ext}$", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    if street_names[0] is not None:
        plt.legend(fontsize=12)
    plt.tight_layout()
    if p_ext_output is not None:
        plt.savefig(p_ext_output)
    plt.show()