import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.gridspec as gridspec
import xarray as xr
import pickle

import sys
sys.path.append("..")

from src.estimation import *
from src.initialization import *
from src.simulation import *
from src.postprocessing import *




figures_folder = "figures" # folder where the figures will be saved
figures_format = "png" # can be changed, e.g. to pdf or eps
show_plots = False # If set to False, the figures will only be saved and not shown on screen





linestyles = ['-', '--', '-.', ':']
markers = ['o', 'v', 's', '+']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

result_folder = "results"


def load_file(s):
    f = open(s, "rb")
    results = pickle.load(f)
    f.close()
    return results


H_range = [0, 1, 2, 5]

E1_results  = load_file(f"{result_folder}/E1_results.pkl")
E21_results = load_file(f"{result_folder}/E21_results.pkl")
E22_results = {}
for H in H_range:
    E22_results[H] = load_file(f"{result_folder}/E22_results_{H}.pkl")
E3_results  = load_file(f"{result_folder}/E3_results.pkl")
E41_results = load_file(f"{result_folder}/E41_results.pkl")
E42_results = {}
for H in H_range:
    E42_results[H] = load_file(f"{result_folder}/E42_results_{H}.pkl")

E5_results_noisy    = load_file(f"{result_folder}/E5_results_noisy.pkl")
E5_results_nonnoisy = load_file(f"{result_folder}/E5_results_nonnoisy.pkl")
E6_results_noisy    = load_file(f"{result_folder}/E6_results_noisy.pkl")
E6_results_nonnoisy = load_file(f"{result_folder}/E6_results_nonnoisy.pkl")
E7_results_noisy    = load_file(f"{result_folder}/E7_results_noisy.pkl")
E7_results_nonnoisy = load_file(f"{result_folder}/E7_results_nonnoisy.pkl")
E8_results = load_file(f"{result_folder}/E8_results.pkl")


p_c = np.array([0.461, 0.652, 0.743, 0.796, 0.831, 0.856, 0.874, 0.888, 0.899, 0.909, 0.916]) # critical probabilities p_c(H)




# 1) Assessment of the extinction risk




## 1. Nice data


# Plot with E21, E22, E41, E42
figure_titles = [
    "Fig. C.1",
    "Fig. C.2",
    "Fig. C.3",
    "Fig. C.4",
]
fig_count = 0
for N in [50,100]:
    for T in [5,10]:
        figure(figsize=(10,8))
        for i, H in enumerate(H_range):
            subplot(2,2,i+1)
            a = E21_results.sel(N=N, T=T, result="GER").mean(dim=["s", "k"])
            b = E22_results[H].sel(N=N, T=T, result="GER").mean(dim=["s", "k"])

            x1, y1 = a.coords["p_ext"].data, a.sel(H=H).data
            x2, y2 = b.coords["p_ext"].data, b.sel(H=H).data
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            order = x.argsort()
            x = x[order]
            y = y[order]

            for e, eps in enumerate(a.coords["eps"].data):
                plot(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", ls=linestyles[e])

            a = E41_results.sel(N=N, T=T, result="GER").mean(dim=["s", "k"])
            b = E42_results[H].sel(N=N, T=T, result="GER").mean(dim=["s", "k"])

            x1, y1 = a.coords["p_ext"].data, a.sel(H=H).data
            x2, y2 = b.coords["p_ext"].data, b.sel(H=H).data
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            order = x.argsort()
            x = x[order]
            y = y[order]

            for e, eps in enumerate(a.coords["eps"].data):
                if eps != 0:
                    plot(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", ls=linestyles[e])
            axvline(p_c[H], color="black", label=f"$p_c({H})$")
            xlabel(r"Patch extinction probability $p_{ext}$", fontsize=15)
            xticks(np.linspace(0.1, 0.9, 9))
            ylabel("Average GER", fontsize=15)
            ylim(-0.05,1.05)
            legend(fontsize=13)
            title(f"H = {H}", fontsize=15)
        tight_layout()
        savefig(f"{figures_folder}/{figure_titles[fig_count]} - GER - N={N} - T={T}.{figures_format}")
        fig_count += 1
        if show_plots: show()
        close("all")

# Plot with E21, E22, E41, E42
figure_titles = [
    "Fig. C.5",
    "Fig. C.6",
    "Fig. C.7",
    "Fig. C.8",
]
fig_count = 0
for N in [50,100]:
    for T in [5,10]:
        figure(figsize=(10,8))
        for i, H in enumerate(H_range):
            subplot(2,2,i+1)
            a = E21_results.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])
            b = E22_results[H].sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

            x1, y1 = a.coords["p_ext"].data, a.sel(H=H).data
            x2, y2 = b.coords["p_ext"].data, b.sel(H=H).data
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            order = x.argsort()
            x = x[order]
            y = y[order]

            for e, eps in enumerate(a.coords["eps"].data):
                plot(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", ls=linestyles[e])

            a = E41_results.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])
            b = E42_results[H].sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

            x1, y1 = a.coords["p_ext"].data, a.sel(H=H).data
            x2, y2 = b.coords["p_ext"].data, b.sel(H=H).data
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            order = x.argsort()
            x = x[order]
            y = y[order]

            for e, eps in enumerate(a.coords["eps"].data):
                if eps != 0:
                    plot(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", ls=linestyles[e])
            axvline(p_c[H], color="black", label=f"$p_c({H})$")
            xlabel(r"Patch extinction probability $p_{ext}$", fontsize=15)
            xticks(np.linspace(0.1, 0.9, 9))
            ylim(-0.05,1.05)
            ylabel("Average MaxGER", fontsize=15)
            legend(fontsize=13)
            title(f"H = {H}", fontsize=15)
        tight_layout()
        savefig(f"{figures_folder}/{figure_titles[fig_count]} - MaxGER - N={N} - T={T}.{figures_format}")
        fig_count += 1
        if show_plots: show()
        close("all")

## 2. Noisy data

# Average MaxGER distribution across experiments with eps=0.05

fig_count = 0
figure_titles = [
    "Fig. C.9",
    "Fig. C.10",
    "Fig. C.11",
    "Fig. C.12",
]
for N in [50, 100]:
    for T in [5, 10]:
        figure(figsize=(10,8))
        for i, H in enumerate(H_range):
            subplot(2,2,i+1)

            a = E41_results.sel(N=N, T=T, result="extinction_test_2").mean(dim=["s", "k"])
            a_inf = E41_results.sel(N=N, T=T, result="extinction_test_2").quantile(q=0.10, dim=["s", "k"])
            a_sup = E41_results.sel(N=N, T=T, result="extinction_test_2").quantile(q=0.90, dim=["s", "k"])
            b = E42_results[H].sel(N=N, T=T, result="extinction_test_2").mean(dim=["s", "k"])
            b_inf = E42_results[H].sel(N=N, T=T, result="extinction_test_2").quantile(q=0.10, dim=["s", "k"])
            b_sup = E42_results[H].sel(N=N, T=T, result="extinction_test_2").quantile(q=0.90, dim=["s", "k"])

            x1, y1 = a.coords["p_ext"].data, a.sel(H=H).data
            x2, y2 = b.coords["p_ext"].data, b.sel(H=H).data
            y1_inf, y1_sup = a_inf.sel(H=H).data, a_sup.sel(H=H).data
            y2_inf, y2_sup = b_inf.sel(H=H).data, b_sup.sel(H=H).data
            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))
            y_inf = np.concatenate((y1_inf, y2_inf))
            y_sup = np.concatenate((y1_sup, y2_sup))
            order = x.argsort()
            x = x[order]
            y = y[order]
            y_inf = y_inf[order]
            y_sup = y_sup[order]

            for e, eps in enumerate(a.coords["eps"].data):
                if eps == 0.05:
                    plot(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", ls="-")
                    print(1-y_sup[:,e] - (1-y_inf[:,e]))
                    fill_between(x=x, y1=1-y_inf[:,e], y2=1-y_sup[:,e], alpha=0.5)
            axvline(p_c[H], color="black", label=f"$p_c({H})$")
            xlabel(r"Patch extinction probability $p_{ext}$", fontsize=15)
            xticks(np.linspace(0.1, 0.9, 9))
            ylim(-0.05,1.05)
            ylabel("Average MaxGER", fontsize=15)
            legend(fontsize=13)
            title(f"H = {H}", fontsize=15)
        tight_layout()
        savefig(f"{figures_folder}/{figure_titles[i]} - MaxGER distribution - N={N} - T={T}.pdf")
        fig_count += 1
        show()

# Average MaxGER across experiments, with varying eps

N = 50
T = 10

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
for k, H in enumerate(E5_results_noisy.coords["H"].data):
    a = E5_results_nonnoisy.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data
    for e, eps in enumerate(a.coords["eps"].data):
        axes[0,k].scatter(x, 1-y[:,e], label=r"$\varepsilon_{col}$ = "+str(eps), marker=markers[e])
    
    axes[0,k].axvline(p_c[H], c="black", label=f"$p_c(H)$")
    axes[0,k].set_ylim(-0.05,1.05)
    axes[0,k].set_title(f"H = {H}", fontsize=16)
    
    #####
    
    a = E5_results_noisy.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data

    for e, eps in enumerate(a.coords["eps"].data):
        axes[1,k].scatter(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", marker=markers[e])
    
    axes[1,k].axvline(p_c[H], c="black", label=f"$p_c(H)$")
    axes[1,k].set_ylim(-0.05,1.05)
    axes[1,k].set_title(f"H = {H}", fontsize=16)
    axes[1,k].set_xlabel("Value of $p_{ext}$", fontsize=12)



axes[0,0].set_ylabel("MaxGER", fontsize=12)
axes[1,0].set_ylabel("MaxGER", fontsize=12)

axes[0,0].annotate("BOA", xy=(0.07, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[1,0].annotate("Noisy BOA", xy=(0.07, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
            ha='center', va='baseline', fontsize=16)


for k in range(2):
    axes[k,-1].axis("off")
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.85, 0.4), fontsize=12)
fig.tight_layout()
savefig(f"{figures_folder}/Fig. C.11 - External colonization - MaxGER value.{figures_format}")
if show_plots: show()
close("all")


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
for k, H in enumerate(E6_results_noisy.coords["H"].data):
    a = E6_results_nonnoisy.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data
    for e, eps in enumerate(a.coords["eps"].data):
        axes[0,k].scatter(x, 1-y[:,e], label=r"$\varepsilon_{pos}$ = "+str(eps), marker=markers[e])
    
    axes[0,k].axvline(p_c[H], c="black", label=f"$p_c(H)$")
    axes[0,k].set_ylim(-0.05,1.05)
    axes[0,k].set_title(f"H = {H}", fontsize=16)
    
    #####
    
    a = E6_results_noisy.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data

    for e, eps in enumerate(a.coords["eps"].data):
        axes[1,k].scatter(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", marker=markers[e])
    
    axes[1,k].axvline(p_c[H], c="black", label=f"$p_c(H)$")
    axes[1,k].set_ylim(-0.05,1.05)
    axes[1,k].set_title(f"H = {H}", fontsize=16)
    axes[1,k].set_xlabel("Value of $p_{ext}$", fontsize=12)


axes[0,0].set_ylabel("MaxGER", fontsize=12)
axes[1,0].set_ylabel("MaxGER", fontsize=12)

axes[0,0].annotate("BOA", xy=(0.07, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[1,0].annotate("Noisy BOA", xy=(0.07, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
            ha='center', va='baseline', fontsize=16)


for k in range(2):
    axes[k,-1].axis("off")
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.85, 0.4), fontsize=12)
fig.tight_layout()
savefig(f"{figures_folder}/Fig. C.10 - False positives - MaxGER value.{figures_format}")
if show_plots: show()
close("all")


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
for k, H in enumerate(E7_results_noisy.coords["H"].data):
    a = E7_results_nonnoisy.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data
    for e, eps in enumerate(a.coords["eps"].data):
        axes[0,k].scatter(x, 1-y[:,e], label=r"$\varepsilon_{neg}$ = "+str(eps), marker=markers[e])
    
    axes[0,k].axvline(p_c[H], c="black", label=f"$p_c(H)$")
    axes[0,k].set_ylim(-0.05,1.05)
    axes[0,k].set_title(f"H = {H}", fontsize=16)
    
    #####
    
    a = E7_results_noisy.sel(N=N, T=T, result="MaxGER").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data

    for e, eps in enumerate(a.coords["eps"].data):
        axes[1,k].scatter(x, 1-y[:,e], label=fr"$\varepsilon$ = {eps}", marker=markers[e])
    
    axes[1,k].axvline(p_c[H], c="black", label=f"$p_c(H)$")
    axes[1,k].set_ylim(-0.05,1.05)
    axes[1,k].set_title(f"H = {H}", fontsize=16)
    axes[1,k].set_xlabel("Value of $p_{ext}$", fontsize=12)

axes[0,0].set_ylabel("MaxGER", fontsize=12)
axes[1,0].set_ylabel("MaxGER", fontsize=12)

axes[0,0].annotate("BOA", xy=(0.07, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[1,0].annotate("Noisy BOA", xy=(0.07, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
            ha='center', va='baseline', fontsize=16)
    

for k in range(2):
    axes[k,-1].axis("off")
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.85, 0.4), fontsize=12)
fig.tight_layout()
savefig(f"{figures_folder}/Fig. C.9 - False negatives - MaxGER value.{figures_format}")
if show_plots: show()
close("all")




# 2) Estimation of p_ext




fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
for k, H in enumerate(E5_results_noisy.coords["H"].data):
    a = E5_results_nonnoisy.sel(N=N, T=T, result="rmse_p_ext").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data
    for e, eps in enumerate(a.coords["eps"].data):
        axes[0,k].scatter(x, y[:,e], label=r"$\varepsilon_{col}$ = "+str(eps), marker=markers[e])
    
    axes[0,k].set_ylim(0,0.2)
    axes[0,k].set_xticks(E5_results_noisy.coords["p_ext"].data)
    axes[0,k].set_title(f"H = {H}", fontsize=16)
    
    #####
    
    a = E5_results_noisy.sel(N=N, T=T, result="rmse_p_ext").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data

    for e, eps in enumerate(a.coords["eps"].data):
        axes[1,k].scatter(x, y[:,e], label=r"$\varepsilon_{col}$ = "+str(eps), marker=markers[e])
    
    axes[1,k].set_ylim(0,0.2)
    axes[1,k].set_xticks(E5_results_noisy.coords["p_ext"].data)
    axes[1,k].set_title(f"H = {H}", fontsize=16)
    axes[1,k].set_xlabel("Value of $p_{ext}$", fontsize=12)


axes[0,0].set_ylabel("Posterior RMSE of $p_{ext}$", fontsize=12)
axes[1,0].set_ylabel("Posterior RMSE of $p_{ext}$", fontsize=12)

axes[0,0].annotate("BOA", xy=(0.07, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[1,0].annotate("Noisy BOA", xy=(0.07, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
            ha='center', va='baseline', fontsize=16)


for k in range(2):
    axes[k,-1].axis("off")
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.85, 0.4), fontsize=12)
fig.tight_layout()
savefig(f"{figures_folder}/Fig. C.14 - External colonization - RMSE of p_ext.{figures_format}")
if show_plots: show()
close("all")

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
for k, H in enumerate(E6_results_noisy.coords["H"].data):
    a = E6_results_nonnoisy.sel(N=N, T=T, result="rmse_p_ext").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data
    for e, eps in enumerate(a.coords["eps"].data):
        axes[0,k].scatter(x, y[:,e], label=r"$\varepsilon_{pos}$ = "+str(eps), marker=markers[e])
    
    axes[0,k].set_ylim(0,0.2)
    axes[0,k].set_xticks(E6_results_noisy.coords["p_ext"].data)
    axes[0,k].set_title(f"H = {H}", fontsize=16)
    
    #####
    
    a = E6_results_noisy.sel(N=N, T=T, result="rmse_p_ext").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data

    for e, eps in enumerate(a.coords["eps"].data):
        axes[1,k].scatter(x, y[:,e], label=r"$\varepsilon_{pos}$ = "+str(eps), marker=markers[e])
    
    axes[1,k].set_ylim(0,0.2)
    axes[1,k].set_xticks(E6_results_noisy.coords["p_ext"].data)
    axes[1,k].set_title(f"H = {H}", fontsize=16)
    axes[1,k].set_xlabel("Value of $p_{ext}$", fontsize=12)

axes[0,0].set_ylabel("Posterior RMSE of $p_{ext}$", fontsize=12)
axes[1,0].set_ylabel("Posterior RMSE of $p_{ext}$", fontsize=12)

axes[0,0].annotate("BOA", xy=(0.07, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[1,0].annotate("Noisy BOA", xy=(0.07, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
            ha='center', va='baseline', fontsize=16)
    
for k in range(2):
    axes[k,-1].axis("off")
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.85, 0.4), fontsize=12)
fig.tight_layout()
savefig(f"{figures_folder}/Fig. C.13 - False positives - RMSE of p_ext.{figures_format}")
if show_plots: show()
close("all")

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
for k, H in enumerate(E7_results_noisy.coords["H"].data):
    a = E7_results_nonnoisy.sel(N=N, T=T, result="rmse_p_ext").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data
    for e, eps in enumerate(a.coords["eps"].data):
        axes[0,k].scatter(x, y[:,e], label=r"$\varepsilon_{neg}$ = "+str(eps), marker=markers[e])
    
    axes[0,k].set_ylim(0,0.2)
    axes[0,k].set_xticks(E7_results_noisy.coords["p_ext"].data)
    axes[0,k].set_title(f"H = {H}", fontsize=16)
    
    #####
    
    a = E7_results_noisy.sel(N=N, T=T, result="rmse_p_ext").mean(dim=["s", "k"])

    x, y = a.coords["p_ext"].data, a.sel(H=H).data

    for e, eps in enumerate(a.coords["eps"].data):
        axes[1,k].scatter(x, y[:,e], label=r"$\varepsilon_{neg}$ = "+str(eps), marker=markers[e])
    
    axes[1,k].set_ylim(0,0.2)
    axes[1,k].set_xticks(E7_results_noisy.coords["p_ext"].data)
    axes[1,k].set_title(f"H = {H}", fontsize=16)
    axes[1,k].set_xlabel("Value of $p_{ext}$", fontsize=12)


axes[0,0].set_ylabel("Posterior RMSE of $p_{ext}$", fontsize=12)
axes[1,0].set_ylabel("Posterior RMSE of $p_{ext}$", fontsize=12)

axes[0,0].annotate("BOA", xy=(0.07, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[1,0].annotate("Noisy BOA", xy=(0.07, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
            ha='center', va='baseline', fontsize=16)


for k in range(2):
    axes[k,-1].axis("off")
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.85, 0.4), fontsize=12)
fig.tight_layout()
savefig(f"{figures_folder}/Fig. C.12 - False negatives - RMSE of p_ext.{figures_format}")
if show_plots: show()
close("all")





# 3) Estimation of H





H_tensor = E5_results_nonnoisy.coords["H"].data[None,None,None,:,None,None,None]



fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
top_lim = 6

H_inf = E5_results_nonnoisy.sel(result='H_inf')
H_tensor = E5_results_nonnoisy.coords["H"].data[None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["N", "T", "s", "k", "p_ext"])

x = a.coords["H"].data
for e, eps in enumerate(a.coords["eps"].data):
    y = a.sel(eps=eps).data
    axes[0,0].plot(x, y, label=r"Noise level = "+f"{eps}", marker=markers[e])

axes[0,0].set_ylim(0, top_lim)
axes[0,0].set_xticks(x)

#####

H_inf = E5_results_noisy.sel(result='H_inf')
H_tensor = E5_results_noisy.coords["H"].data[None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["N", "T", "s", "k", "p_ext"])

x = a.coords["H"].data
for e, eps in enumerate(a.coords["eps"].data):
    y = a.sel(eps=eps).data
    axes[1,0].plot(x, y, marker=markers[e])
    
axes[1,0].set_ylim(0, top_lim)
axes[1,0].set_xticks(x)

#####

H_inf = E6_results_nonnoisy.sel(result='H_inf')
H_tensor = E6_results_nonnoisy.coords["H"].data[None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["N", "T", "s", "k", "p_ext"])

x = a.coords["H"].data
for e, eps in enumerate(a.coords["eps"].data):
    y = a.sel(eps=eps).data
    axes[0,1].plot(x, y, marker=markers[e])

axes[0,1].set_ylim(0, top_lim)
axes[0,1].set_xticks(x)

#####

H_inf = E6_results_noisy.sel(result='H_inf')
H_tensor = E6_results_noisy.coords["H"].data[None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["N", "T", "s", "k", "p_ext"])

x = a.coords["H"].data
for e, eps in enumerate(a.coords["eps"].data):
    y = a.sel(eps=eps).data
    axes[1,1].plot(x, y, marker=markers[e])
    
axes[1,1].set_ylim(0, top_lim)
axes[1,1].set_xticks(x)

#####

H_inf = E7_results_nonnoisy.sel(result='H_inf')
H_tensor = E7_results_nonnoisy.coords["H"].data[None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["N", "T", "s", "k", "p_ext"])

x = a.coords["H"].data
for e, eps in enumerate(a.coords["eps"].data):
    y = a.sel(eps=eps).data
    axes[0,2].plot(x, y, marker=markers[e])

axes[0,2].set_ylim(0, top_lim)
axes[0,2].set_xticks(x)

#####

H_inf = E7_results_noisy.sel(result='H_inf')
H_tensor = E7_results_noisy.coords["H"].data[None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["N", "T", "s", "k", "p_ext"])

x = a.coords["H"].data
for e, eps in enumerate(a.coords["eps"].data):
    y = a.sel(eps=eps).data
    axes[1,2].plot(x, y, marker=markers[e])
    
axes[1,2].set_ylim(0, top_lim)
axes[1,2].set_xticks(x)

###################################

axes[0,0].set_ylabel("Average value\nof $|H_{inf} - H|$", fontsize=12)
axes[1,0].set_ylabel("Average value\nof $|H_{inf} - H|$", fontsize=12)

axes[0,0].annotate("BOA", xy=(0, 0.4), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=18)
axes[1,0].annotate("Noisy BOA", xy=(0, 0.2), xytext=(-70, 0), rotation=90,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=18)

axes[0,0].annotate("External\ncolonization", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[0,1].annotate("False positives", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)
axes[0,2].annotate("False negatives", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=16)

for k in range(2):
    axes[k,-1].axis("off")
for k in range(3):
    axes[1,k].set_xlabel("Value of $H$", fontsize=12)
lines, labels = axes[0,0].get_legend_handles_labels()
fig.legend(lines, labels, loc=(0.75, 0.38), fontsize=12)
fig.subplots_adjust()
savefig(f"{figures_folder}/Fig. C.15 - Estimation of H.{figures_format}")
if show_plots: show()
close("all")




# 4) Estimation on multiple streets




a = E8_results.sel(result="H_recovery").mean(["H", "p_ext", "s", "eps", "k"])

M_range = E8_results.coords["M"].data
N_range = E8_results.coords["N"].data[:-1]
T_range = E8_results.coords["T"].data
for e, N in enumerate(N_range):
    for f, T in enumerate(T_range):
        plot(M_range, a.sel(N=N, T=T), label=f"N={N}, T={T}", marker=markers[2*e+f])
legend(fontsize=12)
ylim(bottom=0)
xlabel("Number of streets", fontsize=16)
ylabel("Proportion of cases where H = H_inf")
if show_plots: show()
close("all")

figure(figsize=(10, 4))

subplot(1,2,1)

H_inf = E8_results.sel(result='H_inf')
H_tensor = E8_results.coords["H"].data[None,None,None,None,:,None,None,None]
a = np.abs(H_inf - H_tensor).mean(["H", "p_ext", "s", "k", "eps"])

M_range = E8_results.coords["M"].data
N_range = E8_results.coords["N"].data[:-1]
T_range = E8_results.coords["T"].data
for e, N in enumerate(N_range):
    for f, T in enumerate(T_range):
        plot(M_range, a.sel(N=N, T=T), label=f"N={N}, T={T}", marker=markers[2*e+f])
legend(fontsize=12)
ylim(bottom=0)
xlabel("Number of streets", fontsize=15)
ylabel("Average value of $|H_{inf} - H|$", fontsize=15)

subplot(1,2,2)

a = E8_results.sel(result="H_recovery").mean(["H", "p_ext", "s", "eps", "k"])

M_range = E8_results.coords["M"].data
N_range = E8_results.coords["N"].data[:-1]
T_range = E8_results.coords["T"].data
for e, N in enumerate(N_range):
    for f, T in enumerate(T_range):
        plot(M_range, a.sel(N=N, T=T), label=f"N={N}, T={T}", marker=markers[2*e+f])
legend(fontsize=12)
ylim(bottom=0)
xlabel("Number of streets", fontsize=15)
ylabel("Proportion of cases s.t. $H_{inf} = H$", fontsize=15)

tight_layout()
savefig(f"{figures_folder}/Fig. C.16 - Multiple streets.{figures_format}")
if show_plots: show()
close("all")