# Code for the simulation and inference of the BOA process

The code consists in the following files:
- `initialization.py` and `simulation.py` contain functions to simulate the BOA and noisy BOA processes from various initial seed bank configurations.
- `estimation.py` contains the code to sample from the posterior density of the model parameters given observed data one one or multiple streets, via MCMC. It relies on `likelihood.py`, which computes the complete model likelihood.
- `postprocessing.py` contains various functions to trim the MCMC result and compute relevant summary statistics, especially the GER and MaxGER metric we propose in the paper. It also contains functions to plot the MCMC convergence and the posterior distributions.

The file `critical_prob.csv` provides the precomputed list of critical probabilities `p_c[H]` such that, if `p_ext > p_c[H]`, the BOA process goes extinct with probability 1.