import numpy as np

import src.estimation
import src.postprocessing

# Read the data file
O = np.loadtxt("data.csv", delimiter=",")

# Run the estimation procedure of the non-noisy BOA for 10000 steps
estimation_result = src.estimation.estimate_single_street(O, niter=10000, noisy_BOA=False)

# Remove the first 5000 MCMC steps to ensure convergence
estimation_result = src.postprocessing.simplify_history(estimation_result, keep_length=5000)

# Summarizes the MCMC result, and print the output
summary = src.postprocessing.compute_summary(estimation_result)
print(summary)

# Plot the posterior distributions of p_ext and H
src.postprocessing.plot_posterior(estimation_result, summary)