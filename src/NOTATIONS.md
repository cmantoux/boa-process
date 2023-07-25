# Notations used throughout the code

**Size variables:**
- `N` = the number of trees in a given street
- `T` = the number of observation years in a given data set
- `K`, or `M` = the number of streets

**Model parameters:**
- `p_ext` = local extinction probability
- `H` = maximal seed dormancy duration
- `s` = proportion of viable seeds at initialization
- `eps` = external colonization probability (noisy BOA only)

Remark: `H_max` denotes the maximal authorized value of `H` in the estimation procedure. In this work, it is set to 10, which is greater than (or equal to) the maximum number of observation years.

**Model variables:**
- `O` = (N, T)-shaped numpy array of binary observations
- `L` = (N, H_max)-shaped numpy array containing the initial seed age in each tree base. L[i,h] is the age of the seeds in tree base i in the case where the true value of `H` is h.
- `C` = (N, T)-shaped array giving the age of the seeds at each tree base and each observation year. It is only used when simulating the data. `C` can be computed from `O` and `L` using `src.likelihood.compute_C(O, L, H)`.