"""
This file proposes different types of initializations
to generate realizations of the BOA process.
"""


import numpy as np


def init_full(N):
    """
    Return an array of viable seeds with age 0
    """
    return np.zeros(N)


def init_rectangle(N, H, a, b):
    """
    Return an array with the interval [a, b[
    filled with zeros, and the remainder with
    age H+1 (non-viable)
    """
    seeds = np.zeros(N) + H+1
    seeds[a:b] = 0
    return seeds


def init_half(N, H):
    """
    Return an array with the left part filled
    with seeds with age 0, and the remainder
    with age H+1 (non-viable)
    """
    return init_rectangle(N, H, 0, N//2)


def init_random(N, H, s=0.5):
    """
    Returns an array full of viable seeds,
    with age randomly chosen from 0 to H.
    A proportion s of the seeds have age H+1 (non-viable)
    """
    seeds = np.random.randint(0, H+1, size=N)
    viable = np.random.rand(N) < s
    return viable*seeds + (1-viable)*(H+1)