"""Helper functions for group-wise estimation."""

import numpy as np


def _grr_helper(t: np.array, eps: float, d: int, rng: np.random.Generator) -> np.array:
    """
    Apply the d-ary randomized response mechanism to the input data.

    Args:
        t: The private data of n clients (vector of size n)
        eps: The privacy budget epsilon.
        d: The domain size.
        rng: A numpy random number Generator

    Return:
        The modified data.
    """
    p = np.exp(eps) / (np.exp(eps) + d - 1)

    # randomly pick "offsets" in the range {1, 2, ..., d-1} for each client
    offset = rng.integers(1, d, size=t.shape)

    # create a mask for the true value
    mask = rng.binomial(1, p, size=t.shape)

    # return the true value if the mask is true, otherwise shift the value by the offset
    return np.where(mask, t, (t + offset) % d)


def _grr_freqs(r: np.array, eps: float, d: int) -> np.array:
    """
    Estimate the frequencies of the private data based on the perturbed responses.

    Args:
        r: The perturbed responses.
        eps: The privacy budget epsilon.
        d: The domain size.

    Return:
        The estimated frequencies and counts.
    """
    n = r.shape[0]
    freqs = np.zeros(d)

    p = np.exp(eps) / (np.exp(eps) + d - 1)
    q = (1 - p) / (d - 1)

    for i in range(d):
        mask = r == i
        freqs[i] = 1 / (n * (p - q)) * (np.sum(mask) - n * q)

    return freqs
