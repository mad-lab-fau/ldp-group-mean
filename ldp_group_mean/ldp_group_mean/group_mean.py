"""Abstract class for group-wise mean estimation."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GroupMean(ABC):
    """Group-wise mean estimation."""

    eps1: float = 0
    eps2: float = 0

    def __init__(self, eps: float | tuple[float, float], input_range: np.array, rng: np.random.Generator = None):
        self.eps = eps
        self.input_range = input_range
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    @abstractmethod
    def mechanism(self, num_groups: int, x: np.array) -> np.array:
        """
        Apply the LDP mechanism to the private data.

        Args:
            num_groups: The number of groups.
            x:  The private data.

        Return:
            The perturbed responses.
        """

    @abstractmethod
    def mean(
        self, num_groups: int, z: np.array, adjust_count: bool = True
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Estimate the mean of the private data per group based on the perturbed responses.

        Args:
            num_groups: The number of groups.
            z: The perturbed responses.
            adjust_count: Whether to adjust the counts.

        Return:
            A tuple (estimated sums, estimated means, estimated counts, estimated groups).
        """

    def _produce_output(self, responses, num_groups, est_func, adjust_count):
        group = responses[:, 0]
        data = responses[:, 1]

        n = len(data)

        data = est_func(data)

        sums = np.zeros(num_groups)
        means = np.zeros(num_groups)
        counts = np.zeros(num_groups)
        counts_est = np.zeros(num_groups)
        for g in range(num_groups):
            mask = group == g
            counts[g] = np.sum(mask)
            counts_est[g] = self._est_counts(counts[g], n, num_groups)

            if counts[g] == 0:
                sums[g] = 0
            else:
                sums[g] = np.sum(data[mask])

            if adjust_count:
                sums[g] = self._transform_sum(sums[g], counts_est[g])
            else:
                sums[g] = self._transform_sum(sums[g], counts[g])

            if adjust_count:
                means[g] = (1 / (counts_est[g])) * sums[g]
            else:
                means[g] = (1 / (counts[g])) * sums[g]

        return sums, means, counts, group

    def _est_counts(self, counts, n, num_groups):
        p = np.exp(self.eps1) / (np.exp(self.eps1) + num_groups - 1)
        q = (1 - p) / (num_groups - 1)

        return 1 / (p - q) * (counts - n * q)

    def _transform_sum(self, sums, counts):
        r = self.input_range[0]
        s = self.input_range[1]

        return ((s - r) / 2) * sums + counts * (r + (s - r) / 2)

    def _transform_input(self, x: np.array) -> np.array:
        return np.clip(2 * (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0]) - 1, -1, 1)

    def _transform_output(self, x: np.array) -> np.array:
        return ((x + 1) / 2) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
