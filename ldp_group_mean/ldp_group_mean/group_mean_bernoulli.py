"""Group-wise Mean Estimation based on Bernoulli mean estimation."""
from __future__ import annotations

import math

import numpy as np

from ldp_group_mean.ldp_group_mean.group_mean import GroupMean
from ldp_group_mean.ldp_group_mean.helpers import _grr_helper


class GroupMeanBernoulli(GroupMean):
    """
    Group-wise Mean Estimation based on Bernoulli mean estimation.

    Implementation based on the paper [1] M. Juarez and A. Korolova, “`You Can`t Fix What You Can`t Measure`:
    Privately Measuring Demographic Performance Disparities in Federated Learning,” in Proceedings of the Workshop on
    Algorithmic Fairness through the Lens of Causality and Privacy, PMLR, Jun. 2023, pp. 67-85. Accessed: Aug. 19,
    2024. [Online]. Available: https://proceedings.mlr.press/v214/juarez23a.html.

    Fixed split of epsilon into eps1 and eps2 based on our paper.
    """

    def __init__(self, eps: float | tuple[float, float], input_range: np.array, rng: np.random.Generator = None):
        if isinstance(eps, tuple):
            eps1, eps2 = eps
            self.eps1 = eps1
            self.eps2 = eps2
        else:
            self.eps1 = eps - math.log(2 * math.exp(eps) / (math.exp(eps) + 1))
            self.eps2 = eps

        # Sanity Check
        max_eps = max(self.eps1 + math.log(2 * math.exp(eps) / (math.exp(eps) + 1)), self.eps2)
        assert math.isclose(max_eps, eps) or max_eps < eps

        super().__init__(eps, input_range, rng)

    def mechanism(self, num_groups: int, x: np.array) -> np.array:
        """
        Apply the LDP mechanism to the private data.

        Args:
            num_groups: The number of groups.
            x:  The private data.

        Return:
            The perturbed responses.
        """
        n = x.shape[0]

        group = x[:, 0]
        data = self._transform_input(x[:, 1])

        # Apply the GRR mechanism
        group_ = _grr_helper(group, self.eps1, num_groups, self.rng)

        change_mask = group_ != group

        # Replace the values with the center of the input range if the group was changed
        data_ = np.where(change_mask, 0, data)  # 0 for x in [-1, 1]

        # Sample from Bernoulli
        b = self.rng.binomial(n=1, p=(1 + data_) / 2, size=n)

        # Apply GRR
        data_ = _grr_helper(b, self.eps2, 2, self.rng)

        data_ = 2 * data_ - 1

        return np.column_stack((group_, data_))

    def mean(self, num_groups: int, responses: np.array, adjust_count: bool = True) -> np.array:
        """
        Estimate the mean of the private data per group based on the perturbed responses.

        Args:
            num_groups: The number of groups.
            responses: The perturbed responses.
            adjust_count: Whether to adjust the counts.

        Return:
            The estimated means per group.
        """
        a = np.exp(self.eps1) / (np.exp(self.eps1) + num_groups - 1)
        b = np.exp(self.eps2) / (1 + np.exp(self.eps2))

        def est_func(x):
            return x / (a * (2 * b - 1))

        return self._produce_output(responses, num_groups, est_func, adjust_count)
