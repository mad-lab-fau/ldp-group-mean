"""Group-wise Mean Estimation based on the NPRR mechanism."""
from __future__ import annotations

import math

import numpy as np

from ldp_group_mean.ldp_group_mean.group_mean import GroupMean
from ldp_group_mean.ldp_group_mean.helpers import _grr_helper


class GroupMeanWaudbySmith(GroupMean):
    """
    Group-wise Mean Estimation based on the NPRR mechanism.

    The NPRR mechanism was first introduced by [1].

    [1] I. Waudby-Smith, S. Wu, and A. Ramdas, “Nonparametric Extensions of
    Randomized Response for Private Confidence Sets,” in Proceedings of the 40th International Conference on Machine
    Learning, PMLR, Jul. 2023, pp. 36748-36789. Available: https://proceedings.mlr.press/v202/waudby-smith23a.html
    """

    def __init__(
        self, eps: float | tuple[float, float], input_range: np.array, rng: np.random.Generator = None, k: int = 8
    ):
        if isinstance(eps, tuple):
            eps1, eps2 = eps
            self.eps1 = eps1
            self.eps2 = eps2
        else:
            self.eps1 = eps - math.log((k + 1) * math.exp(eps) / (math.exp(eps) + k))
            self.eps2 = eps

        # Sanity Check
        max_eps = max(self.eps1 + math.log((k + 1) * math.exp(eps) / (math.exp(eps) + k)), self.eps2)
        assert math.isclose(max_eps, eps) or max_eps < eps

        self.k = k

        super().__init__(eps, input_range, rng)

    def _nprr(self, data: np.array) -> np.array:
        v_ = (data + 1) / 2
        vf = np.floor(v_ * self.k) / self.k

        b = self.rng.binomial(1, self.k * (v_ - vf))
        y = vf + b / self.k

        return 2 * _grr_helper(y * self.k, self.eps2, self.k + 1, self.rng) / self.k - 1

    def mechanism(self, num_groups: int, x: np.array) -> np.array:
        """
        Apply the LDP mechanism to the private data.

        Args:
            num_groups: The number of groups.
            x:  The private data.

        Return:
            The perturbed responses.
        """
        group = x[:, 0]
        data = self._transform_input(x[:, 1])  # Transform to [-1, 1]

        # Apply the GRR mechanism
        group_ = _grr_helper(group, self.eps1, num_groups, self.rng)

        change_mask = group_ != group

        # Replace the values with the center of the input range if the group was changed
        data_ = np.where(change_mask, 0, data)  # 0 for x in [-1, 1]

        data_ = self._nprr(data_)

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
        b = (np.exp(self.eps2) - 1) / (self.k + np.exp(self.eps2))

        def est_func(x):
            return x / (a * b)

        return self._produce_output(responses, num_groups, est_func, adjust_count)
