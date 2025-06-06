"""Group-wise Mean Estimation based on Laplace mean estimation."""
from __future__ import annotations

import math

import numpy as np

from ldp_group_mean.ldp_group_mean.group_mean import GroupMean
from ldp_group_mean.ldp_group_mean.helpers import _grr_helper


class GroupMeanLaplace(GroupMean):
    """
    Group-wise Mean Estimation based on Laplace mean estimation.

    Implementation based on the paper [1] M. Juarez and A. Korolova, “`You Can`t Fix What You Can`t Measure`:
    Privately Measuring Demographic Performance Disparities in Federated Learning,” in Proceedings of the Workshop on
    Algorithmic Fairness through the Lens of Causality and Privacy, PMLR, Jun. 2023, pp. 67-85. Accessed: Aug. 19,
    2024. [Online]. Available: https://proceedings.mlr.press/v214/juarez23a.html.
    """

    def __init__(
        self,
        eps: float | tuple[float, float],
        input_range: np.array,
        rng: np.random.Generator = None,
    ):
        if isinstance(eps, tuple):
            eps1, eps2 = eps
            self.eps1 = eps1
            self.eps2 = eps2
        else:
            self.eps2 = eps
            self.eps1 = eps / 2

        # Sanity Check
        max_eps = max(
            self.eps2,
            self.eps2 / 2 + self.eps1,
        )

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

        # Generate noise with scale 2 for normal values and scale k for changed values
        noise = self.rng.laplace(0, 2 / self.eps2, (n,))
        noise_0 = self.rng.laplace(0, 2 / self.eps2, (n,))

        # Add noise
        data_ = data_ + np.where(change_mask, noise_0, noise)

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

        def est_func(x):
            return x / a

        return self._produce_output(responses, num_groups, est_func, adjust_count)
