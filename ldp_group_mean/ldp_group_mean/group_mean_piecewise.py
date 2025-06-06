"""Group-wise mean estimation based on the piecewise mechanism."""

from __future__ import annotations

import math

import numpy as np
from sok_ldp_analysis.ldp.mean.wang2019 import _piecewise_1d

from ldp_group_mean.ldp_group_mean.group_mean import GroupMean
from ldp_group_mean.ldp_group_mean.helpers import _grr_helper


class GroupMeanPiecewise(GroupMean):
    """
    Group-wise mean estimation based on piecewise mean estimation.

    Piecewise mean estimation was first introduced by Wang et al. 2019 [1].

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.
    """

    def __init__(
        self,
        eps: float | tuple[float, float],
        input_range: np.array,
        rng: np.random.Generator = None,
        auto_eps: bool = True,
    ):
        if isinstance(eps, tuple):
            eps1, eps2 = eps
            self.eps1 = eps1
            self.eps2 = eps2
            eps = eps1 + eps2
            auto_eps = False
        else:
            self.eps1 = eps / 2
            self.eps2 = eps / 2

        # Sanity Check
        max_eps = max(
            self.eps2,
            self.eps1,
            self.eps1 + self.eps2,
        )

        assert math.isclose(max_eps, eps) or max_eps < eps

        self.auto_eps = auto_eps

        super().__init__(eps, input_range, rng)

    def mechanism(self, num_groups: int, x: np.array) -> np.array:
        """
        Apply the LDP mechanism to the private data.

        Args:
            num_groups: The number of groups.
            x: The private data.

        Return:
            The perturbed responses.
        """
        if self.auto_eps:
            ratio = 0.156 * np.log(num_groups) + 0.148

            self.eps1 = self.eps * ratio
            self.eps2 = self.eps - self.eps1

        group = x[:, 0]
        data = self._transform_input(x[:, 1])

        group_ = _grr_helper(group, self.eps1, num_groups, self.rng)

        # Replace the values with the center of the input range if the group was changed
        data_ = np.where(group_ != group, 0, data)

        # Apply the piecewise mechanism
        res = _piecewise_1d(data_, self.eps2, self.rng)

        return np.column_stack((group_, res))

    def mean(self, num_groups: int, responses: np.array, adjust_count: bool = True) -> np.array:
        """
        Estimate the mean of the private data per group based on the perturbed responses.

        Args:
            num_groups: The number of groups.
            z: The perturbed responses.
            adjust_count: Whether to adjust the counts.

        Return:
            The estimated means.
        """
        a = np.exp(self.eps1) / (np.exp(self.eps1) + num_groups - 1)

        def est_func(x):
            return x / a

        return self._produce_output(responses, num_groups, est_func, adjust_count)


class GroupMeanPiecewiseHalf(GroupMeanPiecewise):
    """
    Group-wise mean estimation based on piecewise mean estimation with the input range being [-1, 1].

    This implementation sets eps1 = eps2 = eps / 2.
    Piecewise mean estimation was first introduced by Wang et al. 2019 [1].

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.
    """

    def __init__(
        self,
        eps: float | tuple[float, float],
        input_range: np.array,
        rng: np.random.Generator = None,
    ):
        eps1 = eps / 2
        eps2 = eps - eps1
        super().__init__((eps1, eps2), input_range, rng, auto_eps=False)
