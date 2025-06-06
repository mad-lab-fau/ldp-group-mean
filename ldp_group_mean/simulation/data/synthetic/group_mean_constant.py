"""Synthetic data with K groups with constant values per group and equal spread between groups."""

from __future__ import annotations

import numpy as np

from ldp_group_mean.simulation.data.synthetic.synthetic import SyntheticContinuousGroupData


class KGroupsConstantEqualSpread(SyntheticContinuousGroupData):
    """K groups with constant values per group and equal spread between groups."""

    def __init__(  # noqa: PLR0913
        self,
        num_groups: int,
        num_samples: int,
        input_range: tuple[float, float],
        name: str | None = None,
        data_rng: np.random.Generator | None = None,
        group_ratio: float | None = None,
    ):
        super().__init__(input_range, num_groups, name, data_rng)

        samples_per_group_list = None
        samples_per_group = num_samples // num_groups

        # Group ratio only for 2 groups
        if group_ratio is not None:
            assert num_groups == 2, "Group ratio only for 2 groups."
            samples_per_group_list = [
                num_samples,
                int(round(num_samples / group_ratio)),
            ]
            num_samples = sum(samples_per_group_list)

        data = np.zeros((num_samples, 2))

        means = np.linspace(input_range[0], input_range[1], num_groups + 2)[1:-1]

        start = 0
        for i in range(num_groups):
            if samples_per_group_list is not None:
                samples_per_group = samples_per_group_list[i]
            elif i == num_groups - 1:
                samples_per_group = num_samples - i * samples_per_group

            data[start : start + samples_per_group, 0] = i
            data[start : start + samples_per_group, 1] = np.repeat(means[i], samples_per_group)

            start = start + samples_per_group

        data[:, 1] = np.clip(data[:, 1], *input_range)

        self.data = self.data_rng.permutation(data)
