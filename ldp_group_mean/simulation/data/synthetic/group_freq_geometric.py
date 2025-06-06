"""
Synthetic data for frequency estimation for groups sampled from a geometric distribution.

The distribution within each group is inspired by the geometric distribution with p = 5/d as in Kairouz et al. (2016).
"""
from __future__ import annotations

import numpy as np

from ldp_group_mean.simulation.data.synthetic.synthetic import SyntheticDiscreteGroupData


class GeometricData(SyntheticDiscreteGroupData):
    """Synthetic data for frequency estimation for groups sampled from a geometric distribution."""

    def __init__(
        self,
        num_groups: int,
        num_samples: int,
        domain_size: int,
        equal_groups: bool = True,
        data_rng: np.random.Generator | None = None,
    ):
        """
        Initialize the data.

        Args:
            num_groups: The number of groups.
            num_samples: The total number of samples.
            domain_size: The size of the domain.
            equal_groups: Whether to have equal groups. If True, the groups will have the same size.
            data_rng: The random number generator.
        """
        super().__init__(
            domain_size, num_groups, None, name=f"Geometric Data({num_groups}, {domain_size})", data_rng=data_rng
        )
        self.num_samples = num_samples
        self.equal_groups = equal_groups

        self.data = self._generate_data()

    def _generate_data(self) -> np.array:
        # Generate the group sizes
        if self.equal_groups:
            group_sizes = np.full(self.num_groups, self.num_samples // self.num_groups)
        else:
            raise NotImplementedError("Unequal groups are not yet implemented.")

        # Generate the probabilities for the geometric distribution
        p = 5 / self.domain_size if self.domain_size > 5 else 0.5
        probabilities = [p * (1 - p) ** (i - 1) for i in range(1, self.domain_size + 1)]

        # normalize the probabilities to sum to 1
        probabilities = np.array(probabilities) / sum(probabilities)

        # Generate the data
        data = np.zeros((self.num_samples, 2), dtype=int)
        idx = 0
        for i, size in enumerate(group_sizes):
            # Adjust the size of the last group
            if i == self.num_groups - 1:
                size = self.num_samples - idx  # noqa: PLW2901

            # Sample the data from a geometric distribution
            data[idx : idx + size, 0] = i
            data[idx : idx + size, 1] = self.data_rng.choice(np.arange(0, self.domain_size), size, p=probabilities)
            idx += size

        self.data_rng.shuffle(data)

        return data
