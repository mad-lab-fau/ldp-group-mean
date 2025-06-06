"""Synthetic group data for simulation experiments."""
from __future__ import annotations

import numpy as np

from ldp_group_mean.simulation.data.group_data import ContinuousGroupData, DiscreteGroupData


class SyntheticContinuousGroupData(ContinuousGroupData):
    """Base class for synthetic continuous group data."""

    def __init__(
        self,
        input_range: tuple[float, float],
        num_groups: int,
        name: str | None = None,
        data_rng: np.random.Generator | None = None,
    ):
        super().__init__(input_range, num_groups, name)

        self.data_rng = data_rng or np.random.default_rng()


class SyntheticDiscreteGroupData(DiscreteGroupData):
    """Base class for synthetic discrete group data."""

    def __init__(
        self,
        domain_size: int,
        num_groups: int,
        domain_mapping: dict[int, any] | None = None,
        name: str | None = None,
        data_rng: np.random.Generator | None = None,
    ):
        super().__init__(domain_size, num_groups, domain_mapping, name)

        self.data_rng = data_rng or np.random.default_rng()
