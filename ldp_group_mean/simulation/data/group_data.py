"""Base class for group data."""
from __future__ import annotations

from abc import ABC

import numpy as np
from sok_ldp_analysis.simulation.data.real import RealDataset


class GroupDataBase(ABC):
    """Base class for group data."""

    num_groups: int
    _data: np.array
    name: str

    def __init__(
        self,
        num_groups: int | None = None,
        name: str | None = None,
        data: np.array = None,
        group_mapping: dict[int, any] | None = None,
    ):
        if num_groups is None:
            self.num_groups = len(np.unique(data[:, 0]))
        else:
            self.num_groups = num_groups

        if data is not None:
            self.data = data

        self.group_mapping = group_mapping

        self.name = name

    def load_data(self):  # noqa: B027
        """Load data."""

    @property
    def data(self) -> np.array:
        """Data of the group data."""
        return self._data

    @data.setter
    def data(self, data):
        assert data.shape[1] == 2, "Data must have two columns."
        assert self.num_groups == len(np.unique(data[:, 0])), "Number of groups does not match the data."
        self._data = data

    @property
    def groups(self):
        """Groups of the group data."""
        return self.data[:, 0]

    @property
    def values(self):
        """Values of the group data."""
        return self.data[:, 1]

    def __len__(self):
        """Length of the group data."""
        return len(self.data)

    def __getitem__(self, key):
        """Get item of the group data."""
        return self.data[key, :]

    def __iter__(self):
        """Iterate over the group data."""
        return np.nditer(self.data)

    def __str__(self):
        """Return a string representation of the group data."""
        if self.name is None:
            return type(self).__name__
        return self.name


class ContinuousGroupData(GroupDataBase):
    """Base class for continuous group data."""

    input_range: tuple[float, float]

    def __init__(
        self,
        input_range: tuple[float, float],
        num_groups: int,
        name: str | None = None,
        data: np.array = None,
        group_mapping: dict[int, any] | None = None,
    ):
        self.input_range = input_range
        super().__init__(num_groups, name, data, group_mapping)

    def _preprocess(self, tmp_file):
        pass


class RealContinuousGroupData(ContinuousGroupData, RealDataset):
    """Base class for real continuous group data."""

    def __init__(
        self,
        input_range: tuple[float, float],
        num_groups: int,
        name: str | None = None,
        data: np.array = None,
        group_mapping: dict[int, any] | None = None,
    ):
        super().__init__(input_range, num_groups, name, data, group_mapping)

    def _preprocess(self, tmp_file):
        pass


class DiscreteGroupData(GroupDataBase):
    """Base class for discrete group data."""

    domain_size: int

    def __init__(
        self,
        domain_size: int,
        num_groups: int,
        domain_mapping: dict[int, any] | None = None,
        name: str | None = None,
        data: np.array = None,
    ):
        super().__init__(num_groups, name, data)
        self.domain_size = domain_size
        self.domain_mapping = domain_mapping


class RealDiscreteGroupData(DiscreteGroupData, RealDataset):
    """Base class for real discrete group data."""

    def __init__(
        self,
        domain_size: int,
        num_groups: int,
        domain_mapping: dict[int, any] | None = None,
        name: str | None = None,
        data: np.array = None,
    ):
        super().__init__(domain_size, num_groups, domain_mapping, name, data)

    def _preprocess(self, tmp_file):
        pass
