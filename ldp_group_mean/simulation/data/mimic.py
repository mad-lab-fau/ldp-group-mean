"""Loader for the MIMIC-IV dataset."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
from icdmappings.mappers import ICD9toICD10, ICD10toChapters

from ldp_group_mean.simulation.data.group_data import RealContinuousGroupData


class MIMICLoader:
    """Loader for the MIMIC-IV dataset."""

    groups: ClassVar[list] = [
        "admission_type",
        "admission_location",
        "discharge_location",
        "race",
        "gender",
        "dead",
        "icd_chapter",
    ]

    values: ClassVar[list] = [
        "age",
        "anchor_age",
    ]

    range_mapping: ClassVar[dict[str, tuple[float, float]]] = {
        "age": (18, 100),
        "anchor_age": (18, 100),
    }

    def __init__(self, data_path: Path | str | None = None):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        if data_path is None:
            file_dir = Path(__file__).parent
            data_path = file_dir / ".." / ".." / ".." / ".." / "data" / "mimic"
            data_path = data_path.resolve()

        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        self._data = pd.read_csv(self.data_path / "mimic_adm.csv")
        self._data["dead"] = self._data["dod"].apply(lambda x: type(x) is str)

        # Map ICD-9 codes to ICD-10 codes
        self._data.loc[self._data["icd_version"] == 9, "icd_code"] = ICD9toICD10().map(
            x.strip() for x in self._data.loc[self._data["icd_version"] == 9, "icd_code"].tolist()
        )
        self._data.loc[self._data["icd_version"] == 9, "icd_version"] = 10

        self._data["icd_chapter"] = ICD10toChapters().map(self._data["icd_code"])

    def get_numeric_dataset(self, group: str, value: str) -> RealContinuousGroupData:
        """
        Get the group data for a specific group and value.

        Args:
            group: The group to get the data for.
            value: The value to get the data for.

        Return:
            The group data.
        """
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")  # noqa: TRY003

        if value not in self.values:
            raise ValueError(f"Value {value} not in {self.values}")  # noqa: TRY003

        data = self._data[[group, value]]

        # Remove NaN values
        data = data.dropna()

        cat = data[group].astype("category")
        data[group] = cat.cat.codes

        data_array = data.to_numpy()

        data_array[:, 0] = data_array[:, 0].astype(int)
        data_array[:, 1] = data_array[:, 1].astype(float)

        return RealContinuousGroupData(
            self.range_mapping[value],
            data[group].nunique(),
            name=f"MIMIC_{group}_{value}",
            data=data_array,
            group_mapping=dict(enumerate(cat.cat.categories)),
        )

    def get_all_numeric_datasets(self) -> dict[str, RealContinuousGroupData]:
        """
        Get all numeric datasets.

        Return:
            Dictionary with the dataset name as key and the GroupData object as value.
        """
        return {
            f"{group}_{value}": self.get_numeric_dataset(group, value) for group in self.groups for value in self.values
        }
