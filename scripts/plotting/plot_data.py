import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ldp_group_mean.simulation.data.mimic import MIMICLoader
from ldp_group_mean.simulation.data.synthetic.group_mean_constant import KGroupsConstantEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_extreme import KGroupsExtremeEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_normal import KGroupsNormalEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_uniform import KGroupsUniform
from scripts.plotting.plot_paper import _figsize, _init_latex

dataset_mapping = {
    "KGroupsUniform": "Uniform",
    "KGroupsNormalEqualSpread": "Normal",
    "KGroupsConstantEqualSpread": "Constant",
    "KGroupsExtremeEqualSpread": "Extreme",
    "MIMIC_admission_location_age": "MIMIC-IV - Admission Location - Age (Years)",
    "MIMIC_admission_type_age": "MIMIC-IV - Admission Type - Age (Years)",
    "MIMIC_gender_age": "MIMIC-IV - Gender - Age (Years)",
    "MIMIC_icd_chapter_age": "MIMIC-IV - ICD Chapter - Age (Years)",
    "MIMIC_dead_age": "MIMIC-IV - Deceased - Age (Years)",
    "MIMIC_discharge_location_age": "MIMIC-IV - Discharge Location - Age (Years)",
    "AQUILA_disease_age": "AQUILA - Disease - Age (Years)",
    "AQUILA_disease_bmi": r"AQUILA - Disease - BMI ($\si{\kilo\gram\per\square\meter}$)",
    "AQUILA_disease_height": r"AQUILA - Disease - Height ($\si{\meter}$)",
    "AQUILA_disease_weight": r"AQUILA - Disease - Weight ($\si{\kilo\gram}$)",
    "AQUILA_sex_age": "AQUILA - Gender - Age (Years)",
    "AQUILA_sex_bmi": r"AQUILA - Gender - BMI ($\si{\kilo\gram\per\square\meter}$)",
    "AQUILA_sex_height": r"AQUILA - Gender - Height ($\si{\meter}$)",
    "AQUILA_sex_weight": r"AQUILA - Gender - Weight ($\si{\kilo\gram}$)",
}


def plot_datasets_real(output_path):
    output_path = output_path / "datasets"
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = []

    # AQUILA omitted from the published code

    mimic_path = Path("../../data/mimic")
    if mimic_path.exists():
        datasets.extend(MIMICLoader(data_path=mimic_path).get_all_numeric_datasets().values())

    datasets = [ds for ds in datasets if ds.num_groups == 2]
    datasets = [ds for ds in datasets if "anchor_age" not in ds.name]

    fig = plt.figure(figsize=_figsize(two_columns=True, height=7))
    gs = fig.add_gridspec(
        math.ceil(len(datasets) / 2), 2, left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.2, hspace=0.5
    )
    axes = gs.subplots(sharex=False, sharey=False).flatten()

    for ax, dataset in zip(axes, datasets):
        data = dataset.data

        q75, q25 = np.percentile(dataset.data, [75, 25])
        iqr = q75 - q25
        h = 2 * iqr / (len(dataset.data) ** (1 / 3))
        maximum = dataset.input_range[1]
        minimum = dataset.input_range[0]
        bins = int((maximum - minimum) / h)
        xlim = dataset.input_range

        bins = max(bins, 10)

        # Calculate histogram and bins with numpy
        groups = np.unique(data[:, 0])

        for i, group in enumerate(groups):
            hist, bin_edges = np.histogram(data[:, 1][data[:, 0] == group], bins=bins, range=xlim, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            width = bin_centers[1] - bin_centers[0]
            ax.bar(bin_centers, hist, width=width, alpha=0.5, label=f"Group {i+1}", lw=0.5, edgecolor="black")

        ax.set_xlim(xlim)
        ax.set_title(dataset_mapping[dataset.name])

        ax.spines[["right", "top"]].set_visible(False)

        # Remove y-axis labels
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_yticks([])
        ax.set_ylabel("Density")

        ax.spines[["right", "top"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=2)

    plt.savefig(output_path / "datasets_real.pdf")


def plot_datasets_synthetic(output_path):
    output_path = output_path / "datasets"
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = [
        KGroupsUniform,
        KGroupsNormalEqualSpread,
        KGroupsConstantEqualSpread,
        KGroupsExtremeEqualSpread,
    ]

    group_sizes = [2, 3, 4]

    num_samples = 1000000

    fig = plt.figure(figsize=_figsize(two_columns=True, height=7))
    gs = fig.add_gridspec(
        len(datasets), len(group_sizes), left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.2, hspace=0.4
    )
    axes = gs.subplots(sharex=False, sharey=False)

    for row, dataset in enumerate(datasets):
        for col, group_size in enumerate(group_sizes):
            ds = dataset(group_size, num_samples, input_range=(-1, 1), data_rng=data_rng)
            data = ds.data

            ax = axes[row, col]

            # optimal histogram bin size - Freedman-Diaconis rule
            q75, q25 = np.percentile(ds.data, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr / (len(ds.data) ** (1 / 3))
            maximum = ds.input_range[1]
            minimum = ds.input_range[0]
            bins = int((maximum - minimum) / h)
            xlim = ds.input_range

            # Calculate histogram and bins with numpy
            groups = np.unique(data[:, 0])

            num_groups = len(groups)

            for i, group in enumerate(groups):
                hist, bin_edges = np.histogram(data[:, 1][data[:, 0] == group], bins=bins, range=xlim, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                width = bin_centers[1] - bin_centers[0]
                ax.bar(bin_centers, hist, width=width, alpha=0.5, label=f"Group {i+1}", lw=0.5, edgecolor="black")

            ax.set_xlim(xlim)
            ax.set_title(dataset_mapping[ds.__class__.__name__])

            # Remove y-axis labels
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_yticks([])
            ax.set_ylabel("Density")

            ax.spines[["right", "top"]].set_visible(False)

    handles, labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=4)

    plt.savefig(output_path / "datasets.pdf")


if __name__ == "__main__":
    _init_latex()

    data_rng = np.random.default_rng(0)

    output_dir = Path("plots_paper")
    plot_datasets_synthetic(output_dir)
    plot_datasets_real(output_dir)
