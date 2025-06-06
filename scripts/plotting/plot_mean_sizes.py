from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.plotting.plot_mean_cases import method_mapping_group_mean
from scripts.plotting.plot_mean_nprr import dataset_mapping
from scripts.plotting.plot_paper import _add_subplot_label, _figsize, _init_latex, _read_all_pkls


def plot_mean_sizes_summary(df_grouped, output_dir, error=None):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if error is None:
        error = "scaled_abs_error"

    grid_horizontal = "num_groups"
    # horizontal_values = df_grouped[grid_horizontal].unique()
    horizontal_values = [2, 8, 64]

    colors = "method"
    color_values = df_grouped[colors].unique()
    # Sort ascending
    color_values = np.sort(color_values)

    # two_columns = error == "scaled_abs_error"
    two_columns = True
    height = 2 if two_columns else 1.6

    fig = plt.figure(figsize=_figsize(two_columns=two_columns, height=height))
    gs = fig.add_gridspec(1, len(horizontal_values), hspace=0, wspace=0)
    if two_columns:
        gs.update(left=0.08, right=0.98, top=0.9, bottom=0.30, wspace=0.15, hspace=0.2)
    else:
        gs.update(left=0.15, right=0.98, top=0.9, bottom=0.35, wspace=0.12, hspace=0.2)

    axes = gs.subplots(sharey=True, sharex=True)

    for col, horizontal_value in enumerate(horizontal_values):
        ax = axes[col]

        n = 10_000 * horizontal_value

        for color_value in color_values:
            df_filtered = df_grouped[
                (df_grouped["data"] == "all")
                & (df_grouped[grid_horizontal] == horizontal_value)
                & (df_grouped[colors] == color_value)
                & (df_grouped["n"] == n)
            ]

            suffix = "" if f"{error}_mean_all" not in df_filtered.columns else "_all"

            method = method_mapping_group_mean[color_value]

            ax.plot(
                df_filtered["eps"],
                df_filtered[f"{error}_mean{suffix}"],
                label=f"{method}",
            )

            ax.fill_between(
                df_filtered["eps"],
                df_filtered[f"{error}_mean{suffix}"] - df_filtered[f"{error}_std{suffix}"],
                df_filtered[f"{error}_mean{suffix}"] + df_filtered[f"{error}_std{suffix}"],
                alpha=0.2,
            )

        ax.spines[["right", "top"]].set_visible(False)
        ax.label_outer()

        ax.set_yscale("log")
        ax.set_xlabel("$\\varepsilon$")

        ax.set_xticks([0, 2, 4, 6, 8, 10])

        ax.grid(which="major", linestyle="--", linewidth=0.5)
        ax.grid(which="minor", linestyle="--", linewidth=0.5)
        ax.yaxis.set_tick_params(which="both", labelleft=True)

        axes[col].set_title(f"{horizontal_value} groups")

        if error == "scaled_abs_error":
            axes[0].set_ylabel("Scaled MAE")
        else:
            axes[0].set_ylabel("Group Error Ratio")

    # Add a legend
    handles, labels = axes[0].get_legend_handles_labels()

    if two_columns:
        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=len(color_values))
    else:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="outside lower center",
            ncol=len(color_values),
            fontsize=6,
            handlelength=1,
        )

    output_path = output_dir / f"{error}_summary.pdf"
    plt.savefig(output_path)


def plot_mean_sizes(df_grouped, output_dir, error=None):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if error is None:
        error = "scaled_abs_error"

    input_ranges = df_grouped["input_range"].unique()

    grid_horizontal = "num_groups"
    # horizontal_values = df_grouped[grid_horizontal].unique()
    horizontal_values = [2, 8, 64]

    grid_vertical = "data"
    vertical_values = df_grouped[grid_vertical].unique()
    # filter nan
    vertical_values = vertical_values[~pd.isna(vertical_values)]
    # sort ascending
    vertical_values = np.sort(vertical_values)

    colors = "method"
    color_values = df_grouped[colors].unique()
    # Sort ascending
    color_values = np.sort(color_values)

    for input_range in input_ranges:
        fig = plt.figure(figsize=_figsize(two_columns=True, height=1.6 * len(vertical_values)))
        gs = fig.add_gridspec(len(vertical_values), len(horizontal_values), hspace=0, wspace=0)
        gs.update(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.15, hspace=0.1)
        axes = gs.subplots(sharey="row", sharex=True)

        for row, vertical_value in enumerate(vertical_values):
            for col, horizontal_value in enumerate(horizontal_values):
                ax = axes[row, col]

                n = 10_000 * horizontal_value

                for color_value in color_values:
                    df_filtered = df_grouped[
                        (df_grouped[grid_vertical] == vertical_value)
                        & (df_grouped[grid_horizontal] == horizontal_value)
                        & (df_grouped[colors] == color_value)
                        & (df_grouped["n"] == n)
                        & (df_grouped["input_range"] == input_range)
                    ]

                    suffix = "" if f"{error}_mean_all" not in df_filtered.columns else "_all"

                    method = method_mapping_group_mean[color_value]

                    ax.plot(
                        df_filtered["eps"],
                        df_filtered[f"{error}_mean{suffix}"],
                        label=f"{method}",
                    )

                    ax.fill_between(
                        df_filtered["eps"],
                        df_filtered[f"{error}_mean{suffix}"] - df_filtered[f"{error}_std{suffix}"],
                        df_filtered[f"{error}_mean{suffix}"] + df_filtered[f"{error}_std{suffix}"],
                        alpha=0.2,
                    )

                ax.spines[["right", "top"]].set_visible(False)
                ax.label_outer()

                ax.set_yscale("log")
                ax.set_xlabel("$\\varepsilon$")
                ax.set_xticks([0, 2, 4, 6, 8, 10])

                ax.grid(which="major", linestyle="--", linewidth=0.5)
                ax.grid(which="minor", linestyle="--", linewidth=0.5)
                ax.yaxis.set_tick_params(which="both", labelleft=True)

                _add_subplot_label(ax, fig, f"Data: {dataset_mapping[vertical_value]}", 0.0, 0.2)

                axes[0, col].set_title(f"{horizontal_value} groups")

            if error == "scaled_abs_error":
                axes[row, 0].set_ylabel("Scaled MAE")
            else:
                axes[row, 0].set_ylabel("Group Error Ratio")

        # Add a legend
        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=len(color_values))

        output_path = output_dir / f"{error}_{input_range}.pdf"
        plt.savefig(output_path)


if __name__ == "__main__":
    _init_latex()

    df_grouped = _read_all_pkls("group_mean_sizes")

    output_dir = Path("plots_paper") / "group_mean_sizes"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_sizes_summary(df_grouped, output_dir)
    plot_mean_sizes_summary(df_grouped, output_dir, "total_group_error_ratio")

    plot_mean_sizes(df_grouped, output_dir)
    plot_mean_sizes(df_grouped, output_dir, "total_group_error_ratio")
