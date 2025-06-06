"""Plot individual runs."""
import copy
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from ldp_group_mean.simulation.processing.processing import read_pickle
from scripts.plotting.plot_paper import _add_subplot_label, _figsize, _init_latex

method_mapping_group_mean = {
    "GroupMeanBernoulli": "Bernoulli",
    "GroupMeanLaplace": "Laplace",
    "GroupMeanPiecewiseHalf": r"Piecewise ($50\%$)",
    "GroupMeanWaudbySmith": "NPRR (k=8)",
    "GroupMeanWaudbySmith1": "NPRR (k=1)",
    "GroupMeanWaudbySmith16": "NPRR (k=16)",
    "GroupMeanWaudbySmith32": "NPRR (k=32)",
}

methods = ["GroupMeanBernoulli", "GroupMeanLaplace", "GroupMeanPiecewiseHalf", "GroupMeanWaudbySmith"]
datasets = [
    "MIMIC_admission_location_age",
    "MIMIC_admission_type_age",
    "MIMIC_dead_age",
    "MIMIC_discharge_location_age",
    "MIMIC_gender_age",
    "MIMIC_icd_chapter_age",
    # "AQUILA_disease_age",
    # "AQUILA_disease_height",
    # "AQUILA_disease_weight",
    # "AQUILA_disease_bmi",
    # "AQUILA_sex_age",
    # "AQUILA_sex_height",
    # "AQUILA_sex_weight",
    # "AQUILA_sex_bmi",
]

value_mapping = {
    "MIMIC_admission_location_age": "Age in Years",
    "MIMIC_admission_type_age": "Age in Years",
    "MIMIC_gender_age": "Age in Years",
    "MIMIC_icd_chapter_age": "Age in Years",
    "MIMIC_dead_age": "Age in Years",
    "MIMIC_discharge_location_age": "Age in Years",
    # "AQUILA_disease_age": "Age in Years",
    # "AQUILA_disease_bmi": r"BMI in $\si{\kilo\gram\per\square\meter}$",
    # "AQUILA_disease_height": r"Height in $\si{\meter}$",
    # "AQUILA_disease_weight": r"Weight in $\si{\kilo\gram}$",
    # "AQUILA_sex_age": "Age in Years",
    # "AQUILA_sex_bmi": r"BMI in $\si{\kilo\gram\per\square\meter}$",
    # "AQUILA_sex_height": r"Height in $\si{\meter}$",
    # "AQUILA_sex_weight": r"Weight in $\si{\kilo\gram}$",
}


def _get_group_mappings():
    pkl_file = Path("../../results_grouped/group_mappings.pkl")

    with pkl_file.open("rb") as f:
        mappings = pickle.load(f)

    return mappings


def plot_mean_case_single_col(output_dir):  # noqa: C901, PLR0915
    input_path = Path("../../results/group_mean")

    output_dir = output_dir / "usecase"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    group_mappings = _get_group_mappings()

    # epsilons = [x.stem for x in (input_path / methods[0] / datasets[0]).glob("*.pkl")]
    epsilons = ["0.5", "1.0", "4.0", "10.0"]
    print(epsilons)

    for dataset in datasets:
        print(f"Processing {dataset}")

        group_mapping = group_mappings[dataset]

        # only use the group mapping if no entry is longer than 4 characters
        if dataset == "MIMIC_dead_age":
            use_group_mapping = True
            group_mapping = {k: "yes" if v else "no" for k, v in group_mapping.items()}
        else:
            use_group_mapping = all(len(group) <= 4 for group in group_mapping.values())

        fig = plt.figure(figsize=_figsize(two_columns=False, height=3))
        gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
        gs.update(left=0.15, right=0.98, top=0.98, bottom=0.25, wspace=0.2, hspace=0.1)
        axes = gs.subplots(sharey="row", sharex=False)

        for i, eps in enumerate(epsilons):
            col = i % 2
            row = i // 2
            ax = axes[row, col]
            method_dfs = []

            # Load raw data
            for method in methods:
                file = input_path / method / dataset / f"{eps}.pkl"

                df = read_pickle(file)
                max_n = df["n"].max()
                df = df[df["n"] == max_n].reset_index()

                method_dfs.append(df)

            num_groups = method_dfs[0]["num_groups"].iloc[0]

            horizontal_width = 0.25

            # Add one line for the true mean for each group
            ax.hlines(
                method_dfs[0].iloc[0]["sample_mean"],
                xmin=np.arange(num_groups) - 1.1 * horizontal_width,
                xmax=np.arange(num_groups) + 1.1 * horizontal_width,
                label="True mean",
                color="black",
                linestyle="-",
            )

            num_methods = len(methods)
            method_offsets = np.linspace(-horizontal_width, horizontal_width, num_methods)
            for df, method, offset in zip(method_dfs, methods, method_offsets):
                # Shape groups x runs
                est_means = np.stack(df["est_mean"].to_numpy())

                n = est_means.shape[0]
                est_means_flat = est_means.flatten()
                small_rnd_noise = np.random.uniform(-0.01, 0.01, n * num_groups)

                order = np.arange(num_groups)
                x = np.tile(np.arange(num_groups), n)
                ax.scatter(
                    x + small_rnd_noise + offset,
                    est_means_flat,
                    marker="o",
                    alpha=0.5,
                    label=method_mapping_group_mean[method],
                    s=0.6,
                    rasterized=True,
                )

            # _add_subplot_label(ax, fig, f"$\\varepsilon$={eps}", 0.7, 0.3)
            ax.set_title(f"$\\varepsilon$={eps}", y=0.85)

            input_range = method_dfs[0].iloc[0]["input_range"]
            print(input_range)
            # Get current ylim
            ylim = ax.get_ylim()

            # Cap ylim to the input range
            ylim = (max(ylim[0], input_range[0]), min(ylim[1], input_range[1]))

            ax.set_ylim(ylim)

            ax.set_xticks(range(num_groups))
            # Label x axis with categories
            if row == 1:
                if use_group_mapping:
                    ax.set_xticklabels([group_mapping[i] for i in range(num_groups)])
                else:
                    ax.set_xlabel("Group")

            ax.grid(which="major", linestyle="--", linewidth=0.5)
            ax.grid(which="minor", linestyle="--", linewidth=0.5)

            # ax.set_xlabel("Group")
            if col == 0:
                ax.set_ylabel(f"Mean {value_mapping[dataset]}")
            ax.spines[["right", "top"]].set_visible(False)
            # ax.label_outer()

        # Add a legend on the bottom
        handles, labels = axes[0, 0].get_legend_handles_labels()
        # Make a copy of the handles
        handles = [copy.copy(h) for h in handles]
        # Set the opacity of the handles to 100% and increase the size
        for handle in handles:
            handle.set_alpha(1)
            if hasattr(handle, "_sizes"):
                handle.set_sizes([20])
        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=3)

        fig.align_ylabels(axes)

        plt.savefig(output_dir / f"{dataset}.pdf")

        plt.close()


def plot_mean_diff(output_dir):
    """
    Plot the difference between estimated group means and compare it to the true difference in means.

    Args:
        output_dir: The output directory to save the plots.
    """
    input_path = Path("../../results/group_mean")

    output_dir = output_dir / "diff"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    epsilons = ["0.5", "1.0", "4.0", "10.0"]

    box_width = 0.8

    for dataset in datasets:
        fig = plt.figure(figsize=_figsize(two_columns=True, height=1.6))
        gs = fig.add_gridspec(1, 4, hspace=0, wspace=0)
        gs.update(left=0.1, right=0.98, top=0.9, bottom=0.2, wspace=0.3, hspace=0.1)
        axes = gs.subplots(sharey=False, sharex=False)

        two_groups = False

        for i, eps in enumerate(epsilons):
            ax = axes[i]
            method_dfs = []

            # Load raw data
            for method in methods:
                print(f"Processing {dataset} {method} {eps}")
                file = input_path / method / dataset / f"{eps}.pkl"

                df = read_pickle(file)

                if df["num_groups"].iloc[0] == 2:
                    two_groups = True
                else:
                    break

                max_n = df["n"].max()
                df = df[df["n"] == max_n].reset_index()

                # Calculate the difference between the mean in group 0 and group 1
                df["diff"] = df["est_mean"].apply(lambda x: x[0] - x[1])
                df["true_diff"] = df["sample_mean"].apply(lambda x: x[0] - x[1])

                method_dfs.append(df)
            else:
                # Continue if the loop was not broken (i.e. we have 2 groups only)
                num_groups = method_dfs[0]["num_groups"].iloc[0]

                # Plot boxplots for the differences
                data = [df["diff"] for df in method_dfs]

                for j, method in enumerate(methods):
                    # get color from default color cycle
                    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][j]

                    parts = ax.violinplot(
                        data[j],
                        positions=[j],
                        widths=box_width,
                        showextrema=True,
                        showmeans=True,
                        showmedians=False,
                    )

                    for pc in parts["bodies"]:
                        pc.set_facecolor(color)
                        pc.set_edgecolor("black")
                        pc.set_alpha(0.7)
                        pc.set_zorder(2)
                        pc.set_label(method_mapping_group_mean[method])

                # Add the true difference as a line
                true_diff = method_dfs[0].iloc[0]["true_diff"]
                ax.hlines(
                    true_diff,
                    xmin=-0.5,
                    xmax=3.5,
                    label=f"True difference ({true_diff:.2f})",
                    color="black",
                    linestyle="--",
                )

                ylim = ax.get_ylim()

                order = np.sign(true_diff)
                if order > 0:  # Color above 0 as "incorrect"
                    ax.fill_between(
                        [-0.5, 3.5],
                        ylim[0],
                        color="lightgray",
                        alpha=0.5,
                        label="Incorrect order",
                        zorder=1,
                    )
                else:  # Color below 0 as "incorrect"
                    ax.fill_between(
                        [-0.5, 3.5],
                        ylim[1],
                        color="lightgray",
                        alpha=0.5,
                        label="Incorrect order",
                        zorder=1,
                    )

                ax.set_ylim(ylim)
                ax.set_xlim(-0.5, 3.5)

                ax.set_title(f"$\\varepsilon$={eps}", y=0.95)

                ax.grid(which="major", linestyle="--", linewidth=0.5)
                ax.grid(which="minor", linestyle="--", linewidth=0.5)

                # Remove x-ticks
                ax.set_xticks([])

                # ax.set_xlabel("Group")
                if i == 0:
                    ax.set_ylabel(f"Group Mean Difference\n({value_mapping[dataset]})")
                ax.spines[["right", "top"]].set_visible(False)

        if two_groups:
            # Add a legend on the bottom
            handles, labels = axes[0].get_legend_handles_labels()
            # Make a copy of the handles
            handles = [copy.copy(h) for h in handles]
            # Set the opacity of the handles to 100% and increase the size
            for handle in handles:
                handle.set_alpha(1)
                if hasattr(handle, "_sizes"):
                    handle.set_sizes([20])
            fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=6)

            fig.align_ylabels(axes)

            plt.savefig(output_dir / f"{dataset}.pdf")

        plt.close()


def plot_mean_diff_order(output_dir):
    """
    Plot the difference between estimated group means and compare it to the true difference in means.

    Args:
        output_dir: The output directory to save the plots.
    """
    input_path = Path("../../results/group_mean")

    output_dir = output_dir / "diff_order"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # epsilons = ["0.5", "1.0", "4.0", "10.0"]

    epsilons = [x.stem for x in (input_path / methods[0] / datasets[0]).glob("*.pkl")]
    # sort epsilons by their float() value
    epsilons = sorted(epsilons, key=lambda x: float(x))

    for dataset in datasets:
        fig = plt.figure(figsize=_figsize(two_columns=False, height=1.5))
        gs = fig.add_gridspec(1, 1)
        gs.update(left=0.2, right=0.98, top=0.98, bottom=0.4, wspace=0.2, hspace=0.1)
        ax = gs.subplots()

        two_groups = False

        true_diff = 0
        method_eps_ratios = np.zeros((len(methods), len(epsilons)))
        for i, eps in enumerate(epsilons):
            # Load raw data
            for method in methods:
                print(f"Processing {dataset} {method} {eps}")
                file = input_path / method / dataset / f"{eps}.pkl"

                df = read_pickle(file)

                if df["num_groups"].iloc[0] == 2:
                    two_groups = True
                else:
                    break

                max_n = df["n"].max()
                df = df[df["n"] == max_n].reset_index()

                # Calculate the difference between the mean in group 0 and group 1
                df["diff"] = df["est_mean"].apply(lambda x: x[0] - x[1])
                df["true_diff"] = df["sample_mean"].apply(lambda x: x[0] - x[1])
                df["correct_order"] = df["diff"] * df["true_diff"] > 0

                if true_diff != 0:
                    assert true_diff == df["true_diff"].iloc[0]

                true_diff = df["true_diff"].iloc[0]

                # Count the number of correct orders for different epsilons by grouping
                df_grouped = df.groupby("eps").agg({"correct_order": "sum"}).reset_index()

                method_eps_ratios[methods.index(method), i] = df_grouped["correct_order"].iloc[0] / 200  # 200 runs

        if two_groups:
            for j, method in enumerate(methods):
                ax.plot(
                    [float(eps) for eps in epsilons],
                    method_eps_ratios[j] * 100,
                    label=method_mapping_group_mean[method],
                )

            ax.grid(which="major", linestyle="--", linewidth=0.5)
            ax.grid(which="minor", linestyle="--", linewidth=0.5)

            _add_subplot_label(ax, fig, f"Group Difference: {true_diff:.2f}", 0.5, 0.25)

            ax.set_xlim(0, 10)
            ax.set_ylim(top=100)

            # ax.set_xlabel("Group")
            ax.set_ylabel("Results With\nCorrect Order (\\%)")
            ax.set_xlabel("$\\varepsilon$")
            ax.spines[["right", "top"]].set_visible(False)
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

            # Add a legend on the bottom
            handles, labels = ax.get_legend_handles_labels()
            # Make a copy of the handles
            handles = [copy.copy(h) for h in handles]
            # Set the opacity of the handles to 100% and increase the size
            for handle in handles:
                if hasattr(handle, "_sizes"):
                    handle.set_sizes([20])
            fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=4, fontsize="small")

            plt.savefig(output_dir / f"{dataset}.pdf")

        plt.close()


if __name__ == "__main__":
    _init_latex()

    output_dir = Path("plots_paper") / "group_mean_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_diff_order(output_dir)
    plot_mean_diff(output_dir)
    plot_mean_case_single_col(output_dir)
