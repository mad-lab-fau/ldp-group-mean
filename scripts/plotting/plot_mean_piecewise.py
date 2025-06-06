from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plotting.plot_paper import _init_latex, _read_all_pkls

dataset_mapping = {
    "KGroupsUniform": "Uniform",
    "KGroupsNormalEqualSpread": "Normal",
    "KGroupsConstantEqualSpread": "Constant",
    "KGroupsExtremeEqualSpread": "Extremum",
    "all": "All",
}


def plot_mean_piecewise_table(df_grouped, output_dir):
    """
    Produce a LaTeX table summarizing the mean piecewise results.

    Args:
        df_grouped: DataFrame containing the grouped results.
        output_dir:  Path to the output directory where the LaTeX table will be saved.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    error = "scaled_abs_error"

    grid_horizontal = "num_groups"
    # horizontal_values = df_grouped[grid_horizontal].unique()
    horizontal_values = [2, 8, 64]

    # Make three pivot tables (one per horizontal value)
    # columns is the ratio, index is the eps
    with Path.open(output_dir / f"mean_piecewise_summary_{error}.tex", "w") as f:
        f.write("")

        suffix = "" if f"{error}_mean_all" not in df_grouped.columns else "_all"
        for horizontal_value in horizontal_values:
            print(f"Groups: {horizontal_value}")
            df_filtered = df_grouped[
                (df_grouped[grid_horizontal] == horizontal_value)
                & (df_grouped["data"] == "all")
                & (df_grouped["n"] == 10_000 * horizontal_value)
                & (df_grouped["input_range"] == "all")
            ]

            df_count = df_filtered.pivot_table(
                index="eps",
                columns="ratio",
                values=f"{error}_mean{suffix}",
                aggfunc="count",
            )

            # Assert that all counts are 1
            assert df_count.max().max() == 1, f"Counts are not 1 for {horizontal_value} groups"

            df_mean = df_filtered.pivot_table(
                index="eps",
                columns="ratio",
                values=f"{error}_mean{suffix}",
                aggfunc="mean",
            )

            df_std = df_filtered.pivot_table(
                index="eps",
                columns="ratio",
                values=f"{error}_std{suffix}",
                aggfunc="mean",
            )

            mean_with_uncertainty = False

            if mean_with_uncertainty:
                # Highlight min values using the numeric mean DataFrame
                df_styled = df_mean.style.highlight_min(axis=1, props="bfseries:--nowrap;")

                # Format indices before combining (these affect the final LaTeX table)
                df_styled = df_styled.format_index("{:.1f}", axis=0)
                df_styled = df_styled.format_index("{:.1f}", axis=1)

                def format_sci_uncertainty(mean, std):
                    if pd.isna(mean) or pd.isna(std):
                        return ""
                    exponent = int(np.floor(np.log10(abs(mean)))) if mean != 0 else 0
                    base = mean / (10**exponent)
                    uncertainty_scaled = std / (10**exponent)
                    # Get uncertainty as integer in last digits
                    uncertainty_digits = int(round(uncertainty_scaled * 10**2))  # 2 sig. digits
                    return f"{base:.2f}({uncertainty_digits})e{exponent}"

                df_combined = pd.DataFrame(
                    np.vectorize(format_sci_uncertainty)(df_mean.values, df_std.abs().values),
                    index=df_mean.index,
                    columns=df_mean.columns,
                )

                # Now set the styled DataFrame's values to the combined strings
                df_styled.data = df_combined
            else:  # place the standard deviation in the row below the mean
                # add "_mean" and "_std" to the index
                df_mean.index = pd.MultiIndex.from_product([df_mean.index, ["mean"]])
                df_std.index = pd.MultiIndex.from_product([df_std.index, ["std"]])
                # flatten the index
                df_mean.index = df_mean.index.map(lambda x: f"{x[0]}_{x[1]}")
                df_std.index = df_std.index.map(lambda x: f"{x[0]}_{x[1]}")

                df_concat = pd.concat([df_mean, df_std]).sort_index(key=lambda x: x.str.split("_").str[0].astype(float))

                num_entries = len(df_concat)
                df_styled = df_concat.style.format("{:.2e}", subset=(df_concat.index[0:num_entries:2], slice(None)))
                df_styled = df_styled.format("({:.2e})", subset=(df_concat.index[1:num_entries:2], slice(None)))

                # mark lowest value in each mean row
                df_styled = df_styled.highlight_min(
                    axis=1, props="bfseries:--nowrap;", subset=(df_concat.index[0:num_entries:2], slice(None))
                )

                df_styled = df_styled.format_index(lambda x: f"{float(x.split('_')[0]):.1f}", axis=0)
                df_styled = df_styled.format_index("{:.1f}", axis=1)

            # render as latex table
            str_out = df_styled.to_latex(
                hrules=True,
                siunitx=True,
                label=f"tab:mean_piecewise_summary_{horizontal_value}",
                caption=f"Mean {error.replace('_', ' ')} for {horizontal_value} groups",
            )

            f.write(str_out)


if __name__ == "__main__":
    _init_latex()

    df_grouped = _read_all_pkls("group_mean_piecewise_fixed")

    output_dir = Path("plots_paper") / "group_mean_piecewise"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_piecewise_table(df_grouped, output_dir)
