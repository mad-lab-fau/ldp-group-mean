"""Processing results from the group mean simulation experiments."""
import numpy as np
import pandas as pd
from sok_ldp_analysis.ldp.distribution.util import project_onto_prob_simplex


def _calculate_errors(df: pd.DataFrame, true_col="sample_mean", est_col="est_mean", prefix=""):
    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix = prefix + "_"

    # add data range to df
    df["data_range_size"] = df["input_range"].apply(lambda x: x[1] - x[0])

    df[prefix + "error"] = df[est_col] - df[true_col]
    df[prefix + "abs_error"] = df[prefix + "error"].abs()
    df[prefix + "mse"] = df[prefix + "error"] ** 2

    df[prefix + "group_err_mean"] = df["errors_per_group"].apply(lambda x: np.mean(x)) / df["n"]
    df[prefix + "group_err_std"] = df["errors_per_group"].apply(lambda x: np.std(x)) / df["n"]

    df[prefix + "total_group_error_ratio"] = df["total_group_errors"] / df["n"]

    df[prefix + "scaled_abs_error"] = df[prefix + "abs_error"] / df["data_range_size"]
    df[prefix + "scaled_mse"] = (df[prefix + "error"] / df["data_range_size"]) ** 2

    errors = [
        "error",
        "abs_error",
        "mse",
        "scaled_abs_error",
        "scaled_mse",
        "group_err_mean",
        "group_err_std",
        "total_group_error_ratio",
    ]

    errors = [prefix + e for e in errors] + ["data_range_size"]

    return df, errors


def _agg_mean(x):
    printed = False
    for y in x.to_numpy():
        if np.isnan(y).any() and not np.all(np.isnan(y)) and not printed:
            print("Some values are nan, but not all", x.name)
            printed = True

    values = [y for y in x.to_numpy() if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.mean(np.stack(values), axis=0)


def _agg_std(x):
    values = [y for y in x.to_numpy() if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.std(np.stack(values), axis=0)


def _agg_mean_all(x):
    printed = False
    for y in x.to_numpy():
        if np.isnan(y).any() and not np.all(np.isnan(y)) and not printed:
            print("Some values are nan, but not all", x.name)
            printed = True

    values = [y for y in x.to_numpy() if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.mean(np.concatenate(values))  # TODO: doesn't work for frequency due to different domain sizes - apply
    # mean per value in the list concatenation would work, but is slow


def _agg_std_all(x):
    values = [y for y in x.to_numpy() if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.std(np.concatenate(values))


def _post_process_results_fo(array):
    if np.isnan(array).any():
        return array

    # check if the sum is not close to 1
    if not np.isclose(np.sum(array), 1) or np.any(array < 0):
        return project_onto_prob_simplex(array)
    return array


def group_results_mean(df, output_dir):  # noqa: C901
    """
    Process the results of the group mean simulation experiments.

    Args:
        df: The dataframe with the results.
        output_dir: The directory to store the grouped results.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # filter rows where failed=True
    df = df[~df["failed"]]

    # Round epsilons to 10 decimal places
    df["eps"] = df["eps"].round(10)

    # calculate errors
    df, agg_columns = _calculate_errors(df)

    # Calculate the mean and standard deviation of the run time and the errors
    agg_columns = ["run_time", "est_mean", "sample_mean", "sample_sigma", "resp_var", *agg_columns]

    agg_columns_multi = []
    for col in agg_columns:
        if type(df[col].iloc[0]) is np.ndarray and len(df[col].iloc[0]) > 1:
            agg_columns_multi.append(col)

    other_group_cols = []
    if "k" in df.columns:
        other_group_cols.append("k")

    if "ratio" in df.columns:
        other_group_cols.append("ratio")

    if "group_ratio" in df.columns:
        other_group_cols.append("group_ratio")

    df_groupby = df.groupby(["method", "data", "input_range", "num_groups", "n", "eps", *other_group_cols])
    df_groupby_ir = df.groupby(["method", "data", "num_groups", "n", "eps", *other_group_cols])
    df_groupby_all_data = df.groupby(["method", "num_groups", "n", "eps", *other_group_cols])

    grouped_dfs = []
    for groupby in [df_groupby, df_groupby_ir, df_groupby_all_data]:
        df_grouped = groupby.agg(
            **{
                "run_time_mean": pd.NamedAgg(column="run_time", aggfunc="mean"),
                "run_time_std": pd.NamedAgg(column="run_time", aggfunc="std"),
                **{f"{k}_mean": pd.NamedAgg(column=k, aggfunc=_agg_mean) for k in agg_columns},
                **{f"{k}_std": pd.NamedAgg(column=k, aggfunc=_agg_std) for k in agg_columns},
                **{f"{k}_mean_all": pd.NamedAgg(column=k, aggfunc=_agg_mean_all) for k in agg_columns_multi},
                **{f"{k}_std_all": pd.NamedAgg(column=k, aggfunc=_agg_std_all) for k in agg_columns_multi},
            }
        )

        df_grouped = df_grouped.reset_index()

        if "data" not in groupby.keys:
            df_grouped["data"] = "all"

        if "input_range" not in groupby.keys:
            df_grouped["input_range"] = "all"

        grouped_dfs.append(df_grouped)

    # Merge the two dataframes
    df_grouped = pd.concat(grouped_dfs, ignore_index=True)

    # Store results
    method_name = df_grouped["method"].iloc[0]
    if "k" in df_grouped.columns:
        k = df_grouped["k"].iloc[0]
        output_path = output_dir / f"{method_name}_{k}.pkl"
    elif "ratio" in df_grouped.columns:
        ratio = df_grouped["ratio"].iloc[0]
        output_path = output_dir / f"{method_name}_{ratio}.pkl"
    else:
        output_path = output_dir / f"{method_name}.pkl"

    df_grouped.to_pickle(output_path)
