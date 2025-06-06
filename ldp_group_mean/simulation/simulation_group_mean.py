"""Simulation for the group mean estimation."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
from sok_ldp_analysis.simulation.simulation import simulation_core_loop

from ldp_group_mean.ldp_group_mean.group_mean_bernoulli import GroupMeanBernoulli
from ldp_group_mean.ldp_group_mean.group_mean_laplace import GroupMeanLaplace
from ldp_group_mean.ldp_group_mean.group_mean_piecewise import GroupMeanPiecewiseHalf
from ldp_group_mean.ldp_group_mean.group_mean_waudby import GroupMeanWaudbySmith
from ldp_group_mean.simulation.data.mimic import MIMICLoader
from ldp_group_mean.simulation.util import _calc_group_errors, calc_mean_counts


def _run_mean_group(args):
    store_path, n_range, dataset, eps, method, num_runs, seed = args

    num_groups = dataset.num_groups
    data = dataset.data
    input_range = dataset.input_range
    rng = np.random.default_rng(seed)

    # Remove all n from n_range that are larger than the dataset size
    n_range = n_range[n_range <= len(data)]
    n_range = [len(data), *list(n_range)]

    result_list = []

    for i in range(num_runs):
        for n in n_range:
            data_ = data[:n]

            try:
                # Initialize the mechanism
                mechanism = method(eps=eps, rng=rng, input_range=input_range)

                # Run the mechanism
                start = time.time()
                resp = mechanism.mechanism(num_groups, data_)
                est_sum, est_mean, est_counts, est_group = mechanism.mean(num_groups, resp)
                end = time.time()

                true_group = data_[:, 0]
                errors_per_group, total_group_errors = _calc_group_errors(true_group, est_group, num_groups)
                est_counts_resp = np.bincount(est_group.astype(int), minlength=num_groups)

                eps1 = mechanism.eps1
                eps2 = mechanism.eps2

                resp_mean = np.mean(resp)
                resp_var = np.var(resp)

                run_time = end - start
                failed = False
            except (ValueError, AssertionError, RuntimeWarning) as e:
                print(e)
                print(traceback.format_exc())
                eps1 = np.nan
                eps2 = np.nan
                est_mean = np.nan
                est_sum = np.nan
                est_counts = np.nan
                resp_mean = np.nan
                resp_var = np.nan
                run_time = np.nan
                failed = True
                errors_per_group = np.nan
                total_group_errors = np.nan
                est_counts_resp = np.nan

            sample_sum, sample_mean, sample_sigma, sample_counts = calc_mean_counts(data_, num_groups)

            result_list.append(
                {
                    "method": method.__name__,
                    "data": str(dataset),
                    "num_groups": num_groups,
                    "input_range": input_range,
                    "n": n,
                    "eps": eps,
                    "eps1": eps1,
                    "eps2": eps2,
                    "run": i,  # Run number
                    "failed": failed,  # Whether the run failed
                    "run_time": run_time,  # Run time of the mechanism
                    "sample_mean": sample_mean,  # Mean of the private data per group
                    "sample_sum": sample_sum,  # Sum of the private data per group
                    "sample_sigma": sample_sigma,  # Standard deviation of the private data per group
                    "sample_counts": sample_counts,  # True counts per group from the private data
                    "est_mean": est_mean,  # Estimated mean per group
                    "est_sum": est_sum,  # Estimated sum per group
                    "est_counts": est_counts,  # Counts from the response with adjustment
                    "resp_mean": resp_mean,  # Mean of the response
                    "resp_var": resp_var,  # Variance of the response
                    "errors_per_group": errors_per_group,  # Errors per group
                    "total_group_errors": total_group_errors,  # Total number of errors
                    "est_counts_resp": est_counts_resp,  # Counts from the response without adjustment
                }
            )

    return store_path, result_list


def simulate_group_mean(output_path, use_mpi=False, i=None) -> None:
    """
    Run the group mean simulation.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        i: The index of the dataset to run. If None, all datasets are run.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    n_range = np.array([100, 1000, 10_000, 100_000, 1_000_000][::-1])
    # eps_range = np.logspace(-1, 1, 16)  # 10^(-1) to 10^1
    eps_range = np.array([0.1, 0.5, 1, 2, 4, 6, 8, 10])
    num_runs = [200]

    # Set up the data
    datasets = []

    # AQUILA omitted from the published code

    mimic_path = Path("../data/mimic")
    if mimic_path.exists() and (i is None or i == 1):
        datasets.extend(MIMICLoader(data_path=mimic_path).get_all_numeric_datasets().values())
    else:
        print("No MIMIC-IV.")

    # Select the methods
    methods = [
        GroupMeanBernoulli,
        GroupMeanLaplace,
        GroupMeanPiecewiseHalf,
        GroupMeanWaudbySmith,
    ]

    simulation_core_loop(output_path, _run_mean_group, methods, datasets, rng, num_runs, n_range, eps_range, use_mpi)
