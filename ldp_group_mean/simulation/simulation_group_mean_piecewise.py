"""Simulation for the group mean estimation."""
import itertools
import time
import traceback

import numpy as np
from sok_ldp_analysis.simulation.simulation import filter_eps_range, simulation_loop

from ldp_group_mean.ldp_group_mean.group_mean_piecewise import GroupMeanPiecewise
from ldp_group_mean.simulation.data.synthetic.group_mean_constant import KGroupsConstantEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_extreme import KGroupsExtremeEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_normal import KGroupsNormalEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_uniform import KGroupsUniform
from ldp_group_mean.simulation.util import _calc_group_errors, calc_mean_counts


def _run_mean_piecewise(args):
    store_path, n_range, dataset, eps, ratio, num_runs, seed = args

    method = GroupMeanPiecewise

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
                eps1 = eps * ratio
                eps2 = eps - eps1

                # Initialize the mechanism
                mechanism = method(eps=(eps1, eps2), rng=rng, input_range=input_range)

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
                    "ratio": ratio,
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


def simulate_group_mean_piecewise(output_path, use_mpi=False, i=None, fixed_group_size=False) -> None:
    """
    Run the group mean simulation with synthetic data and different group sizes.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        i: The index of the method to run. If None, all methods are run.
        fixed_group_size: Whether to fix the group size for different number of groups.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    if fixed_group_size:
        n_range = np.array([100, 1000, 10_000][::-1], dtype=int)
    else:
        n_range = np.array([100, 1000, 10_000, 100_000, 1_000_000][::-1], dtype=int)

    eps_range = np.array([0.1, 0.5, 1, 2, 4, 6, 8, 10])
    num_runs = [200]

    # Set up the data
    num_groups_range = [2, 3, 4, 8, 16, 32, 64][::-1]
    datasets = [
        KGroupsUniform,
        KGroupsNormalEqualSpread,
        KGroupsConstantEqualSpread,
        KGroupsExtremeEqualSpread,
    ]
    data_rng = rng.spawn(1)[0]

    # different ratios for eps1/eps2
    ratio_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][::-1]

    if i is not None:
        ratio_range = [ratio_range[i]]

    tasks = []
    for ratio in ratio_range:
        for dataset in datasets:
            for num_groups in num_groups_range:
                print()
                print(ratio)
                print(dataset.__name__)
                print(num_groups)

                store_path = output_path / str(ratio) / dataset.__name__ / str(num_groups)
                store_path.mkdir(parents=True, exist_ok=True)

                # Load/Generate the data
                n_range_ = num_groups * n_range if fixed_group_size else n_range

                dataset_size = max(n_range_)
                dataset_ = dataset(num_groups, dataset_size, (-1, 1), data_rng=data_rng.spawn(1)[0])

                # Filter runs that are already done
                eps_range_ = filter_eps_range(store_path, eps_range, len(n_range_) * num_runs[0])
                print(eps_range_)
                if len(eps_range_) == 0:
                    continue

                # Prepare random seeds
                seeds = rng.spawn(len(num_runs))

                # Set up the task list
                tasks.extend(
                    list(
                        itertools.product(
                            [store_path],
                            [n_range_],
                            [dataset_],
                            eps_range_,
                            [ratio],
                            num_runs,
                            seeds,
                        )
                    )
                )

    # Run the simulation
    simulation_loop(_run_mean_piecewise, tasks, use_mpi=use_mpi)
