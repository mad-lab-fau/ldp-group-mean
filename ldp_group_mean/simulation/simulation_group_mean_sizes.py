"""Simulation for the group mean estimation."""
from __future__ import annotations

import itertools

import numpy as np
from sok_ldp_analysis.simulation.simulation import filter_eps_range, simulation_loop

from ldp_group_mean.ldp_group_mean.group_mean_bernoulli import GroupMeanBernoulli
from ldp_group_mean.ldp_group_mean.group_mean_laplace import GroupMeanLaplace
from ldp_group_mean.ldp_group_mean.group_mean_piecewise import GroupMeanPiecewiseHalf
from ldp_group_mean.ldp_group_mean.group_mean_waudby import GroupMeanWaudbySmith
from ldp_group_mean.simulation.data.synthetic.group_mean_constant import KGroupsConstantEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_extreme import KGroupsExtremeEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_normal import KGroupsNormalEqualSpread
from ldp_group_mean.simulation.data.synthetic.group_mean_uniform import KGroupsUniform
from ldp_group_mean.simulation.simulation_group_mean import _run_mean_group


def simulate_group_mean_sizes(output_path, use_mpi=False, i=None) -> None:
    """
    Run the group mean simulation with synthetic data and different group sizes.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        i: The index of the method to run. If None, all methods are run.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    n_range = np.array([100, 1000, 10_000][::-1], dtype=int)
    # eps_range = np.logspace(-1, 1, 16)  # 10^(-1) to 10^1
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

    input_ranges = [
        (-1, 1),
        (0, 100),
        (-50, 50),
    ]

    # Select the methods
    methods = [
        GroupMeanBernoulli,
        GroupMeanLaplace,
        GroupMeanPiecewiseHalf,
        GroupMeanWaudbySmith,
        GroupMeanWaudbySmith16,
        GroupMeanWaudbySmith32,
    ]

    if i is not None:
        methods = [methods[i]]

    tasks = []
    for method in methods:
        for dataset in datasets:
            for input_range in input_ranges:
                for num_groups in num_groups_range:
                    print()
                    print(method.__name__)
                    print(dataset.__name__)
                    print(num_groups)

                    store_path = output_path / method.__name__ / dataset.__name__ / str(input_range) / str(num_groups)
                    store_path.mkdir(parents=True, exist_ok=True)

                    # Load/Generate the data
                    n_range_ = num_groups * n_range

                    dataset_size = max(n_range_)
                    dataset_ = dataset(num_groups, dataset_size, input_range, data_rng=data_rng.spawn(1)[0])

                    # Filter runs that are already done
                    eps_range_ = filter_eps_range(store_path, eps_range, len(n_range) * num_runs[0])
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
                                [n_range],
                                [dataset_],
                                eps_range_,
                                [method],
                                num_runs,
                                seeds,
                            )
                        )
                    )

    # Run the simulation
    simulation_loop(_run_mean_group, tasks, use_mpi=use_mpi)


class GroupMeanWaudbySmith16(GroupMeanWaudbySmith):  # noqa: D101
    def __init__(self, eps: float | tuple[float, float], input_range: np.array, rng: np.random.Generator = None):
        super().__init__(eps, input_range, rng, k=16)


class GroupMeanWaudbySmith32(GroupMeanWaudbySmith):  # noqa: D101
    def __init__(self, eps: float | tuple[float, float], input_range: np.array, rng: np.random.Generator = None):
        super().__init__(eps, input_range, rng, k=32)
