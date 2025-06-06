"""Script for preprocessing the results of the simulation experiments."""
import argparse

from ldp_group_mean.simulation.processing.processing import group_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true", help="Use MPI for simulations")
    args = parser.parse_args()

    use_mpi = args.mpi

    if use_mpi:
        print("Using MPI for simulations")
    else:
        print("Using multiprocessing for simulations")

    group_results("group_mean", use_mpi=use_mpi)
    group_results("group_mean_sizes", use_mpi=use_mpi)
    group_results("group_mean_nprr_fixed", use_mpi=use_mpi)
    group_results("group_mean_imbalance", use_mpi=use_mpi)
