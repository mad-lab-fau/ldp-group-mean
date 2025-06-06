"""Script to run the simulation experiments for the group comparison project."""

import argparse
import pathlib

from ldp_group_mean.simulation.processing.processing import group_results
from ldp_group_mean.simulation.simulation_group_mean import simulate_group_mean
from ldp_group_mean.simulation.simulation_group_mean_imbalance import simulate_group_mean_imbalance
from ldp_group_mean.simulation.simulation_group_mean_nprr import simulate_group_mean_nprr
from ldp_group_mean.simulation.simulation_group_mean_piecewise import simulate_group_mean_piecewise
from ldp_group_mean.simulation.simulation_group_mean_sizes import simulate_group_mean_sizes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("type", type=str, default="mean", help="The type of simulation to run")
    parser.add_argument("--output", type=str, default="../results", help="The output path")
    parser.add_argument("--mpi", action="store_true", help="Use MPI for simulations")
    parser.add_argument("--i", type=int, default=None, help="The index of the simulation")
    args = parser.parse_args()

    use_mpi = args.mpi

    if use_mpi:
        print("Using MPI for simulations")
    else:
        print("Using multiprocessing for simulations")

    output_path = pathlib.Path(args.output)

    sim_type = args.type

    if sim_type == "group_mean_nprr_fixed":
        simulate_group_mean_nprr(
            output_path / "group_mean_nprr_fixed", use_mpi=use_mpi, i=args.i, fixed_group_size=True
        )
        group_results("group_mean_nprr_fixed", use_mpi=use_mpi)

    if sim_type == "group_mean_piecewise_fixed":
        simulate_group_mean_piecewise(
            output_path / "group_mean_piecewise_fixed", use_mpi=use_mpi, i=args.i, fixed_group_size=True
        )
        group_results("group_mean_piecewise_fixed", use_mpi=use_mpi)

    if sim_type == "group_mean":
        # Run the group mean simulation
        simulate_group_mean(output_path / "group_mean", use_mpi=use_mpi, i=args.i)
        group_results("group_mean", use_mpi=use_mpi)

    if sim_type == "group_mean_sizes":
        # Run the group mean simulation with different group sizes
        simulate_group_mean_sizes(output_path / "group_mean_sizes", use_mpi=use_mpi, i=args.i)
        group_results("group_mean_sizes", use_mpi=use_mpi)

    if sim_type == "group_mean_imbalance":
        simulate_group_mean_imbalance(output_path / "group_mean_imbalance", use_mpi=use_mpi, i=args.i)
        group_results("group_mean_imbalance", use_mpi=use_mpi)
