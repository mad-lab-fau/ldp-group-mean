"""General methods for processing simulation data."""

import multiprocessing
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ldp_group_mean.simulation.processing.process_mean import group_results_mean

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    MPIPoolExecutor = None


def read_pickle(filename):
    """
    Read a pickle file and return a DataFrame.

    Args:
        filename: The path to the pickle file.

    Returns: The DataFrame containing the data from the pickle file.
    """
    try:
        df = pd.read_pickle(filename)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading {filename}: {e}")
        return pd.DataFrame()

    if type(df) is list:
        df = pd.DataFrame(df, columns=df[0].keys())

    return df


def _process(args):
    method_dir, output_dir, group_method = args
    print(method_dir)
    # join all .pkl files into one dataframe
    files = list(method_dir.glob("**/*.pkl"))

    if len(files) == 0:
        print(f"No files found in {method_dir}. Skipping.")
        return

    df = pd.concat(map(read_pickle, files))
    df = df.reset_index(drop=True)
    group_method(df, output_dir)


def group_results(method_type: str, use_mpi: bool = False, show_progress: bool = True):
    """
    Group the results from the simulation experiments.

    Args:
        method_type: The type of simulation to group.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        show_progress: Whether to show a progress bar or not. Default is True. No effect if use_mpi is True.
    """
    base_path = Path.cwd()

    # Check if current working directory is "scripts"
    if base_path.name == "scripts":
        # Change to the parent directory
        base_path = base_path.parent

    output_dir = base_path / "results_grouped" / method_type

    # delete output dir
    # shutil.rmtree(output_dir, ignore_errors=True)

    # Input path
    path = base_path / "results" / method_type

    # Check if the input path exists
    if not path.exists():
        print(f"Path {path} does not exist. Skipping.")
        return

    # Iterate over all directories
    directories = [x for x in path.iterdir() if x.is_dir()]

    if method_type in (
        "group_mean",
        "group_mean_sizes",
        "piecewise_split",
        "group_mean_nprr",
        "group_mean_nprr_fixed",
        "group_mean_imbalance",
        "group_mean_piecewise",
        "group_mean_piecewise_fixed",
    ):
        group_method = group_results_mean
    else:
        raise ValueError(f"Unknown method type: {method_type}")  # noqa: TRY003

    pool_ = MPIPoolExecutor if use_mpi and MPIPoolExecutor is not None else multiprocessing.Pool
    max_workers = None if use_mpi else os.cpu_count()

    tasks = [(x, output_dir, group_method) for x in directories]

    with pool_(max_workers) as pool:
        if use_mpi:
            mapping = pool.map(_process, tasks, chunksize=1, unordered=True)
        elif show_progress:
            mapping = tqdm(pool.imap_unordered(_process, tasks, chunksize=1), total=len(tasks))
        else:
            mapping = pool.imap_unordered(_process, tasks, chunksize=1)

        for _ in mapping:
            pass
