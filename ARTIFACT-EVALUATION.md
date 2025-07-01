# Artifact Appendix

Paper title: **Estimating Group Means Under Local Differential Privacy**

Artifacts HotCRP Id: **#1**

Requested Badge: **Reproduced, Functional and Available**

## Description
This repository contains the code for the simulation experiments in the paper "Estimating Group Means Under Local Differential Privacy" accepted at PETS 2025.
It provides both the code to run the experiments and the code to generate the figures in the paper.

The code for the AQUILA data has been omitted as AQUILA is not freely available.
The code for MIMIC-IV is contained in this repository, but the data needs to be requested from PhysioNet (https://physionet.org/content/mimiciv/2.2/) if the experiments on MIMIC-IV are to be run.

The repository contains the following experiments:
- Experiment 1: Choice of the discretization parameter for Group NPRR
- Experiment 2: Privacy budget assignment for the Group Piecewise method
- Experiment 3: Mean estimation (all methods on synthetic data)
- Experiment 4: Impact of group imbalance on mean estimation
- Experiment 5: Mean estimation on MIMIC-IV data (requires access to MIMIC-IV)

### Security/Privacy Issues and Ethical Concerns
After installing the necessary packages, the code can be run without any internet connection and without any security or privacy issues.

## Basic Requirements

### Hardware Requirements
All experiments can be run on a standard laptop or desktop computer, however, running them on more powerful hardware will speed up the process.

### Software Requirements
The code has only been tested on Linux operating systems running Python 3.9 or 3.10, but should work on other OSs and
newer Python versions as well.

Make sure you have Python 3.9 (or 3.10) and Poetry installed on your system (see README.md for installation
instructions). We use Poetry to manage the dependencies of the project (see "Environment" section below).

On the artifact evaluation VM (we recommend the "compute VM"), you may run the following commands to install the
necessary software:

```
sudo apt install python3-pip python3-venv
pip install pipx
pipx ensurepath
pipx install poetry
```

To reproduce the plots from the paper, you need a LaTeX installation on your system. The following packages are
minimally required (example for Ubuntu / artifact evaluation VM):

```
sudo apt install texlive-latex-recommended cm-super texlive-science texlive-fonts-extra dvipng
```

If you are using a different Linux distribution, you need to install the equivalent packages for your distribution. 

### Estimated Time and Storage Consumption
These estimates are based on a rather old laptop with 4 cores (Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz) and 16 GB of RAM.
Runtimes will be significantly shorter on more powerful hardware (especially with more cores).
All simulations show progress bars, so the user can get a rough idea of how long the simulations will take.

- Experiment 1: ~8 hours
- Experiment 2: ~8 hours
- Experiment 3: ~12 hours
- Experiment 4: ~2 hours
- Experiment 5: ~2 hours

## Environment 

### Accessibility
The project code is available at https://github.com/mad-lab-fau/ldp-group-mean. It is licensed under the MIT license.

### Set up the environment
Make sure you have Python 3.9 and Poetry installed on your system (see README.md for installation instructions).
Clone the repository from the provided link.
Navigate to the repository root and run the following command to install all dependencies:

```bash
poetry install
```

This will create a new virtual environment in the directory .venv and install all required dependencies.

If you intend to run the simulation on an HPC cluster and make use of MPI support, you need to install the `mpi4py`
package:

```bash
poetry add mpi4py
```

### Testing the Environment
If the `poetry install` command ran without any errors, the environment should be set up correctly.

## Artifact Evaluation

### Main Results and Claims
Our paper discusses and compare four methods for estimating group means under local differential privacy.
The artifact allows the reviewer to reproduce the results and figures presented in the paper.
The results map to the paper as follows:

- Experiment 1: Choice of the discretization parameter for Group NPRR; section 5.2.1, figures 1, 2, (figures 7, 8 in the appendix)
- Experiment 2: Privacy budget assignment for the Group Piecewise method; section 5.2.2, tables 3, 4, 5, (tables 8, 9, 10 in the appendix)
- Experiment 3: Mean estimation (all methods on synthetic data); section 5.2.3, figures 3, 4 (figures 9, 10 in the appendix)
- Experiment 4: Impact of group imbalance on mean estimation; section 5.2.4, figure 5, (figure 11 in the appendix)
- Experiment 5: Mean estimation on MIMIC-IV data (requires access to MIMIC-IV); section 5.3.2, figures 6c, 6d (figures 14, 17, 20 in the appendix)

### Experiments 

All experiments can be run with the provided scripts and expect the working directory to be the `scripts` directory.
Switch to the `scripts` directory before running the commands below:
```bash
cd scripts
```

If you want to run the experiments on an HPC cluster, make sure to use the `run_simulation.py` script with the `--mpi` flag in your job script and ensure that the `mpi4py` package is installed in your environment (see "Set up the environment" section above).
Example: `poetry run python run_simulation.py --mpi group_mean_nprr_fixed`

#### Experiment 1: Choice of the discretization parameter for Group NPRR
Run the simulation with the following command:
```bash
poetry run python run_simulation.py group_mean_nprr_fixed
```

The results will be saved in the `results/group_mean_nprr_fixed` directory.
The simulation will also pre-process the results for plotting and save them in the `results_grouped` directory.
Pre-processing can also be run manually with the following command:
```bash
poetry run python preprocess_results.py
```

Re-create the figures from the paper with the following command:
```bash
cd plotting
poetry run python plot_mean_nprr.py
```

This creates four plots in the `scripts/plotting/plots_paper/group_mean_nprr` directory corresponding to the figures 1, 2, 7, and 8 in the paper.

#### Experiment 2: Privacy budget assignment for the Group Piecewise method
Run the simulation with the following command:
```bash
poetry run python run_simulation.py group_mean_piecewise_fixed
```

The results will be saved in the `results/group_mean_piecewise_fixed` directory.
The simulation will also pre-process the results for plotting and save them in the `results_grouped` directory.
Pre-processing can also be run manually with the following command:
```bash
poetry run python preprocess_results.py
```

Re-create the figures from the paper with the following command:
```bash
cd plotting
poetry run python plot_mean_piecewise.py
```

This creates the LaTeX source for tables 8, 9, and 10 in `scripts/plotting/plots_paper/group_mean_piecewise/mean_piecewise_summary_scaled_abs_error.tex`.
Tables 3, 4, and 5 in the paper are the same as tables 8, 9, and 10, but without the standard deviation (every second row omitted).

#### Experiment 3: Mean estimation (all methods on synthetic data)
Run the simulation with the following command:
```bash
poetry run python run_simulation.py group_mean_sizes
```

The results will be saved in the `results/group_mean_sizes` directory.
The simulation will also pre-process the results for plotting and save them in the `results_grouped` directory.
Pre-processing can also be run manually with the following command:
```bash
poetry run python preprocess_results.py
```

Re-create the figures from the paper with the following command:
```bash
cd plotting
poetry run python plot_mean_sizes.py
```

This creates figures in the `scripts/plotting/plots_paper/group_mean_sizes` directory corresponding to figures 3, 4, 9, and 10 in the paper.
It additionally creates figures for the different input ranges (included in the file name) which are not shown in the paper.

#### Experiment 4: Impact of group imbalance on mean estimation
Run the simulation with the following command:
```bash
poetry run python run_simulation.py group_mean_imbalance
```

The results will be saved in the `results/group_mean_imbalance` directory.
The simulation will also pre-process the results for plotting and save them in the `results_grouped` directory.
Pre-processing can also be run manually with the following command:
```bash
poetry run python preprocess_results.py
```

Re-create the figures from the paper with the following command:
```bash
cd plotting
poetry run python plot_mean_imbalance.py
```

This creates figures in the `scripts/plotting/plots_paper/group_mean_imbalance` directory corresponding to figures 5, 11 in the paper and some additional figures not shown in the paper.

#### Experiment 5: Mean estimation on MIMIC-IV data (requires access to MIMIC-IV)
To run this experiment, you first need to prepare the MIMIC-IV data.
Assuming you have access to the MIMIC-IV dataset (version 2.2) and imported it into a PostgreSQL database, you can run the SQL query in `scripts/sql/extract_adm.sql` to extract the relevant data.
Store the results in a CSV file named `mimic_adm.csv` in the `data/mimic` directory.

Run the simulation with the following command:
```bash
poetry run python run_simulation.py group_mean
```

The results will be saved in the `results/group_mean` directory.
Pre-process the results with the following command:
```bash
poetry run python preprocess_results.py
```

Re-create the figures from the paper with the following command:
```bash
cd plotting
poetry run python generate_group_mappings.py
poetry run python plot_mean_cases.py
```

This creates figures in the `scripts/plotting/plots_paper/group_mean_cases` directory including figures 6c, 6d, 14, 17, and 20 in the paper.

## Limitations
The real-world data experiments are limited to the MIMIC-IV dataset, as AQUILA is not freely available.
Furthermore, access to MIMIC-IV requires a PhysioNet account and approval for data access.
All simulations on synthetic data can be reproduced without any limitations.

## Notes on Reusability
The `ldp_group_mean` package can be reused for other experiments on group mean estimation under local differential privacy.
The four methods discussed in the paper are implemented in the package and can be used for other datasets as well.
Note that the library is optimized for performance and does not fully simulate the communication-aspects of a real-world implementation.