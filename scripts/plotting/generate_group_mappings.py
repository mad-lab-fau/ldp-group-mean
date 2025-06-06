import pickle
from pathlib import Path

import numpy as np

from ldp_group_mean.simulation.data.mimic import MIMICLoader

if __name__ == "__main__":
    datasets = []

    # AQUILA omitted from the published code

    mimic_path = Path("../../data/mimic")
    if mimic_path.exists():
        datasets.extend(MIMICLoader(data_path=mimic_path).get_all_numeric_datasets().values())

    mappings = {}
    for dataset in datasets:
        name = dataset.name

        # Get group mapping
        gm = dataset.group_mapping
        mappings[name] = gm

        # Get number of rows for each group
        arr = dataset.data
        unique, counts = np.unique(arr[:, 0], return_counts=True)
        group_counts_ids = dict(zip(unique, counts))
        group_counts = {gm[k]: v for k, v in zip(unique, counts)}

        print("Dataset:", name)
        print(", ".join(f"{int(k)!s}: \\num{{{v}}}" for k, v in group_counts_ids.items()))
        print(", ".join(f"{k}: \\num{{{v}}}" for k, v in group_counts.items()))
        print()

        # Print group mapping
        print("Group mapping:")
        print(", ".join(f"{v!s} ({k})" for k, v in gm.items()))
        print()

    print(mappings)

    # Store as pickle
    output_file = Path("../../results_grouped/group_mappings.pkl")
    with output_file.open("wb") as f:
        pickle.dump(mappings, f)
