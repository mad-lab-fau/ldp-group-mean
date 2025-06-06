"""Utility functions for the simulation."""

import numpy as np


def calc_mean_counts(data: np.array, num_groups: int) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Calculate the means, standard deviations and counts for each group.

    Args:
        data: The data array of shape (n, 2) where the first column is the group and the second column is the value.
        num_groups: The number of groups.

    Return:
        means: The means of the groups.
        stds: The standard deviations of the groups.
        counts: The counts of the groups
    """
    sums = np.zeros(num_groups)
    means = np.zeros(num_groups)
    stds = np.zeros(num_groups)
    counts = np.zeros(num_groups)

    for g in range(num_groups):
        mask = data[:, 0] == g
        counts[g] = np.sum(mask)
        if counts[g] == 0:
            continue
        sums[g] = np.sum(data[mask, 1])
        means[g] = sums[g] / counts[g]
        stds[g] = np.std(data[mask, 1])

    return sums, means, stds, counts


def _calc_group_errors(true_group, est_group, num_groups):
    errors_per_group = np.zeros(num_groups)

    for i in range(num_groups):
        mask = true_group == i
        errors_per_group[i] = np.sum(true_group[mask] != est_group[mask])

    total_errors = np.sum(errors_per_group)

    return errors_per_group, total_errors
