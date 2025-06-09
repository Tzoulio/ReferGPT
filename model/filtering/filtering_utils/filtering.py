from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import torch


def dynamic_hierarchical_grouping(values, distance_threshold, linkage_method='single'):
    """
    Group values using hierarchical clustering. All parameters are config-driven.
    Args:
        values: list of float/int/torch.Tensor
        distance_threshold: float, from config
        linkage_method: str, from config or default
    Returns:
        grouped_values: list of groups
        group_with_highest_mean: group with highest mean
        group_with_lowest_mean: group with lowest mean
    """
    scores_filtered = []
    for score in values:
        if score is None:
            continue
        elif isinstance(score, (float, int)):
            if score < 200:
                scores_filtered.append(float(score))
        elif isinstance(score, (torch.Tensor, np.floating, np.ndarray)):
            if float(score) < 200:
                scores_filtered.append(float(score))
        else:
            raise ValueError(f"Unsupported type: {type(score)}")
    if not scores_filtered:
        return []
    values = np.sort(scores_filtered)
    Z = linkage(values.reshape(-1, 1), method=linkage_method)
    labels = fcluster(Z, t=distance_threshold, criterion='distance')
    grouped_values = []
    for label in np.unique(labels):
        grouped_values.append(list(values[labels == label]))
    group_with_highest_mean = max(grouped_values, key=lambda group: np.mean(group))
    group_with_lowest_mean = min(grouped_values, key=lambda group: np.mean(group))
    return grouped_values, group_with_highest_mean, group_with_lowest_mean

def remove_outliers_and_scale(values, max_valid=9):
    """
    Remove outliers and scale values to [0, 100]. All parameters are config-driven.
    Args:
        values: list of float/int
        max_valid: float, from config or default
    Returns:
        scaled_values: list
        min_val: float
        max_val: float
    """
    numeric_values = np.array([v for v in values if v is not None and v < max_valid])
    min_val = min(numeric_values)
    max_val = max(numeric_values)
    if min_val == max_val:
        return [100 if v is not None else None for v in numeric_values], min_val, max_val
    scaled_values = [
        ((v - min_val) / (max_val - min_val) * 100) if v is not None else None
        for v in numeric_values
    ]
    return scaled_values, min_val, max_val

def scale_back(scaled_value, original_min, original_max):
    """
    Scale a value from [0, 100] back to original range.
    """
    if scaled_value is None:
        return None
    original_value = (scaled_value / 100) * (original_max - original_min) + original_min
    return original_value