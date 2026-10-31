import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def defect_score(logits):
    probabilities = torch.softmax(logits, dim=1)
    defect_probabilities = 1 - probabilities[:, 0] # class 0 is normal, the rest are defect
    return defect_probabilities

def rates_at(scores: torch.Tensor, is_defect: torch.Tensor, tau: float):
    """
    returns (miss rate, false alarm rate) at threshold tau
    """
    # Calculate predicted defect labels based on the threshold tau
    predicted_defect = scores >= tau
    
    # Calculate misses
    misses = torch.sum((predicted_defect == 0) & (is_defect == 1)).item()
    total_defects = torch.sum(is_defect == 1).item()
    miss_rate = misses / total_defects if total_defects > 0 else 0.0

    # Calculate false alarms
    false_alarms = torch.sum((predicted_defect == 1) & (is_defect == 0)).item()
    total_normals = torch.sum(is_defect == 0).item()
    false_alarm_rate = false_alarms / total_normals if total_normals > 0 else 0.0

    return miss_rate, false_alarm_rate

def threshold_sweep(scores: torch.Tensor, is_defect: torch.Tensor):
    """
    Evaluate every unique score as the threshold tau, largest first.

    The rates only change where tau crosses a score, so the unique scores are exactly the distinct
    operating points; a fixed grid would skip some whenever two scores sit closer together than
    its spacing.

    Returns a list of (threshold, miss_rate, false_alarm_rate) tuples.
    """
    thresholds = sorted(set(scores.tolist()), reverse=True)
    results = []

    for tau in thresholds:
        miss_rate, false_alarm_rate = rates_at(scores, is_defect, tau)
        results.append((tau, miss_rate, false_alarm_rate))

    return results

def plot_roc(sweep, path):
    """
    Draw the ROC curve: x = false-alarm rate, y = 1 - miss rate (the fraction of defects caught),
    with the diagonal as the "guessing" reference.
    """
    thresholds, miss_rates, false_alarm_rates = zip(*sweep)
    # Above the highest score nothing is flagged: the curve starts at (0, 0).
    x = [0.0, *false_alarm_rates]
    y = [0.0, *(1 - m for m in miss_rates)]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', markersize=3, label='model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='guessing')
    plt.title('ROC curve')
    plt.xlabel('False-alarm rate (good joints flagged)')
    plt.ylabel('1 - miss rate (defects caught)')
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(path, dpi=120)
    plt.close()