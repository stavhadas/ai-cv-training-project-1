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

def threshold_sweep(scores: torch.Tensor, is_defect: torch.Tensor, num_thresholds: int = 100):
    """
    Sweep thresholds from 0 to 1 and compute miss rate and false alarm rate at each threshold.
    Returns a list of (threshold, miss_rate, false_alarm_rate) tuples.
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    results = []
    
    for tau in thresholds:
        miss_rate, false_alarm_rate = rates_at(scores, is_defect, tau)
        results.append((tau, miss_rate, false_alarm_rate))
    
    return results

def plot_roc(sweep, path): 

    thresholds, miss_rates, false_alarm_rates = zip(*sweep)
    
    plt.figure(figsize=(8, 6))
    plt.plot(false_alarm_rates, miss_rates, marker='o')
    plt.title('ROC Curve')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Miss Rate')
    plt.grid()
    plt.savefig(path)
    plt.close()