import os.path as osp
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mmengine
import numpy as np
import torch
from alive_progress import alive_it
from transformers.trainer_utils import PredictionOutput

from ..utils import setup_logger


def plot_predictions(preds: PredictionOutput, plots_cfg: Dict[str, Any], work_dir: str) -> None:
    logger = setup_logger('ib-edl')
    save_dir = osp.join(work_dir, 'plots')
    correct_samples_dir = osp.join(save_dir, 'correct')
    wrong_samples_dir = osp.join(save_dir, 'wrong')
    mmengine.mkdir_or_exist(save_dir)
    mmengine.mkdir_or_exist(correct_samples_dir)
    mmengine.mkdir_or_exist(wrong_samples_dir)

    if isinstance(preds.predictions, tuple):
        # PredictionOutput.predictions can be a tuple of (logits, uncertainties)
        logits, uncertainties = preds.predictions
    else:
        logits, uncertainties = preds.predictions, None

    # sanity check. For probabilities, we don't need softmax.
    min_val, max_val = np.min(logits), np.max(logits)
    if min_val >= 0.0 and max_val <= 1.0:
        logger.warning(
            'logits are in the range [0, 1]. It seems that the predictions are already probabilities. '
            'Check your code to ensure that the predictions are logits.')

    pred_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    labels = preds.label_ids

    correct_mask = np.argmax(pred_probs, axis=1) == labels
    incorrect_mask = ~correct_mask
    indices = np.arange(len(labels))

    correct_probs = pred_probs[correct_mask]
    correct_labels = labels[correct_mask]
    correct_indices = indices[correct_mask]
    correct_uncertainties = uncertainties[correct_mask] if uncertainties is not None else None
    plot_and_save_batches(
        correct_probs,
        correct_labels,
        correct_uncertainties,
        correct_indices,
        plots_cfg['plot_grid_size'],
        correct_samples_dir)

    incorrect_probs = pred_probs[incorrect_mask]
    incorrect_labels = labels[incorrect_mask]
    incorrect_indices = indices[incorrect_mask]
    incorrect_uncertainties = uncertainties[incorrect_mask] if uncertainties is not None else None
    plot_and_save_batches(
        incorrect_probs,
        incorrect_labels,
        incorrect_uncertainties,
        incorrect_indices,
        plots_cfg['plot_grid_size'],
        wrong_samples_dir)

    # create a violin plot for the uncertainties of the correct and incorrect samples, placing in the same figure
    fig, ax = plt.subplots(figsize=(8, 6))
    if uncertainties is not None:
        ax.violinplot([correct_uncertainties, incorrect_uncertainties], showmeans=True, showextrema=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.set_ylabel('Uncertainty')
        plt.tight_layout()
        plt.savefig(osp.join(save_dir, 'uncertainties.png'))
        plt.close(fig)

    logger.info(f'Plots saved to {save_dir}')


def plot_and_save_batches(
    pred_probs: np.ndarray,
    gt_labels: np.ndarray,
    uncertainties: Optional[np.ndarray],
    sample_indices: np.ndarray,
    plot_grid_size: Tuple[int, int],
    output_dir: str,
) -> None:
    correct_color = '#2364aa'
    incorrect_color = '#ea7317'

    num_samples = pred_probs.shape[0]
    batch_size = plot_grid_size[0] * plot_grid_size[1]
    num_batches = int(np.ceil(num_samples / batch_size))

    bar_tilte = 'Plotting predictions:'
    for batch_idx in alive_it(range(num_batches), total=num_batches, title=bar_tilte):
        fig, axs = plt.subplots(plot_grid_size[0], plot_grid_size[1], figsize=(15, 15), sharey=True, sharex=True)
        axs = axs.flatten()

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        i = 0
        for i, idx in enumerate(range(start_idx, end_idx)):
            ax = axs[i]
            probabilities = pred_probs[idx]
            true_label = gt_labels[idx]
            sample_index = sample_indices[idx]

            colors = [incorrect_color if j != true_label else correct_color for j in range(len(probabilities))]
            ax.bar(range(len(probabilities)), probabilities, color=colors)
            title = f'ID: {sample_index}'
            if uncertainties is not None:
                title += f', U: {uncertainties[idx]:.2f}'
            ax.set_title(title)
            ax.set_ylim(0, 1)

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.savefig(osp.join(output_dir, f'batch_{batch_idx + 1}.png'))
        plt.close(fig)


def plot_calibration_curve_and_ece(
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        figsize: Tuple[float, float] = (5, 3),
        save_path: Optional[str] = None,
        show: bool = True) -> None:
    assert probs.shape[0] == labels.shape[0], 'Number of samples in probs and labels must be the same.'
    assert save_path is not None or show, 'Either save_path or show must be specified.'

    num_samples = probs.shape[0]
    # Apply softmax if necessary (torchmetrics does this)
    if np.any(probs < 0) or np.any(probs > 1):
        # Assuming logits, apply softmax
        exp_probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))  # For numerical stability
        probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
    # Get the maximum predicted probabilities (top-1 confidences)
    confidences = np.max(probs, axis=1)
    # Get the predicted labels
    pred_labels = np.argmax(probs, axis=1)
    # Compute correctness of predictions
    correct = (pred_labels == labels).astype(int)
    # Bin the predicted confidences
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    # Adjust binning to match torchmetrics
    bin_indices = np.digitize(confidences, bins, right=False) - 1
    bin_indices[bin_indices == num_bins] = num_bins - 1  # Handle edge case where confidence == 1.0

    # Initialize arrays to store bin accuracies and confidences
    bin_total = np.zeros(num_bins)
    bin_correct = np.zeros(num_bins)
    bin_confidence = np.zeros(num_bins)

    for b in range(num_bins):
        bin_mask = (bin_indices == b)
        bin_total[b] = np.sum(bin_mask)
        if bin_total[b] > 0:
            bin_correct[b] = np.sum(correct[bin_mask])
            bin_confidence[b] = np.sum(confidences[bin_mask])

    # Avoid division by zero
    accuracy = np.zeros(num_bins)
    avg_confidence = np.zeros(num_bins)
    nonzero = bin_total > 0
    accuracy[nonzero] = bin_correct[nonzero] / bin_total[nonzero]
    avg_confidence[nonzero] = bin_confidence[nonzero] / bin_total[nonzero]

    # Compute ECE using L1 norm
    bin_weights = bin_total / num_samples
    ece = np.sum(bin_weights[nonzero] * np.abs(avg_confidence[nonzero] - accuracy[nonzero]))

    # Prepare the bar plot
    bin_width = bins[1] - bins[0]
    bar_width = bin_width  # Full width to eliminate gaps
    bin_left_edges = bins[:-1]  # Left edges of the bins

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(
        bin_left_edges,
        accuracy,
        width=bar_width,
        edgecolor='black',
        color='#40916c',
        align='edge',
    )

    ax.plot([0, 1], [0, 1], linestyle='--', color='#ee964b')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Calibration Curve, ECE: {ece:.4f}')
    ax.set_xticks(np.linspace(0, 1, num_bins + 1), minor=True)
    ax.set_xticks(np.linspace(0, 1, 6), minor=False)
    ax.grid(True, which='minor', axis='x', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, pad_inches=0)
    if show:
        plt.show()
