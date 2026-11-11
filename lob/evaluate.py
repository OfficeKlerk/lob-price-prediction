"""Inference over a whole split and the classification metrics we report."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from lob.data import CLASS_NAMES, Split, gather_windows, window_end_indices


# Run a model over a whole split and collect its predictions.
@torch.no_grad()
def predict(
    model: nn.Module, split: Split, window: int, device: torch.device, batch_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over every full window of a split.

    Returns (y_true, y_pred), both of shape (n_windows,).
    """
    model.eval()
    x = torch.from_numpy(split.x).to(device)
    end_idx = torch.from_numpy(window_end_indices(len(split), window)).to(device)

    # Predict batch by batch to keep memory use bounded.
    preds = []
    for start in range(0, len(end_idx), batch_size):
        batch = gather_windows(x, end_idx[start : start + batch_size], window)
        preds.append(model(batch).argmax(dim=1).cpu())

    # The first `window - 1` events have no full window, so they have no prediction.
    y_true = split.y[window - 1 :]
    return y_true, torch.cat(preds).numpy()


# Turn true and predicted labels into the metrics reported in the README.
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro F1, per-class F1 and the confusion matrix.

    Macro F1 is the headline metric: classes are imbalanced, so plain accuracy
    would reward a model that mostly predicts the majority class.
    """
    labels = list(range(len(CLASS_NAMES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_per_class": dict(zip(CLASS_NAMES, map(float, per_class), strict=True)),
        # Rows are true classes, columns are predicted classes.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
