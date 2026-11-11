"""Train all models and write their test metrics to results/metrics.json.

Usage:
    python -m lob.train                 # train logreg, mlp and cnn
    python -m lob.train --models cnn    # train a single model
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lob.data import N_FEATURES, Split, gather_windows, load_splits, window_end_indices
from lob.evaluate import classification_metrics, predict
from lob.models import MODEL_NAMES, build_model, fit_normalizer


# All training hyperparameters in one place.
@dataclass(frozen=True)
class TrainConfig:
    horizon: int = 10  # predict the mid-price move over the next `horizon` events
    window: int = 100  # number of past snapshots the model sees
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 3  # stop after this many epochs without validation improvement
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


# Use a GPU when one is available (CUDA or Apple MPS), otherwise the CPU.
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Write the weights together with the config needed to rebuild the model.
def save_checkpoint(path: Path, model: nn.Module, name: str, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "config": asdict(cfg), "state_dict": model.state_dict()}, path)


# Inverse of `save_checkpoint`.
def load_checkpoint(path: Path) -> tuple[nn.Module, TrainConfig]:
    """Rebuild a trained model (on CPU, in eval mode) from a checkpoint file."""
    ckpt = torch.load(path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    # Normalisation statistics are restored from the state dict; zeros and ones
    # are just placeholders of the right shape.
    model = build_model(
        ckpt["name"], cfg.window, N_FEATURES, torch.zeros(N_FEATURES), torch.ones(N_FEATURES)
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), cfg


# The training loop.
def train_model(
    name: str, splits: dict[str, Split], cfg: TrainConfig, device: torch.device
) -> tuple[nn.Module, list[dict]]:
    """Train one model with early stopping on validation macro F1.

    Returns the model restored to its best validation epoch, and the
    per-epoch training history.
    """
    torch.manual_seed(cfg.seed)
    train = splits["train"]

    # Normalisation statistics come from the training days only.
    mean, std = fit_normalizer(torch.from_numpy(train.x))
    model = build_model(name, cfg.window, N_FEATURES, mean, std).to(device)

    # Optimiser and loss.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # The whole dataset fits in memory, so we keep it on the device and slice
    # windows out of it on the fly instead of materialising every window.
    x = torch.from_numpy(train.x).to(device)
    y = torch.from_numpy(train.y).to(device)
    end_idx = torch.from_numpy(window_end_indices(len(train), cfg.window)).to(device)

    # State for early stopping.
    history: list[dict] = []
    best_f1, best_state, epochs_without_improvement = -1.0, None, 0

    for epoch in range(1, cfg.max_epochs + 1):
        # One pass over the training windows, in a fresh random order.
        model.train()
        started = time.perf_counter()
        shuffled = end_idx[torch.randperm(len(end_idx), device=device)]
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(shuffled), cfg.batch_size):
            batch_idx = shuffled[start : start + cfg.batch_size]
            logits = model(gather_windows(x, batch_idx, cfg.window))
            loss = loss_fn(logits, y[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Evaluate on the validation day and log the epoch.
        val_f1 = classification_metrics(*predict(model, splits["val"], cfg.window, device))[
            "macro_f1"
        ]
        history.append(
            {"epoch": epoch, "train_loss": total_loss / n_batches, "val_macro_f1": val_f1}
        )
        print(
            f"[{name}] epoch {epoch:2d}  loss {total_loss / n_batches:.4f}  "
            f"val macro-F1 {val_f1:.4f}  ({time.perf_counter() - started:.0f}s)"
        )

        # Keep the best weights; stop when validation stops improving.
        if val_f1 > best_f1:
            best_f1, epochs_without_improvement = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    # Roll back to the best validation epoch.
    model.load_state_dict(best_state)
    return model, history


# Trivial reference point: a "model" that ignores its input.
def majority_baseline(splits: dict[str, Split], window: int) -> dict:
    """Metrics of always predicting the most common training class."""
    majority_class = int(np.bincount(splits["train"].y).argmax())
    y_true = splits["test"].y[window - 1 :]
    return classification_metrics(y_true, np.full_like(y_true, majority_class))


# Command-line entry point.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Load the data and pick the device.
    cfg = TrainConfig(horizon=args.horizon, max_epochs=args.max_epochs)
    device = pick_device()
    splits = load_splits(args.data_dir, cfg.horizon)
    print(f"device: {device}; events: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # Merge into the existing file so that retraining one model keeps the others.
    metrics_path = args.results_dir / "metrics.json"
    results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    results["config"] = asdict(cfg)
    results.setdefault("models", {})
    results["models"]["majority"] = {"test": majority_baseline(splits, cfg.window)}

    # Train each requested model, save it and score it on the test days.
    for name in args.models:
        model, history = train_model(name, splits, cfg, device)
        save_checkpoint(args.out_dir / f"{name}.pt", model, name, cfg)

        test_metrics = classification_metrics(*predict(model, splits["test"], cfg.window, device))
        results["models"][name] = {
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "history": history,
            "test": test_metrics,
        }
        print(
            f"[{name}] test macro-F1 {test_metrics['macro_f1']:.4f}  "
            f"accuracy {test_metrics['accuracy']:.4f}"
        )

    # Persist the metrics.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
