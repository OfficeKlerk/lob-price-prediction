"""Three models of increasing capacity, all sharing one input/output contract.

Every model maps a window of raw order book snapshots, shaped
(batch, window, features), to (batch, 3) class logits. Input normalisation is
baked into the model, so the exported ONNX graph accepts raw book values and
the C++ side needs no preprocessing of its own.
"""

from __future__ import annotations

import torch
from torch import nn

# Number of output classes: up, stationary, down.
N_CLASSES = 3


# Input preprocessing shared by all models.
class Normalizer(nn.Module):
    """Make a window of raw book snapshots stationary and unit-scaled.

    Raw prices are useless as features: they sit at a different level for every
    stock and day, while the signal lives in moves of a single tick. So every
    price is first expressed relative to the mid-price of the latest snapshot
    in the window, and then all features are z-scored with statistics fixed
    from the training set (see `fit_normalizer`).
    """

    # Store the price mask and the training statistics inside the model.
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        # Feature layout is [ask price, ask size, bid price, bid size] per level,
        # so prices sit at the even indices.
        is_price = torch.zeros_like(mean)
        is_price[0::2] = 1.0
        # Buffers are saved with the model and exported as ONNX constants.
        self.register_buffer("is_price", is_price)
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    # Centre prices on the latest mid, then z-score every feature.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mid-price of the latest snapshot: (best ask + best bid) / 2, shape (batch, 1, 1).
        mid = (x[:, -1:, 0:1] + x[:, -1:, 2:3]) / 2
        return (x - mid * self.is_price - self.mean) / self.std


# Compute the normalisation statistics from the training data.
def fit_normalizer(train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean and std of training snapshots, prices relative to their own mid.

    `train_x` has shape (n_events, n_features). The scale that matters is the
    spread between book levels (a few ticks), which is what these statistics
    capture.
    """
    mid = (train_x[:, 0:1] + train_x[:, 2:3]) / 2
    is_price = torch.zeros(train_x.shape[1])
    is_price[0::2] = 1.0
    centred = train_x - mid * is_price
    return centred.mean(dim=0), centred.std(dim=0)


# Model 1 of 3: the simplest baseline.
class LogisticRegression(nn.Module):
    """Baseline: a linear classifier on the most recent snapshot only."""

    # A single linear layer: 40 features in, 3 class logits out.
    def __init__(self, norm: Normalizer, n_features: int) -> None:
        super().__init__()
        self.norm = norm
        self.linear = nn.Linear(n_features, N_CLASSES)

    # Normalise the window, then classify its last snapshot.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x)[:, -1, :])


# Model 2 of 3: a fully connected network.
class MLP(nn.Module):
    """Fully connected network on the flattened window."""

    # Two hidden layers: window * features -> 256 -> 64 -> 3.
    def __init__(self, norm: Normalizer, window: int, n_features: int) -> None:
        super().__init__()
        self.norm = norm
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    # Normalise the window, flatten it and classify.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


# Model 3 of 3: a convolutional network over the time axis.
class CNN(nn.Module):
    """Small 1-D CNN: convolutions slide along time, book features are channels.

    The first kernel spans 12 events on purpose: it has to be longer than the
    prediction horizon (10 events) so that a single filter can compare the
    current price with its recent average. With short 5-event kernels the same
    network plateaus about 15 F1 points lower.
    """

    # Three convolutions extract temporal features; a small dense head classifies them.
    def __init__(self, norm: Normalizer, window: int, n_features: int) -> None:
        super().__init__()
        self.norm = norm
        # Convolutional feature extractor (strides shrink the time axis).
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=12),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        # Length of the time axis after the three convolutions (100 -> 89 -> 43 -> 20).
        out_len = window - 12 + 1
        out_len = (out_len - 5) // 2 + 1
        out_len = (out_len - 5) // 2 + 1
        # Classifier head on the flattened feature map.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * out_len, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    # Normalise, move features to the channel axis, convolve and classify.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).transpose(1, 2)  # (batch, window, features) -> (batch, features, window)
        return self.head(self.features(x))


# Names accepted by `build_model` and by the command-line tools.
MODEL_NAMES = ("logreg", "mlp", "cnn")


# Factory: build a model by its name.
def build_model(
    name: str, window: int, n_features: int, mean: torch.Tensor, std: torch.Tensor
) -> nn.Module:
    """Create a model by name. `mean` and `std` come from `fit_normalizer`."""
    norm = Normalizer(mean, std)
    if name == "logreg":
        return LogisticRegression(norm, n_features)
    if name == "mlp":
        return MLP(norm, window, n_features)
    if name == "cnn":
        return CNN(norm, window, n_features)
    raise ValueError(f"unknown model '{name}', expected one of {MODEL_NAMES}")
