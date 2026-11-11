"""FI-2010 limit order book dataset: download, parsing and chronological splits.

FI-2010 holds ten trading days of level-10 order book snapshots for five
Nasdaq Nordic stocks. We use the "NoAuction / DecPre" variant redistributed
with the DeepLOB paper, where each text file is a matrix of shape
(149, n_events): one column per order book event, and

    rows   0..39   raw book: [ask price, ask size, bid price, bid size] x 10 levels
    rows  40..143  hand-crafted features (unused here)
    rows 144..148  mid-price direction labels for horizons k = 10, 20, 30, 50, 100

The split is strictly chronological, so no model ever sees the future:

    train       days 1-7   (Train_..._CF_7)
    validation  day  8     (Test_..._CF_7)
    test        days 9-10  (Test_..._CF_8 + Test_..._CF_9)
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Where the dataset archive is downloaded from (the DeepLOB authors' copy).
DATA_URL = (
    "https://raw.githubusercontent.com/zcakhaa/"
    "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip"
)

# Shape and meaning of the data.
N_FEATURES = 40  # 10 levels x (ask price, ask size, bid price, bid size)
HORIZONS = (10, 20, 30, 50, 100)  # prediction horizons, in order book events
CLASS_NAMES = ("up", "stationary", "down")

# File naming scheme and which files make up each chronological split.
_FILE_TEMPLATE = "{kind}_Dst_NoAuction_DecPre_CF_{fold}.txt"
_SPLIT_FILES = {
    "train": [("Train", 7)],
    "val": [("Test", 7)],
    "test": [("Test", 8), ("Test", 9)],
}


# Container for one split (train, validation or test).
@dataclass(frozen=True)
class Split:
    """One contiguous, time-ordered slice of the dataset."""

    x: np.ndarray  # (n_events, N_FEATURES) float32, raw order book snapshots
    y: np.ndarray  # (n_events,) int64, class index into CLASS_NAMES

    # Number of order book events in the split.
    def __len__(self) -> int:
        return len(self.y)


# Step 1: fetch the raw text files.
def download(data_dir: Path) -> None:
    """Download and unpack the dataset into `data_dir` (no-op if already there)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    first_file = data_dir / _FILE_TEMPLATE.format(kind="Train", fold=7)
    if first_file.exists():
        return

    # Download the archive only if it is not on disk yet, then unpack it.
    archive = data_dir / "data.zip"
    if not archive.exists():
        print(f"Downloading FI-2010 (~56 MB) to {archive} ...")
        urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)


# Step 2: parse one text file into a matrix (with an on-disk cache).
def _load_file(data_dir: Path, kind: str, fold: int) -> np.ndarray:
    """Load one raw text file as a (149, n_events) matrix, caching it as .npy.

    Parsing ~38M text numbers takes a while, so the parsed matrix is cached
    next to the source file and reused on later runs.
    """
    txt_path = data_dir / _FILE_TEMPLATE.format(kind=kind, fold=fold)
    npy_path = txt_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(npy_path)

    # Cache miss: parse the text file once and store the result.
    raw = np.loadtxt(txt_path, dtype=np.float32)
    np.save(npy_path, raw)
    return raw


# Step 3: separate the 40 book features from the label row of one horizon.
def split_features_and_labels(raw: np.ndarray, horizon: int) -> Split:
    """Turn a raw (149, n_events) matrix into features and labels for one horizon."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")

    x = np.ascontiguousarray(raw[:N_FEATURES].T, dtype=np.float32)
    label_row = raw[-len(HORIZONS) + HORIZONS.index(horizon)]
    # The files encode labels as 1 = up, 2 = stationary, 3 = down.
    y = label_row.astype(np.int64) - 1
    return Split(x=x, y=y)


# Step 4: assemble the three chronological splits from their files.
def load_splits(data_dir: Path, horizon: int) -> dict[str, Split]:
    """Return the chronological train / val / test splits for one horizon."""
    download(data_dir)
    splits = {}
    for name, files in _SPLIT_FILES.items():
        parts = [
            split_features_and_labels(_load_file(data_dir, kind, fold), horizon)
            for kind, fold in files
        ]
        splits[name] = Split(
            x=np.concatenate([p.x for p in parts]),
            y=np.concatenate([p.y for p in parts]),
        )
    return splits


# Which events can serve as the last event of a full-length window.
def window_end_indices(n_events: int, window: int) -> np.ndarray:
    """Indices of events that have a full `window` of history behind them.

    Note: each file concatenates the five stocks, so a handful of windows
    (roughly 1% of validation and test windows, measured) straddle two
    stocks. We keep them for simplicity; they only add a little label noise.
    """
    return np.arange(window - 1, n_events)


# Slice a batch of windows out of the event matrix (no copy of the dataset).
def gather_windows(x: torch.Tensor, end_idx: torch.Tensor, window: int) -> torch.Tensor:
    """Build a (batch, window, features) tensor of windows ending at `end_idx`.

    A window ending at event t covers events [t - window + 1, t] and is paired
    with the label of event t, which describes the price move *after* t.
    The model input therefore never contains information from the future.
    """
    offsets = torch.arange(-window + 1, 1, device=x.device)
    return x[end_idx[:, None] + offsets]
