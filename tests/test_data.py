import numpy as np
import pytest
import torch

from lob.data import (
    HORIZONS,
    N_FEATURES,
    gather_windows,
    split_features_and_labels,
    window_end_indices,
)


# Test fixture: a synthetic stand-in for one raw dataset file.
def make_raw(n_events: int) -> np.ndarray:
    """A fake raw file: 149 rows, labels cycling through 1, 2, 3."""
    raw = np.random.default_rng(0).random((149, n_events), dtype=np.float32)
    raw[-len(HORIZONS) :] = np.tile([1, 2, 3], n_events // 3 + 1)[:n_events]
    return raw


# Features and labels are cut out of the raw matrix correctly.
def test_split_features_and_labels_shapes_and_label_mapping():
    raw = make_raw(30)
    split = split_features_and_labels(raw, horizon=10)

    assert split.x.shape == (30, N_FEATURES)
    assert split.y.shape == (30,)
    # Labels 1, 2, 3 in the file become class indices 0, 1, 2.
    assert split.y[:3].tolist() == [0, 1, 2]
    # Row i of x is column i of the raw file.
    np.testing.assert_array_equal(split.x[5], raw[:N_FEATURES, 5])


# Each horizon reads its own label row.
def test_split_picks_the_requested_horizon_row():
    raw = make_raw(9)
    raw[-1] = 3  # horizon 100 is the last label row
    assert set(split_features_and_labels(raw, horizon=100).y) == {2}


# Horizons that are not in the dataset are an error.
def test_split_rejects_unknown_horizon():
    with pytest.raises(ValueError):
        split_features_and_labels(make_raw(9), horizon=7)


# Only events with enough history get a window.
def test_window_end_indices_cover_only_full_windows():
    assert window_end_indices(n_events=10, window=4).tolist() == [3, 4, 5, 6, 7, 8, 9]


# The key no-leakage property of the input pipeline.
def test_gather_windows_never_looks_into_the_future():
    # Feature value equals the event index, so the window content is easy to read.
    x = torch.arange(20, dtype=torch.float32)[:, None].repeat(1, 2)
    windows = gather_windows(x, torch.tensor([4, 19]), window=5)

    assert windows.shape == (2, 5, 2)
    # A window ending at event t holds exactly events t-4 .. t.
    assert windows[0, :, 0].tolist() == [0, 1, 2, 3, 4]
    assert windows[1, :, 0].tolist() == [15, 16, 17, 18, 19]
