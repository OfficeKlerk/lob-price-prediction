import struct

import numpy as np
import pytest
import torch

from lob.data import N_FEATURES
from lob.evaluate import classification_metrics
from lob.export import SAMPLES_MAGIC, export_fp32, run_onnx, write_samples
from lob.models import MODEL_NAMES, build_model

# Window length used by all tests in this file.
WINDOW = 100


# The sample file has exactly the byte layout the C++ loader expects.
def test_write_samples_layout(tmp_path):
    windows = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    labels = np.array([2, 0])
    path = tmp_path / "samples.bin"
    write_samples(path, windows, labels)

    blob = path.read_bytes()
    assert struct.unpack("<4I", blob[:16]) == (SAMPLES_MAGIC, 2, 3, 4)
    np.testing.assert_array_equal(np.frombuffer(blob[16 : 16 + 96], "<f4"), windows.ravel())
    np.testing.assert_array_equal(np.frombuffer(blob[16 + 96 :], "<i4"), labels)


# The exported ONNX graph computes the same function as the PyTorch model.
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_onnx_export_matches_pytorch(name, tmp_path):
    torch.manual_seed(0)
    model = build_model(name, WINDOW, N_FEATURES, torch.zeros(N_FEATURES), torch.ones(N_FEATURES))
    model.eval()
    path = tmp_path / f"{name}.onnx"
    export_fp32(model, WINDOW, path)

    # Batch of 3 also exercises the dynamic batch dimension (exported with batch 1).
    x = torch.rand(3, WINDOW, N_FEATURES)
    with torch.no_grad():
        expected = model(x).numpy()
    np.testing.assert_allclose(run_onnx(path, x.numpy()), expected, atol=1e-4)


# Metrics are checked against values worked out by hand.
def test_classification_metrics_on_a_known_case():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # Per-class F1: up 0.5, stationary 0.8, down 2/3.
    assert metrics["f1_per_class"]["stationary"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
