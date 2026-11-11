"""Export trained models to ONNX (float32 and int8) for the C++ benchmark.

For every checkpoint in artifacts/ this script:
  1. exports a float32 ONNX graph and checks it against PyTorch;
  2. produces an int8 version with post-training static quantisation;
  3. measures the test macro F1 of both, so the accuracy cost of int8 is known.

It also writes artifacts/samples.bin, a small file of real test windows that
the C++ benchmark replays.

Usage:
    python -m lob.export
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from lob.data import N_FEATURES, Split, load_splits, window_end_indices
from lob.evaluate import classification_metrics
from lob.models import MODEL_NAMES
from lob.train import load_checkpoint

# Tensor names in the ONNX graph; the C++ benchmark uses the same names.
INPUT_NAME = "book_window"
OUTPUT_NAME = "logits"

# Benchmark sample file and int8 calibration settings.
SAMPLES_MAGIC = 0x53424F4C  # the bytes "LOBS" when written little-endian
N_BENCH_SAMPLES = 2000
N_CALIBRATION_WINDOWS = 1000
CALIBRATION_PERCENTILE = 99.9


# NumPy counterpart of `lob.data.gather_windows`.
def make_windows(split: Split, end_idx: np.ndarray, window: int) -> np.ndarray:
    """Materialise (len(end_idx), window, features) float32 windows from a split."""
    offsets = np.arange(-window + 1, 1)
    return split.x[end_idx[:, None] + offsets]


# PyTorch -> ONNX, float32.
def export_fp32(model: torch.nn.Module, window: int, path: Path) -> None:
    """Export the model to ONNX with a dynamic batch dimension."""
    example = torch.zeros(1, window, N_FEATURES)
    torch.onnx.export(
        model,
        (example,),
        path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        dynamo=True,
        verbose=False,
        external_data=False,  # keep weights inside the .onnx file
    )


# Adapter that hands calibration data to ONNX Runtime's quantiser.
class _WindowCalibrationReader(CalibrationDataReader):
    """Feeds calibration windows to the quantiser, one batch at a time."""

    # Split the calibration windows into batches up front.
    def __init__(self, windows: np.ndarray, batch_size: int = 100) -> None:
        self._batches = iter(
            {INPUT_NAME: windows[i : i + batch_size]} for i in range(0, len(windows), batch_size)
        )

    # Return the next batch, or None when the data is exhausted.
    def get_next(self) -> dict | None:
        return next(self._batches, None)


# float32 ONNX -> int8 ONNX.
def quantize_int8(fp32_path: Path, int8_path: Path, calibration_windows: np.ndarray) -> None:
    """Post-training static quantisation of weights and activations to int8.

    Only Conv / Gemm / MatMul are quantised. The input normalisation stays in
    float32 on purpose: raw prices need far more than 8 bits (a one-tick move
    is ~0.04% of the price), so the int8 boundary must come after the prices
    have been centred on the mid.

    Activation ranges are calibrated at the 99.9th percentile instead of
    min/max. Book sizes are heavy-tailed, and letting a few outliers set the
    range leaves too few of the 256 levels for typical values (min/max
    calibration dropped the MLP from 0.76 to 0.40 macro F1). The percentile was
    picked by comparing 99.0-99.99 on the validation day in an exploratory run
    that is not part of this repository; here it is simply a constant.
    """
    # Shape inference and graph clean-up, recommended before quantisation.
    preprocessed = int8_path.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32_path), str(preprocessed))
    quantize_static(
        str(preprocessed),
        str(int8_path),
        _WindowCalibrationReader(calibration_windows),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": CALIBRATION_PERCENTILE},
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    preprocessed.unlink()


# Run an ONNX model in Python (used for parity checks and test metrics).
def run_onnx(path: Path, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Return the logits of an ONNX model for a batch of windows."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = [
        session.run([OUTPUT_NAME], {INPUT_NAME: windows[i : i + batch_size]})[0]
        for i in range(0, len(windows), batch_size)
    ]
    return np.concatenate(outputs)


# Save test windows for the C++ benchmark.
def write_samples(path: Path, windows: np.ndarray, labels: np.ndarray) -> None:
    """Write benchmark samples in the binary layout read by cpp/include/lob/samples.hpp.

    Layout (little-endian):
        uint32  magic, n_samples, window, n_features
        float32 windows[n_samples * window * n_features]
        int32   labels[n_samples]
    """
    n, window, n_features = windows.shape
    with path.open("wb") as f:
        f.write(struct.pack("<4I", SAMPLES_MAGIC, n, window, n_features))
        f.write(windows.astype("<f4").tobytes())
        f.write(labels.astype("<i4").tobytes())


# Command-line entry point.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Load the data with the same settings the models were trained with.
    _, cfg = load_checkpoint(args.artifacts_dir / f"{args.models[0]}.pt")
    splits = load_splits(args.data_dir, cfg.horizon)
    rng = np.random.default_rng(cfg.seed)

    # Calibration uses training data only; the test set stays untouched.
    train_ends = window_end_indices(len(splits["train"]), cfg.window)
    calibration = make_windows(
        splits["train"], rng.choice(train_ends, N_CALIBRATION_WINDOWS, replace=False), cfg.window
    )

    # Every full window of the test days, with its label.
    test_ends = window_end_indices(len(splits["test"]), cfg.window)
    test_windows = make_windows(splits["test"], test_ends, cfg.window)
    test_labels = splits["test"].y[test_ends]

    # Benchmark samples: evenly spaced test windows, so both days are covered.
    bench_idx = np.linspace(0, len(test_ends) - 1, N_BENCH_SAMPLES).astype(int)
    write_samples(
        args.artifacts_dir / "samples.bin", test_windows[bench_idx], test_labels[bench_idx]
    )

    # Export, quantise and evaluate each model.
    report: dict = {}
    for name in args.models:
        model, _ = load_checkpoint(args.artifacts_dir / f"{name}.pt")
        fp32_path = args.artifacts_dir / f"{name}.onnx"
        int8_path = args.artifacts_dir / f"{name}.int8.onnx"

        export_fp32(model, cfg.window, fp32_path)
        quantize_int8(fp32_path, int8_path, calibration)

        # Parity check: the exported graph must reproduce PyTorch.
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(test_windows[bench_idx])).numpy()
        max_diff = float(np.abs(run_onnx(fp32_path, test_windows[bench_idx]) - torch_logits).max())
        if max_diff > 1e-3:
            raise RuntimeError(f"{name}: ONNX output differs from PyTorch by {max_diff}")

        report[name] = {"max_abs_logit_diff_vs_pytorch": max_diff}
        # Test metrics of both variants: the accuracy cost of int8.
        for variant, path in (("fp32", fp32_path), ("int8", int8_path)):
            pred = run_onnx(path, test_windows).argmax(axis=1)
            report[name][variant] = {
                "file_size_kb": round(path.stat().st_size / 1024, 1),
                "test": classification_metrics(test_labels, pred),
                # Reference for the C++ benchmark, which replays the same samples.
                "bench_sample_accuracy": float((pred[bench_idx] == test_labels[bench_idx]).mean()),
            }
            print(
                f"[{name}/{variant}] test macro-F1 {report[name][variant]['test']['macro_f1']:.4f}"
                f"  size {report[name][variant]['file_size_kb']:.0f} kB"
            )

    # Persist the report.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "export.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
