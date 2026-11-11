"""Run the C++ latency benchmark for every model and write results/latency.json.

Each model is measured in three configurations:

    fp32-noopt   float32 graph, ONNX Runtime graph optimisations disabled
    fp32         float32 graph, all graph optimisations enabled
    int8         int8-quantised graph, all graph optimisations enabled

at batch size 1 (per-event latency, the number that matters for trading) and
batch size 64 (throughput). Everything runs on a single thread. Each
configuration is run three times and the run with the median p50 is kept, to
damp the noise of a machine that is not dedicated to benchmarking.

Usage:
    python -m lob.bench
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from lob.models import MODEL_NAMES

# Benchmark matrix: every model is run in each variant at each batch size.
# (name, model file suffix, extra benchmark flags)
VARIANTS = (
    ("fp32-noopt", ".onnx", ["--no-opt"]),
    ("fp32", ".onnx", []),
    ("int8", ".int8.onnx", []),
)
# (batch size, timed iterations)
BATCHES = ((1, 20000), (64, 2000))
REPEATS = 3


# Record which machine the numbers come from.
def cpu_name() -> str:
    """Best-effort human-readable CPU name for the results file."""
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.returncode == 0:
            return out.stdout.strip()
    return platform.processor() or platform.machine()


# One invocation of the C++ binary.
def run_benchmark(
    binary: Path, model: Path, samples: Path, batch: int, iters: int, flags: list[str]
) -> dict:
    """Run the benchmark binary once and parse the JSON it prints."""
    cmd = [str(binary), "--model", str(model), "--samples", str(samples)]
    cmd += ["--batch", str(batch), "--iters", str(iters), *flags]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


# Command-line entry point.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--binary", type=Path, default=Path("build/lob_bench"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Inputs produced by `python -m lob.export`.
    samples = args.artifacts_dir / "samples.bin"
    export_report = json.loads((args.results_dir / "export.json").read_text())

    # Run the whole benchmark matrix.
    runs = []
    for model_name in args.models:
        for variant, suffix, flags in VARIANTS:
            for batch, iters in BATCHES:
                model_path = args.artifacts_dir / f"{model_name}{suffix}"
                # Repeat the run and keep the one with the median p50.
                repeats = [
                    run_benchmark(args.binary, model_path, samples, batch, iters, flags)
                    for _ in range(REPEATS)
                ]
                result = sorted(repeats, key=lambda r: r["p50_ns"])[REPEATS // 2]

                # End-to-end check: at batch 1 the C++ benchmark scores every
                # sample, so it must agree with what Python measured at export.
                expected = export_report[model_name][suffix_to_precision(suffix)][
                    "bench_sample_accuracy"
                ]
                if batch == 1 and abs(result["accuracy"] - expected) > 1e-3:
                    raise RuntimeError(
                        f"{model_name}/{variant}: C++ accuracy {result['accuracy']:.4f} "
                        f"!= Python accuracy {expected:.4f}"
                    )

                runs.append({"model": model_name, "variant": variant, **result})
                print(
                    f"[{model_name}/{variant}] batch {batch:2d}: "
                    f"p50 {result['p50_ns'] / 1e3:8.1f} us  p99 {result['p99_ns'] / 1e3:8.1f} us  "
                    f"{result['throughput_per_s']:10.0f} windows/s"
                )

    # Persist the results together with the machine description.
    report = {"machine": {"cpu": cpu_name(), "os": platform.platform()}, "runs": runs}
    out_path = args.results_dir / "latency.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")


# Helper for looking up the reference accuracy of a model file.
def suffix_to_precision(suffix: str) -> str:
    """Map a model file suffix to its key in results/export.json."""
    return "int8" if suffix == ".int8.onnx" else "fp32"


if __name__ == "__main__":
    main()
