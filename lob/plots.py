"""Render the README figures from the JSON files in results/.

Usage:
    python -m lob.plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt

# Human-readable names used in the figures.
MODEL_LABELS = {"logreg": "Logistic regression", "mlp": "MLP", "cnn": "CNN"}
VARIANT_LABELS = {
    "fp32-noopt": "float32, no graph opt.",
    "fp32": "float32, optimised",
    "int8": "int8, optimised",
}

# Colours: two categorical series on a near-white surface, text in neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Thickness of one bar, in axis units (two bars share one row).
BAR_HEIGHT = 0.36


# Visual style shared by both figures.
def _style_axes(ax: plt.Axes) -> None:
    """Shared look: no frame, faint vertical grid behind the bars."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Drawing primitive shared by both figures.
def _grouped_barh(
    ax: plt.Axes,
    labels: list[str],
    first: list[float],
    second: list[float],
    names: tuple[str, str],
    fmt: str,
) -> None:
    """Two horizontal bars per label, each annotated with its value."""
    positions = range(len(labels))
    for values, offset, color, name in (
        (first, -BAR_HEIGHT / 2, BLUE, names[0]),
        (second, BAR_HEIGHT / 2, ORANGE, names[1]),
    ):
        bars = ax.barh(
            [p + offset for p in positions],
            values,
            height=BAR_HEIGHT - 0.04,
            color=color,
            label=name,
        )
        ax.bar_label(bars, fmt=fmt, padding=4, color=INK, fontsize=9)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()  # first label on top
    ax.margins(x=0.18)  # room for the value labels


# Figure 1: prediction quality.
def plot_f1(metrics: dict, export: dict, out_path: Path) -> None:
    """Test macro F1 per model, float32 vs int8, with the majority baseline."""
    models = [m for m in MODEL_LABELS if m in export]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor=SURFACE)
    _style_axes(ax)
    _grouped_barh(
        ax,
        [MODEL_LABELS[m] for m in models],
        [export[m]["fp32"]["test"]["macro_f1"] for m in models],
        [export[m]["int8"]["test"]["macro_f1"] for m in models],
        names=("float32", "int8"),
        fmt="%.3f",
    )

    # Reference line: the majority-class baseline.
    majority = metrics["models"]["majority"]["test"]["macro_f1"]
    ax.axvline(majority, color=MUTED, linewidth=1, linestyle="--")
    ax.text(
        majority,
        -0.62,
        f"majority class only: {majority:.3f}",
        color=MUTED,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    # Axis labels, title, legend.
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 on the test days (higher is better)", color=MUTED)
    ax.set_title("Prediction quality", color=INK, loc="left", fontweight="bold", pad=22)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, labelcolor=INK)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Figure 2: inference latency.
def plot_latency(latency: dict, out_path: Path) -> None:
    """p50 and p99 latency at batch size 1: one panel per model, one row per variant.

    Panels have independent x-axes because the models differ in latency by
    orders of magnitude; compare variants within a panel, not across panels.
    """
    runs = [r for r in latency["runs"] if r["batch"] == 1]
    models = [m for m in MODEL_LABELS if any(r["model"] == m for r in runs)]

    fig, axes = plt.subplots(
        1, len(models), figsize=(4.3 * len(models), 3.0), facecolor=SURFACE, sharey=True
    )
    for ax, model in zip(axes, models, strict=True):
        by_variant = {r["variant"]: r for r in runs if r["model"] == model}
        _style_axes(ax)
        _grouped_barh(
            ax,
            [VARIANT_LABELS[v] for v in VARIANT_LABELS],
            [by_variant[v]["p50_ns"] / 1e3 for v in VARIANT_LABELS],
            [by_variant[v]["p99_ns"] / 1e3 for v in VARIANT_LABELS],
            names=("p50", "p99"),
            fmt="%.1f",
        )
        ax.set_xlim(left=0)
        ax.set_xlabel("Latency per prediction, µs (lower is better)", color=MUTED)
        ax.set_title(MODEL_LABELS[model], color=INK, loc="left", fontweight="bold")
    axes[-1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Command-line entry point.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Read one JSON file from the results directory.
    def load(name: str) -> dict:
        return json.loads((args.results_dir / name).read_text())

    plot_f1(load("metrics.json"), load("export.json"), args.results_dir / "f1.png")
    plot_latency(load("latency.json"), args.results_dir / "latency.png")
    print(f"wrote {args.results_dir / 'f1.png'} and {args.results_dir / 'latency.png'}")


if __name__ == "__main__":
    main()
