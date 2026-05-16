#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing Python dependency. Please install pandas and matplotlib, for example:\n"
        "  python3 -m pip install pandas matplotlib"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT / "results" / "ch4_8panels.csv"
DEFAULT_OUT_PATH = ROOT / "results" / "ch4_8panels.png"

PANELS = [
    ("tiled", "col_major", "right", "Tiled / Column-major / Right-looking"),
    ("tiled", "col_major", "left", "Tiled / Column-major / Left-looking"),
    ("tiled", "row_major", "right", "Tiled / Row-major / Right-looking"),
    ("tiled", "row_major", "left", "Tiled / Row-major / Left-looking"),
    ("contiguous", "col_major", "right", "Contiguous / Column-major / Right-looking"),
    ("contiguous", "col_major", "left", "Contiguous / Column-major / Left-looking"),
    ("contiguous", "row_major", "right", "Contiguous / Row-major / Right-looking"),
    ("contiguous", "row_major", "left", "Contiguous / Row-major / Left-looking"),
]

CURVES = [
    ("fp64", "fp64", "#1f4e79"),
    ("fp32_fp64", "fp32+fp64", "#d9a300"),
    ("fp32", "fp32", "#2e8b57"),
]

BASELINES = [
    ("direct_fp64_mkl", "fp64 (MKL)", "#1f4e79"),
    ("direct_fp32_mkl", "fp32 (MKL)", "#2e8b57"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Chapter 4 eight-panel benchmark results.")
    parser.add_argument("--input", default=str(DEFAULT_CSV_PATH), help="input CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUT_PATH), help="output PNG path")
    return parser.parse_args()


def blocked_medians(df, matrix_layout, data_layout, looking, mode):
    sub = df[
        (df["mode"] == mode)
        & (df["matrix_layout"] == matrix_layout)
        & (df["data_layout"] == data_layout)
        & (df["looking"] == looking)
    ]
    if sub.empty:
        return sub
    return sub.groupby("b", as_index=False)["gflops"].median().sort_values("b")


def plot_blocked_curve(ax, med, label, color):
    if med.empty:
        return
    ax.plot(
        med["b"],
        med["gflops"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color=color,
        label=label,
    )


def compute_y_limits(panel_medians, baseline_values):
    values = []
    for med in panel_medians.values():
        if not med.empty:
            values.extend(med["gflops"].tolist())
    values.extend(value for value in baseline_values.values() if value is not None)

    if not values:
        return None

    ymin = min(values)
    ymax = max(values)
    span = ymax - ymin
    if span <= 0:
        padding = max(abs(ymax) * 0.05, 1.0)
    else:
        padding = span * 0.05
    return max(0.0, ymin - padding), ymax + padding


def plot_direct_baseline(ax, value, label, color):
    if value is None:
        return
    ax.axhline(
        value,
        linewidth=1.4,
        linestyle="--",
        color=color,
        label=label,
    )


def main() -> int:
    args = parse_args()
    csv_path = Path(args.input)
    out_path = Path(args.output)

    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        print("run scripts/run_ch4_eight_panels.sh first", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    required = {"mode", "b", "repeat", "gflops", "matrix_layout", "data_layout", "looking"}
    missing = required.difference(df.columns)
    if missing:
        print(f"error: CSV missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    df["b"] = pd.to_numeric(df["b"])
    df["gflops"] = pd.to_numeric(df["gflops"])

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    baseline_values = {}
    for mode, _, _ in BASELINES:
        sub = df[df["mode"] == mode]
        baseline_values[mode] = None if sub.empty else sub["gflops"].median()

    panel_medians = {}
    for matrix_layout, data_layout, looking, _ in PANELS:
        for mode, _, _ in CURVES:
            panel_medians[(matrix_layout, data_layout, looking, mode)] = blocked_medians(
                df, matrix_layout, data_layout, looking, mode
            )

    y_limits = compute_y_limits(panel_medians, baseline_values)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (matrix_layout, data_layout, looking, title) in zip(axes, PANELS):
        for mode, label, color in CURVES:
            med = panel_medians[(matrix_layout, data_layout, looking, mode)]
            plot_blocked_curve(ax, med, label, color)
        for mode, label, color in BASELINES:
            plot_direct_baseline(ax, baseline_values[mode], label, color)

        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)

    for ax in axes[4:]:
        ax.set_xlabel("Block size b")
    for ax in axes[::4]:
        ax.set_ylabel("Performance (Gflop/s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Chapter 4 Eight-Panel Tree-Parallel Comparison", y=1.06, fontsize=13)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
