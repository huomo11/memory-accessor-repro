#!/usr/bin/env python3

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
CSV_PATH = ROOT / "results" / "ch4_8panels.csv"
OUT_PATH = ROOT / "results" / "ch4_8panels.png"

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


def plot_blocked_curve(ax, df, matrix_layout, data_layout, looking, mode, label, color):
    sub = df[
        (df["mode"] == mode)
        & (df["matrix_layout"] == matrix_layout)
        & (df["data_layout"] == data_layout)
        & (df["looking"] == looking)
    ]
    if sub.empty:
        return
    med = sub.groupby("b", as_index=False)["gflops"].median().sort_values("b")
    ax.plot(
        med["b"],
        med["gflops"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color=color,
        label=label,
    )


def plot_direct_baseline(ax, df, data_layout, mode, label, color):
    sub = df[
        (df["mode"] == mode)
        & (df["matrix_layout"] == "contiguous")
        & (df["data_layout"] == data_layout)
        & (df["looking"] == "direct")
    ]
    if sub.empty:
        return
    ax.axhline(
        sub["gflops"].median(),
        linewidth=1.4,
        linestyle="--",
        color=color,
        label=label,
    )


def main() -> int:
    if not CSV_PATH.exists():
        print(f"error: CSV not found: {CSV_PATH}", file=sys.stderr)
        print("run scripts/run_ch4_eight_panels.sh first", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_PATH)
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

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.5), sharex=True)
    axes = axes.ravel()

    for ax, (matrix_layout, data_layout, looking, title) in zip(axes, PANELS):
        for mode, label, color in CURVES:
            plot_blocked_curve(ax, df, matrix_layout, data_layout, looking, mode, label, color)
        for mode, label, color in BASELINES:
            plot_direct_baseline(ax, df, data_layout, mode, label, color)

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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
