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
CSV_PATH = ROOT / "results" / "ch4_minimal.csv"
FIG41_PATH = ROOT / "results" / "ch4_figure4_1_style.png"


PANEL_ROWS = [
    ("contiguous", "contiguous array"),
    ("tiled", "tiled matrix"),
]

PANEL_COLS = [
    ("col_major", "left", "Column-major / Left-looking"),
    ("col_major", "right", "Column-major / Right-looking"),
    ("row_major", "left", "Row-major / Left-looking"),
    ("row_major", "right", "Row-major / Right-looking"),
]

CURVES = [
    ("fp64", "fp64", "#1f4e79", "-"),
    ("fp32_fp64", "fp32+fp64", "#d9a300", "-"),
    ("fp32", "fp32", "#2e8b57", "-"),
]

BASELINES = [
    ("direct_fp64_mkl", "fp64 (MKL)", "#1f4e79"),
    ("direct_fp32_mkl", "fp32 (MKL)", "#2e8b57"),
]


def require_columns(df: pd.DataFrame) -> None:
    required = {
        "mode",
        "matrix_layout",
        "data_layout",
        "looking",
        "b",
        "gflops",
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"error: CSV missing columns: {sorted(missing)}")


def median_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["mode", "matrix_layout", "data_layout", "looking", "b"], as_index=False)["gflops"]
        .median()
        .sort_values(["mode", "matrix_layout", "data_layout", "looking", "b"])
    )


def main() -> int:
    if not CSV_PATH.exists():
        print(f"error: CSV not found: {CSV_PATH}", file=sys.stderr)
        print("run scripts/run_ch4_minimal.sh first", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_PATH)
    require_columns(df)
    grouped = median_series(df)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True)

    for row_idx, (matrix_layout, row_label) in enumerate(PANEL_ROWS):
        for col_idx, (data_layout, looking, title) in enumerate(PANEL_COLS):
            ax = axes[row_idx][col_idx]
            panel = grouped[
                (grouped["matrix_layout"] == matrix_layout)
                & (grouped["data_layout"] == data_layout)
            ]

            for mode, label, color, linestyle in CURVES:
                sub = panel[(panel["mode"] == mode) & (panel["looking"] == looking)]
                if sub.empty:
                    continue
                ax.plot(
                    sub["b"],
                    sub["gflops"],
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )

            for mode, label, color in BASELINES:
                sub = panel[(panel["mode"] == mode) & (panel["looking"] == "direct")]
                if sub.empty:
                    continue
                ax.plot(
                    sub["b"],
                    sub["gflops"],
                    linewidth=1.5,
                    color=color,
                    linestyle="--",
                    label=label,
                )

            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nPerformance (Gflop/s)")
            if row_idx == 1:
                ax.set_xlabel("Block size b")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
            ax.set_axisbelow(True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)
    fig.suptitle("Section 4 / Figure 4.1 Style Benchmark", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    FIG41_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG41_PATH, dpi=200)
    print(f"wrote {FIG41_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
