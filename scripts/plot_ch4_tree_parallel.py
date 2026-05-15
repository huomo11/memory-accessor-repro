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
CSV_PATH = ROOT / "results" / "ch4_tree_parallel.csv"
OUT_PATH = ROOT / "results" / "ch4_tree_parallel_colmajor_right.png"

CURVES = [
    ("fp64", "fp64", "#1f4e79"),
    ("fp32_fp64", "fp32+fp64", "#d9a300"),
    ("fp32", "fp32", "#2e8b57"),
]

BASELINES = [
    ("direct_fp64_mkl", "fp64 (MKL)", "#1f4e79"),
    ("direct_fp32_mkl", "fp32 (MKL)", "#2e8b57"),
]


def main() -> int:
    if not CSV_PATH.exists():
        print(f"error: CSV not found: {CSV_PATH}", file=sys.stderr)
        print("run scripts/run_ch4_tree_parallel.sh first", file=sys.stderr)
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
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for mode, label, color in CURVES:
        sub = df[
            (df["mode"] == mode)
            & (df["matrix_layout"] == "tiled")
            & (df["data_layout"] == "col_major")
            & (df["looking"] == "right")
        ]
        if sub.empty:
            continue
        med = sub.groupby("b", as_index=False)["gflops"].median().sort_values("b")
        ax.plot(
            med["b"],
            med["gflops"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=label,
        )

    for mode, label, color in BASELINES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        ax.axhline(
            sub["gflops"].median(),
            linewidth=1.6,
            linestyle="--",
            color=color,
            label=label,
        )

    ax.set_title("Tree parallel / Tiled matrix / Column-major / Right-looking")
    ax.set_xlabel("Block size b")
    ax.set_ylabel("Performance (Gflop/s)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
