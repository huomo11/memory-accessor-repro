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
OUT_PATH = ROOT / "results" / "ch4_minimal_gflops.png"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"error: CSV not found: {CSV_PATH}", file=sys.stderr)
        print("run scripts/run_ch4_minimal.sh first", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_PATH)
    required = {"mode", "b", "gflops"}
    missing = required.difference(df.columns)
    if missing:
        print(f"error: CSV missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    grouped = (
        df.groupby(["mode", "b"], as_index=False)["gflops"]
        .median()
        .sort_values(["mode", "b"])
    )

    colors = {
        "fp64": "#1f4e79",
        "fp32_fp64": "#d9a300",
        "fp32": "#2e8b57",
    }
    order = ["fp64", "fp32_fp64", "fp32"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for mode in order:
        sub = grouped[grouped["mode"] == mode]
        if sub.empty:
            continue
        ax.plot(
            sub["b"],
            sub["gflops"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=colors.get(mode),
            label=mode,
        )

    ax.set_title("Tiled matrix / Row-major / Right-looking")
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
