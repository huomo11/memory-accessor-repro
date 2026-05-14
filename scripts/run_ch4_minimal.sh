#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p results

if [[ ! -x build/blocked_trsv_benchmark ]]; then
    echo "error: build/blocked_trsv_benchmark not found or not executable" >&2
    echo "build first with: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

build/blocked_trsv_benchmark --output results/ch4_minimal.csv

echo "wrote results/ch4_minimal.csv"
