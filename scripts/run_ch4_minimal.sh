#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

mkdir -p results

if [[ ! -x build-mkl/blocked_trsv_benchmark ]]; then
    echo "error: build-mkl/blocked_trsv_benchmark not found or not executable" >&2
    echo "build first with: cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda && cmake --build build-mkl -j" >&2
    exit 1
fi

build-mkl/blocked_trsv_benchmark --output results/ch4_minimal.csv

echo "wrote results/ch4_minimal.csv"
