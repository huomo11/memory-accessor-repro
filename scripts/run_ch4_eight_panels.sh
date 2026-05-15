#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MKL_NUM_THREADS=1
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    OMP_NUM_THREADS="$(nproc)"
fi
export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE

# Chapter 4 eight-panel benchmark. This is the systematic comparison path,
# separate from the current single-panel tree-parallel main result.
mkdir -p results

if [[ ! -x build-mkl/blocked_trsv_benchmark ]]; then
    echo "error: build-mkl/blocked_trsv_benchmark not found or not executable" >&2
    echo "build first with: cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda && cmake --build build-mkl -j" >&2
    exit 1
fi

build-mkl/blocked_trsv_benchmark \
    --eight-panels \
    --m 4096 \
    --b 16,32,64,128,256,512 \
    --repeat 10 \
    --warmup 2 \
    --output results/ch4_8panels.csv

echo "wrote results/ch4_8panels.csv"
