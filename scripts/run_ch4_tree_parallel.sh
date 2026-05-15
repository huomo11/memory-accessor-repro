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

# The executable performs unrecorded warmup solves for each mode/block size
# before writing timed repeats, reducing first-call/cache/page-fault outliers.
mkdir -p results

if [[ ! -x build-mkl/blocked_trsv_benchmark ]]; then
    echo "error: build-mkl/blocked_trsv_benchmark not found or not executable" >&2
    echo "build first with: cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda && cmake --build build-mkl -j" >&2
    exit 1
fi

build-mkl/blocked_trsv_benchmark \
    --tree-parallel \
    --m 4096 \
    --b 16,32,64,128,256,512 \
    --repeat 10 \
    --warmup 2 \
    --output results/ch4_tree_parallel.csv

echo "wrote results/ch4_tree_parallel.csv"
