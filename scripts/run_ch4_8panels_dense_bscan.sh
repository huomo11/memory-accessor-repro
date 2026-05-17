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

M="${M:-4096}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"
BLOCKS="${BLOCKS:-16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512}"
OUTPUT="${OUTPUT:-results/ch4_8panels_dense_bscan.csv}"

mkdir -p results

if [[ ! -x build-mkl/blocked_trsv_benchmark ]]; then
    echo "error: build-mkl/blocked_trsv_benchmark not found or not executable" >&2
    echo "build first with: cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda && cmake --build build-mkl -j" >&2
    exit 1
fi

build-mkl/blocked_trsv_benchmark \
    --eight-panels \
    --m "$M" \
    --b "$BLOCKS" \
    --repeat "$REPEAT" \
    --warmup "$WARMUP" \
    --output "$OUTPUT"

echo "wrote $OUTPUT"
