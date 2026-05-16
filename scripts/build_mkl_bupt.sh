#!/bin/bash
set -e

if [[ ! -f CMakeLists.txt ]]; then
    echo "error: scripts/build_mkl_bupt.sh must be run from the repository root" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-20}}"

echo "===== Configure ====="
rm -rf build-mkl
cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/opt/app/anaconda3 \
  -DCBLAS_INCLUDE_DIR=$PWD/third_party/cblas_compat

echo
echo "===== Build ====="
cmake --build build-mkl -j "$OMP_NUM_THREADS"
