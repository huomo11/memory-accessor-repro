#!/bin/bash
#SBATCH --job-name=check_build_mkl
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --comment=alama

set -eo pipefail

export COLORTERM="${COLORTERM:-}"
source /etc/profile || true
module load gcc/10.1.0

export PATH=$HOME/.local/bin:/opt/app/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/opt/app/anaconda3/lib:${LD_LIBRARY_PATH:-}

echo "===== Tools ====="
which g++
g++ --version
which cmake
cmake --version

echo
echo "===== Configure ====="
cd ~/workspace/memory-accessor-repro
rm -rf build-mkl

cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/opt/app/anaconda3 \
  -DCBLAS_INCLUDE_DIR=$PWD/third_party/cblas_compat

echo
echo "===== Build ====="
cmake --build build-mkl -j "$SLURM_CPUS_PER_TASK"

echo
echo "===== Link check ====="
ldd build-mkl/blocked_trsv_benchmark | grep -E "mkl|openblas|blas" || true

echo
echo "===== Done ====="
