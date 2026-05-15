#!/bin/bash
#SBATCH --job-name=check_build_gcc10
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

echo "===== Host ====="
hostname
pwd
echo

echo "===== Load environment ====="
export COLORTERM="${COLORTERM:-}"
source /etc/profile || true

module load gcc/10.1.0

export PATH=$HOME/.local/bin:$PATH
hash -r

echo "PATH=$PATH"
echo

echo "===== Tools ====="
which gcc
gcc --version

which g++
g++ --version

which c++
c++ --version

which cmake
cmake --version

which ninja || true
ninja --version || true
echo

echo "===== Configure ====="
cd ~/workspace/memory-accessor-repro
rm -rf build-gcc10
cmake -S . -B build-gcc10 -DCMAKE_BUILD_TYPE=Release

echo
echo "===== Build ====="
cmake --build build-gcc10 -j "$SLURM_CPUS_PER_TASK"

echo
echo "===== Link check ====="
ldd build-gcc10/blocked_trsv_benchmark | grep -E "mkl|openblas|blas" || true

echo
echo "===== Done ====="
