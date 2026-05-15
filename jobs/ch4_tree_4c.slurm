#!/bin/bash
#SBATCH --job-name=ch4_tree_4c
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --comment=alama

set -eo pipefail

export COLORTERM="${COLORTERM:-}"
source /etc/profile || true
module load gcc/10.1.0

export PATH=$HOME/.local/bin:/opt/app/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/opt/app/anaconda3/lib:${LD_LIBRARY_PATH:-}

export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "===== Job info ====="
date
hostname
pwd
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo

echo "===== Tools ====="
which g++
g++ --version
which cmake
cmake --version
which python3
python3 --version
echo

cd ~/workspace/memory-accessor-repro

echo "===== Configure ====="
rm -rf build-mkl
cmake -S . -B build-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/opt/app/anaconda3 \
  -DCBLAS_INCLUDE_DIR=$PWD/third_party/cblas_compat

echo
echo "===== Build ====="
cmake --build build-mkl -j "$OMP_NUM_THREADS"

echo
echo "===== Link check ====="
ldd build-mkl/blocked_trsv_benchmark | grep -E "mkl|openblas|blas" || true

echo
echo "===== Run benchmark ====="
bash scripts/run_ch4_tree_parallel.sh

echo
echo "===== Plot ====="
python3 scripts/plot_ch4_tree_parallel.py

echo
echo "===== Output files ====="
ls -lh results/ch4_tree_parallel.csv || true
ls -lh results/ch4_tree_parallel_colmajor_right.png || true

echo
echo "===== Done ====="
date
