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

cd ~/workspace/memory-accessor-repro
mkdir -p logs results
source scripts/env_bupt.sh

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

bash scripts/build_mkl_bupt.sh

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
