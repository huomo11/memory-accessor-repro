#!/bin/bash
#SBATCH --job-name=ch4_tree_20c_bind
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --comment=alama

set -eo pipefail

cd ~/workspace/memory-accessor-repro
mkdir -p logs results
source scripts/env_bupt.sh

echo "===== Job info ====="
date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PLACES=$OMP_PLACES"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo

bash scripts/build_mkl_bupt.sh

echo
echo "===== Link check ====="
ldd build-mkl/blocked_trsv_benchmark | grep -E "mkl|gomp|openblas|blas" || true

echo
echo "===== Run benchmark ====="
bash scripts/run_ch4_tree_parallel.sh

echo
echo "===== Plot ====="
/opt/app/anaconda3/bin/python3 scripts/plot_ch4_tree_parallel.py

echo
echo "===== Save 20c bind outputs ====="
cp results/ch4_tree_parallel.csv results/ch4_tree_parallel_20c_bind.csv
cp results/ch4_tree_parallel_colmajor_right.png results/ch4_tree_parallel_colmajor_right_20c_bind.png

ls -lh results/ch4_tree_parallel_20c_bind.csv
ls -lh results/ch4_tree_parallel_colmajor_right_20c_bind.png

echo
echo "===== Done ====="
date
