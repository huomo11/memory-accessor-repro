#!/bin/bash
#SBATCH --job-name=check_blas
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --comment=alama

set -eo pipefail

export COLORTERM="${COLORTERM:-}"
source /etc/profile || true

echo "===== module avail blas/mkl/openblas/intel ====="
module avail 2>&1 | grep -Ei "blas|openblas|mkl|intel|oneapi|lapack|atlas" || true

echo
echo "===== /opt/app top level ====="
ls /opt/app 2>/dev/null | grep -Ei "blas|openblas|mkl|intel|oneapi|lapack|anaconda|conda|gcc" || true

echo
echo "===== common include/lib locations ====="
ls /usr/include/cblas.h 2>/dev/null || true
ls /usr/local/include/cblas.h 2>/dev/null || true

ls /usr/lib64/libblas* 2>/dev/null || true
ls /usr/lib64/libopenblas* 2>/dev/null || true
ls /usr/local/lib64/libopenblas* 2>/dev/null || true

echo
echo "===== gcc module load check ====="
module load gcc/10.1.0
which g++
g++ --version
