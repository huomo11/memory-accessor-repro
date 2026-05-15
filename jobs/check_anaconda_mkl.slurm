#!/bin/bash
#SBATCH --job-name=check_anaconda_mkl
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --comment=alama

echo "===== Check Anaconda include ====="
ls /opt/app/anaconda3/include/cblas.h 2>/dev/null || true
ls /opt/app/anaconda3/include 2>/dev/null | grep -Ei "cblas|mkl|blas" || true

echo
echo "===== Check Anaconda lib ====="
ls /opt/app/anaconda3/lib/libmkl_rt.so* 2>/dev/null || true
ls /opt/app/anaconda3/lib 2>/dev/null | grep -Ei "mkl|openblas|blas|lapack" | head -80 || true

echo
echo "===== Check python ====="
/opt/app/anaconda3/bin/python --version || true
