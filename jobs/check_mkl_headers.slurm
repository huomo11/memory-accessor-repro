#!/bin/bash
#SBATCH --job-name=check_mkl_headers
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --comment=alama

echo "===== /opt/app/anaconda3/include head ====="
ls /opt/app/anaconda3/include 2>/dev/null | head -80 || true

echo
echo "===== find cblas / mkl headers under anaconda3 ====="
find /opt/app/anaconda3 -maxdepth 6 \( -name "cblas.h" -o -name "mkl_cblas.h" -o -name "mkl.h" \) 2>/dev/null | head -80

echo
echo "===== pkgs related to mkl/blas ====="
ls /opt/app/anaconda3/pkgs 2>/dev/null | grep -Ei "mkl|blas|lapack" | head -100 || true
