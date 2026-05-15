#!/bin/bash
#SBATCH --job-name=check_toolchain
#SBATCH --partition=c003t
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --comment=alama

echo "===== Host ====="
hostname
pwd
echo

echo "===== Basic PATH ====="
echo "$PATH"
echo

echo "===== Add user cmake/ninja ====="
export PATH=$HOME/.local/bin:$PATH
which cmake || true
cmake --version || true
which ninja || true
ninja --version || true
echo

echo "===== Default compiler ====="
which g++ || true
g++ --version || true
echo

echo "===== Try devtoolset ====="
ls /opt/rh 2>/dev/null || true

for v in 12 11 10 9 8 7; do
    if [[ -f "/opt/rh/devtoolset-${v}/enable" ]]; then
        echo "Found devtoolset-${v}"
        source "/opt/rh/devtoolset-${v}/enable"
        which g++
        g++ --version
        break
    fi
done
echo

echo "===== Try module ====="
source /etc/profile || true
type module || true
module avail 2>&1 | grep -Ei "gcc|devtoolset|compiler|intel|oneapi|cmake|mkl|openblas|blas" | head -120 || true
echo

echo "===== BLAS/MKL quick check ====="
ls /opt/conda/lib/libmkl_rt.so.2 2>/dev/null || true
ldconfig -p 2>/dev/null | grep -Ei "mkl|openblas|blas" | head -80 || true
echo

echo "===== Try CMake configure ====="
cd ~/workspace/memory-accessor-repro
rm -rf build-test
cmake -S . -B build-test -DCMAKE_BUILD_TYPE=Release || true

echo
echo "===== Done ====="
