#!/bin/bash
set -e

export COLORTERM="${COLORTERM:-}"
source /etc/profile || true
module load gcc/10.1.0

export PATH=$HOME/.local/bin:/opt/app/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/opt/app/anaconda3/lib:${LD_LIBRARY_PATH:-}

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=GNU

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-20}}"
export OMP_DYNAMIC=FALSE
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
