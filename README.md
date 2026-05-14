# memory-accessor-repro

This repository is a minimal reproduction workspace for Section 4 of “BLAS-based Block Memory Accessor with Applications to Mixed Precision Sparse Direct Solvers”.

The current main path is the Section 4 tree-parallel benchmark:

```text
tiled matrix + column-major tile layout + right-looking blocked triangular solve
```

Each OpenMP thread solves one independent triangular system of order `m=4096`. MKL is kept single-threaded; OpenMP provides the outer tree parallelism.

## Modes

Blocked modes:

- `fp64`: double storage, double compute, direct tile pointer, `dtrsv/dgemv`
- `fp32_fp64`: float storage, double compute, float tile to double workspace, `dtrsv/dgemv`
- `fp32`: float storage, float compute, direct tile pointer, `strsv/sgemv`

Direct MKL baselines:

- `direct_fp64_mkl`: full contiguous double upper triangular matrix, one `cblas_dtrsv` per system
- `direct_fp32_mkl`: full contiguous float upper triangular matrix, one `cblas_strsv` per system

## Build

The remote target links MKL runtime directly from:

```text
/opt/conda/lib/libmkl_rt.so.2
```

Build:

```bash
cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda
cmake --build build-mkl -j
```

## Run

```bash
bash scripts/run_ch4_tree_parallel.sh
```

The script sets:

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=20   # only if not already set
export MKL_DYNAMIC=FALSE
```

Output:

```text
results/ch4_tree_parallel.csv
```

CSV fields:

```text
mode,matrix_layout,data_layout,looking,m,b,repeat,num_systems,num_threads,time_ms,gflops,max_rel_error
```

## Plot

```bash
python3 scripts/plot_ch4_tree_parallel.py
```

Output:

```text
results/ch4_tree_parallel_colmajor_right.png
```

The figure is a single plot for:

```text
Tree parallel / Tiled matrix / Column-major / Right-looking
```

It shows the three blocked precision modes as solid curves and the two direct MKL baselines as horizontal dashed lines.

## Legacy Paths

The older single-thread 8-panel Figure 4.1 style path and the breakdown diagnostic code are still present for reference, but they are no longer the default workflow. The default scripts now run the tree-parallel benchmark.

Current non-goals:

- custom precision
- AVX512-VBMI
- BLR
- MUMPS
- MPI
- non-divisible blocks
- left-looking
- row-major
- contiguous blocked variants
- 8-panel figure generation as the main output
