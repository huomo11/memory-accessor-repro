# memory-accessor-repro

This repository is a minimal Chapter 4 reproduction workspace for *BLAS-based Block Memory Accessor with Applications to Mixed Precision Sparse Direct Solvers*.

The detailed technical report is:

```text
doc/ch4_reproduction_notes.md
```

Current scope: qualitative reproduction of Section 4 / Figure 4.1 style performance trends, not a full reproduction of the paper.

## Maintained Benchmarks

Main single-panel path:

```text
tree-parallel / tiled matrix / column-major / right-looking
```

Eight-panel extension:

```text
tiled or contiguous
row_major or col_major
right-looking or left-looking
```

Each OpenMP thread solves one independent dense upper triangular system of order `m=4096`. MKL is kept single-threaded; OpenMP provides the outer tree parallelism.

## Modes

Blocked modes:

- `fp64`: double storage, double compute, `dtrsv/dgemv`
- `fp32_fp64`: float storage, double compute, float tile to double workspace, `dtrsv/dgemv`
- `fp32`: float storage, float compute, `strsv/sgemv`

Direct MKL baselines:

- `direct_fp64_mkl`: full contiguous double triangular solve, one `cblas_dtrsv` per system
- `direct_fp32_mkl`: full contiguous float triangular solve, one `cblas_strsv` per system

For direct MKL baselines, `b` is only a plotting/grouping label; the direct solve itself does not use block size. Direct MKL references in the 8-panel plot use global medians because they do not depend on panel-specific blocked layout settings.

## Build

Remote target:

```text
/opt/conda/lib/libmkl_rt.so.2
```

Build:

```bash
cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda
cmake --build build-mkl -j
```

## Run

Single-panel benchmark:

```bash
bash scripts/run_ch4_tree_parallel.sh
python3 scripts/plot_ch4_tree_parallel.py
```

Outputs:

```text
results/ch4_tree_parallel.csv
results/ch4_tree_parallel_colmajor_right.png
```

Eight-panel benchmark:

```bash
bash scripts/run_ch4_eight_panels.sh
python3 scripts/plot_ch4_8panels.py
```

Outputs:

```text
results/ch4_8panels.csv
results/ch4_8panels.png
```

Dense block-size scan with non-divisor block sizes:

```bash
bash scripts/run_ch4_8panels_dense_bscan.sh
python3 scripts/plot_ch4_8panels.py \
  --input results/ch4_8panels_dense_bscan.csv \
  --output results/ch4_8panels_dense_bscan.png
```

The scripts use:

```bash
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=$(nproc)   # only if not already set
```

For report-quality runs, prefer a 20-core single-socket binding:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=20
```

Do not force `OMP_NUM_THREADS=20` inside a 4-core interactive shell. The benchmark sets `num_systems = omp_get_max_threads()`, so oversubscribing CPUs also increases the number of independent systems.

## CSV

Header:

```text
mode,matrix_layout,data_layout,looking,m,b,repeat,num_systems,num_threads,time_ms,gflops,max_rel_error
```

Throughput formula:

```text
gflops = num_systems * m * m / time_seconds / 1e9
```

Default experiment settings:

```text
m = 4096
b = 16, 32, 64, 128, 256, 512
warmup = 2
repeat = 10
```

Warmup rows are not written to CSV. The `repeat` column records formal timed repeats only.

## Support for Non-Divisor Block Sizes

The benchmark supports nominal block sizes `b` that do not divide `m`. Internally, the number of blocks is `ceil(m / b)`, and the last block uses `min(b, m - k*b)` as its actual extent. Tiled storage still allocates padded `b x b` tiles so BLAS leading dimensions stay simple; BLAS `M/N/n` arguments use the logical block extent.

The CSV `b` column records the nominal block size requested by the user. This makes denser block-size scans possible, closer to the sampling style of Figure 4.1, without claiming an exact reproduction of the paper's plot.

## Current Reference Results

Existing report artifacts include:

```text
results/ch4_8panels_20c_bind.png
results/ch4_tree_parallel_colmajor_right_20c_bind_run1.png
results/ch4_tree_parallel_colmajor_right_20c_bind_run2.png
```

The current main result is the 20-core single-socket bind result on the BUPT CPU cluster node `c003t` with Intel Xeon Gold 6148 CPUs. A 40-core dual-socket run is kept as NUMA sensitivity observation, not as the main conclusion.

## Non-Goals

- exact reproduction of paper Figure 4.1 absolute performance
- custom precision
- AVX512-VBMI
- BLR
- MUMPS
- MPI
- production benchmark framework
