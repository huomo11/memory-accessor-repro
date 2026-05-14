# memory-accessor-repro

本仓库用于复现论文 “BLAS-based Block Memory Accessor with Applications to Mixed Precision Sparse Direct Solvers” 的 Section 4 / Figure 4.1 风格实验。

当前实现 dense upper triangular solve：

```text
U x = y
```

支持 8 种 blocked 配置：

- `contiguous` / `tiled`
- `row_major` / `col_major`
- `left` / `right`

支持三种 blocked mode：

- `fp64`: fp64 storage + fp64 compute
- `fp32_fp64`: fp32 storage + fp64 compute
- `fp32`: fp32 storage + fp32 compute

其中 `fp32_fp64` 使用 block memory accessor 逻辑：`float` tile/sub-block 读入后 upcast 到 `double` workspace `B`，再调用 MKL CBLAS `dtrsv` / `dgemv`。

另外加入两个 direct MKL baseline：

- `direct_fp64_mkl`: full contiguous double matrix + one `cblas_dtrsv`
- `direct_fp32_mkl`: full contiguous float matrix + one `cblas_strsv`

## 远程 MKL 构建

默认目标环境是远程算力平台，MKL 位于：

```text
/opt/conda/lib/libmkl_rt.so.2
```

构建命令：

```bash
cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT=/opt/conda
cmake --build build-mkl -j
```

生成可执行文件：

```text
build-mkl/blocked_trsv_benchmark
```

## 运行 benchmark

```bash
bash scripts/run_ch4_minimal.sh
```

脚本设置：

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
```

输出 CSV：

```text
results/ch4_minimal.csv
```

CSV 格式：

```text
mode,matrix_layout,data_layout,looking,m,b,repeat,time_ms,gflops,rel_error
```

## 画图

```bash
python3 scripts/plot_ch4_minimal.py
```

输出：

```text
results/ch4_figure4_1_style.png
```

图像为 2 行 x 4 列，风格对应 Figure 4.1：

- 第一行：contiguous array
- 第二行：tiled matrix
- 四列：Column-major/Left-looking、Column-major/Right-looking、Row-major/Left-looking、Row-major/Right-looking

每个子图包含 `fp64`、`fp32+fp64`、`fp32` 三条 blocked 曲线，以及 `fp64 (MKL)`、`fp32 (MKL)` 两条 direct baseline 虚线。

## 性能诊断

默认 benchmark 不开启逐块计时。需要分析 mixed mode 时间分布时运行：

```bash
build-mkl/blocked_trsv_benchmark --breakdown
```

输出：

```text
results/ch4_breakdown.csv
```

字段：

```text
mode,matrix_layout,data_layout,looking,m,b,repeat,total_ms,prepare_ms,trsv_ms,gemv_ms,rel_error
```

## 当前限制

当前不包含：

- custom precision
- AVX512-VBMI
- BLR
- MUMPS
- OpenMP tree parallelism
- MPI

`m % b != 0` 的 block size 会被跳过；如果通过 `--b` 显式指定非法 block size，则程序会报错。当前目标是复现 Section 4 / Figure 4.1 的核心 benchmark 结构，不追求论文绝对性能数值。
