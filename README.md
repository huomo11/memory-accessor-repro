# memory-accessor-repro

本仓库用于复现论文 “BLAS-based Block Memory Accessor with Applications to Mixed Precision Sparse Direct Solvers” 的一个最小实验闭环。

当前只做 Chapter 4 minimal reproduction：dense upper triangular solve，tiled matrix，tile 内部 row-major，right-looking Algorithm 2，并比较：

- `fp64`: fp64 storage + fp64 compute
- `fp32`: fp32 storage + fp32 compute
- `fp32_fp64`: fp32 storage + fp64 compute

`fp32_fp64` 模式中，矩阵 tile 用 `float` 存储，每次访问 tile 时 upcast 到 `double` workspace `B`，再调用 `dtrsv` / `dgemv`。

## 依赖

Ubuntu 20.04 示例：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopenblas-dev python3 python3-pip
python3 -m pip install pandas matplotlib
```

## 编译

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

生成可执行文件：

```text
build/blocked_trsv_benchmark
```

## 运行 benchmark

```bash
bash scripts/run_ch4_minimal.sh
```

脚本会设置单线程环境：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

输出：

```text
results/ch4_minimal.csv
```

CSV 格式：

```text
mode,m,b,repeat,time_ms,gflops,rel_error
```

注意：当前 minimal 版本要求 `m % b == 0`。默认 `m=4096` 时，不能整除的 block size 会被跳过并打印 warning。

## 画图

```bash
python3 scripts/plot_ch4_minimal.py
```

输出：

```text
results/ch4_minimal_gflops.png
```

这张图只对应 Figure 4.1 中 “Tiled matrix / Row-major / Right-looking” 子图风格参考，不是整张 8 子图 Figure 4.1 的复现。

## 当前限制

当前不包含：

- custom precision
- AVX512-VBMI
- BLR
- MUMPS
- OpenMP tree parallelism
- left-looking
- contiguous layout
- column-major tile layout

目标不是复现论文绝对性能，而是先跑通 Chapter 4 的最小可编译、可运行、可画图工作流。
