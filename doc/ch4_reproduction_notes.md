# Chapter 4 Minimal Reproduction 知识包

## 1. Algorithm 2 伪代码

目标：求解 upper triangular system：

```text
U x = y
```

其中：

```text
U: m x m upper triangular matrix
U 被划分为 b x b blocks
m = p * b
B: b x b compute-precision workspace
```

核心思想：

```text
U 用 storage precision 存储；
每次读取一个 tile；
如有需要，upcast 到 compute precision workspace B；
然后用 BLAS 做计算。
```

伪代码：

```text
x = y

for j = p-1 down to 0:

    read U_jj
    upcast U_jj to workspace B if needed

    solve:
        B x_j = x_j
    using BLAS trsv

    for i = j-1 down to 0:

        read U_ij
        upcast U_ij to workspace B if needed

        update:
            x_i = x_i - B x_j
        using BLAS gemv
```

三种模式：

```text
fp64:
    U: double
    x/y/B: double
    BLAS: dtrsv / dgemv

fp32:
    U: float
    x/y/B: float
    BLAS: strsv / sgemv

fp32_fp64:
    U: float
    x/y/B: double
    tile access: float -> double workspace B
    BLAS: dtrsv / dgemv
```

## 2. 实验设置

本阶段只复现 Chapter 4 minimal version。

矩阵与布局：

```text
matrix type: dense upper triangular
m = 4096
layout: tiled matrix
tile layout: row-major
algorithm: right-looking
requirement: m % b == 0
```

block size sweep：

```text
b = 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512
```

repeat：

```text
repeat = 5
```

运行环境：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

输出 CSV：

```text
mode,m,b,repeat,time_ms,gflops,rel_error
```

Gflops 简化估算：

```text
flops ~= m * m
gflops = flops / time_seconds / 1e9
```

正确性检查：

```text
生成 x_true
生成稳定 upper triangular U
计算 y = U x_true
用不同 mode 解 U x = y
计算 rel_error = ||x - x_true|| / ||x_true||
```

U 的生成：

```text
固定 random seed
非对角元素用小随机数
对角线加较大正数，例如 10.0 或 m
```

## 3. 补充说明

当前只做：

```text
tiled matrix layout
tile 内部 row-major
right-looking Algorithm 2
fp64 / fp32 / fp32_fp64
OpenBLAS CBLAS
单线程 benchmark
```

当前不做：

```text
custom precision
AVX512-VBMI
BLR
MUMPS
OpenMP tree parallelism
left-looking
contiguous layout
column-major tile layout
```

预期趋势：

```text
b 太小：
    BLAS call 太碎，性能低

b 适中：
    BLAS 效率提高，性能上升

b 太大：
    fp32_fp64 的 double workspace B 可能放不进 cache，
    mixed mode 性能可能下降
```

第一阶段成功标准：

```text
能编译
能运行
能生成 CSV
三种 mode 都有结果
rel_error 不是 NaN / inf
gflops 趋势可以解释
```

当前目标不是复现论文绝对性能，而是先跑通 Chapter 4 的最小工作闭环。
