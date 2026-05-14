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

right-looking：

```text
x = y

for j = p-1 down to 0:
    read U_jj
    use tile pointer directly if storage precision == compute precision
    otherwise upcast U_jj to workspace B

    solve B x_j = x_j using BLAS trsv

    for i = j-1 down to 0:
        read U_ij
        use tile pointer directly if storage precision == compute precision
        otherwise upcast U_ij to workspace B

        update x_i = x_i - B x_j using BLAS gemv
```

left-looking：

```text
x = y

for i = p-1 down to 0:
    for j = p-1 down to i+1:
        read U_ij
        use tile pointer directly if storage precision == compute precision
        otherwise upcast U_ij to workspace B

        update x_i = x_i - B x_j using BLAS gemv

    read U_ii
    use tile pointer directly if storage precision == compute precision
    otherwise upcast U_ii to workspace B

    solve B x_i = x_i using BLAS trsv
```

三种 blocked mode：

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
    block access: float -> double workspace B
    BLAS: dtrsv / dgemv
```

direct MKL baseline：

```text
direct_fp64_mkl:
    full contiguous double matrix
    one cblas_dtrsv

direct_fp32_mkl:
    full contiguous float matrix
    one cblas_strsv
```

## 2. 实验设置

本阶段复现 Section 4 / Figure 4.1 风格实验。

矩阵与布局：

```text
matrix type: dense upper triangular
m = 4096
matrix layout: contiguous / tiled
data layout: col_major / row_major
algorithm: left-looking / right-looking
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
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
```

输出 CSV：

```text
mode,matrix_layout,data_layout,looking,m,b,repeat,time_ms,gflops,rel_error
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
求解 U x = y
计算 rel_error = ||x - x_true|| / ||x_true||
```

## 3. 补充说明

当前只做：

```text
dense upper triangular solve
contiguous / tiled matrix layout
row-major / column-major data layout
left-looking / right-looking blocked traversal
fp64 / fp32 / fp32_fp64
direct fp64/fp32 MKL baselines
MKL CBLAS
single-thread benchmark
```

当前不做：

```text
custom precision
AVX512-VBMI
BLR
MUMPS
OpenMP tree parallelism
MPI
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

当前目标不是复现论文绝对性能，而是先跑通 Section 4 / Figure 4.1 风格的最小工作闭环。
