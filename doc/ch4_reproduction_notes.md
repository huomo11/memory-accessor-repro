# Chapter 4 Tree-Parallel Reproduction Notes

## 1. 目标

本项目复现论文 *BLAS-based Block Memory Accessor with Applications to Mixed Precision Sparse Direct Solvers* 中 Chapter 4 / Figure 4.1 相关的核心性能现象。重点不是完整复现论文所有实验，而是构造一个可运行、可检查、可画图的 C++ prototype，用于观察：

- triangular solve 中 block size `b` 对性能的影响；
- storage precision 与 compute precision 分离后的性能变化；
- tiled block memory accessor 与 direct MKL reference 的相对趋势；
- tree-parallel 场景下，多独立三角系统并行求解的 aggregate throughput。

当前已经完成：

- dense upper triangular solve；
- tiled / contiguous 两类 matrix layout；
- row_major / col_major 两类 data layout；
- right-looking / left-looking 两类 blocked traversal；
- tree-parallel benchmark；
- `fp64`、`fp32_fp64`、`fp32` 三种 blocked mode；
- `direct_fp64_mkl`、`direct_fp32_mkl` 两条 direct MKL baseline；
- single-panel 主结果与 8-panel 扩展图。

当前不复现：

- custom precision；
- AVX512-VBMI；
- BLR；
- MUMPS；
- MPI；
- 非整除 block；
- 论文完整 sparse direct solver pipeline。

因此本文档中的结果应理解为 qualitative reproduction，而不是论文 Figure 4.1 的逐点数值复现。

## 2. 论文目标现象

Figure 4.1 关注的是 BLAS-based block memory accessor 在不同 block size 下的性能趋势。核心变量包括：

- block size `b`：太小会导致 BLAS call 粒度太碎；适中时 BLAS 效率提高；太大时 workspace/cache 压力上升；
- storage precision：矩阵 `U` 用 fp64 或 fp32 存储；
- compute precision：计算时使用 fp64 或 fp32 BLAS；
- mixed precision：`fp32_fp64` 使用 fp32 storage，但在访问 tile 时 upcast 到 fp64 workspace，再调用 fp64 BLAS；
- direct MKL reference：整矩阵 contiguous triangular solve，用作参考线，而不是 blocked accessor 的同类实现。

论文原图比较了多个 layout / looking 组合。本项目当前保留两条结果路径：

- single-panel 主线：`tiled + col_major + right-looking`；
- 8-panel 扩展：`tiled/contiguous × col_major/row_major × right/left`。

direct MKL baseline 在图中用于提供参考位置。它不是 blocked algorithm，也不使用 block size；CSV 中的 `b` 只是 plotting/grouping label。

## 3. Prototype 设计

数学问题为：

```text
U x = y
```

其中 `U` 是 dense upper triangular matrix，`m = 4096`。blocked benchmark 要求 `m % b == 0`。

tree-parallel 设置：

```text
num_systems = OMP_NUM_THREADS
```

每个 OpenMP thread 求解一个独立的 triangular system。MKL 内部保持单线程，外层 OpenMP 提供 tree parallelism：

```bash
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
```

主结果使用的 block size sweep：

```text
b = 16, 32, 64, 128, 256, 512
```

正式运行配置：

```text
warmup = 2
repeat = 10
```

warmup 不写入 CSV，只用于降低首次 BLAS call、OpenMP/MKL runtime 初始化、cache/page fault 等造成的 outlier。`repeat` 列只记录正式计时编号，当前为 `0..9`。

CSV header 固定为：

```text
mode,matrix_layout,data_layout,looking,m,b,repeat,num_systems,num_threads,time_ms,gflops,max_rel_error
```

性能使用 aggregate throughput：

```text
gflops = num_systems * m * m / time_seconds / 1e9
```

这里的 flops 是统一简化估算，用于比较不同配置的趋势，不用于精确 flop accounting。

正确性检查：

```text
生成 x_true
生成稳定 upper triangular U
计算 y = U x_true
求解 U x = y
计算 max_rel_error = max_system ||x - x_true||_2 / ||x_true||_2
```

## 4. Precision Modes

`fp64`：

```text
storage precision: double
compute precision: double
matrix tile: double
x/y/workspace: double
BLAS: dtrsv / dgemv
```

在 tiled uniform precision 路径中，tile 本身已经是 compact BLAS operand，因此直接把 tile pointer 传给 BLAS。

`fp32`：

```text
storage precision: float
compute precision: float
matrix tile: float
x/y/workspace: float
BLAS: strsv / sgemv
```

同样，在 tiled uniform precision 路径中直接使用 tile pointer。

`fp32_fp64`：

```text
storage precision: float
compute precision: double
matrix tile: float
x/y/workspace: double
BLAS: dtrsv / dgemv
```

这是本 prototype 中最接近 block memory accessor 思想的模式：矩阵 `U` 以 fp32 存储以降低 memory traffic；每次访问一个 tile 时，将 float tile upcast 到 double workspace `B`；随后使用 fp64 BLAS 做 `trsv/gemv`。因此 mixed mode 的性能受两个因素共同影响：

- fp32 storage 带来的读带宽优势；
- float -> double workspace copy/upcast 与 fp64 BLAS 带来的额外成本。

`direct_fp64_mkl`：

```text
full contiguous upper triangular double matrix
one cblas_dtrsv per system
```

`direct_fp32_mkl`：

```text
full contiguous upper triangular float matrix
one cblas_strsv per system
```

direct MKL baseline 不走 blocked accessor，也不使用 `b`。图中将其画为水平虚线。

## 5. Platform

主要实验平台：

```text
BUPT CPU cluster
node: c003t
CPU: Intel Xeon Gold 6148
topology: 2 sockets x 20 cores
```

当前主结果采用 20-core single-socket bind。这样做的目的是尽量减少 NUMA 干扰，使 tree-parallel 的 aggregate throughput 更容易解释。

40-core dual-socket bind 结果作为 NUMA sensitivity observation。该结果不是当前主结论来源，主要用于观察跨 socket 后内存访问和调度带来的影响。

该平台没有 AVX512-VBMI，因此论文中涉及 custom precision / VBMI 的路径没有复现。

构建使用远程 MKL runtime：

```text
/opt/app/anaconda3/lib/libmkl_rt.so.2
```

运行时建议：

```bash
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

## 6. Results

### 6.1 Single-Panel Result

single-panel 主线配置为：

```text
tiled matrix / col_major / right-looking
```

主要图文件：

```text
results/ch4_tree_parallel_colmajor_right_20c_bind_run1.png
results/ch4_tree_parallel_colmajor_right_20c_bind_run2.png
```

对应 CSV：

```text
results/ch4_tree_parallel_20c_bind_run1.csv
results/ch4_tree_parallel_20c_bind_run2.csv
```

20-core single-socket bind 下，run1/run2 的趋势稳定。以 median gflops 看：

- `fp64` 最好点约为 `26.3 Gflop/s`，通常在 `b=128` 附近；
- `fp32` 最好点约为 `51.7 Gflop/s`，通常在 `b=64` 附近；
- `fp32_fp64` 最好点约为 `48.1-48.3 Gflop/s`，通常在 `b=128` 附近；
- `fp32_fp64` 在 `b=512` 明显下降到约 `13.9 Gflop/s`；
- direct MKL fp64 baseline 约 `26.0 Gflop/s`；
- direct MKL fp32 baseline 约 `50.0-50.6 Gflop/s`。

这些数值不应与论文机器的绝对性能直接比较；更有意义的是三条 blocked 曲线随 block size 的相对趋势。

### 6.2 8-Panel Result

8-panel 扩展图文件：

```text
results/ch4_8panels_20c_bind.png
```

对应 CSV：

```text
results/ch4_8panels_20c_bind.csv
```

8 个 panel 分别为：

- `tiled + col_major + right`
- `tiled + col_major + left`
- `tiled + row_major + right`
- `tiled + row_major + left`
- `contiguous + col_major + right`
- `contiguous + col_major + left`
- `contiguous + row_major + right`
- `contiguous + row_major + left`

20-core single-socket bind 下，tiled panels 的整体表现强于 contiguous panels。典型观察：

- tiled panels 中 `fp32` 峰值约 `52-54 Gflop/s`；
- tiled panels 中 `fp32_fp64` 在中等 block size 约 `48-49 Gflop/s`；
- tiled panels 中 `fp64` 峰值约 `26-27 Gflop/s`；
- contiguous panels 的 blocked 曲线整体低于 tiled panels，尤其 mixed mode 与 fp32 mode 的上限更低。

这说明当前 prototype 的 tiled compact block 路径更符合 BLAS small dense block operand 的访问模式；contiguous blocked 子块虽然数学正确，但在部分路径中需要依赖 `lda=m` 或 copy/upcast，局部性与 call 粒度更难达到 tiled 路径的效果。

### 6.3 run1/run2 Reproducibility

single-panel 20c bind 的 run1/run2 结果非常接近。峰值位置和曲线形状基本一致：

- `fp64` 峰值均在 `b=128` 附近，约 `26.3 Gflop/s`；
- `fp32` 峰值均在 `b=64` 附近，约 `51.7 Gflop/s`；
- `fp32_fp64` 峰值均在 `b=128` 附近，约 `48 Gflop/s`；
- `fp32_fp64` 在 `b=512` 的下降在两次运行中都出现。

这说明 warmup/repeat 与 single-socket binding 后，主要趋势具有可重复性。

## 7. Interpretation

1. 为什么 `fp32` 高：

`fp32` 的矩阵存储、向量和 BLAS compute 都使用 float。相比 `fp64`，它的 memory traffic 更低，BLAS kernel 的吞吐也更高。因此在合适 block size 下，`fp32` 通常接近或略高于 direct fp32 MKL reference。

2. 为什么 `fp64` 低：

`fp64` 使用 double storage 和 double compute，单位数据量更大，BLAS kernel 的峰值吞吐也低于 fp32。当前结果中 `fp64` blocked 曲线与 direct fp64 MKL baseline 大致处于同一量级。

3. 为什么 mixed 在中等 block size 接近 `fp32`：

`fp32_fp64` 的 storage 是 fp32，因此读取 `U` 的带宽压力接近 `fp32`。当 block size 适中时，float -> double workspace 的成本可以被后续 fp64 BLAS compute 部分摊薄；同时 tile 粒度足够大，BLAS call overhead 不再主导。因此 mixed mode 在 `b=64/128` 附近可以接近 `fp32` 曲线。

4. 为什么 mixed 在 `b=512` 明显下降：

`fp32_fp64` 每次访问 tile 都需要 double workspace `B`。当 `b=512` 时，一个 double workspace tile 约为：

```text
512 * 512 * 8 bytes = 2 MiB
```

这会显著增加 cache 压力。再加上 upcast copy、fp64 BLAS compute、多个 independent systems 并行运行，mixed mode 的优势被削弱，性能明显下降。

5. 为什么 direct MKL baseline 只是 reference：

direct MKL baseline 是 full contiguous triangular solve，一次调用 `cblas_trsv`。blocked accessor 则把矩阵拆成 tile，通过多次 `trsv/gemv` 组合完成求解。二者算法结构、memory access pattern、BLAS call 粒度都不同。因此 direct MKL baseline 应作为参考线，而不是“必须超过”的对象。

6. 为什么不要写“我们的代码超过 MKL”：

当前 prototype 与 direct MKL baseline 并非同一种实现。blocked path 使用了额外的 `gemv` 分块更新，direct MKL 是整矩阵 `trsv`。在某些 block size 下 blocked curve 可能高于 direct baseline，但这不能简单表述为“超过 MKL”。更准确的说法是：在本 prototype 和本平台上，blocked memory accessor 的某些配置达到或接近 direct MKL reference 的 aggregate throughput，并呈现出与论文目标一致的 qualitative trend。

## 8. Limitations

- 当前是 qualitative reproduction only，不是论文 Figure 4.1 的完整逐点复现。
- 硬件平台、cache 层次、内存系统与论文平台不同。
- compiler、MKL runtime、调度策略与论文环境不同。
- 没有 AVX512-VBMI，未复现 custom precision 路径。
- 没有 BLR、MUMPS，也没有完整 sparse direct solver pipeline。
- contiguous panels 整体弱于 tiled panels；这说明 prototype 对 tiled compact tile 路径更友好，也提示 contiguous blocked path 仍有优化空间。
- 当前 benchmark 使用 dense upper triangular matrix，不能直接代表 sparse direct solver 中所有 supernode / frontal matrix 行为。
- direct MKL baseline 与 blocked modes 不是同构算法，只能作为 reference。
- 当前仍是 prototype，代码目标是清楚、可 debug、可复现实验趋势，而不是生产级 benchmark framework。

## 9. Next Steps

可选后续工作：

- 增加更细粒度 timing breakdown，区分 prepare/upcast、`trsv`、`gemv` 的耗时；
- 做 parallel first-touch 和更严格的 NUMA 控制；
- 在更接近论文平台的 CPU 上重新测试；
- 实现 custom precision prototype，并在有 AVX512-VBMI 的平台上验证；
- 扩展到 BLR 或 sparse direct solver 场景；
- 对 contiguous blocked path 做更系统的 locality 与 copy/upcast 优化；
- 增加自动化 run summary，用于比较 run1/run2 和 20c/40c 的稳定性。
