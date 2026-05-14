# memory-accessor-repro

Reproduction project for BLAS-based block memory accessor experiments.

Current focus:

- dense blocked triangular solve
- fp32 storage + fp64 compute
- tiled matrix layout
- block size sweep
- BLAS dtrsv / dgemv benchmark
