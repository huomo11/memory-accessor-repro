#pragma once

#include "tiled_matrix.hpp"

#include <cblas.h>

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

inline CBLAS_LAYOUT cblas_layout(DataLayout layout)
{
    return layout == DataLayout::RowMajor ? CblasRowMajor : CblasColMajor;
}

template <typename Compute>
struct Blas;

template <>
struct Blas<float> {
    static void trsv(CBLAS_LAYOUT layout, std::size_t n, const float* a, std::size_t lda, float* x)
    {
        cblas_strsv(layout, CblasUpper, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), a, static_cast<int>(lda), x, 1);
    }

    static void gemv(CBLAS_LAYOUT layout, std::size_t n, const float* a, std::size_t lda, const float* x, float* y)
    {
        cblas_sgemv(layout, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(n),
                    -1.0f, a, static_cast<int>(lda), x, 1, 1.0f, y, 1);
    }
};

template <>
struct Blas<double> {
    static void trsv(CBLAS_LAYOUT layout, std::size_t n, const double* a, std::size_t lda, double* x)
    {
        cblas_dtrsv(layout, CblasUpper, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), a, static_cast<int>(lda), x, 1);
    }

    static void gemv(CBLAS_LAYOUT layout, std::size_t n, const double* a, std::size_t lda, const double* x, double* y)
    {
        cblas_dgemv(layout, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(n),
                    -1.0, a, static_cast<int>(lda), x, 1, 1.0, y, 1);
    }
};

template <typename Compute>
struct BlasOperand {
    const Compute* data = nullptr;
    std::size_t lda = 0;
};

struct BreakdownStats {
    double total_ms = 0.0;
    double prepare_ms = 0.0;
    double trsv_ms = 0.0;
    double gemv_ms = 0.0;
};

template <typename Storage, typename Compute>
void copy_compact_block_as_compute(const Storage* src, Compute* dst, std::size_t b, DataLayout layout)
{
    if (layout == DataLayout::RowMajor) {
        for (std::size_t row = 0; row < b; ++row) {
            for (std::size_t col = 0; col < b; ++col) {
                const std::size_t k = row * b + col;
                dst[k] = static_cast<Compute>(src[k]);
            }
        }
    } else {
        for (std::size_t col = 0; col < b; ++col) {
            for (std::size_t row = 0; row < b; ++row) {
                const std::size_t k = col * b + row;
                dst[k] = static_cast<Compute>(src[k]);
            }
        }
    }
}

template <typename Storage, typename Compute>
void copy_contiguous_block_as_compute(const Storage* src,
                                      Compute* dst,
                                      std::size_t m,
                                      std::size_t b,
                                      std::size_t row0,
                                      std::size_t col0,
                                      DataLayout layout)
{
    if (layout == DataLayout::RowMajor) {
        for (std::size_t row = 0; row < b; ++row) {
            const Storage* src_row = src + (row0 + row) * m + col0;
            Compute* dst_row = dst + row * b;
            for (std::size_t col = 0; col < b; ++col) {
                dst_row[col] = static_cast<Compute>(src_row[col]);
            }
        }
    } else {
        for (std::size_t col = 0; col < b; ++col) {
            const Storage* src_col = src + (col0 + col) * m + row0;
            Compute* dst_col = dst + col * b;
            for (std::size_t row = 0; row < b; ++row) {
                dst_col[row] = static_cast<Compute>(src_col[row]);
            }
        }
    }
}

template <typename Storage, typename Compute>
BlasOperand<Compute> prepare_block_for_blas(const TiledUpperMatrix<Storage>& u,
                                            std::size_t tile_i,
                                            std::size_t tile_j,
                                            std::vector<Compute>& workspace)
{
    const std::size_t b = u.b();
    if constexpr (std::is_same_v<Storage, Compute>) {
        // Uniform precision tiled blocks are already compact b x b BLAS operands.
        return {u.tile(tile_i, tile_j), b};
    } else {
        // Mixed precision block memory accessor: storage tile -> compute workspace.
        copy_compact_block_as_compute(u.tile(tile_i, tile_j), workspace.data(), b, u.data_layout());
        return {workspace.data(), b};
    }
}

template <typename Storage, typename Compute>
BlasOperand<Compute> prepare_block_for_blas(const ContiguousUpperMatrix<Storage>& u,
                                            std::size_t tile_i,
                                            std::size_t tile_j,
                                            std::vector<Compute>& workspace)
{
    const std::size_t m = u.m();
    const std::size_t b = u.b();
    const std::size_t row0 = tile_i * b;
    const std::size_t col0 = tile_j * b;

    if constexpr (std::is_same_v<Storage, Compute>) {
        const std::size_t offset = dense_index(row0, col0, m, u.data_layout());
        return {u.data() + offset, m};
    } else {
        copy_contiguous_block_as_compute(u.data(), workspace.data(), m, b, row0, col0, u.data_layout());
        return {workspace.data(), b};
    }
}

template <typename Matrix, typename Compute>
void blocked_trsv_right_looking(const Matrix& u, std::vector<Compute>& x)
{
    const std::size_t m = u.m();
    const std::size_t b = u.b();
    const std::size_t p = u.p();

    if (x.size() != m) {
        throw std::invalid_argument("x size must match matrix size m");
    }

    std::vector<Compute> workspace(b * b);
    const CBLAS_LAYOUT layout = cblas_layout(u.data_layout());

    for (std::size_t j = p; j-- > 0;) {
        const BlasOperand<Compute> diag = prepare_block_for_blas(u, j, j, workspace);
        Blas<Compute>::trsv(layout, b, diag.data, diag.lda, x.data() + j * b);

        for (std::size_t i = j; i-- > 0;) {
            const BlasOperand<Compute> block = prepare_block_for_blas(u, i, j, workspace);
            Blas<Compute>::gemv(layout, b, block.data, block.lda, x.data() + j * b, x.data() + i * b);
        }
    }
}

template <typename Matrix, typename Compute>
BreakdownStats blocked_trsv_right_looking_breakdown(const Matrix& u, std::vector<Compute>& x)
{
    const std::size_t m = u.m();
    const std::size_t b = u.b();
    const std::size_t p = u.p();

    if (x.size() != m) {
        throw std::invalid_argument("x size must match matrix size m");
    }

    auto elapsed_ms = [](const auto& start, const auto& stop) {
        return std::chrono::duration<double, std::milli>(stop - start).count();
    };

    BreakdownStats stats;
    std::vector<Compute> workspace(b * b);
    const CBLAS_LAYOUT layout = cblas_layout(u.data_layout());

    const auto total_start = std::chrono::steady_clock::now();

    for (std::size_t j = p; j-- > 0;) {
        const auto prepare_start = std::chrono::steady_clock::now();
        const BlasOperand<Compute> diag = prepare_block_for_blas(u, j, j, workspace);
        const auto prepare_stop = std::chrono::steady_clock::now();
        stats.prepare_ms += elapsed_ms(prepare_start, prepare_stop);

        const auto trsv_start = std::chrono::steady_clock::now();
        Blas<Compute>::trsv(layout, b, diag.data, diag.lda, x.data() + j * b);
        const auto trsv_stop = std::chrono::steady_clock::now();
        stats.trsv_ms += elapsed_ms(trsv_start, trsv_stop);

        for (std::size_t i = j; i-- > 0;) {
            const auto block_prepare_start = std::chrono::steady_clock::now();
            const BlasOperand<Compute> block = prepare_block_for_blas(u, i, j, workspace);
            const auto block_prepare_stop = std::chrono::steady_clock::now();
            stats.prepare_ms += elapsed_ms(block_prepare_start, block_prepare_stop);

            const auto gemv_start = std::chrono::steady_clock::now();
            Blas<Compute>::gemv(layout, b, block.data, block.lda, x.data() + j * b, x.data() + i * b);
            const auto gemv_stop = std::chrono::steady_clock::now();
            stats.gemv_ms += elapsed_ms(gemv_start, gemv_stop);
        }
    }

    const auto total_stop = std::chrono::steady_clock::now();
    stats.total_ms = elapsed_ms(total_start, total_stop);
    return stats;
}

template <typename Matrix, typename Compute>
void blocked_trsv_left_looking(const Matrix& u, std::vector<Compute>& x)
{
    const std::size_t m = u.m();
    const std::size_t b = u.b();
    const std::size_t p = u.p();

    if (x.size() != m) {
        throw std::invalid_argument("x size must match matrix size m");
    }

    std::vector<Compute> workspace(b * b);
    const CBLAS_LAYOUT layout = cblas_layout(u.data_layout());

    for (std::size_t i = p; i-- > 0;) {
        for (std::size_t j = p; j-- > i + 1;) {
            const BlasOperand<Compute> block = prepare_block_for_blas(u, i, j, workspace);
            Blas<Compute>::gemv(layout, b, block.data, block.lda, x.data() + j * b, x.data() + i * b);
        }

        const BlasOperand<Compute> diag = prepare_block_for_blas(u, i, i, workspace);
        Blas<Compute>::trsv(layout, b, diag.data, diag.lda, x.data() + i * b);
    }
}

template <typename Matrix, typename Compute>
void blocked_trsv(const Matrix& u, std::vector<Compute>& x, Looking looking)
{
    if (looking == Looking::Right) {
        blocked_trsv_right_looking(u, x);
    } else {
        blocked_trsv_left_looking(u, x);
    }
}

template <typename Compute>
void direct_trsv(const ContiguousUpperMatrix<Compute>& u, std::vector<Compute>& x)
{
    if (x.size() != u.m()) {
        throw std::invalid_argument("x size must match matrix size m");
    }
    Blas<Compute>::trsv(cblas_layout(u.data_layout()), u.m(), u.data(), u.m(), x.data());
}
