#pragma once

#include "tiled_matrix.hpp"

#include <mkl_cblas.h>

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

template <typename Matrix, typename Compute>
void copy_block_as_compute(const Matrix& u, std::size_t tile_i, std::size_t tile_j, Compute* dst)
{
    const std::size_t b = u.b();
    const DataLayout layout = u.data_layout();
    for (std::size_t row = 0; row < b; ++row) {
        for (std::size_t col = 0; col < b; ++col) {
            dst[dense_index(row, col, b, layout)] = static_cast<Compute>(u.block_value(tile_i, tile_j, row, col));
        }
    }
}

template <typename Storage, typename Compute>
const Compute* prepare_block_for_blas(const TiledUpperMatrix<Storage>& u,
                                      std::size_t tile_i,
                                      std::size_t tile_j,
                                      std::vector<Compute>& workspace)
{
    if constexpr (std::is_same_v<Storage, Compute>) {
        // Uniform precision tiled blocks are already compact b x b BLAS operands.
        return u.tile(tile_i, tile_j);
    } else {
        // Mixed precision block memory accessor: storage tile -> compute workspace.
        copy_block_as_compute(u, tile_i, tile_j, workspace.data());
        return workspace.data();
    }
}

template <typename Storage, typename Compute>
const Compute* prepare_block_for_blas(const ContiguousUpperMatrix<Storage>& u,
                                      std::size_t tile_i,
                                      std::size_t tile_j,
                                      std::vector<Compute>& workspace)
{
    // A b x b sub-block of a full matrix is strided, so keep the compact workspace path.
    copy_block_as_compute(u, tile_i, tile_j, workspace.data());
    return workspace.data();
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
        const Compute* diag = prepare_block_for_blas(u, j, j, workspace);
        Blas<Compute>::trsv(layout, b, diag, b, x.data() + j * b);

        for (std::size_t i = j; i-- > 0;) {
            const Compute* block = prepare_block_for_blas(u, i, j, workspace);
            Blas<Compute>::gemv(layout, b, block, b, x.data() + j * b, x.data() + i * b);
        }
    }
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
            const Compute* block = prepare_block_for_blas(u, i, j, workspace);
            Blas<Compute>::gemv(layout, b, block, b, x.data() + j * b, x.data() + i * b);
        }

        const Compute* diag = prepare_block_for_blas(u, i, i, workspace);
        Blas<Compute>::trsv(layout, b, diag, b, x.data() + i * b);
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
