#pragma once

#include "tiled_matrix.hpp"

#include <cblas.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename Compute>
struct Blas;

template <>
struct Blas<float> {
    static void trsv(std::size_t n, const float* a, std::size_t lda, float* x)
    {
        cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), a, static_cast<int>(lda), x, 1);
    }

    static void gemv(std::size_t n, const float* a, std::size_t lda, const float* x, float* y)
    {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(n),
                    -1.0f, a, static_cast<int>(lda), x, 1, 1.0f, y, 1);
    }
};

template <>
struct Blas<double> {
    static void trsv(std::size_t n, const double* a, std::size_t lda, double* x)
    {
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), a, static_cast<int>(lda), x, 1);
    }

    static void gemv(std::size_t n, const double* a, std::size_t lda, const double* x, double* y)
    {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(n), static_cast<int>(n),
                    -1.0, a, static_cast<int>(lda), x, 1, 1.0, y, 1);
    }
};

template <typename Storage, typename Compute>
void load_tile_as_compute(const Storage* src, Compute* dst, std::size_t b)
{
    // Storage precision may be lower than compute precision; mixed mode upcasts here.
    for (std::size_t k = 0; k < b * b; ++k) {
        dst[k] = static_cast<Compute>(src[k]);
    }
}

template <typename Storage, typename Compute>
void blocked_trsv_right_looking(const TiledUpperMatrix<Storage>& u, std::vector<Compute>& x)
{
    const std::size_t m = u.m();
    const std::size_t b = u.b();
    const std::size_t p = u.p();

    if (x.size() != m) {
        throw std::invalid_argument("x size must match matrix size m");
    }

    std::vector<Compute> workspace(b * b);

    for (std::size_t j = p; j-- > 0;) {
        load_tile_as_compute(u.tile(j, j), workspace.data(), b);
        Blas<Compute>::trsv(b, workspace.data(), b, x.data() + j * b);

        for (std::size_t i = j; i-- > 0;) {
            load_tile_as_compute(u.tile(i, j), workspace.data(), b);
            Blas<Compute>::gemv(b, workspace.data(), b, x.data() + j * b, x.data() + i * b);
        }
    }
}
