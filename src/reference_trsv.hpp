#pragma once

#include "tiled_matrix.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

template <typename Compute>
std::vector<Compute> make_x_true(std::size_t m, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<Compute> x(m);
    for (Compute& value : x) {
        value = static_cast<Compute>(dist(gen));
    }
    return x;
}

template <typename Matrix, typename Compute>
std::vector<Compute> upper_matvec(const Matrix& u, const std::vector<Compute>& x)
{
    const std::size_t m = u.m();

    if (x.size() != m) {
        throw std::invalid_argument("x size must match matrix size m");
    }

    std::vector<Compute> y(m, Compute{0});
    for (std::size_t row = 0; row < m; ++row) {
        Compute sum = Compute{0};
        for (std::size_t col = row; col < m; ++col) {
            sum += static_cast<Compute>(u.at(row, col)) * x[col];
        }
        y[row] = sum;
    }
    return y;
}

template <typename Compute>
double relative_error(const std::vector<Compute>& x, const std::vector<Compute>& x_true)
{
    if (x.size() != x_true.size()) {
        throw std::invalid_argument("relative_error input sizes do not match");
    }

    long double diff2 = 0.0L;
    long double ref2 = 0.0L;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const long double diff = static_cast<long double>(x[i]) - static_cast<long double>(x_true[i]);
        const long double ref = static_cast<long double>(x_true[i]);
        diff2 += diff * diff;
        ref2 += ref * ref;
    }

    if (ref2 == 0.0L) {
        return 0.0;
    }
    return static_cast<double>(std::sqrt(diff2 / ref2));
}
