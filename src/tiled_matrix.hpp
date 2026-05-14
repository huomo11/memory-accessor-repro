#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

template <typename Storage>
class TiledUpperMatrix {
public:
    using value_type = Storage;

    TiledUpperMatrix(std::size_t m, std::size_t b)
        : m_(m), b_(b), p_(b == 0 ? 0 : m / b)
    {
        if (b == 0) {
            throw std::invalid_argument("block size b must be positive");
        }
        if (m == 0) {
            throw std::invalid_argument("matrix size m must be positive");
        }
        if (m % b != 0) {
            throw std::invalid_argument("matrix size m must be divisible by block size b");
        }
        data_.assign(p_ * p_ * b_ * b_, Storage{0});
    }

    std::size_t m() const { return m_; }
    std::size_t b() const { return b_; }
    std::size_t p() const { return p_; }

    Storage* tile(std::size_t tile_i, std::size_t tile_j)
    {
        return data_.data() + tile_offset(tile_i, tile_j);
    }

    const Storage* tile(std::size_t tile_i, std::size_t tile_j) const
    {
        return data_.data() + tile_offset(tile_i, tile_j);
    }

    void fill_stable_upper(unsigned seed, double diagonal_shift = 10.0)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> offdiag_dist(-0.01, 0.01);
        std::uniform_real_distribution<double> diag_dist(0.0, 1.0);

        for (std::size_t ti = 0; ti < p_; ++ti) {
            for (std::size_t tj = 0; tj < p_; ++tj) {
                Storage* t = tile(ti, tj);
                for (std::size_t ii = 0; ii < b_; ++ii) {
                    for (std::size_t jj = 0; jj < b_; ++jj) {
                        const std::size_t row = ti * b_ + ii;
                        const std::size_t col = tj * b_ + jj;
                        double value = 0.0;
                        if (col > row) {
                            value = offdiag_dist(gen);
                        } else if (col == row) {
                            value = diagonal_shift + diag_dist(gen);
                        }
                        t[ii * b_ + jj] = static_cast<Storage>(value);
                    }
                }
            }
        }
    }

private:
    std::size_t tile_offset(std::size_t tile_i, std::size_t tile_j) const
    {
        if (tile_i >= p_ || tile_j >= p_) {
            throw std::out_of_range("tile index out of range");
        }
        return (tile_i * p_ + tile_j) * b_ * b_;
    }

    std::size_t m_;
    std::size_t b_;
    std::size_t p_;
    std::vector<Storage> data_;
};
