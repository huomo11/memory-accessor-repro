#pragma once

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

enum class DataLayout {
    RowMajor,
    ColMajor
};

enum class MatrixLayout {
    Contiguous,
    Tiled
};

enum class Looking {
    Left,
    Right
};

inline const char* to_string(DataLayout layout)
{
    return layout == DataLayout::RowMajor ? "row_major" : "col_major";
}

inline const char* to_string(MatrixLayout layout)
{
    return layout == MatrixLayout::Contiguous ? "contiguous" : "tiled";
}

inline const char* to_string(Looking looking)
{
    return looking == Looking::Right ? "right" : "left";
}

inline std::size_t dense_index(std::size_t row, std::size_t col, std::size_t ld, DataLayout layout)
{
    return layout == DataLayout::RowMajor ? row * ld + col : col * ld + row;
}

inline std::size_t ceil_div(std::size_t n, std::size_t d)
{
    return (n + d - 1) / d;
}

template <typename Storage>
class TiledUpperMatrix {
public:
    using value_type = Storage;

    TiledUpperMatrix(std::size_t m, std::size_t b, DataLayout data_layout)
        : m_(m), b_(b), p_(b == 0 ? 0 : ceil_div(m, b)), data_layout_(data_layout)
    {
        validate_shape(m, b);
        data_.assign(p_ * p_ * b_ * b_, Storage{0});
    }

    std::size_t m() const { return m_; }
    std::size_t b() const { return b_; }
    std::size_t p() const { return p_; }
    std::size_t size() const { return m_; }
    std::size_t block_size() const { return b_; }
    std::size_t num_blocks() const { return p_; }
    std::size_t block_begin(std::size_t k) const { return k * b_; }
    std::size_t block_extent(std::size_t k) const
    {
        const std::size_t begin = block_begin(k);
        if (begin >= m_) {
            throw std::out_of_range("block index out of range");
        }
        return std::min(b_, m_ - begin);
    }
    std::size_t tile_ld() const { return b_; }
    DataLayout data_layout() const { return data_layout_; }

    Storage* tile(std::size_t tile_i, std::size_t tile_j)
    {
        return data_.data() + tile_offset(tile_i, tile_j);
    }

    const Storage* tile(std::size_t tile_i, std::size_t tile_j) const
    {
        return data_.data() + tile_offset(tile_i, tile_j);
    }

    Storage block_value(std::size_t tile_i, std::size_t tile_j, std::size_t row, std::size_t col) const
    {
        if (row >= block_extent(tile_i) || col >= block_extent(tile_j)) {
            throw std::out_of_range("block entry is outside logical matrix extent");
        }
        return tile(tile_i, tile_j)[dense_index(row, col, b_, data_layout_)];
    }

    Storage at(std::size_t row, std::size_t col) const
    {
        if (row >= m_ || col >= m_) {
            throw std::out_of_range("matrix index out of range");
        }
        return block_value(row / b_, col / b_, row % b_, col % b_);
    }

    void fill_stable_upper(unsigned seed, double diagonal_shift = 10.0)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> offdiag_dist(-0.01, 0.01);
        std::uniform_real_distribution<double> diag_dist(0.0, 1.0);

        for (std::size_t ti = 0; ti < p_; ++ti) {
            for (std::size_t tj = 0; tj < p_; ++tj) {
                Storage* t = tile(ti, tj);
                const std::size_t bi = block_extent(ti);
                const std::size_t bj = block_extent(tj);
                for (std::size_t ii = 0; ii < bi; ++ii) {
                    for (std::size_t jj = 0; jj < bj; ++jj) {
                        const std::size_t row = ti * b_ + ii;
                        const std::size_t col = tj * b_ + jj;
                        t[dense_index(ii, jj, b_, data_layout_)] = make_upper_value(row, col, gen, offdiag_dist, diag_dist, diagonal_shift);
                    }
                }
            }
        }
    }

private:
    static void validate_shape(std::size_t m, std::size_t b)
    {
        if (b == 0) {
            throw std::invalid_argument("block size b must be positive");
        }
        if (m == 0) {
            throw std::invalid_argument("matrix size m must be positive");
        }
    }

    static Storage make_upper_value(std::size_t row,
                                    std::size_t col,
                                    std::mt19937& gen,
                                    std::uniform_real_distribution<double>& offdiag_dist,
                                    std::uniform_real_distribution<double>& diag_dist,
                                    double diagonal_shift)
    {
        double value = 0.0;
        if (col > row) {
            value = offdiag_dist(gen);
        } else if (col == row) {
            value = diagonal_shift + diag_dist(gen);
        }
        return static_cast<Storage>(value);
    }

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
    DataLayout data_layout_;
    std::vector<Storage> data_;
};

template <typename Storage>
class ContiguousUpperMatrix {
public:
    using value_type = Storage;

    ContiguousUpperMatrix(std::size_t m, std::size_t b, DataLayout data_layout)
        : m_(m), b_(b), p_(b == 0 ? 0 : ceil_div(m, b)), data_layout_(data_layout)
    {
        if (b == 0) {
            throw std::invalid_argument("block size b must be positive");
        }
        if (m == 0) {
            throw std::invalid_argument("matrix size m must be positive");
        }
        data_.assign(m_ * m_, Storage{0});
    }

    std::size_t m() const { return m_; }
    std::size_t b() const { return b_; }
    std::size_t p() const { return p_; }
    std::size_t size() const { return m_; }
    std::size_t block_size() const { return b_; }
    std::size_t num_blocks() const { return p_; }
    std::size_t block_begin(std::size_t k) const { return k * b_; }
    std::size_t block_extent(std::size_t k) const
    {
        const std::size_t begin = block_begin(k);
        if (begin >= m_) {
            throw std::out_of_range("block index out of range");
        }
        return std::min(b_, m_ - begin);
    }
    std::size_t tile_ld() const { return m_; }
    DataLayout data_layout() const { return data_layout_; }

    const Storage* data() const { return data_.data(); }

    Storage at(std::size_t row, std::size_t col) const
    {
        if (row >= m_ || col >= m_) {
            throw std::out_of_range("matrix index out of range");
        }
        return data_[dense_index(row, col, m_, data_layout_)];
    }

    Storage block_value(std::size_t tile_i, std::size_t tile_j, std::size_t row, std::size_t col) const
    {
        if (row >= block_extent(tile_i) || col >= block_extent(tile_j)) {
            throw std::out_of_range("block entry is outside logical matrix extent");
        }
        return at(tile_i * b_ + row, tile_j * b_ + col);
    }

    void fill_stable_upper(unsigned seed, double diagonal_shift = 10.0)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> offdiag_dist(-0.01, 0.01);
        std::uniform_real_distribution<double> diag_dist(0.0, 1.0);

        for (std::size_t row = 0; row < m_; ++row) {
            for (std::size_t col = 0; col < m_; ++col) {
                double value = 0.0;
                if (col > row) {
                    value = offdiag_dist(gen);
                } else if (col == row) {
                    value = diagonal_shift + diag_dist(gen);
                }
                data_[dense_index(row, col, m_, data_layout_)] = static_cast<Storage>(value);
            }
        }
    }

private:
    std::size_t m_;
    std::size_t b_;
    std::size_t p_;
    DataLayout data_layout_;
    std::vector<Storage> data_;
};
