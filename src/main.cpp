#include "blocked_trsv.hpp"
#include "reference_trsv.hpp"
#include "tiled_matrix.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::size_t m = 4096;
    int repeat = 5;
    std::string output = "results/ch4_minimal.csv";
    std::vector<std::size_t> blocks = {16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512};
    bool explicit_block_list = false;
    bool breakdown = false;
    bool tree_parallel = false;
};

struct TreeTiming {
    double time_ms = 0.0;
    int num_threads = 0;
};

std::size_t parse_size(const std::string& text, const std::string& name)
{
    try {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != text.size() || value == 0) {
            throw std::invalid_argument("bad value");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
}

int parse_int(const std::string& text, const std::string& name)
{
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size() || value <= 0) {
            throw std::invalid_argument("bad value");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
}

std::vector<std::size_t> parse_blocks(const std::string& text)
{
    std::vector<std::size_t> blocks;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? comma : comma - start);
        blocks.push_back(parse_size(token, "block size"));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (blocks.empty()) {
        throw std::invalid_argument("block list must not be empty");
    }
    return blocks;
}

Options parse_options(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(flag + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--m") {
            opt.m = parse_size(require_value(arg), "m");
        } else if (arg == "--repeat") {
            opt.repeat = parse_int(require_value(arg), "repeat");
        } else if (arg == "--output") {
            opt.output = require_value(arg);
        } else if (arg == "--b") {
            opt.blocks = parse_blocks(require_value(arg));
            opt.explicit_block_list = true;
        } else if (arg == "--breakdown") {
            opt.breakdown = true;
        } else if (arg == "--tree-parallel") {
            opt.tree_parallel = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: blocked_trsv_benchmark --tree-parallel\n"
                         "Legacy options: [--m 4096] [--repeat 5] [--b 16,32,64] "
                         "[--output results/ch4_minimal.csv] [--breakdown]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return opt;
}

void write_row(std::ofstream& csv,
               const std::string& mode,
               MatrixLayout matrix_layout,
               DataLayout data_layout,
               const std::string& looking,
               std::size_t m,
               std::size_t b,
               int repeat,
               double time_ms,
               double gflops,
               double rel_error)
{
    csv << mode << ','
        << to_string(matrix_layout) << ','
        << to_string(data_layout) << ','
        << looking << ','
        << m << ','
        << b << ','
        << repeat << ','
        << std::setprecision(10) << time_ms << ','
        << std::setprecision(10) << gflops << ','
        << std::setprecision(10) << rel_error << '\n';
}

void write_breakdown_row(std::ofstream& csv,
                         const std::string& mode,
                         MatrixLayout matrix_layout,
                         DataLayout data_layout,
                         Looking looking,
                         std::size_t m,
                         std::size_t b,
                         int repeat,
                         const BreakdownStats& stats,
                         double rel_error)
{
    csv << mode << ','
        << to_string(matrix_layout) << ','
        << to_string(data_layout) << ','
        << to_string(looking) << ','
        << m << ','
        << b << ','
        << repeat << ','
        << std::setprecision(10) << stats.total_ms << ','
        << std::setprecision(10) << stats.prepare_ms << ','
        << std::setprecision(10) << stats.trsv_ms << ','
        << std::setprecision(10) << stats.gemv_ms << ','
        << std::setprecision(10) << rel_error << '\n';
}

void write_tree_row(std::ofstream& csv,
                    const std::string& mode,
                    MatrixLayout matrix_layout,
                    const std::string& looking,
                    std::size_t m,
                    std::size_t b,
                    int repeat,
                    int num_systems,
                    int num_threads,
                    double time_ms,
                    double gflops,
                    double max_rel_error)
{
    csv << mode << ','
        << to_string(matrix_layout) << ','
        << to_string(DataLayout::ColMajor) << ','
        << looking << ','
        << m << ','
        << b << ','
        << repeat << ','
        << num_systems << ','
        << num_threads << ','
        << std::setprecision(10) << time_ms << ','
        << std::setprecision(10) << gflops << ','
        << std::setprecision(10) << max_rel_error << '\n';
}

template <typename Matrix, typename Compute>
void run_blocked_repeats(const std::string& mode,
                         MatrixLayout matrix_layout,
                         Looking looking,
                         const Matrix& u,
                         int repeat,
                         std::ofstream& csv)
{
    const std::size_t m = u.m();
    const std::size_t b = u.b();
    const std::vector<Compute> x_true = make_x_true<Compute>(m, 67890);
    const std::vector<Compute> y = upper_matvec<Matrix, Compute>(u, x_true);
    const double flops = static_cast<double>(m) * static_cast<double>(m);

    for (int r = 0; r < repeat; ++r) {
        std::vector<Compute> x = y;

        const auto start = std::chrono::steady_clock::now();
        blocked_trsv<Matrix, Compute>(u, x, looking);
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = stop - start;
        const double seconds = elapsed.count();
        const double time_ms = seconds * 1000.0;
        const double gflops = flops / seconds / 1.0e9;
        const double rel_error = relative_error(x, x_true);

        write_row(csv, mode, matrix_layout, u.data_layout(), to_string(looking),
                  m, b, r, time_ms, gflops, rel_error);
    }
}

template <typename Storage, typename Compute>
void run_blocked_case(const std::string& mode,
                      MatrixLayout matrix_layout,
                      DataLayout data_layout,
                      Looking looking,
                      std::size_t m,
                      std::size_t b,
                      int repeat,
                      std::ofstream& csv)
{
    if (matrix_layout == MatrixLayout::Tiled) {
        TiledUpperMatrix<Storage> u(m, b, data_layout);
        u.fill_stable_upper(12345);
        run_blocked_repeats<TiledUpperMatrix<Storage>, Compute>(mode, matrix_layout, looking, u, repeat, csv);
    } else {
        ContiguousUpperMatrix<Storage> u(m, b, data_layout);
        u.fill_stable_upper(12345);
        run_blocked_repeats<ContiguousUpperMatrix<Storage>, Compute>(mode, matrix_layout, looking, u, repeat, csv);
    }
}

template <typename Compute>
void run_direct_case(const std::string& mode,
                     MatrixLayout panel_layout,
                     DataLayout data_layout,
                     std::size_t m,
                     std::size_t b,
                     int repeat,
                     std::ofstream& csv)
{
    ContiguousUpperMatrix<Compute> u(m, b, data_layout);
    u.fill_stable_upper(12345);

    const std::vector<Compute> x_true = make_x_true<Compute>(m, 67890);
    const std::vector<Compute> y = upper_matvec<ContiguousUpperMatrix<Compute>, Compute>(u, x_true);
    const double flops = static_cast<double>(m) * static_cast<double>(m);

    for (int r = 0; r < repeat; ++r) {
        std::vector<Compute> x = y;

        const auto start = std::chrono::steady_clock::now();
        direct_trsv<Compute>(u, x);
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = stop - start;
        const double seconds = elapsed.count();
        const double time_ms = seconds * 1000.0;
        const double gflops = flops / seconds / 1.0e9;
        const double rel_error = relative_error(x, x_true);

        write_row(csv, mode, panel_layout, data_layout, "direct",
                  m, b, r, time_ms, gflops, rel_error);
    }
}

void run_all_for_block(std::size_t m, std::size_t b, int repeat, std::ofstream& csv)
{
    const std::vector<MatrixLayout> matrix_layouts = {MatrixLayout::Contiguous, MatrixLayout::Tiled};
    const std::vector<DataLayout> data_layouts = {DataLayout::ColMajor, DataLayout::RowMajor};
    const std::vector<Looking> lookings = {Looking::Left, Looking::Right};

    for (const MatrixLayout matrix_layout : matrix_layouts) {
        for (const DataLayout data_layout : data_layouts) {
            for (const Looking looking : lookings) {
                std::cerr << "running b=" << b
                          << " matrix_layout=" << to_string(matrix_layout)
                          << " data_layout=" << to_string(data_layout)
                          << " looking=" << to_string(looking)
                          << " mode=fp64\n";
                run_blocked_case<double, double>("fp64", matrix_layout, data_layout, looking, m, b, repeat, csv);

                std::cerr << "running b=" << b
                          << " matrix_layout=" << to_string(matrix_layout)
                          << " data_layout=" << to_string(data_layout)
                          << " looking=" << to_string(looking)
                          << " mode=fp32_fp64\n";
                run_blocked_case<float, double>("fp32_fp64", matrix_layout, data_layout, looking, m, b, repeat, csv);

                std::cerr << "running b=" << b
                          << " matrix_layout=" << to_string(matrix_layout)
                          << " data_layout=" << to_string(data_layout)
                          << " looking=" << to_string(looking)
                          << " mode=fp32\n";
                run_blocked_case<float, float>("fp32", matrix_layout, data_layout, looking, m, b, repeat, csv);
            }

            // Direct MKL baselines are full contiguous solves; rows are duplicated per panel for plotting.
            run_direct_case<double>("direct_fp64_mkl", matrix_layout, data_layout, m, b, repeat, csv);
            run_direct_case<float>("direct_fp32_mkl", matrix_layout, data_layout, m, b, repeat, csv);
        }
    }
}

template <typename Storage, typename Compute>
void run_breakdown_mode(const std::string& mode,
                        DataLayout data_layout,
                        std::size_t m,
                        std::size_t b,
                        int repeat,
                        std::ofstream& csv)
{
    TiledUpperMatrix<Storage> u(m, b, data_layout);
    u.fill_stable_upper(12345);

    const std::vector<Compute> x_true = make_x_true<Compute>(m, 67890);
    const std::vector<Compute> y = upper_matvec<TiledUpperMatrix<Storage>, Compute>(u, x_true);

    for (int r = 0; r < repeat; ++r) {
        std::vector<Compute> x = y;
        const BreakdownStats stats = blocked_trsv_right_looking_breakdown<TiledUpperMatrix<Storage>, Compute>(u, x);
        const double rel_error = relative_error(x, x_true);

        write_breakdown_row(csv, mode, MatrixLayout::Tiled, data_layout, Looking::Right,
                            m, b, r, stats, rel_error);
    }
}

void run_breakdown(std::size_t m)
{
    const std::vector<std::size_t> blocks = {32, 64, 128, 256, 512};
    const std::vector<DataLayout> data_layouts = {DataLayout::RowMajor, DataLayout::ColMajor};
    constexpr int repeat = 3;

    std::ofstream csv("results/ch4_breakdown.csv");
    if (!csv) {
        throw std::runtime_error("failed to open output CSV: results/ch4_breakdown.csv");
    }

    csv << "mode,matrix_layout,data_layout,looking,m,b,repeat,total_ms,prepare_ms,trsv_ms,gemv_ms,rel_error\n";

    for (const std::size_t b : blocks) {
        if (m % b != 0) {
            std::cerr << "warning: skipping breakdown block size because m=" << m
                      << " is not divisible by b=" << b << '\n';
            continue;
        }

        for (const DataLayout data_layout : data_layouts) {
            std::cerr << "breakdown b=" << b
                      << " matrix_layout=tiled"
                      << " data_layout=" << to_string(data_layout)
                      << " looking=right"
                      << " mode=fp64\n";
            run_breakdown_mode<double, double>("fp64", data_layout, m, b, repeat, csv);

            std::cerr << "breakdown b=" << b
                      << " matrix_layout=tiled"
                      << " data_layout=" << to_string(data_layout)
                      << " looking=right"
                      << " mode=fp32_fp64\n";
            run_breakdown_mode<float, double>("fp32_fp64", data_layout, m, b, repeat, csv);

            std::cerr << "breakdown b=" << b
                      << " matrix_layout=tiled"
                      << " data_layout=" << to_string(data_layout)
                      << " looking=right"
                      << " mode=fp32\n";
            run_breakdown_mode<float, float>("fp32", data_layout, m, b, repeat, csv);
        }
    }
}

template <typename Storage, typename Compute>
struct TiledTreeSystem {
    TiledUpperMatrix<Storage> u;
    std::vector<Compute> x_true;
    std::vector<Compute> y;
    std::vector<Compute> x;

    TiledTreeSystem(std::size_t m, std::size_t b, int system_id)
        : u(m, b, DataLayout::ColMajor),
          x_true(make_x_true<Compute>(m, static_cast<unsigned>(67890 + system_id)))
    {
        u.fill_stable_upper(static_cast<unsigned>(12345 + system_id));
        y = upper_matvec<TiledUpperMatrix<Storage>, Compute>(u, x_true);
        x = y;
    }
};

template <typename Compute>
struct DirectTreeSystem {
    ContiguousUpperMatrix<Compute> u;
    std::vector<Compute> x_true;
    std::vector<Compute> y;
    std::vector<Compute> x;

    DirectTreeSystem(std::size_t m, int system_id)
        : u(m, m, DataLayout::ColMajor),
          x_true(make_x_true<Compute>(m, static_cast<unsigned>(67890 + system_id)))
    {
        u.fill_stable_upper(static_cast<unsigned>(12345 + system_id));
        y = upper_matvec<ContiguousUpperMatrix<Compute>, Compute>(u, x_true);
        x = y;
    }
};

template <typename Storage, typename Compute>
void run_tree_blocked_mode(const std::string& mode,
                           std::size_t m,
                           std::size_t b,
                           int repeat,
                           int warmup,
                           int num_systems,
                           std::ofstream& csv)
{
    std::vector<TiledTreeSystem<Storage, Compute>> systems;
    systems.reserve(static_cast<std::size_t>(num_systems));
    for (int system_id = 0; system_id < num_systems; ++system_id) {
        systems.emplace_back(m, b, system_id);
    }

    const double flops = static_cast<double>(num_systems) * static_cast<double>(m) * static_cast<double>(m);

    auto solve_all_systems = [&]() {
        for (auto& system : systems) {
            system.x = system.y;
        }

        TreeTiming timing;
        const auto start = std::chrono::steady_clock::now();

#pragma omp parallel
        {
#pragma omp single
            timing.num_threads = omp_get_num_threads();

#pragma omp for schedule(static)
            for (int system_id = 0; system_id < num_systems; ++system_id) {
                blocked_trsv_right_looking<TiledUpperMatrix<Storage>, Compute>(
                    systems[static_cast<std::size_t>(system_id)].u,
                    systems[static_cast<std::size_t>(system_id)].x);
            }
        }

        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = stop - start;
        timing.time_ms = elapsed.count() * 1000.0;
        return timing;
    };

    for (int w = 0; w < warmup; ++w) {
        (void)solve_all_systems();
    }

    for (int r = 0; r < repeat; ++r) {
        const TreeTiming timing = solve_all_systems();
        const double seconds = timing.time_ms / 1000.0;
        const double gflops = flops / seconds / 1.0e9;

        double max_rel_error = 0.0;
        for (const auto& system : systems) {
            max_rel_error = std::max(max_rel_error, relative_error(system.x, system.x_true));
        }

        write_tree_row(csv, mode, MatrixLayout::Tiled, "right", m, b, r,
                       num_systems, timing.num_threads, timing.time_ms, gflops, max_rel_error);
    }
}

template <typename Compute>
void run_tree_direct_mode(const std::string& mode,
                          std::size_t m,
                          std::size_t b,
                          int repeat,
                          int warmup,
                          int num_systems,
                          std::ofstream& csv)
{
    std::vector<DirectTreeSystem<Compute>> systems;
    systems.reserve(static_cast<std::size_t>(num_systems));
    for (int system_id = 0; system_id < num_systems; ++system_id) {
        systems.emplace_back(m, system_id);
    }

    const double flops = static_cast<double>(num_systems) * static_cast<double>(m) * static_cast<double>(m);

    auto solve_all_systems = [&]() {
        for (auto& system : systems) {
            system.x = system.y;
        }

        TreeTiming timing;
        const auto start = std::chrono::steady_clock::now();

#pragma omp parallel
        {
#pragma omp single
            timing.num_threads = omp_get_num_threads();

#pragma omp for schedule(static)
            for (int system_id = 0; system_id < num_systems; ++system_id) {
                direct_trsv<Compute>(
                    systems[static_cast<std::size_t>(system_id)].u,
                    systems[static_cast<std::size_t>(system_id)].x);
            }
        }

        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = stop - start;
        timing.time_ms = elapsed.count() * 1000.0;
        return timing;
    };

    for (int w = 0; w < warmup; ++w) {
        (void)solve_all_systems();
    }

    for (int r = 0; r < repeat; ++r) {
        const TreeTiming timing = solve_all_systems();
        const double seconds = timing.time_ms / 1000.0;
        const double gflops = flops / seconds / 1.0e9;

        double max_rel_error = 0.0;
        for (const auto& system : systems) {
            max_rel_error = std::max(max_rel_error, relative_error(system.x, system.x_true));
        }

        write_tree_row(csv, mode, MatrixLayout::Contiguous, "direct", m, b, r,
                       num_systems, timing.num_threads, timing.time_ms, gflops, max_rel_error);
    }
}

void run_tree_parallel(std::size_t m)
{
    const std::vector<std::size_t> blocks = {16, 32, 64, 128, 256, 512};
    constexpr int repeat = 10;
    constexpr int warmup = 2;
    const int num_systems = omp_get_max_threads();

    std::ofstream csv("results/ch4_tree_parallel.csv");
    if (!csv) {
        throw std::runtime_error("failed to open output CSV: results/ch4_tree_parallel.csv");
    }

    csv << "mode,matrix_layout,data_layout,looking,m,b,repeat,num_systems,num_threads,time_ms,gflops,max_rel_error\n";

    for (const std::size_t b : blocks) {
        if (m % b != 0) {
            throw std::invalid_argument("tree-parallel benchmark requires m % b == 0");
        }

        std::cerr << "tree-parallel b=" << b << " mode=fp64 warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_blocked_mode<double, double>("fp64", m, b, repeat, warmup, num_systems, csv);

        std::cerr << "tree-parallel b=" << b << " mode=fp32_fp64 warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_blocked_mode<float, double>("fp32_fp64", m, b, repeat, warmup, num_systems, csv);

        std::cerr << "tree-parallel b=" << b << " mode=fp32 warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_blocked_mode<float, float>("fp32", m, b, repeat, warmup, num_systems, csv);

        std::cerr << "tree-parallel b=" << b << " mode=direct_fp64_mkl warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_direct_mode<double>("direct_fp64_mkl", m, b, repeat, warmup, num_systems, csv);

        std::cerr << "tree-parallel b=" << b << " mode=direct_fp32_mkl warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_direct_mode<float>("direct_fp32_mkl", m, b, repeat, warmup, num_systems, csv);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_options(argc, argv);

        if (opt.breakdown) {
            run_breakdown(opt.m);
            return 0;
        }

        if (opt.tree_parallel) {
            run_tree_parallel(opt.m);
            return 0;
        }

        std::ofstream csv(opt.output);
        if (!csv) {
            throw std::runtime_error("failed to open output CSV: " + opt.output);
        }

        csv << "mode,matrix_layout,data_layout,looking,m,b,repeat,time_ms,gflops,rel_error\n";

        for (const std::size_t b : opt.blocks) {
            if (opt.m % b != 0) {
                const std::string message = "m=" + std::to_string(opt.m)
                    + " is not divisible by b=" + std::to_string(b)
                    + "; this implementation requires m % b == 0";
                if (opt.explicit_block_list) {
                    throw std::invalid_argument(message);
                }
                std::cerr << "warning: skipping block size: " << message << '\n';
                continue;
            }

            run_all_for_block(opt.m, b, opt.repeat, csv);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
