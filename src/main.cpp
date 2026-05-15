#include "blocked_trsv.hpp"
#include "reference_trsv.hpp"
#include "tiled_matrix.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
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
    int warmup = 0;
    std::string output = "results/ch4_tree_parallel.csv";
    std::vector<std::size_t> blocks = {16, 32, 64, 128, 256, 512};
    bool tree_parallel = false;
    bool eight_panels = false;
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

int parse_nonnegative_int(const std::string& text, const std::string& name)
{
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size() || value < 0) {
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
        } else if (arg == "--warmup") {
            opt.warmup = parse_nonnegative_int(require_value(arg), "warmup");
        } else if (arg == "--output") {
            opt.output = require_value(arg);
        } else if (arg == "--b") {
            opt.blocks = parse_blocks(require_value(arg));
        } else if (arg == "--tree-parallel") {
            opt.tree_parallel = true;
        } else if (arg == "--eight-panels") {
            opt.eight_panels = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: blocked_trsv_benchmark --tree-parallel | --eight-panels\n"
                         "Options: [--m 4096] [--b 16,32,64] [--repeat 10] [--warmup 2] "
                         "[--output results/ch4_tree_parallel.csv]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return opt;
}

void ensure_parent_directory(const std::string& output)
{
    const std::filesystem::path path(output);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_tree_row(std::ofstream& csv,
                    const std::string& mode,
                    MatrixLayout matrix_layout,
                    DataLayout data_layout,
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
        << to_string(data_layout) << ','
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
struct TreeSystem {
    Matrix u;
    std::vector<Compute> x_true;
    std::vector<Compute> y;
    std::vector<Compute> x;

    TreeSystem(std::size_t m, std::size_t b, DataLayout data_layout, int system_id)
        : u(m, b, data_layout),
          x_true(make_x_true<Compute>(m, static_cast<unsigned>(67890 + system_id)))
    {
        u.fill_stable_upper(static_cast<unsigned>(12345 + system_id));
        y = upper_matvec<Matrix, Compute>(u, x_true);
        x = y;
    }
};

template <typename Compute>
struct DirectTreeSystem {
    ContiguousUpperMatrix<Compute> u;
    std::vector<Compute> x_true;
    std::vector<Compute> y;
    std::vector<Compute> x;

    DirectTreeSystem(std::size_t m, DataLayout data_layout, int system_id)
        : u(m, m, data_layout),
          x_true(make_x_true<Compute>(m, static_cast<unsigned>(67890 + system_id)))
    {
        u.fill_stable_upper(static_cast<unsigned>(12345 + system_id));
        y = upper_matvec<ContiguousUpperMatrix<Compute>, Compute>(u, x_true);
        x = y;
    }
};

template <typename Matrix, typename Compute>
void run_tree_blocked_mode(const std::string& mode,
                           MatrixLayout matrix_layout,
                           DataLayout data_layout,
                           Looking looking,
                           std::size_t m,
                           std::size_t b,
                           int repeat,
                           int warmup,
                           int num_systems,
                           std::ofstream& csv)
{
    std::vector<TreeSystem<Matrix, Compute>> systems;
    systems.reserve(static_cast<std::size_t>(num_systems));
    for (int system_id = 0; system_id < num_systems; ++system_id) {
        systems.emplace_back(m, b, data_layout, system_id);
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
                blocked_trsv<Matrix, Compute>(
                    systems[static_cast<std::size_t>(system_id)].u,
                    systems[static_cast<std::size_t>(system_id)].x,
                    looking);
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

        write_tree_row(csv, mode, matrix_layout, data_layout, to_string(looking), m, b, r,
                       num_systems, timing.num_threads, timing.time_ms, gflops, max_rel_error);
    }
}

template <typename Compute>
void run_tree_direct_mode(const std::string& mode,
                          DataLayout data_layout,
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
        systems.emplace_back(m, data_layout, system_id);
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

        write_tree_row(csv, mode, MatrixLayout::Contiguous, data_layout, "direct", m, b, r,
                       num_systems, timing.num_threads, timing.time_ms, gflops, max_rel_error);
    }
}

void validate_tree_options(std::size_t m, const std::vector<std::size_t>& blocks, int repeat, int warmup)
{
    if (repeat <= 0) {
        throw std::invalid_argument("tree-parallel repeat must be > 0");
    }
    if (warmup < 0) {
        throw std::invalid_argument("tree-parallel warmup must be >= 0");
    }
    if (blocks.empty()) {
        throw std::invalid_argument("tree-parallel block list must not be empty");
    }
    for (const std::size_t b : blocks) {
        if (m % b != 0) {
            throw std::invalid_argument("tree-parallel benchmark requires m % b == 0; got m="
                                        + std::to_string(m) + ", b=" + std::to_string(b));
        }
    }
}

template <typename Matrix64, typename Matrix32>
void run_tree_blocked_panel(MatrixLayout matrix_layout,
                            DataLayout data_layout,
                            Looking looking,
                            std::size_t m,
                            std::size_t b,
                            int repeat,
                            int warmup,
                            int num_systems,
                            std::ofstream& csv)
{
    std::cerr << "[progress] tree-parallel b=" << b << " matrix_layout=" << to_string(matrix_layout)
              << " data_layout=" << to_string(data_layout) << " looking=" << to_string(looking)
              << " mode=fp64 warmup=" << warmup << " repeat=" << repeat << '\n';
    run_tree_blocked_mode<Matrix64, double>("fp64", matrix_layout, data_layout, looking, m, b, repeat, warmup, num_systems, csv);

    std::cerr << "[progress] tree-parallel b=" << b << " matrix_layout=" << to_string(matrix_layout)
              << " data_layout=" << to_string(data_layout) << " looking=" << to_string(looking)
              << " mode=fp32_fp64 warmup=" << warmup << " repeat=" << repeat << '\n';
    run_tree_blocked_mode<Matrix32, double>("fp32_fp64", matrix_layout, data_layout, looking, m, b, repeat, warmup, num_systems, csv);

    std::cerr << "[progress] tree-parallel b=" << b << " matrix_layout=" << to_string(matrix_layout)
              << " data_layout=" << to_string(data_layout) << " looking=" << to_string(looking)
              << " mode=fp32 warmup=" << warmup << " repeat=" << repeat << '\n';
    run_tree_blocked_mode<Matrix32, float>("fp32", matrix_layout, data_layout, looking, m, b, repeat, warmup, num_systems, csv);
}

void run_tree_panel(MatrixLayout matrix_layout,
                    DataLayout data_layout,
                    Looking looking,
                    std::size_t m,
                    const std::vector<std::size_t>& blocks,
                    int repeat,
                    int warmup,
                    std::ofstream& csv)
{
    const int num_systems = omp_get_max_threads();

    for (const std::size_t b : blocks) {
        if (matrix_layout == MatrixLayout::Tiled) {
            run_tree_blocked_panel<TiledUpperMatrix<double>, TiledUpperMatrix<float>>(
                matrix_layout, data_layout, looking, m, b, repeat, warmup, num_systems, csv);
        } else {
            run_tree_blocked_panel<ContiguousUpperMatrix<double>, ContiguousUpperMatrix<float>>(
                matrix_layout, data_layout, looking, m, b, repeat, warmup, num_systems, csv);
        }

        // For direct MKL baselines, b is only a plotting/grouping label.
        std::cerr << "[progress] tree-parallel b=" << b << " matrix_layout=" << to_string(matrix_layout)
                  << " data_layout=" << to_string(data_layout) << " looking=" << to_string(looking)
                  << " mode=direct_fp64_mkl warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_direct_mode<double>("direct_fp64_mkl", data_layout, m, b, repeat, warmup, num_systems, csv);

        std::cerr << "[progress] tree-parallel b=" << b << " matrix_layout=" << to_string(matrix_layout)
                  << " data_layout=" << to_string(data_layout) << " looking=" << to_string(looking)
                  << " mode=direct_fp32_mkl warmup=" << warmup << " repeat=" << repeat << '\n';
        run_tree_direct_mode<float>("direct_fp32_mkl", data_layout, m, b, repeat, warmup, num_systems, csv);
    }
}

void write_tree_header(std::ofstream& csv)
{
    csv << "mode,matrix_layout,data_layout,looking,m,b,repeat,num_systems,num_threads,time_ms,gflops,max_rel_error\n";
}

void run_tree_parallel(std::size_t m,
                       const std::vector<std::size_t>& blocks,
                       int repeat,
                       int warmup,
                       const std::string& output)
{
    validate_tree_options(m, blocks, repeat, warmup);
    ensure_parent_directory(output);

    std::ofstream csv(output);
    if (!csv) {
        throw std::runtime_error("failed to open output CSV: " + output);
    }
    write_tree_header(csv);

    run_tree_panel(MatrixLayout::Tiled, DataLayout::ColMajor, Looking::Right,
                   m, blocks, repeat, warmup, csv);
}

void run_eight_panels(std::size_t m,
                      const std::vector<std::size_t>& blocks,
                      int repeat,
                      int warmup,
                      const std::string& output)
{
    validate_tree_options(m, blocks, repeat, warmup);
    ensure_parent_directory(output);

    std::ofstream csv(output);
    if (!csv) {
        throw std::runtime_error("failed to open output CSV: " + output);
    }
    write_tree_header(csv);

    const std::vector<MatrixLayout> matrix_layouts = {MatrixLayout::Tiled, MatrixLayout::Contiguous};
    const std::vector<DataLayout> data_layouts = {DataLayout::ColMajor, DataLayout::RowMajor};
    const std::vector<Looking> lookings = {Looking::Right, Looking::Left};

    for (const MatrixLayout matrix_layout : matrix_layouts) {
        for (const DataLayout data_layout : data_layouts) {
            for (const Looking looking : lookings) {
                run_tree_panel(matrix_layout, data_layout, looking, m, blocks, repeat, warmup, csv);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_options(argc, argv);

        if (opt.tree_parallel) {
            run_tree_parallel(opt.m, opt.blocks, opt.repeat, opt.warmup, opt.output);
            return 0;
        }

        if (opt.eight_panels) {
            run_eight_panels(opt.m, opt.blocks, opt.repeat, opt.warmup, opt.output);
            return 0;
        }

        throw std::invalid_argument("no active benchmark selected; use --tree-parallel or --eight-panels");
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
