#include "blocked_trsv.hpp"
#include "reference_trsv.hpp"
#include "tiled_matrix.hpp"

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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: blocked_trsv_benchmark [--m 4096] [--repeat 5] "
                         "[--b 16,32,64] [--output results/ch4_minimal.csv]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return opt;
}

template <typename Storage, typename Compute>
void run_mode(const std::string& mode, std::size_t m, std::size_t b, int repeat, std::ofstream& csv)
{
    TiledUpperMatrix<Storage> u(m, b);
    u.fill_stable_upper(12345);

    const std::vector<Compute> x_true = make_x_true<Compute>(m, 67890);
    const std::vector<Compute> y = upper_matvec<Storage, Compute>(u, x_true);
    const double flops = static_cast<double>(m) * static_cast<double>(m);

    for (int r = 0; r < repeat; ++r) {
        std::vector<Compute> x = y;

        const auto start = std::chrono::steady_clock::now();
        blocked_trsv_right_looking<Storage, Compute>(u, x);
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = stop - start;
        const double seconds = elapsed.count();
        const double time_ms = seconds * 1000.0;
        const double gflops = flops / seconds / 1.0e9;
        const double rel_error = relative_error(x, x_true);

        csv << mode << ','
            << m << ','
            << b << ','
            << r << ','
            << std::setprecision(10) << time_ms << ','
            << std::setprecision(10) << gflops << ','
            << std::setprecision(10) << rel_error << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_options(argc, argv);

        std::ofstream csv(opt.output);
        if (!csv) {
            throw std::runtime_error("failed to open output CSV: " + opt.output);
        }

        csv << "mode,m,b,repeat,time_ms,gflops,rel_error\n";

        for (const std::size_t b : opt.blocks) {
            if (opt.m % b != 0) {
                const std::string message = "m=" + std::to_string(opt.m)
                    + " is not divisible by b=" + std::to_string(b)
                    + "; this minimal tiled implementation requires m % b == 0";
                if (opt.explicit_block_list) {
                    throw std::invalid_argument(message);
                }
                std::cerr << "warning: skipping block size: " << message << '\n';
                continue;
            }

            std::cerr << "running b=" << b << " mode=fp64\n";
            run_mode<double, double>("fp64", opt.m, b, opt.repeat, csv);

            std::cerr << "running b=" << b << " mode=fp32_fp64\n";
            run_mode<float, double>("fp32_fp64", opt.m, b, opt.repeat, csv);

            std::cerr << "running b=" << b << " mode=fp32\n";
            run_mode<float, float>("fp32", opt.m, b, opt.repeat, csv);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
