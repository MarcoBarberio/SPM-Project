#include "hpc_helpers.hpp"
#include "mapping_baseline.hpp"
#include "utilities.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <k> <seed> <warmup> [--json] [--print]\n";
        return 1;
    }

    const size_t n = std::stoull(argv[1]);
    const int k = std::stoi(argv[2]);
    const int seed = std::stoi(argv[3]);
    const int warmup = std::stoi(argv[4]);

    bool json = false;
    bool print_array = false;

    for (int i = 5; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--json")
            json = true;
        if (arg == "--print")
            print_array = true;
    }

    auto keys = generate_random_keys(n, seed);
    auto [a, b] = generate_params(seed);

    std::vector<uint64_t> mapping(n);

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        generate_mapping_baseline(keys, mapping, n, k, a, b);
    }

    TIMERSTART(run)
    generate_mapping_baseline(keys, mapping, n, k, a, b);
    TIMERSTOP(run)

    double time = elapsed_run;
    uint64_t chk = checksum(mapping);

    if (json)
    {
        std::string filename = "result_n" + std::to_string(n) + "_k" + std::to_string(k) + ".json";

        std::ofstream file(filename);

        file << "{\n";
        file << "  \"n\": " << n << ",\n";
        file << "  \"k\": " << k << ",\n";
        file << "  \"seed\": " << seed << ",\n";
        file << "  \"warmup\": " << warmup << ",\n";
        file << "  \"time\": " << time << ",\n";
        file << "  \"checksum\": " << chk;

        if (print_array && n < 500)
        {
            file << ",\n  \"mapping\": [";
            for (size_t i = 0; i < n; ++i)
            {
                file << mapping[i];
                if (i != n - 1)
                    file << ", ";
            }
            file << "]";
        }

        file << "\n}\n";
        file.close();

        std::cout << "JSON file written to " << filename << "\n";
    }
    else
    {
        std::cout << "n = " << n << "\n";
        std::cout << "k = " << k << "\n";
        std::cout << "seed = " << seed << "\n";
        std::cout << "warmup = " << warmup << "\n";
        std::cout << "time = " << time << " s\n";
        std::cout << "checksum = " << chk << "\n";

        if (print_array && n < 500)
        {
            std::cout << "mapping = [";
            for (size_t i = 0; i < n; ++i)
            {
                std::cout << mapping[i];
                if (i != n - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }

    return 0;
}