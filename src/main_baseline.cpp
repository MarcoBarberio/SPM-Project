#include "hpc_helpers.hpp"
#include "mapping_baseline.hpp"
#include "utilities.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{

    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <k> <seed> <warmup> <csv>\n";
        return 1;
    }

    const size_t n = std::stoull(argv[1]);
    const int k = std::stoi(argv[2]);
    const int seed = std::stoi(argv[3]);
    const int warmup = std::stoi(argv[4]);
    const int csv = std::stoi(argv[5]);

    auto keys = generate_random_keys(n, seed);
    auto [a, b] = generate_params(seed);

    std::vector<uint64_t> mapping(n);

    for (int i = 0; i < warmup; ++i)
    {
        generate_mapping_baseline(keys, mapping, n, k, a, b);
    }
    TIMERSTART(run)
    generate_mapping_baseline(keys, mapping, n, k, a, b);
    TIMERSTOP(run)

    double time = elapsed_run;
    uint64_t chk = checksum(mapping);
    if (csv == 1)
    {
        std::cout << n << "," << k << "," << seed << "," << warmup << "," << time << "," << chk << "\n";
    }
    else
    {
        std::cout << "n = " << n << "\n";
        std::cout << "k = " << k << "\n";
        std::cout << "seed = " << seed << "\n";
        std::cout << "warmup = " << warmup << "\n";
        std::cout << "time = " << time << " s\n";
        std::cout << "checksum = " << chk << "\n";
    }
    return 0;
}