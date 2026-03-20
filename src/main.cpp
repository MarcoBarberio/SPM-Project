#include "utilities.hpp"
#include "mapping.hpp"
#include "hpc_helpers.hpp"
#include <vector>
#include <iostream>

#include <iostream>
#include <cstdlib>   // std::stoull, std::stoi

int main(int argc, char* argv[]) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <n> <k> <seed> <warmup>\n";
        return 1;
    }

    const size_t n = std::stoull(argv[1]);   
    const int k = std::stoi(argv[2]);  
    const int seed = std::stoi(argv[3]);     
    const int warmup = std::stoi(argv[4]);   

    auto keys = generate_random_keys(n, seed);
    auto [a, b] = generate_params(seed);

    std::vector<uint64_t> mapping(n);

    const uint64_t* in = keys.data();
    uint64_t* out = mapping.data();

    for (int i = 0; i < warmup; ++i) {
        generate_mapping(in, out, n, k, a, b);
    }

    TIMERSTART(run)
    generate_mapping(in, out, n, k, a, b);
    TIMERSTOP(run)

    std::cout << "n = " << n << "\n";
    std::cout << "k = " << k << "\n";
    std::cout << "seed = " << seed << "\n";
    std::cout << "warmup = " << warmup << "\n";

    return 0;
}