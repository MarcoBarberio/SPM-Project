#include "hash.hpp"
#include "hpc_helpers.hpp"
#include "mapping.hpp"
#include <vector>
#include <iostream>

int main() {
    std::vector<uint64_t> keys = {1,2,3,4,5};
    int k = 16;

    TIMERSTART(run);
    auto mapping = generate_mapping(keys, k);
    TIMERSTOP(run);

    for (auto v : mapping) {
        std::cout << v << "\n";
    }

    return 0;
}