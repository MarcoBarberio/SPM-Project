#include "hash.hpp"
#include "mapping.hpp"
#include "hpc_helpers.hpp"
#include <vector>
#include <iostream>

int main() {

    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 100000000; ++i) {
        keys.push_back(i);
    }
    int k = 64;

    auto mapping = generate_mapping(keys, k);
    
    return 0;
}