#include "hash.hpp"
#include <vector>
#include <cstdint>
std::vector<uint64_t> generate_mapping(const std::vector<uint64_t>& keys, int k) {
    uint64_t seed = 24;
    auto [a, b] = hash::generate_params(seed);
    std::vector<uint64_t> mapping(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        mapping[i] = hash::multiply_shift_add(keys[i], a, b, k);
    }

    return mapping;
}