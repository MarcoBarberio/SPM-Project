#include "hash.hpp"
#include <vector>
#include <cstdint>
#include "mapping.hpp"
#include "hpc_helpers.hpp"
#define SEED 24
/**Generates a hash mapping for the given keys using the multiply-shift-add hash function. */
std::vector<uint64_t> generate_mapping(const std::vector<uint64_t>& keys, int k) {
    auto [a, b] = hash::generate_params(SEED);
    const auto size=keys.size();
    std::vector<uint64_t> mapping(size);
    size_t i=0;
    TIMERSTART(run)
    for (; i < size; ++i) {
        mapping[i] = hash::multiply_shift_add(keys[i], a, b, k);
    }
    TIMERSTOP(run)
    return mapping;
}

