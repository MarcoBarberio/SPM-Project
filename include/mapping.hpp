#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <vector>
#include <cstdint>

void generate_mapping(
    const uint64_t* __restrict__ in,
    uint64_t* __restrict__ out,
    size_t n,
    int k,
    uint64_t a,
    uint64_t b
);

#endif