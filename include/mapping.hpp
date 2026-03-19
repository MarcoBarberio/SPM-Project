#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <vector>
#include <cstdint>

std::vector<uint64_t> generate_mapping(
    const std::vector<uint64_t>& keys,
    int k
);

#endif