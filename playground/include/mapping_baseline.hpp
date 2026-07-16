#ifndef MAPPING_BASELINE_HPP
#define MAPPING_BASELINE_HPP

#include <cstdint>
#include <vector>


void generate_mapping_baseline(const std::vector<uint64_t>& data, std::vector<uint64_t>& mapping, std::size_t n, int k,
                               uint64_t a, uint64_t b);

#endif