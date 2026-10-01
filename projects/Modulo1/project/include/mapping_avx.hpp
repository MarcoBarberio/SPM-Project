#ifndef MAPPING_AVX_HPP
#define MAPPING_AVX_HPP

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <vector>

inline __m256i mul64_avx2(__m256i x, __m256i a);
void generate_mapping_avx(const std::vector<uint64_t>& data, std::vector<uint64_t>& mapping, size_t n, int k,
                          uint64_t a, uint64_t b);

#endif