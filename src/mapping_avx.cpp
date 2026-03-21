#include "utilities.hpp"
#include <vector>
#include <cstdint>
#include "mapping_avx.hpp"
#include <immintrin.h>
#include <cstddef>

/** Generates a hash mapping for the given keys using the multiply-shift-add hash function using AVX2 instructions. */
void generate_mapping_avx(
    const std::vector<uint64_t> &data,
    std::vector<uint64_t> &mapping,
    size_t n,
    int k,
    uint64_t a,
    uint64_t b
) {
    const int shift = 64 - k;
    size_t i = 0;

    const uint64_t* __restrict__ in  = data.data();
    uint64_t* __restrict__ out = mapping.data();

    __m256i va = _mm256_set1_epi64x(a);
    __m256i vb = _mm256_set1_epi64x(b);

    for (; i + 4 <= n; i += 4) {
        __m256i x = _mm256_loadu_si256((__m256i const*)(in + i));
        __m256i mul = _mm256_mullo_epi64(x, va);
        __m256i add = _mm256_add_epi64(mul, vb);
        __m256i res = _mm256_srli_epi64(add, shift);
        _mm256_storeu_si256((__m256i*)(out + i), res);
    }

    // tail loop
    for (; i < n; ++i) {
        out[i] = (a * in[i] + b) >> shift;
    }
}
    
