#include "mapping_avx.hpp"
#include "utilities.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <vector>

/** Compute 64-bit multiplication using AVX2 instructions, using the relation
 * (a⋅x)=(alo​⋅xlo​)+((ahi​⋅xlo​+alo​⋅xhi​)<<32) */
inline __m256i mul64_avx2(__m256i x, __m256i a)
{

    // split x and a into low and high 32-bit parts
    // it is possible to store the low part directly with the original 64-bit value because the multiplication will only
    // use the low 32 bits of each operand
    __m256i x_lo = x;
    __m256i a_lo = a;

    __m256i x_hi = _mm256_srli_epi64(x, 32);
    __m256i a_hi = _mm256_srli_epi64(a, 32);

    // a_lo * x_lo
    __m256i lo = _mm256_mul_epu32(x_lo, a_lo);

    // a_hi * x_lo
    __m256i hi1 = _mm256_mul_epu32(x_hi, a_lo);

    // a_lo * x_hi
    __m256i hi2 = _mm256_mul_epu32(x_lo, a_hi);

    // sum hi1 and hi2
    __m256i hi = _mm256_add_epi64(hi1, hi2);

    // shift << 32
    hi = _mm256_slli_epi64(hi, 32);

    return _mm256_add_epi64(lo, hi);
}
/** Generates a hash mapping for the given keys using the multiply-shift-add hash function using AVX2 instructions. 
  * @param data The input vector containing the keys to be hashed.
 * @param mapping The output vector where the resulting hash values will be stored.
 * @param n The number of elements in the input and output vectors.
 * @param k The number of bits to use for the hash output (the output will be in the range [0, 2^k - 1]).
 * @param a The multiplier for the hash function.
 * @param b The addend for the hash function.
*/
void generate_mapping_avx(const std::vector<uint64_t>& data, std::vector<uint64_t>& mapping, size_t n, int k,
                          uint64_t a, uint64_t b)
{
    const int shift = 64 - k;
    size_t i = 0;
    // pointers for AVX2 processing
    const uint64_t* __restrict__ in = data.data();
    uint64_t* __restrict__ out = mapping.data();

    // broadcast a and b to 256-bit vectors
    __m256i va = _mm256_set1_epi64x(a);
    __m256i vb = _mm256_set1_epi64x(b);

    // main loop processing 4 elements at a time
    for (; i + 4 <= n; i += 4)
    {
        // load 4 keys into an AVX2 register
        __m256i x = _mm256_loadu_si256((__m256i const*)(in + i));
        // compute a * x using the custom 64-bit multiplication
        __m256i mul = mul64_avx2(x, va);
        // sum b
        __m256i add = _mm256_add_epi64(mul, vb);
        // shift right by (64 - k)
        __m256i res = _mm256_srli_epi64(add, shift);
        // store the result back to the output array
        _mm256_storeu_si256((__m256i*)(out + i), res);
    }

    // process possible remaining elements
    for (; i < n; ++i)
    {
        out[i] = (a * in[i] + b) >> shift;
    }
}
