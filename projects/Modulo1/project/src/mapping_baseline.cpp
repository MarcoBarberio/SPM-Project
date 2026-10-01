#include "mapping_baseline.hpp"
#include "utilities.hpp"
#include <cstdint>
#include <vector>

/** Generates a hash mapping for the given keys using the multiply-shift-add hash function using AVX2 instructions.
 * @param data The input vector containing the keys to be hashed.
 * @param mapping The output vector where the resulting hash values will be stored.
 * @param n The number of elements in the input and output vectors.
 * @param k The number of bits to use for the hash output (the output will be in the range [0, 2^k - 1]).
 * @param a The multiplier for the hash function.
 * @param b The addend for the hash function.
 */
void generate_mapping_baseline(const std::vector<uint64_t>& data, std::vector<uint64_t>& mapping, size_t n, int k,
                               uint64_t a, uint64_t b)
{
    const int shift = 64 - k;
    
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i)
    {
        mapping[i] = (a * data[i] + b) >> shift;
    }
}