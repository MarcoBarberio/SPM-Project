#include "utilities.hpp"
#include <vector>
#include <cstdint>
#include "mapping.hpp"
#include "hpc_helpers.hpp"

/**Generates a hash mapping for the given keys using the multiply-shift-add hash function. */
void generate_mapping(
    const uint64_t* __restrict__ in,
    uint64_t* __restrict__ out,
    size_t n,
    int k,
    uint64_t a,
    uint64_t b
) {
    const int shift = 64 - k;

    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        out[i] = (a * in[i] + b) >> shift;
    }
}