#include "mapping_baseline.hpp"
#include "utilities.hpp"
#include <cstdint>
#include <vector>


/**Generates a hash mapping for the given keys using the multiply-shift-add hash function. */
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