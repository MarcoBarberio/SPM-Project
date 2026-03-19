#include <cstdint>
#include <random>

namespace hash {
    /** Generates a pair of hash parameters (a, b) for the multiply-shift-add hash function. */
    inline std::pair<uint64_t, uint64_t> generate_params(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist;

        uint64_t a = dist(rng) | 1ULL;  // must be odd
        uint64_t b = dist(rng);

        return {a, b};
    }
    /** Multiplies the input by 'a', adds 'b', and then shifts the result right by (64-k) bits. */
    inline uint64_t multiply_shift_add(uint64_t x, uint64_t a, uint64_t b, int k) {
        return (a * x + b) >> (64-k);
    }
}