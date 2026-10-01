#include <cstdint>
#include <random>

inline std::vector<uint64_t> generate_random_keys(size_t n, uint64_t seed)
{
    /**Generates a vector of n random 64-bit keys using the given seed. */
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist;
    std::vector<uint64_t> keys(n);
    for (size_t i = 0; i < n; ++i)
    {
        keys[i] = dist(rng);
    }
    return keys;
}
inline std::pair<uint64_t, uint64_t> generate_params(uint64_t seed)
{
    /** Generates random parameters for the hash function. */
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist;

    uint64_t a = dist(rng) | 1ULL; // must be odd
    uint64_t b = dist(rng);

    return {a, b};
}
inline uint64_t checksum(const std::vector<uint64_t>& mapping)
{
    /** Computes a simple checksum of the mapping for verification. */
    return std::accumulate(mapping.begin(), mapping.end(), 0ULL);
}

