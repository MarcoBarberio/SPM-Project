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

/**
 * Maps a single key to a partition ID using the multiply-shift-add hash function.
 * @param key The input key to be hashed.
 * @param a The multiplier for the hash function.
 * @param b The addend for the hash function.
 * @param shift The number of bits to shift right to obtain the final partition ID (calculated as 64 - k).
 * @return The computed partition ID for the given key.
 */
int inline map_single_key(std::uint64_t key, std::uint64_t a, std::uint64_t b, int shift) {
    return (a * key + b) >> shift;
}