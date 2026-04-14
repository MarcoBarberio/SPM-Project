// hashjoin_seq.cpp
//
// Sequential reference for Module 2
// Partitioned Hash Join with Duplicates
//
// This code is intentionally written to be simple and readable.
// It is meant as a reference baseline and as a starting point for
// the parallel version. You can modify it for improving performance,
// provided you do not change the overall algorithm.
//
//
// IMPORTANT:
// The function compute_partition_id(...) below is intentionally very simple.
// Students must replace it with their own mapping function from Module 1.
// The same mapping function must be used consistently in both the sequential
// and parallel versions.
//
// Run example:
//   ./hashjoin_seq -nr 5 -ns 8 -seed 13 -max-key 8 -p 4
//
// Output:
//   join_count
//   checksum1
//   checksum2
//
//
// The code follows these phases:
//
//   1. Input generation
//      Generate two relations R and S with deterministic keys.
//
//   2. Partitioning of R and S
//      The goal of this phase is to reorganize the data so that
//      records belonging to the same partition are stored contiguously.
//
//      This is done in three steps:
//
//      - mapping key -> partition id 
//        Each key is mapped to a partition identifier in [0, P).
//
//      - histogram
//        Count how many records are assigned to each partition.
//        This tells us how much space each partition will occupy.
//
//      - prefix sum (offset computation)
//        Convert counts into starting positions (offsets) for each partition
//        in the output array.
//
//      - scatter
//        Move each record to its correct position so that all records
//        of the same partition are stored contiguously.
//
//      After this phase, each partition corresponds to a contiguous
//      segment of the array, and can be processed independently.
//
//   3. Local join per partition
//      For each partition p:
//
//      - build
//        Scan the R partition and count how many times each key appears.
//
//      - probe
//        Scan the corresponding S partition.
//        For each key, if it exists in R, add as many matches as its multiplicity.
//
//   4. Final output
//      Accumulate results across all partitions.
//
// The result does NOT materialize the join pairs.
// It only computes:
//   - total number of matches
//   - two checksums for correctness verification
//

#include "utilities.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
// a and b are fixed parameters for the hash function. In this case I use the same constants as splitmix64.
#define a 0x9E3779B97F4A7C15ULL
#define b 0xBF58476D1CE4E5B9ULL
// ------------------------------------------------------------
// Record definition
// ------------------------------------------------------------
//
// For this reference implementation we only store the key.
// You may extend the record with a payload in later versions if desired.
//
struct Record
{
    std::uint64_t key{};
};
using Clock = std::chrono::steady_clock;
// ------------------------------------------------------------
// Join result
// ------------------------------------------------------------
struct JoinResult
{
    std::uint64_t join_count = 0;
    std::uint64_t checksum1 = 0;
    std::uint64_t checksum2 = 0;
};

struct PartitionTimes
{
    double histogram = 0.0;
    double prefix = 0.0;
    double scatter = 0.0;
    double total = 0.0;
};

struct PhaseTimes
{
    PartitionTimes R;
    PartitionTimes S;
    double join_local = 0.0;
    double accumulation = 0.0;
    double total = 0.0;
};

static inline double elapsed_sec(const Clock::time_point& t0, const Clock::time_point& t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}
// ------------------------------------------------------------
// Utility: command-line parsing
// ------------------------------------------------------------
static bool read_arg_string(int argc, char** argv, const std::string& name, std::string& out)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (name == argv[i])
        {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}
static bool read_arg_u64(int argc, char** argv, const std::string& name, std::uint64_t& out)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (name == argv[i])
        {
            out = std::strtoull(argv[i + 1], nullptr, 10);
            return true;
        }
    }
    return false;
}

static void write_run_json_seq(const std::string& path,
                               std::size_t NR,
                               std::size_t NS,
                               std::uint32_t P,
                               std::uint64_t seed,
                               std::uint64_t max_key,
                               const JoinResult& result,
                               const PhaseTimes& times)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Error: cannot open JSON output file: " << path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"mode\": \"seq\",\n";
    out << "  \"NR\": " << NR << ",\n";
    out << "  \"NS\": " << NS << ",\n";
    out << "  \"P\": " << P << ",\n";
    out << "  \"seed\": " << seed << ",\n";
    out << "  \"max_key\": " << max_key << ",\n";

    out << "  \"join_count\": " << result.join_count << ",\n";
    out << "  \"checksum1\": " << result.checksum1 << ",\n";
    out << "  \"checksum2\": " << result.checksum2 << ",\n";

    out << "  \"time_histogram_R\": " << times.R.histogram << ",\n";
    out << "  \"time_prefix_R\": " << times.R.prefix << ",\n";
    out << "  \"time_scatter_R\": " << times.R.scatter << ",\n";
    out << "  \"time_partition_R\": " << times.R.total << ",\n";

    out << "  \"time_histogram_S\": " << times.S.histogram << ",\n";
    out << "  \"time_prefix_S\": " << times.S.prefix << ",\n";
    out << "  \"time_scatter_S\": " << times.S.scatter << ",\n";
    out << "  \"time_partition_S\": " << times.S.total << ",\n";

    out << "  \"time_join\": " << times.join_local << ",\n";
    out << "  \"time_accumulation\": " << times.accumulation << ",\n";
    out << "  \"time_total\": " << times.total << "\n";
    out << "}\n";
}

static void usage(const char* prog)
{
    std::cerr << "Usage:\n"
              << "  " << prog << " -nr NR -ns NS -seed SEED -max-key K -p P\n\n"
              << "Parameters:\n"
              << "  -nr         Number of records in relation R\n"
              << "  -ns         Number of records in relation S\n"
              << "  -seed       Deterministic seed\n"
              << "  -max-key    Keys are generated in [0, max-key)\n"
              << "  -p          Number of partitions (power of two required in this reference code)\n"
              << "  -json       Optional path to save run information as JSON\n";
}
static bool is_power_of_two(std::uint32_t x)
{
    return x != 0 && (x & (x - 1U)) == 0;
}

// ------------------------------------------------------------
// Deterministic pseudo-random generation
// ------------------------------------------------------------
//
// We use splitmix64 to generate reproducible keys and also for checksum.
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
//
// splitmix64_next is used as a deterministic pseudo-random generator step,
// while splitmix64 is used as a stateless 64-bit mixing function for checksums.
//
static inline std::uint64_t splitmix64_mix(std::uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}
static inline std::uint64_t splitmix64(std::uint64_t x)
{
    return splitmix64_mix(x + 0x9e3779b97f4a7c15ULL);
}
static inline std::uint64_t splitmix64_next(std::uint64_t& state)
{
    state += 0x9e3779b97f4a7c15ULL;
    return splitmix64_mix(state);
}

static std::vector<Record> generate_relation(std::size_t n, std::uint64_t seed, std::uint64_t max_key)
{
    std::vector<Record> out(n);
    std::uint64_t state = seed;

    for (std::size_t i = 0; i < n; ++i)
    {
        const std::uint64_t r = splitmix64_next(state);
        out[i].key = (max_key == 0) ? 0ULL : (r % max_key);
    }
    return out;
}

// ------------------------------------------------------------
// Intentionally simple partition mapping
// ------------------------------------------------------------
//
// This mapping is deliberately minimal.
// It is here only so that the reference code is complete and runnable.
//
// Students must replace this function with their own implementation from Module 1.
// The same mapping function must be used consistently in both the sequential
// and parallel versions to ensure a fair performance comparison.
//
// If P is a power of two, then key & (P-1) maps into [0, P).
// This is fast, but intentionally simplistic.
//
static inline std::uint32_t compute_partition_id(std::uint64_t key, std::uint32_t p)
{
    int shift = 64 - static_cast<int>(std::log2(p));
    return map_single_key(key, a, b, shift);
}

// ------------------------------------------------------------
// Histogram
// ------------------------------------------------------------
//
// Count how many records go to each partition.
//
// hist[pid] = number of records whose key maps to pid
//
static std::vector<std::size_t> compute_histogram(const std::vector<Record>& rel, std::uint32_t p)
{
    std::vector<std::size_t> hist(p, 0);

    for (const auto& rec : rel)
    {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        ++hist[pid];
    }
    return hist;
}

// ------------------------------------------------------------
// Prefix sum (exclusive scan)
// ------------------------------------------------------------
//
// Given a histogram, compute the begin offsets of each partition.
//
// Example:
//   hist  = [0, 1, 2, 5]
//   begin = [0, 0, 1, 3]
//
// Then partition p occupies [begin[p], begin[p] + hist[p]).
//
static std::vector<std::size_t> exclusive_prefix_sum(const std::vector<std::size_t>& hist)
{
    std::vector<std::size_t> begin(hist.size(), 0);

    std::size_t running = 0;
    for (std::size_t p = 0; p < hist.size(); ++p)
    {
        begin[p] = running;
        running += hist[p];
    }
    return begin;
}

// ------------------------------------------------------------
// Scatter into a partitioned array
// ------------------------------------------------------------
//
// Reorder records so that all records belonging to the same partition become
// contiguous in memory.
//
// We use a write cursor per partition, initialized from the begin offsets.
//
static std::vector<Record> scatter_partitioned(const std::vector<Record>& rel, std::uint32_t p,
                                               const std::vector<std::size_t>& begin)
{
    std::vector<Record> out(rel.size());

    // Current write position for each partition.
    std::vector<std::size_t> next = begin;

    for (const auto& rec : rel)
    {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        out[next[pid]++] = rec;
    }

    return out;
}

// ------------------------------------------------------------
// Partitioned relation metadata
// ------------------------------------------------------------
//
// This stores:
//   - the partitioned array
//   - the begin offset of each partition
//   - the end offset of each partition
//
// So partition p is data[begin[p] .. end[p]).
//
struct PartitionedRelation
{
    std::vector<Record> data;
    std::vector<std::size_t> begin;
    std::vector<std::size_t> end;
};

// ------------------------------------------------------------
// Full partitioning pipeline for one relation
// ------------------------------------------------------------
//
// This groups together the steps:
//   histogram -> prefix sum -> scatter -> end offsets
//
// After this phase, all records belonging to the same partition
// are stored contiguously in memory, enabling independent processing.
//
static PartitionedRelation partition_relation(const std::vector<Record>& rel, std::uint32_t p,
                                              PartitionTimes& times)
{
    const auto t0 = Clock::now();
    const auto hist = compute_histogram(rel, p);
    const auto t1 = Clock::now();

    const auto begin = exclusive_prefix_sum(hist);
    const auto t2 = Clock::now();

    auto data = scatter_partitioned(rel, p, begin);
    const auto t3 = Clock::now();

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid)
    {
        end[pid] = begin[pid] + hist[pid];
    }
    const auto t4 = Clock::now();

    times.histogram = elapsed_sec(t0, t1);
    times.prefix    = elapsed_sec(t1, t2);
    times.scatter   = elapsed_sec(t2, t3);
    times.total     = elapsed_sec(t0, t4);

    return PartitionedRelation{.data = std::move(data), .begin = begin, .end = end};
}


// ------------------------------------------------------------
// Local join on one partition
// ------------------------------------------------------------
//
// We process one partition p as follows:
//
//   Build:
//     Scan R_p and compute countR[key]
//
//   Probe:
//     Scan S_p
//     If key k is present in countR, then each occurrence in S_p matches
//     countR[k] occurrences in R_p.
//
// Duplicates are handled by counting occurrences in R.
// Each record in S contributes as many matches as the multiplicity
// of its key in the corresponding partition of R.
//
static JoinResult join_one_partition(const PartitionedRelation& Rpart, const PartitionedRelation& Spart,
                                     std::uint32_t pid)
{
    JoinResult result{};

    const std::size_t r_begin = Rpart.begin[pid];
    const std::size_t r_end = Rpart.end[pid];
    const std::size_t s_begin = Spart.begin[pid];
    const std::size_t s_end = Spart.end[pid];

    // Nothing to do if either partition is empty.
    if (r_begin == r_end || s_begin == s_end)
    {
        return result;
    }

    // Build phase:
    // count how many times each key appears in R_p.
    //
    // NOTE: Adopting std::unordered_map is an implementation choice
    // of the reference code, not a mandatory part of the algorithm itself.
    // Students may discuss its impact on performance and, if properly justified,
    // replace it with alternative structures in their analysis or optimized versions,
    // provided that the overall join logic remains unchanged
    //
    std::unordered_map<std::uint64_t, std::uint32_t> countR;
    countR.reserve((r_end - r_begin) * 2);

    for (std::size_t i = r_begin; i < r_end; ++i)
    {
        ++countR[Rpart.data[i].key];
    }

    // Probe phase:
    // for each key in S_p, if it exists in countR, add countR[key] matches.
    for (std::size_t i = s_begin; i < s_end; ++i)
    {
        const std::uint64_t key = Spart.data[i].key;
        const auto it = countR.find(key);
        if (it != countR.end())
        {
            const std::uint64_t multiplicity = it->second;

            result.join_count += multiplicity;
            result.checksum1 += splitmix64(key) * multiplicity;
            result.checksum2 += splitmix64(key ^ 0x9e3779b97f4a7c15ULL) * multiplicity;
        }
    }

    return result;
}

// ------------------------------------------------------------
// Full sequential partitioned hash join
// ------------------------------------------------------------
//
// This is the end-to-end baseline:
//
//   1. Partition R
//   2. Partition S
//   3. For each partition p:
//        local build + local probe
//   4. Accumulate results
//
// Each partition can be processed independently.
// This property is the basis for parallelization in Module 2.
//
static JoinResult partitioned_hash_join_sequential(const std::vector<Record>& R, const std::vector<Record>& S,
                                                   std::uint32_t p, PhaseTimes& times)
{
    const auto t0 = Clock::now();

    const PartitionedRelation Rpart = partition_relation(R, p, times.R);
    const PartitionedRelation Spart = partition_relation(S, p, times.S);

    std::vector<JoinResult> locals(p);

    const auto tj0 = Clock::now();
    for (std::uint32_t pid = 0; pid < p; ++pid)
    {
        locals[pid] = join_one_partition(Rpart, Spart, pid);
    }
    const auto tj1 = Clock::now();

    JoinResult total{};
    const auto ta0 = Clock::now();
    for (const auto& local : locals)
    {
        total.join_count += local.join_count;
        total.checksum1 += local.checksum1;
        total.checksum2 += local.checksum2;
    }
    const auto ta1 = Clock::now();

    const auto t1 = Clock::now();

    times.join_local   = elapsed_sec(tj0, tj1);
    times.accumulation = elapsed_sec(ta0, ta1);
    times.total        = elapsed_sec(t0, t1);

    return total;
}

// ------------------------------------------------------------
// Verifier for very small inputs
// ------------------------------------------------------------
//
// This is useful only for debugging and correctness testing on tiny
// datasets. It checks all pairs directly, so its complexity is O(|R|*|S|).
//
static JoinResult naive_join_verifier(const std::vector<Record>& R, const std::vector<Record>& S)
{
    JoinResult result{};

    for (const auto& r : R)
    {
        for (const auto& s : S)
        {
            if (r.key == s.key)
            {
                result.join_count += 1;
                result.checksum1 += splitmix64(r.key);
                result.checksum2 += splitmix64(r.key ^ 0x9e3779b97f4a7c15ULL);
            }
        }
    }
    return result;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0;
    std::string json_path;
    const bool save_json = read_arg_string(argc, argv, "-json", json_path);

    if (!read_arg_u64(argc, argv, "-nr", nr) || !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) || !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p))
    {
        usage(argv[0]);
        return 1;
    }

    if (p > std::numeric_limits<std::uint32_t>::max())
    {
        std::cerr << "Error: P too large.\n";
        return 1;
    }

    const std::uint32_t P = static_cast<std::uint32_t>(p);

    if (!is_power_of_two(P))
    {
        std::cerr << "Error: in this reference implementation, P must be a power of two.\n";
        return 1;
    }

    const std::size_t NR = static_cast<std::size_t>(nr);
    const std::size_t NS = static_cast<std::size_t>(ns);

    const auto R = generate_relation(NR, seed, max_key);
    const auto S = generate_relation(NS, seed ^ 0xdeadebdecdeedef1ULL, max_key);

    PhaseTimes times;
    const JoinResult result = partitioned_hash_join_sequential(R, S, P, times);

    std::cout << "NR=" << NR << " NS=" << NS << " P=" << P << " seed=" << seed << " [0, " << max_key << ")\n";

    std::cout << "join_count=" << result.join_count << "\n";
    std::cout << "checksum1=" << result.checksum1 << "\n";
    std::cout << "checksum2=" << result.checksum2 << "\n";

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "time_histogram_R=" << times.R.histogram << "\n";
    std::cout << "time_prefix_R=" << times.R.prefix << "\n";
    std::cout << "time_scatter_R=" << times.R.scatter << "\n";
    std::cout << "time_partition_R=" << times.R.total << "\n";

    std::cout << "time_histogram_S=" << times.S.histogram << "\n";
    std::cout << "time_prefix_S=" << times.S.prefix << "\n";
    std::cout << "time_scatter_S=" << times.S.scatter << "\n";
    std::cout << "time_partition_S=" << times.S.total << "\n";

    std::cout << "time_join=" << times.join_local << "\n";
    std::cout << "time_accumulation=" << times.accumulation << "\n";
    std::cout << "time_total=" << times.total << "\n";
    std::cout << "time_sec=" << times.total << "\n";

    if (NR <= 500 && NS <= 500)
    {
        const JoinResult naive = naive_join_verifier(R, S);
        std::cout << "naive_join_count=" << naive.join_count << "\n";
        std::cout << "naive_checksum1=" << naive.checksum1 << "\n";
        std::cout << "naive_checksum2=" << naive.checksum2 << "\n";
    }

    if (save_json)
    {
        write_run_json_seq(json_path, NR, NS, P, seed, max_key, result, times);
    }

    return 0;
}