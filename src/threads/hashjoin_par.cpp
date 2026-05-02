
#include "mapping_baseline.hpp"
#include "threadPool.hpp"
#include "utilities.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fstream>
//a and b are fixed parameters for the hash function. 
//In this case I use the same constants as splitmix64.

#define a 0x9E3779B97F4A7C15ULL
#define b 0xBF58476D1CE4E5B9ULL
struct Record
{
    std::uint64_t key{};
};
// ------------------------------------------------------------
// Join result
// ------------------------------------------------------------
struct JoinResult
{
    std::uint64_t join_count = 0;
    std::uint64_t checksum1 = 0;
    std::uint64_t checksum2 = 0;
};

// ------------------------------------------------------------
// Timing utilities
// ------------------------------------------------------------
using Clock = std::chrono::steady_clock;

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

// ------------------------------------------------------------
// JSON output
// ------------------------------------------------------------
static void write_run_json_par(const std::string& path,
                               std::size_t NR,
                               std::size_t NS,
                               std::uint32_t P,
                               std::size_t T,
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
    out << "  \"mode\": \"par\",\n";
    out << "  \"NR\": " << NR << ",\n";
    out << "  \"NS\": " << NS << ",\n";
    out << "  \"P\": " << P << ",\n";
    out << "  \"T\": " << T << ",\n";
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
              << "  " << prog << " -nr NR -ns NS -seed SEED -max-key K -p P -t T\n\n"
              << "Parameters:\n"
              << "  -nr         Number of records in relation R\n"
              << "  -ns         Number of records in relation S\n"
              << "  -seed       Deterministic seed\n"
              << "  -max-key    Keys are generated in [0, max-key)\n"
              << "  -p          Number of partitions (power of two required in this reference code)\n"
              << "  -t          Number of threads\n"
              << "  -json       Optional path to save run information as JSON\n";
}
static bool is_power_of_two(std::uint32_t x)
{
    return x != 0 && (x & (x - 1U)) == 0;
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

static inline std::uint32_t compute_partition_id(std::uint64_t key, std::uint32_t p)
{
    int shift = 64 - static_cast<int>(std::log2(p));
    return map_single_key(key, a, b, shift);
}

struct HistogramData
{
    std::vector<std::size_t> hist;
    std::vector<std::vector<std::size_t>> local_hists;
};

// ------------------------------------------------------------
// Histogram
// ------------------------------------------------------------
//
// Count how many records go to each partition.
// In order to avoid contention on the histogram counters, each thread computes a local histogram,
// and then we sum them up at the end.
// hist[pid] = number of records whose key maps to pid
//
static HistogramData compute_histogram(const std::vector<Record>& rel, std::uint32_t p, threadPool& pool,
                                       std::size_t nthreads)
{
    const std::size_t n = rel.size();

    std::vector<std::vector<std::size_t>> local_hists(nthreads, std::vector<std::size_t>(p, 0));

    std::vector<std::future<void>> futures;
    futures.reserve(nthreads);

    for (std::size_t tid = 0; tid < nthreads; ++tid)
    {
        futures.emplace_back(pool.submit(
            [&, tid]
            {
                const std::size_t chunk = (n + nthreads - 1) / nthreads;
                const std::size_t begin = tid * chunk;
                const std::size_t end = std::min(begin + chunk, n);

                auto& hist = local_hists[tid];
                for (std::size_t i = begin; i < end; ++i)
                {
                    const std::uint32_t pid = compute_partition_id(rel[i].key, p);
                    ++hist[pid];
                }
            }));
    }

    for (auto& f : futures)
        f.get();

    std::vector<std::size_t> hist(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid)
    {
        std::size_t sum = 0;
        for (std::size_t tid = 0; tid < nthreads; ++tid)
            sum += local_hists[tid][pid];
        hist[pid] = sum;
    }

    return HistogramData{std::move(hist), std::move(local_hists)};
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
// Reorder records so that all records belonging to the same partition become
// contiguous in memory.
//
// For each partition, we precompute a private output range for every thread.
// These per-thread ranges are derived from the global begin offsets and from
// the local histograms computed during the histogram phase.
//
// During scatter, each thread processes the same input chunk used in the
// histogram phase and maintains a private write cursor for each partition,
// initialized from its own precomputed starting offsets.
// Therefore, threads write only inside disjoint regions of the output array,
// and no atomic operations or locks are needed.
    
static std::vector<Record> scatter_partitioned(const std::vector<Record>& rel, std::uint32_t p,
                                               const std::vector<std::size_t>& begin,
                                               const std::vector<std::vector<std::size_t>>& local_hists,
                                               threadPool& pool, std::size_t nthreads)
{
    const std::size_t n = rel.size();
    std::vector<Record> out(n);

    std::vector<std::vector<std::size_t>> thread_begin(nthreads, std::vector<std::size_t>(p, 0));

    for (std::uint32_t pid = 0; pid < p; ++pid)
    {
        std::size_t offset = begin[pid];
        for (std::size_t tid = 0; tid < nthreads; ++tid)
        {
            thread_begin[tid][pid] = offset;
            offset += local_hists[tid][pid];
        }
    }

    std::vector<std::future<void>> futures;
    futures.reserve(nthreads);

    for (std::size_t tid = 0; tid < nthreads; ++tid)
    {
        futures.emplace_back(pool.submit(
            [&, tid]
            {
                const std::size_t chunk = (n + nthreads - 1) / nthreads;
                const std::size_t i_begin = tid * chunk;
                const std::size_t i_end = std::min(i_begin + chunk, n);

                std::vector<std::size_t> next = thread_begin[tid];

                for (std::size_t i = i_begin; i < i_end; ++i)
                {
                    const auto& rec = rel[i];
                    const std::uint32_t pid = compute_partition_id(rec.key, p);
                    out[next[pid]++] = rec;
                }
            }));
    }

    for (auto& f : futures)
        f.get();

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
static PartitionedRelation partition_relation(const std::vector<Record>& rel, std::uint32_t p, threadPool& pool,
                                              std::size_t nthreads, PartitionTimes& times)
{
    const auto t0 = Clock::now();
    const HistogramData hdata = compute_histogram(rel, p, pool, nthreads);
    const auto t1 = Clock::now();

    const auto begin = exclusive_prefix_sum(hdata.hist);
    const auto t2 = Clock::now();

    auto data = scatter_partitioned(rel, p, begin, hdata.local_hists, pool, nthreads);
    const auto t3 = Clock::now();

    std::vector<std::size_t> end(p, 0);
    for (std::uint32_t pid = 0; pid < p; ++pid)
    {
        end[pid] = begin[pid] + hdata.hist[pid];
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


// This is the end-to-end baseline:
//
//   1. Partition R
//   2. Partition S
//   3. For each partition p:
//        local build + local probe
//   4. Accumulate results
//
// Each partition can be processed independently,so we can parallelize step 3 
// by having multiple threads process different partitions concurrently.
//
static JoinResult partitioned_hash_join_parallel(const std::vector<Record>& R, const std::vector<Record>& S,
                                                 std::uint32_t p, threadPool& pool, std::size_t nthreads,
                                                 PhaseTimes& times)
{
    const auto t0 = Clock::now();

    const PartitionedRelation Rpart = partition_relation(R, p, pool, nthreads, times.R);
    const PartitionedRelation Spart = partition_relation(S, p, pool, nthreads, times.S);

    std::atomic<std::uint32_t> next_pid{0};
    std::vector<JoinResult> partials(nthreads);
    std::vector<std::future<void>> futures;
    futures.reserve(nthreads);

    const auto tj0 = Clock::now();

    for (std::size_t tid = 0; tid < nthreads; ++tid)
    {
        futures.emplace_back(pool.submit(
            [&, tid]
            {
                JoinResult local{};

                while (true)
                {
                    const std::uint32_t pid = next_pid.fetch_add(1, std::memory_order_relaxed);
                    if (pid >= p)
                        break;

                    const JoinResult jr = join_one_partition(Rpart, Spart, pid);
                    local.join_count += jr.join_count;
                    local.checksum1 += jr.checksum1;
                    local.checksum2 += jr.checksum2;
                }

                partials[tid] = local;
            }));
    }

    for (auto& f : futures)
        f.get();

    const auto tj1 = Clock::now();

    JoinResult total{};
    const auto ta0 = Clock::now();
    for (const auto& x : partials)
    {
        total.join_count += x.join_count;
        total.checksum1 += x.checksum1;
        total.checksum2 += x.checksum2;
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
    std::uint64_t nr = 0, ns = 0, seed = 0, max_key = 0, p = 0, t = 0;
    std::string json_path;
    const bool save_json = read_arg_string(argc, argv, "-json", json_path);

    if (!read_arg_u64(argc, argv, "-nr", nr) || !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) || !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p) || !read_arg_u64(argc, argv, "-t", t))
    {
        usage(argv[0]);
        return 1;
    }

    if (p > std::numeric_limits<std::uint32_t>::max())
    {
        std::cerr << "Error: P too large.\n";
        return 1;
    }

    if (t == 0)
    {
        std::cerr << "Error: number of threads must be >= 1.\n";
        return 1;
    }

    if (t > std::numeric_limits<std::size_t>::max())
    {
        std::cerr << "Error: T too large.\n";
        return 1;
    }

    const std::uint32_t P = static_cast<std::uint32_t>(p);
    const std::size_t T = static_cast<std::size_t>(t);
    threadPool pool(T);

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
    const JoinResult result = partitioned_hash_join_parallel(R, S, P, pool, T, times);

    std::cout << "NR=" << NR << " NS=" << NS << " P=" << P << " T=" << T << " seed=" << seed
              << " [0, " << max_key << ")\n";

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
        write_run_json_par(json_path, NR, NS, P, T, seed, max_key, result, times);
    }

    return 0;
}