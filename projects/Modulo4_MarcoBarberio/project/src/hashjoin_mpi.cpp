// hashjoin_mpi.cpp
//
// Partitioned Hash Join with Duplicates - MPI version
//
// Compile:
//   mpicxx -Wall -O3 -std=c++20 -I./include hashjoin_mpi.cpp -o build/hashjoin_mpi
//
// Example:
//   srun --mpi=pmix -N 2 -n 2 ./build/hashjoin_mpi -nr 1000000 -ns 1000000 -seed 13 -max-key 100000 -p 1024
//
// MPI pipeline:
//
//   1. Each rank generates only its local slice of R and S.
//   2. Each local record is mapped to a partition.
//   3. The owner rank of a partition is computed as pid % world_size.
//   4. Records are redistributed among ranks with MPI_Alltoallv.
//   5. Each rank locally partitions the records it received.
//   6. Each rank joins only the partitions assigned to it.
//   7. The final result is obtained with MPI_Reduce.
//

#include "utilities.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Same constants used in the previous modules for the partition mapping.
#define a 0x9E3779B97F4A7C15ULL
#define b 0xBF58476D1CE4E5B9ULL

struct Record
{
    std::uint64_t key{};
};

using Clock = std::chrono::steady_clock;

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

    // New timings for the MPI version.
    // Redistribution is the communication phase that replaces the purely local scatter
    // of the shared-memory implementations.
    double redistribution_R = 0.0;
    double redistribution_S = 0.0;

    double join_local = 0.0;
    double accumulation = 0.0;
    double total = 0.0;
};

static inline double elapsed_sec(const Clock::time_point& t0, const Clock::time_point& t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

// ------------------------------------------------------------
// Command-line parsing
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

static void usage(const char* prog)
{
    std::cerr << "Usage:\n"
              << "  " << prog << " -nr NR -ns NS -seed SEED -max-key K -p P\n\n"
              << "Parameters:\n"
              << "  -nr              Number of records in relation R\n"
              << "  -ns              Number of records in relation S\n"
              << "  -seed            Deterministic seed\n"
              << "  -max-key         Keys are generated in [0, max-key)\n"
              << "  -p               Number of partitions, power of two required\n"
              << "  -workload        uniform | skewed, default uniform\n"
              << "  -hot-partitions  Number of hot partitions for skewed workload, default P/64\n"
              << "  -skew-percent    Percentage of records assigned to hot partitions, default 90\n"
              << "  -json            Optional path to save run information as JSON\n";
}

static bool is_power_of_two(std::uint32_t x)
{
    return x != 0 && (x & (x - 1U)) == 0;
}

// ------------------------------------------------------------
// Deterministic pseudo-random generation
// ------------------------------------------------------------

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

static inline std::uint32_t compute_partition_id(std::uint64_t key, std::uint32_t p);

// ------------------------------------------------------------
// Block decomposition helper
// ------------------------------------------------------------
//
// The global input is split into contiguous slices.
// Rank r generates only its assigned slice, avoiding a centralized rank-0 generation.

static std::size_t block_start(std::size_t n, int rank, int world_size)
{
    const std::size_t base = n / static_cast<std::size_t>(world_size);
    const std::size_t rem = n % static_cast<std::size_t>(world_size);

    return static_cast<std::size_t>(rank) * base + std::min<std::size_t>(static_cast<std::size_t>(rank), rem);
}

static std::size_t block_count(std::size_t n, int rank, int world_size)
{
    const std::size_t base = n / static_cast<std::size_t>(world_size);
    const std::size_t rem = n % static_cast<std::size_t>(world_size);

    return base + (static_cast<std::size_t>(rank) < rem ? 1ULL : 0ULL);
}

// ------------------------------------------------------------
// Input generation: local range
// ------------------------------------------------------------
//
// These functions generate exactly the same logical global sequence as the
// sequential generator, but each rank materializes only its own range.

static std::vector<Record> generate_relation_uniform_range(std::size_t begin, std::size_t count, std::uint64_t seed,
                                                           std::uint64_t max_key)
{
    std::vector<Record> out(count);

    // Offset the pseudo-random state according to the global starting index.
    // This allows each rank to generate its local slice while preserving the same
    // logical sequence that would be produced by a sequential generator.
    std::uint64_t state = seed + begin * 0x9e3779b97f4a7c15ULL;

    for (std::size_t i = 0; i < count; ++i)
    {
        const std::uint64_t r = splitmix64_next(state);
        out[i].key = (max_key == 0) ? 0ULL : (r % max_key);
    }

    return out;
}

static std::vector<Record> generate_relation_skewed_range(std::size_t begin, std::size_t count, std::uint64_t seed,
                                                          std::uint64_t max_key, std::uint32_t p,
                                                          std::uint32_t hot_partitions,
                                                          std::uint32_t skew_percent)
{
    if (max_key == 0)
    {
        throw std::runtime_error("Skewed workload requires max_key > 0.");
    }

    if (hot_partitions == 0 || hot_partitions > p)
    {
        throw std::runtime_error("Invalid number of hot partitions.");
    }

    if (skew_percent > 100)
    {
        throw std::runtime_error("skew_percent must be in [0, 100].");
    }

    std::vector<std::uint64_t> hot_keys;
    std::vector<std::uint64_t> cold_keys;

    for (std::uint64_t key = 0; key < max_key; ++key)
    {
        const std::uint32_t pid = compute_partition_id(key, p);

        if (pid < hot_partitions)
        {
            hot_keys.push_back(key);
        }
        else
        {
            cold_keys.push_back(key);
        }
    }

    if (hot_keys.empty())
    {
        throw std::runtime_error("No keys map to the selected hot partitions. Increase max_key or hot_partitions.");
    }

    if (cold_keys.empty() && skew_percent < 100)
    {
        throw std::runtime_error("No keys map to cold partitions. Increase max_key or reduce hot_partitions.");
    }

    std::vector<Record> out(count);

    // The sequential skewed generator consumes two random values per record:
    // one to choose hot/cold and one to choose the key inside that set.
    std::uint64_t state = seed + (2ULL * begin) * 0x9e3779b97f4a7c15ULL;

    for (std::size_t i = 0; i < count; ++i)
    {
        const std::uint64_t r = splitmix64_next(state);
        const bool choose_hot = (r % 100) < skew_percent || cold_keys.empty();

        const std::uint64_t r_key = splitmix64_next(state);

        if (choose_hot)
        {
            out[i].key = hot_keys[r_key % hot_keys.size()];
        }
        else
        {
            out[i].key = cold_keys[r_key % cold_keys.size()];
        }
    }

    return out;
}

static std::vector<Record> generate_relation_range(std::size_t begin, std::size_t count, std::uint64_t seed,
                                                   std::uint64_t max_key, std::uint32_t p,
                                                   const std::string& workload, std::uint32_t hot_partitions,
                                                   std::uint32_t skew_percent)
{
    if (workload == "uniform")
    {
        return generate_relation_uniform_range(begin, count, seed, max_key);
    }

    if (workload == "skewed")
    {
        return generate_relation_skewed_range(begin, count, seed, max_key, p, hot_partitions, skew_percent);
    }

    throw std::runtime_error("Unknown workload. Use 'uniform' or 'skewed'.");
}

// ------------------------------------------------------------
// Partition mapping
// ------------------------------------------------------------
//

static inline std::uint32_t compute_partition_id(std::uint64_t key, std::uint32_t p)
{
    int shift = 64 - static_cast<int>(std::log2(p));
    return map_single_key(key, a, b, shift);
}

// ------------------------------------------------------------
// Local partitioning utilities
// ------------------------------------------------------------
//
// These functions are the same logical phases used in previous modules:
// histogram -> prefix sum -> scatter.
// In the MPI version they are applied after redistribution, on the records
// owned by the current rank.

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

static std::vector<Record> scatter_partitioned(const std::vector<Record>& rel, std::uint32_t p,
                                               const std::vector<std::size_t>& begin)
{
    std::vector<Record> out(rel.size());
    std::vector<std::size_t> next = begin;

    for (const auto& rec : rel)
    {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        out[next[pid]++] = rec;
    }

    return out;
}

struct PartitionedRelation
{
    std::vector<Record> data;
    std::vector<std::size_t> begin;
    std::vector<std::size_t> end;
};

static PartitionedRelation partition_relation(const std::vector<Record>& rel, std::uint32_t p, PartitionTimes& times)
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
    times.prefix = elapsed_sec(t1, t2);
    times.scatter = elapsed_sec(t2, t3);
    times.total = elapsed_sec(t0, t4);

    return PartitionedRelation{.data = std::move(data), .begin = begin, .end = end};
}

// ------------------------------------------------------------
// Local join on one partition
// ------------------------------------------------------------
//
// This is unchanged with respect to the previous modules.
// The MPI version changes which rank owns which partitions, not the internal
// logic used to join a single partition.

static JoinResult join_one_partition(const PartitionedRelation& Rpart, const PartitionedRelation& Spart,
                                     std::uint32_t pid)
{
    JoinResult result{};

    const std::size_t r_begin = Rpart.begin[pid];
    const std::size_t r_end = Rpart.end[pid];
    const std::size_t s_begin = Spart.begin[pid];
    const std::size_t s_end = Spart.end[pid];

    if (r_begin == r_end || s_begin == s_end)
    {
        return result;
    }

    // Build phase: count the multiplicity of each key in R_p.
    std::unordered_map<std::uint64_t, std::uint32_t> countR;
    countR.reserve((r_end - r_begin) * 2);

    for (std::size_t i = r_begin; i < r_end; ++i)
    {
        ++countR[Rpart.data[i].key];
    }

    // Probe phase: each S key contributes as many matches as its multiplicity in R_p.
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
// MPI helper functions
// ------------------------------------------------------------

static int owner_of_partition(std::uint32_t pid, int world_size)
{
    // Round-robin partition ownership:
    // partition 0 -> rank 0, partition 1 -> rank 1, ..., partition k -> rank k % world_size.
    //
    // This is usually safer than assigning contiguous partition ranges, especially for skewed data.
    return static_cast<int>(pid % static_cast<std::uint32_t>(world_size));
}

static double mpi_max_double(double local_value)
{
    double global_value = 0.0;

    // Parallel time is determined by the slowest rank.
    // Therefore phase timings are aggregated with MPI_MAX, not MPI_SUM.
    MPI_Reduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return global_value;
}
// For parallel timings, the relevant execution time is the slowest rank time.
// Therefore, phase times are reduced using MPI_MAX rather than MPI_SUM.

static PhaseTimes collect_phase_times_max(const PhaseTimes& local)
{
    PhaseTimes global{};

    global.redistribution_R = mpi_max_double(local.redistribution_R);
    global.redistribution_S = mpi_max_double(local.redistribution_S);

    global.R.histogram = mpi_max_double(local.R.histogram);
    global.R.prefix = mpi_max_double(local.R.prefix);
    global.R.scatter = mpi_max_double(local.R.scatter);
    global.R.total = mpi_max_double(local.R.total);

    global.S.histogram = mpi_max_double(local.S.histogram);
    global.S.prefix = mpi_max_double(local.S.prefix);
    global.S.scatter = mpi_max_double(local.S.scatter);
    global.S.total = mpi_max_double(local.S.total);

    global.join_local = mpi_max_double(local.join_local);
    global.accumulation = mpi_max_double(local.accumulation);
    global.total = mpi_max_double(local.total);

    return global;
}

// ------------------------------------------------------------
// MPI redistribution
// ------------------------------------------------------------
//
// This is the central MPI-specific phase.
//
// Before the join, all records belonging to the same partition must be located
// on the same rank. Each rank therefore sends each local record to the owner
// of its partition.
//
// MPI_Alltoallv is used because different ranks may send different numbers
// of records to each destination rank.

static std::vector<Record> redistribute_relation(const std::vector<Record>& local_rel, std::uint32_t p,
                                                 int world_size, MPI_Datatype mpi_record,
                                                 double& time_redistribution)
{
    const double t0 = MPI_Wtime();

    if (local_rel.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Local relation too large for MPI_Alltoallv int counts.");
    }

    std::vector<int> sendcounts(world_size, 0);

    // First pass: count how many records must be sent to each destination rank.
    for (const auto& rec : local_rel)
    {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        const int dst = owner_of_partition(pid, world_size);

        if (sendcounts[dst] == std::numeric_limits<int>::max())
        {
            throw std::runtime_error("sendcount overflow in redistribute_relation.");
        }

        ++sendcounts[dst];
    }

    // Displacements define where each destination block starts in sendbuf.
    std::vector<int> sdispls(world_size, 0);
    for (int i = 1; i < world_size; ++i)
    {
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
    }

    std::vector<Record> sendbuf(local_rel.size());
    std::vector<int> cursor = sdispls;

    // Second pass: pack records contiguously by destination rank.
    for (const auto& rec : local_rel)
    {
        const std::uint32_t pid = compute_partition_id(rec.key, p);
        const int dst = owner_of_partition(pid, world_size);

        sendbuf[static_cast<std::size_t>(cursor[dst]++)] = rec;
    }

    std::vector<int> recvcounts(world_size, 0);

    // Exchange only the counts first, so every rank knows how large its receive buffer must be.
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> rdispls(world_size, 0);

    int total_recv = 0;
    for (int i = 0; i < world_size; ++i)
    {
        rdispls[i] = total_recv;

        if (recvcounts[i] > std::numeric_limits<int>::max() - total_recv)
        {
            throw std::runtime_error("recvcount overflow in redistribute_relation.");
        }

        total_recv += recvcounts[i];
    }

    std::vector<Record> recvbuf(static_cast<std::size_t>(total_recv));

    // Actual all-to-all redistribution of records.
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), mpi_record, recvbuf.data(), recvcounts.data(),
                  rdispls.data(), mpi_record, MPI_COMM_WORLD);

    const double t1 = MPI_Wtime();
    time_redistribution = t1 - t0;

    return recvbuf;
}

// ------------------------------------------------------------
// MPI partitioned hash join
// ------------------------------------------------------------

static JoinResult partitioned_hash_join_mpi(const std::vector<Record>& R_local, const std::vector<Record>& S_local,
                                            std::uint32_t p, int rank, int world_size, MPI_Datatype mpi_record,
                                            PhaseTimes& times)
{
    // Synchronize before starting the measured region.
    MPI_Barrier(MPI_COMM_WORLD);

    const double t0 = MPI_Wtime();

    // Redistribute both relations so that each rank receives the records
    // belonging to the partitions it owns.
    std::vector<Record> R_owned =
        redistribute_relation(R_local, p, world_size, mpi_record, times.redistribution_R);

    std::vector<Record> S_owned =
        redistribute_relation(S_local, p, world_size, mpi_record, times.redistribution_S);

    // Local partitioning after MPI redistribution.
    // The data is now distributed across ranks, but each rank still needs contiguous
    // partition ranges to reuse the same join_one_partition logic.
    const PartitionedRelation Rpart = partition_relation(R_owned, p, times.R);
    const PartitionedRelation Spart = partition_relation(S_owned, p, times.S);

    JoinResult local_result{};

    const double tj0 = MPI_Wtime();

    // Each rank joins only the partitions assigned to it.
    // With owner(pid) = pid % world_size, rank r owns:
    // r, r + world_size, r + 2*world_size, ...
    for (std::uint32_t pid = static_cast<std::uint32_t>(rank); pid < p;
         pid += static_cast<std::uint32_t>(world_size))
    {
        const JoinResult part = join_one_partition(Rpart, Spart, pid);

        local_result.join_count += part.join_count;
        local_result.checksum1 += part.checksum1;
        local_result.checksum2 += part.checksum2;
    }

    const double tj1 = MPI_Wtime();
    times.join_local = tj1 - tj0;

    const std::uint64_t local_values[3] = {
        local_result.join_count,
        local_result.checksum1,
        local_result.checksum2,
    };

    std::uint64_t global_values[3] = {0, 0, 0};

    const double ta0 = MPI_Wtime();

    // Final global accumulation.
    // This replaces the final sequential accumulation over partition results.
    MPI_Reduce(local_values, global_values, 3, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    // Only rank 0 receives the global result. Other ranks return an empty
    // JoinResult because only rank 0 prints and writes the final output.
    const double ta1 = MPI_Wtime();
    times.accumulation = ta1 - ta0;

    const double t1 = MPI_Wtime();
    times.total = t1 - t0;

    JoinResult global_result{};

    if (rank == 0)
    {
        global_result.join_count = global_values[0];
        global_result.checksum1 = global_values[1];
        global_result.checksum2 = global_values[2];
    }

    return global_result;
}

// ------------------------------------------------------------
// JSON output
// ------------------------------------------------------------
//
// The Python orchestrator reads this JSON file to collect timings, checksums,
// and metadata for the experimental campaign.

static void write_run_json_mpi(const std::string& path, std::size_t NR, std::size_t NS, std::uint32_t P,
                               std::uint64_t seed, std::uint64_t max_key, const std::string& workload,
                               std::uint32_t hot_partitions, std::uint32_t skew_percent, int world_size,
                               const JoinResult& result, const PhaseTimes& times)
{
    std::ofstream out(path);

    if (!out)
    {
        std::cerr << "Error: cannot open JSON output file: " << path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"mode\": \"mpi\",\n";
    out << "  \"ranks\": " << world_size << ",\n";

    out << "  \"NR\": " << NR << ",\n";
    out << "  \"NS\": " << NS << ",\n";
    out << "  \"P\": " << P << ",\n";
    out << "  \"seed\": " << seed << ",\n";
    out << "  \"max_key\": " << max_key << ",\n";
    out << "  \"workload\": \"" << workload << "\",\n";
    out << "  \"hot_partitions\": " << hot_partitions << ",\n";
    out << "  \"skew_percent\": " << skew_percent << ",\n";

    out << "  \"join_count\": " << result.join_count << ",\n";
    out << "  \"checksum1\": " << result.checksum1 << ",\n";
    out << "  \"checksum2\": " << result.checksum2 << ",\n";

    out << "  \"time_redistribution_R\": " << times.redistribution_R << ",\n";
    out << "  \"time_redistribution_S\": " << times.redistribution_S << ",\n";

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

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv)
{
    // MPI must be initialized before using any MPI routine.
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::uint64_t nr = 0;
    std::uint64_t ns = 0;
    std::uint64_t seed = 0;
    std::uint64_t max_key = 0;
    std::uint64_t p = 0;

    std::string json_path;
    const bool save_json = read_arg_string(argc, argv, "-json", json_path);

    std::string workload = "uniform";
    read_arg_string(argc, argv, "-workload", workload);

    std::uint64_t hot_partitions_arg = 0;
    std::uint64_t skew_percent_arg = 90;

    read_arg_u64(argc, argv, "-hot-partitions", hot_partitions_arg);
    read_arg_u64(argc, argv, "-skew-percent", skew_percent_arg);

    if (!read_arg_u64(argc, argv, "-nr", nr) || !read_arg_u64(argc, argv, "-ns", ns) ||
        !read_arg_u64(argc, argv, "-seed", seed) || !read_arg_u64(argc, argv, "-max-key", max_key) ||
        !read_arg_u64(argc, argv, "-p", p))
    {
        if (rank == 0)
        {
            usage(argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    if (p == 0 || p > std::numeric_limits<std::uint32_t>::max())
    {
        if (rank == 0)
        {
            std::cerr << "Error: P must be in [1, UINT32_MAX].\n";
        }

        MPI_Finalize();
        return 1;
    }

    const std::uint32_t P = static_cast<std::uint32_t>(p);

    if (!is_power_of_two(P))
    {
        if (rank == 0)
        {
            std::cerr << "Error: in this implementation, P must be a power of two.\n";
        }

        MPI_Finalize();
        return 1;
    }

    if (workload != "uniform" && workload != "skewed")
    {
        if (rank == 0)
        {
            std::cerr << "Error: workload must be either 'uniform' or 'skewed'.\n";
        }

        MPI_Finalize();
        return 1;
    }

    std::uint32_t hot_partitions = 0;

    if (hot_partitions_arg == 0)
    {
        hot_partitions = std::max<std::uint32_t>(1, P / 64);
    }
    else
    {
        if (hot_partitions_arg > std::numeric_limits<std::uint32_t>::max())
        {
            if (rank == 0)
            {
                std::cerr << "Error: hot-partitions too large.\n";
            }

            MPI_Finalize();
            return 1;
        }

        hot_partitions = static_cast<std::uint32_t>(hot_partitions_arg);
    }

    if (hot_partitions == 0 || hot_partitions > P)
    {
        if (rank == 0)
        {
            std::cerr << "Error: hot-partitions must be in [1, P].\n";
        }

        MPI_Finalize();
        return 1;
    }

    if (skew_percent_arg > 100)
    {
        if (rank == 0)
        {
            std::cerr << "Error: skew-percent must be in [0, 100].\n";
        }

        MPI_Finalize();
        return 1;
    }

    const std::uint32_t skew_percent = static_cast<std::uint32_t>(skew_percent_arg);

    const std::size_t NR = static_cast<std::size_t>(nr);
    const std::size_t NS = static_cast<std::size_t>(ns);

    // The global relation is split into almost equally sized contiguous ranges.
    // The first ranks receive one extra element when n is not divisible by
    // world_size. This keeps generation distributed and deterministic.
    const std::size_t R_begin = block_start(NR, rank, world_size);
    const std::size_t R_count = block_count(NR, rank, world_size);

    const std::size_t S_begin = block_start(NS, rank, world_size);
    const std::size_t S_count = block_count(NS, rank, world_size);

    std::vector<Record> R_local;
    std::vector<Record> S_local;

    try
    {
        R_local = generate_relation_range(R_begin, R_count, seed, max_key, P, workload, hot_partitions, skew_percent);

        S_local = generate_relation_range(S_begin, S_count, seed ^ 0xdeadebdecdeedef1ULL, max_key, P, workload,
                                          hot_partitions, skew_percent);
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
        {
            std::cerr << "Error while generating local input relation: " << e.what() << "\n";
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // MPI datatype for Record.
    // Record currently contains only one uint64_t key, so a contiguous datatype is enough.
    MPI_Datatype mpi_record;
    MPI_Type_contiguous(1, MPI_UINT64_T, &mpi_record);
    MPI_Type_commit(&mpi_record);

    PhaseTimes local_times;
    JoinResult result{};

    try
    {
        result = partitioned_hash_join_mpi(R_local, S_local, P, rank, world_size, mpi_record, local_times);
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
        {
            std::cerr << "Error during MPI hash join: " << e.what() << "\n";
        }

        MPI_Type_free(&mpi_record);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Gather timing information using MPI_MAX for each phase.
    // Only rank 0 receives meaningful global_times values.
    PhaseTimes global_times = collect_phase_times_max(local_times);

    if (rank == 0)
    {
        std::cout << "mode=mpi"
                  << " ranks=" << world_size
                  << " NR=" << NR
                  << " NS=" << NS
                  << " P=" << P
                  << " seed=" << seed
                  << " workload=" << workload
                  << " hot_partitions=" << hot_partitions
                  << " skew_percent=" << skew_percent
                  << " [0, " << max_key << ")\n";

        std::cout << "join_count=" << result.join_count << "\n";
        std::cout << "checksum1=" << result.checksum1 << "\n";
        std::cout << "checksum2=" << result.checksum2 << "\n";

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "time_redistribution_R=" << global_times.redistribution_R << "\n";
        std::cout << "time_redistribution_S=" << global_times.redistribution_S << "\n";

        std::cout << "time_histogram_R=" << global_times.R.histogram << "\n";
        std::cout << "time_prefix_R=" << global_times.R.prefix << "\n";
        std::cout << "time_scatter_R=" << global_times.R.scatter << "\n";
        std::cout << "time_partition_R=" << global_times.R.total << "\n";

        std::cout << "time_histogram_S=" << global_times.S.histogram << "\n";
        std::cout << "time_prefix_S=" << global_times.S.prefix << "\n";
        std::cout << "time_scatter_S=" << global_times.S.scatter << "\n";
        std::cout << "time_partition_S=" << global_times.S.total << "\n";

        std::cout << "time_join=" << global_times.join_local << "\n";
        std::cout << "time_accumulation=" << global_times.accumulation << "\n";
        std::cout << "time_total=" << global_times.total << "\n";
        std::cout << "time_sec=" << global_times.total << "\n";

        if (save_json)
        {
            write_run_json_mpi(json_path, NR, NS, P, seed, max_key, workload, hot_partitions, skew_percent,
                               world_size, result, global_times);
        }
    }

    MPI_Type_free(&mpi_record);
    MPI_Finalize();

    return 0;
}