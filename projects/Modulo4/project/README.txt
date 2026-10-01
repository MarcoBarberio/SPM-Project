Compilation using the Makefile:

- make

Builds:

- build/hashjoin_seq
- build/hashjoin_mpi

Clean:

- make clean


Direct compilation:

Sequential baseline:

- g++ -I./include -std=c++20 -Wall -pthread -O3 src/hashjoin_seq.cpp -o build/hashjoin_seq

Pure MPI implementation:

- mpicxx -I./include -std=c++20 -Wall -pthread -O3 src/hashjoin_mpi.cpp -o build/hashjoin_mpi


PROGRAM EXECUTION


1) Sequential baseline

Command:

- ./build/hashjoin_seq -nr NR -ns NS -seed SEED -max-key K -p P -workload WORKLOAD -hot-partitions H -skew-percent SKEW [-json PATH]

Parameters:

- -nr              number of records in relation R
- -ns              number of records in relation S
- -seed            deterministic seed used for data generation
- -max-key         keys generated in [0, max-key)
- -p               number of partitions
- -workload        input workload type: uniform or skewed
- -hot-partitions  number of hot partitions used for the skewed workload
- -skew-percent    percentage of records assigned to hot partitions in the skewed workload
- -json            optional, saves run data in a JSON file

Constraints:

- P must be a power of 2
- workload must be either uniform or skewed
- hot-partitions must be at least 1
- skew-percent must be between 0 and 100

Examples:

Uniform workload:

- ./build/hashjoin_seq -nr 1000000 -ns 1000000 -seed 1 -max-key 100000 -p 1024 -workload uniform -hot-partitions 16 -skew-percent 90

Skewed workload:

- ./build/hashjoin_seq -nr 1000000 -ns 1000000 -seed 1 -max-key 100000 -p 1024 -workload skewed -hot-partitions 16 -skew-percent 90

With JSON output:

- ./build/hashjoin_seq -nr 1000 -ns 1000 -seed 1 -max-key 1000 -p 16 -workload uniform -hot-partitions 4 -skew-percent 90 -json results/seq_run.json

Output:

- join_count
- checksum1
- checksum2
- phase times
- time_total / time_sec

For very small inputs, namely NR <= 500 and NS <= 500, it also prints:

- naive_join_count
- naive_checksum1
- naive_checksum2


2) Pure MPI implementation

The MPI executable must be launched with an MPI launcher such as srun or mpirun.
On the spmcluster, the experiments were executed with srun.

Command:

- srun -N NODES -n RANKS --ntasks-per-node RANKS_PER_NODE ./build/hashjoin_mpi -nr NR -ns NS -seed SEED -max-key K -p P -workload WORKLOAD -hot-partitions H -skew-percent SKEW [-json PATH]

Parameters:

- -nr              number of records in relation R
- -ns              number of records in relation S
- -seed            deterministic seed used for data generation
- -max-key         keys generated in [0, max-key)
- -p               number of partitions
- -workload        input workload type: uniform or skewed
- -hot-partitions  number of hot partitions used for the skewed workload
- -skew-percent    percentage of records assigned to hot partitions in the skewed workload
- -json            optional, saves run data in a JSON file

MPI launch parameters:

- -N               number of nodes
- -n               total number of MPI ranks
- --ntasks-per-node number of MPI ranks per node

Constraints:

- P must be a power of 2
- workload must be either uniform or skewed
- hot-partitions must be at least 1
- skew-percent must be between 0 and 100
- the number of MPI ranks must be at least 1

Examples:

Uniform workload, 4 nodes and 1 rank per node:

- srun -N 4 -n 4 --ntasks-per-node 1 ./build/hashjoin_mpi -nr 1000000 -ns 1000000 -seed 1 -max-key 100000 -p 1024 -workload uniform -hot-partitions 16 -skew-percent 90

Skewed workload, 4 nodes and 1 rank per node:

- srun -N 4 -n 4 --ntasks-per-node 1 ./build/hashjoin_mpi -nr 1000000 -ns 1000000 -seed 1 -max-key 100000 -p 1024 -workload skewed -hot-partitions 16 -skew-percent 90

With JSON output:

- srun -N 2 -n 2 --ntasks-per-node 1 ./build/hashjoin_mpi -nr 1000 -ns 1000 -seed 1 -max-key 1000 -p 16 -workload uniform -hot-partitions 4 -skew-percent 90 -json results/mpi_run.json

Output:

- join_count
- checksum1
- checksum2
- phase times
- time_total / time_sec

The MPI implementation also reports communication-related phase times, including:

- time_redistribution_R
- time_redistribution_S

Only rank 0 prints the final aggregated output and writes the JSON file.


WORKLOADS


1) Uniform workload

In the uniform workload, keys are generated so that records are approximately evenly distributed across partitions.

Example:

- -workload uniform


2) Skewed workload

In the skewed workload, a given percentage of records is forced to map to a small subset of hot partitions.

Example:

- -workload skewed -hot-partitions 16 -skew-percent 90

This means that approximately 90% of the records are assigned to 16 hot partitions.

The purpose of the skewed workload is to create load imbalance across partitions and evaluate how the MPI implementation behaves when the amount of local join work and the amount of received data are less evenly distributed across ranks.


CORRECTNESS CHECKS

Correctness is verified by comparing:

- join_count
- checksum1
- checksum2

For small inputs, the sequential program may also compare the result against a naive reference join.

For larger inputs, correctness is checked by comparing the final aggregate values produced by:

- sequential baseline
- pure MPI implementation

The orchestrator automatically stores the field:

- checksum_correct

checksum_correct = true means that the MPI implementation produced the same:

- join_count
- checksum1
- checksum2

as the sequential baseline.

If checksum_correct is false for any run, the orchestrator stops with an error.


ORCHESTRATOR

Script:

- python3 orchestrator.py [parameters]

The orchestrator is used to execute:

- correctness experiments
- strong scaling experiments
- weak scaling experiments

It automatically runs the sequential baseline and the MPI implementation on the same input parameters, collects the JSON output of each run, verifies correctness using join_count and checksums, and stores the results in a single JSON file.

MPI executions are grouped by node count. With the default node list 1, 2, 4, 8, the orchestrator uses one srun invocation per node count.


Available parameters:


General:

- --run correctness strong weak
  chooses which experiment groups to run

- --workloads uniform skewed
  chooses which workloads to test

- --nodes N1 N2 ...
  list of node counts for MPI runs

- --ranks-per-node R
  number of MPI ranks per node

- --ps P1 P2 ...
  list of partition counts

- --seeds S1 S2 ...
  list of seeds

- --max-key K
  value of max_key

- --hot-partitions H
  number of hot partitions for the skewed workload

- --skew-percent SKEW
  percentage of records assigned to hot partitions for skewed workload

- --timeout-sec X
  timeout for a single MPI command inside each srun batch, in seconds

- --results-file PATH
  output JSON file for results

- --config-file PATH
  output JSON file for the experimental configuration

- --append
  appends results to the existing file instead of overwriting it


Correctness:

- --correctness-cases-file PATH
  optional JSON file containing correctness cases


Strong scaling:

- --strong-ns N1 N2 ...
  fixed problem sizes for strong scaling experiments


Weak scaling:

- --weak-base-n N
  base workload per MPI rank


DEFAULT PARAMETERS

If no parameters are passed, the defaults are:

- --run correctness strong weak
- --workloads uniform skewed
- --nodes 1 2 4 8
- --ranks-per-node 1
- --ps 1024
- --seeds 0 1 2 3 4
- --max-key 100000
- --hot-partitions 16
- --skew-percent 90
- --timeout-sec 60

Strong scaling defaults:

- --strong-ns 10000000

Weak scaling defaults:

- --weak-base-n 20000


ORCHESTRATOR EXAMPLES


1) Only correctness:

- python3 orchestrator.py --run correctness


2) Only strong scaling:

- python3 orchestrator.py --run strong --strong-ns 10000000 --nodes 1 2 4 8 --ranks-per-node 1


3) Only weak scaling:

- python3 orchestrator.py --run weak --weak-base-n 20000 --nodes 1 2 4 8 --ranks-per-node 1


4) Uniform workload only:

- python3 orchestrator.py --workloads uniform


5) Skewed workload only:

- python3 orchestrator.py --workloads skewed --hot-partitions 16 --skew-percent 90


6) Complete run with correctness, strong scaling, and weak scaling:

- python3 orchestrator.py --run correctness strong weak


7) Complete run with custom parameters:

- python3 orchestrator.py --run correctness strong weak --workloads uniform skewed --nodes 1 2 4 8 --ranks-per-node 1 --ps 1024 --seeds 0 1 2 --max-key 100000 --hot-partitions 16 --skew-percent 90 --strong-ns 10000000 --weak-base-n 20000


8) Parameters used in my analysis:

python3 orchestrator.py \
  --run correctness strong weak \
  --workloads uniform skewed \
  --nodes 1 2 4 8 \
  --ranks-per-node 1 \
  --ps 1024 \
  --seeds 0 1 2 3 4 \
  --max-key 100000 \
  --hot-partitions 16 \
  --skew-percent 90 \
  --strong-ns 10000000 \
  --weak-base-n 20000 \
  --timeout-sec 300


GENERATED FILES

The orchestrator saves:

- results/results_modulo4.json
- results/experiment_config_modulo4.json

Temporary JSON files and logs are created during each run inside:

- results/tmp_modulo4_batch/

Per-node batch scripts and logs may be kept for debugging. Per-case JSON files are removed automatically after being parsed.

Each run with -json saves a JSON file containing:

- run parameters
- join_count
- checksum1
- checksum2
- phase times
- time_total


RESULT FIELDS

Each result record contains, among others:

- experiment_type
- implementation
- NR
- NS
- P
- seed
- max_key
- workload
- hot_partitions
- skew_percent
- nodes
- ranks
- ranks_per_node
- N_per_rank
- time_seq
- time_mpi
- speedup_vs_seq
- weak_efficiency
- join_count_seq
- join_count_mpi
- checksum1_seq
- checksum1_mpi
- checksum2_seq
- checksum2_mpi
- checksum_correct
- phase timings for the sequential version
- phase timings for the MPI implementation


PHASE TIMES

The programs may report the following phase timings:

- time_redistribution_R
- time_redistribution_S
- time_partition_R
- time_partition_S
- time_histogram_R
- time_prefix_R
- time_scatter_R
- time_histogram_S
- time_prefix_S
- time_scatter_S
- time_join
- time_accumulation
- time_total

These phase timings are useful to analyze where the execution time is spent and to compare the cost of communication/redistribution with the cost of local partitioning and local joining.
