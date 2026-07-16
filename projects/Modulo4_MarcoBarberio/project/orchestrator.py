# This script orchestrates the experimental campaign for Module 4.
# It executes the sequential baseline and the pure MPI implementation
# on the same inputs, collects their JSON outputs, and stores all results
# MPI executions are grouped by node count to reduce the number of srun
# launches. With the default nodes [1, 2, 4, 8], the MPI part uses one
# srun call per node count.
import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path


# PATHS

BUILD_DIR = Path("./build")

SEQ_EXE = BUILD_DIR / "hashjoin_seq"
MPI_EXE = BUILD_DIR / "hashjoin_mpi"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "results_modulo4.json"
CONFIG_FILE = RESULTS_DIR / "experiment_config_modulo4.json"

TMP_DIR = RESULTS_DIR / "tmp_modulo4_batch"
TMP_DIR.mkdir(exist_ok=True)


# DEFAULT PARAMETERS

DEFAULT_PS = [1024]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_MAX_KEY = 100_000

# Node counts required by the assignment. 
# The pure MPI implementation uses one rank per node. 
DEFAULT_NODES = [1, 2, 4, 8]
DEFAULT_RANKS_PER_NODE = 1

DEFAULT_WORKLOADS = ["uniform", "skewed"]

DEFAULT_HOT_PARTITIONS = 16
DEFAULT_SKEW_PERCENT = 90

# Small input cases used to validate correctness. 
# The MPI output is compared against the sequential baseline using 
# join_count, checksum1, and checksum2.
DEFAULT_CORRECTNESS_CASES = [
    {"NR": 0, "NS": 0, "P": 4},
    {"NR": 1, "NS": 1, "P": 4},
    {"NR": 100, "NS": 100, "P": 8},
    {"NR": 500, "NS": 500, "P": 16},
]

DEFAULT_STRONG_NS = [10_000_000]
DEFAULT_WEAK_BASE_N = 20_000

# This is used as timeout per individual MPI command inside the batch script.
DEFAULT_TIMEOUT_SEC = 60

# Phase timing keys expected from the C++ executables. 
# These values are copied into the final result records to allow a breakdown
# of communication, local partitioning, joining, and accumulation phases.
PHASE_KEYS = [
    "time_redistribution_R",
    "time_redistribution_S",
    "time_partition_R",
    "time_partition_S",
    "time_join",
    "time_accumulation",
    "time_total",
    "time_histogram_R",
    "time_prefix_R",
    "time_scatter_R",
    "time_histogram_S",
    "time_prefix_S",
    "time_scatter_S",
]


# ARGUMENT PARSING

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Module 4 experiments with only one srun per node count"
    )

    parser.add_argument(
        "--run",
        nargs="+",
        choices=["correctness", "strong", "weak"],
        default=["correctness", "strong", "weak"],
        help="Experiment groups to run",
    )

    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["uniform", "skewed"],
        default=DEFAULT_WORKLOADS,
        help="Workloads to test",
    )

    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=DEFAULT_NODES,
        help="Node counts for MPI runs",
    )

    parser.add_argument(
        "--ranks-per-node",
        type=int,
        default=DEFAULT_RANKS_PER_NODE,
        help="MPI ranks per node",
    )

    parser.add_argument(
        "--ps",
        nargs="+",
        type=int,
        default=DEFAULT_PS,
        help="Partition counts P",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to use",
    )

    parser.add_argument(
        "--max-key",
        type=int,
        default=DEFAULT_MAX_KEY,
        help="max_key parameter",
    )

    parser.add_argument(
        "--hot-partitions",
        type=int,
        default=DEFAULT_HOT_PARTITIONS,
        help="Number of hot partitions for skewed workload",
    )

    parser.add_argument(
        "--skew-percent",
        type=int,
        default=DEFAULT_SKEW_PERCENT,
        help="Percentage of records assigned to hot partitions for skewed workload",
    )

    parser.add_argument(
        "--correctness-cases-file",
        type=str,
        default=None,
        help="Optional JSON file containing correctness cases",
    )

    parser.add_argument(
        "--strong-ns",
        nargs="+",
        type=int,
        default=DEFAULT_STRONG_NS,
        help="Fixed problem sizes N for strong scaling",
    )

    parser.add_argument(
        "--weak-base-n",
        type=int,
        default=DEFAULT_WEAK_BASE_N,
        help="Base N per MPI rank for weak scaling",
    )

    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Timeout per individual MPI command inside each srun batch",
    )

    parser.add_argument(
        "--results-file",
        type=str,
        default=str(RESULTS_FILE),
        help="Output results JSON path",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default=str(CONFIG_FILE),
        help="Output config JSON path",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing results file instead of overwriting",
    )

    return parser.parse_args()


# BASIC HELPERS

def check_executables():
    if not SEQ_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {SEQ_EXE}")

    if not MPI_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {MPI_EXE}")


def validate_power_of_two(p):
    return p > 0 and (p & (p - 1)) == 0


def load_correctness_cases(path):
    if path is None:
        return DEFAULT_CORRECTNESS_CASES

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_hot_partitions(hot_partitions, p, workload, skew_percent):
    p = int(p)
    hot = max(1, min(int(hot_partitions), p))

    # For skewed workload with skew_percent < 100, keep at least one cold partition.
    if workload == "skewed" and int(skew_percent) < 100 and p > 1:
        hot = min(hot, p - 1)

    return hot


def common_args(nr, ns, seed, max_key, p, workload, hot_partitions, skew_percent):
    if not validate_power_of_two(p):
        raise ValueError(f"P must be a power of two. Got P={p}")

    hot = normalize_hot_partitions(hot_partitions, p, workload, skew_percent)

    return [
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
        "-workload", workload,
        "-hot-partitions", str(hot),
        "-skew-percent", str(skew_percent),
    ]


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def case_key(experiment_type, nr, ns, p, seed, workload, nodes, ranks_per_node):
    return (
        experiment_type,
        int(nr),
        int(ns),
        int(p),
        int(seed),
        str(workload),
        int(nodes),
        int(ranks_per_node),
    )


def seq_key(nr, ns, p, seed, workload):
    return (
        int(nr),
        int(ns),
        int(p),
        int(seed),
        str(workload),
    )


def get_time(run_data):
    return run_data.get("time_total", run_data.get("time_sec"))


def checksums_equal(a, b):
    return (
        a["join_count"] == b["join_count"]
        and a["checksum1"] == b["checksum1"]
        and a["checksum2"] == b["checksum2"]
    )


# Runs the sequential baseline once and asks it to write detailed measurements 
# to a temporary JSON file. Standard output is suppressed because the 
# orchestrator reads the generated JSON directly.
def run_seq(nr, ns, seed, max_key, p, workload, hot_partitions, skew_percent, timeout_sec):
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="seq_", suffix=".json", dir=TMP_DIR)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    err_path = tmp_path.with_suffix(".err")

    cmd = [str(SEQ_EXE)] + common_args(
        nr, ns, seed, max_key, p, workload, hot_partitions, skew_percent
    ) + ["-json", str(tmp_path)]

    try:
        with open(err_path, "w", encoding="utf-8") as err_file:
            subprocess.run(
                cmd,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
                text=True,
                timeout=timeout_sec,
            )

        if not tmp_path.exists():
            raise RuntimeError(f"Expected JSON output file not found: {tmp_path}")

        return load_json_file(tmp_path)

    except subprocess.TimeoutExpired:
        print("\nSequential command timed out:")
        print(shell_join(cmd))
        if err_path.exists():
            print(err_path.read_text(encoding="utf-8", errors="replace"))
        raise

    except subprocess.CalledProcessError as exc:
        print("\nSequential command failed:")
        print(shell_join(cmd))
        print(f"Return code: {exc.returncode}")
        if err_path.exists():
            print(err_path.read_text(encoding="utf-8", errors="replace"))
        raise

    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if err_path.exists():
            err_path.unlink()


# ONE SRUN PER NODE COUNT

def run_mpi_batch_for_nodes(cases, nodes, ranks_per_node, timeout_sec):
    """
    Runs all MPI cases associated with a given node count using a single srun call.

    The generated bash script is executed by all MPI tasks. Each task launches
    the same hashjoin_mpi command, so each command forms one MPI execution for
    the current node count. This reduces Slurm launch overhead while preserving
    the same MPI configuration for each experiment.
    """

    if not cases:
        return {}

    ranks = nodes * ranks_per_node

    batch_dir = TMP_DIR / f"nodes_{nodes}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    script_path = batch_dir / f"mpi_batch_nodes_{nodes}.sh"
    log_path = batch_dir / f"mpi_batch_nodes_{nodes}.log"

    json_paths = {}

    with open(script_path, "w", encoding="utf-8") as script:
        script.write("#!/usr/bin/env bash\n")
        script.write("set -euo pipefail\n\n")
        script.write('rank="${SLURM_PROCID:-0}"\n\n')

        for i, c in enumerate(cases):
            json_path = batch_dir / f"case_{i:05d}.json"
            json_paths[c["case_id"]] = json_path

            mpi_cmd = [str(MPI_EXE)] + common_args(
                c["NR"],
                c["NS"],
                c["seed"],
                c["max_key"],
                c["P"],
                c["workload"],
                c["hot_partitions"],
                c["skew_percent"],
            ) + ["-json", str(json_path)]

            exp_name = str(c["experiment_type"]).upper()

            label = (
                f"[{exp_name}][MPI][{c['workload']}] "
                f"NR={c['NR']} NS={c['NS']} P={c['P']} "
                f"seed={c['seed']} nodes={nodes} ranks={ranks}"
            )

            # Only the shell running on rank 0 prints progress, but all ranks
            # execute the MPI binary on the next line.
            script.write(f'if [ "$rank" = "0" ]; then echo {shlex.quote(label)}; fi\n')

            # All rank-shells execute the same MPI command simultaneously. 
            # Normal stdout/stderr is redirected to per-rank logs because the orchestrator 
            # reads the JSON output files to build the final result records.
            
            raw_log = batch_dir / f"case_{i:05d}_rank_${{rank}}.out"
            script.write(
                f"timeout --kill-after=10s {int(timeout_sec)}s {shell_join(mpi_cmd)} "
                f"> {shlex.quote(str(raw_log))} 2>&1\n\n"
            )

    os.chmod(script_path, 0o755)

    srun_cmd = [
        "srun",
        "--kill-on-bad-exit=1",
        "--mpi=pmix",
        "-N", str(nodes),
        "-n", str(ranks),
        "--ntasks-per-node", str(ranks_per_node),
        "bash", str(script_path),
    ]

    # Give the whole batch a larger Python-side timeout.
    # Individual MPI commands are already protected by the shell timeout above.
    batch_timeout = int(timeout_sec) * max(1, len(cases)) + 120

    print("\n==============================")
    print(f"MPI BATCH: nodes={nodes}, ranks={ranks}, cases={len(cases)}")
    print("==============================")

    try:
        # Show the srun output live on the terminal, while also saving it
        # to the batch log file. The shell uses pipefail so that a failing
        # srun still makes this subprocess fail even though output is piped
        # through tee.
        tee_cmd = [
            "bash",
            "-lc",
            f"set -o pipefail; {shell_join(srun_cmd)} 2>&1 | tee {shlex.quote(str(log_path))}",
        ]

        subprocess.run(
            tee_cmd,
            check=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=batch_timeout,
        )

    except subprocess.TimeoutExpired:
        print("\nMPI batch timed out:")
        print(shell_join(srun_cmd))
        if log_path.exists():
            print("\nBATCH LOG:")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
        raise

    except subprocess.CalledProcessError as exc:
        print("\nMPI batch failed:")
        print(shell_join(srun_cmd))
        print(f"Return code: {exc.returncode}")
        if log_path.exists():
            print("\nBATCH LOG:")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
        raise

    out = {}

    for cid, path in json_paths.items():
        if not path.exists():
            raise RuntimeError(f"Expected JSON output file not found: {path}")

        out[cid] = load_json_file(path)

    # Keep script/log for debugging, remove only per-case JSONs.
    for path in json_paths.values():
        if path.exists():
            path.unlink()

    return out


# Adds one MPI experiment case to the list associated with its node count.
# Cases are grouped by node count so that they can be executed in batches.

def add_case(cases_by_nodes, experiment_type, nr, ns, p, seed, workload,
             nodes, ranks_per_node, cfg):
    cid = "|".join(map(str, case_key(
        experiment_type, nr, ns, p, seed, workload, nodes, ranks_per_node
    )))

    cases_by_nodes.setdefault(nodes, []).append({
        "case_id": cid,
        "experiment_type": experiment_type,
        "NR": int(nr),
        "NS": int(ns),
        "P": int(p),
        "seed": int(seed),
        "max_key": int(cfg["max_key"]),
        "workload": workload,
        "hot_partitions": int(cfg["hot_partitions"]),
        "skew_percent": int(cfg["skew_percent"]),
        "nodes": int(nodes),
        "ranks_per_node": int(ranks_per_node),
    })

# Builds all MPI cases requested by the configuration.
# Strong scaling keeps the input size fixed, while weak scaling increases 
# the input size proportionally to the number of MPI ranks.

def build_all_mpi_cases(cfg):
    cases_by_nodes = {int(n): [] for n in cfg["nodes"]}

    if "correctness" in cfg["run"]:
        correctness_nodes = [n for n in cfg["nodes"] if n in (1, 2)]
        if not correctness_nodes:
            correctness_nodes = [cfg["nodes"][0]]

        for workload in cfg["workloads"]:
            for case in cfg["correctness_cases"]:
                nr = int(case["NR"])
                ns = int(case["NS"])
                p = int(case["P"])

                for seed in cfg["seeds"]:
                    for nodes in correctness_nodes:
                        add_case(
                            cases_by_nodes, "correctness",
                            nr, ns, p, seed, workload,
                            nodes, cfg["ranks_per_node"], cfg
                        )

    if "strong" in cfg["run"]:
        for workload in cfg["workloads"]:
            for p in cfg["ps"]:
                for n in cfg["strong_ns"]:
                    nr = int(n)
                    ns = int(n)

                    for seed in cfg["seeds"]:
                        for nodes in cfg["nodes"]:
                            add_case(
                                cases_by_nodes, "strong",
                                nr, ns, p, seed, workload,
                                nodes, cfg["ranks_per_node"], cfg
                            )

    if "weak" in cfg["run"]:
        for workload in cfg["workloads"]:
            for p in cfg["ps"]:
                for seed in cfg["seeds"]:
                    for nodes in cfg["nodes"]:
                        ranks = nodes * cfg["ranks_per_node"]
                        nr = int(cfg["weak_base_n"] * ranks)
                        ns = int(cfg["weak_base_n"] * ranks)

                        add_case(
                            cases_by_nodes, "weak",
                            nr, ns, p, seed, workload,
                            nodes, cfg["ranks_per_node"], cfg
                        )

    return cases_by_nodes


# Builds the final result entry stored in results_modulo4.json. 
# Each record contains experiment metadata, sequential and MPI timings,
# speedup, weak-scaling efficiency, checksum comparisons, and phase timings.

def build_result_record(case, seq_data, mpi_data, weak_efficiency=None):
    nodes = int(case["nodes"])
    ranks_per_node = int(case["ranks_per_node"])
    ranks = nodes * ranks_per_node

    nr = int(case["NR"])
    ns = int(case["NS"])
    p = int(case["P"])
    seed = int(case["seed"])
    workload = case["workload"]
    skew_percent = int(case["skew_percent"])
    hot_partitions = int(case["hot_partitions"])

    seq_time = get_time(seq_data)
    mpi_time = get_time(mpi_data)

    checksum_correct = checksums_equal(seq_data, mpi_data)
    speedup_vs_seq = seq_time / mpi_time if mpi_time and mpi_time > 0 else 0.0

    record = {
        "experiment_type": case["experiment_type"],
        "implementation": "mpi",
        "NR": nr,
        "NS": ns,
        "P": p,
        "seed": seed,
        "max_key": int(case["max_key"]),
        "workload": workload,
        "hot_partitions": normalize_hot_partitions(hot_partitions, p, workload, skew_percent),
        "skew_percent": skew_percent,

        "nodes": nodes,
        "ranks": ranks,
        "ranks_per_node": ranks_per_node,
        "N_per_rank": nr / ranks if ranks > 0 else None,

        "time_seq": seq_time,
        "time_mpi": mpi_time,
        "speedup_vs_seq": speedup_vs_seq,
        "weak_efficiency": weak_efficiency,

        "join_count_seq": seq_data["join_count"],
        "join_count_mpi": mpi_data["join_count"],
        "checksum1_seq": seq_data["checksum1"],
        "checksum1_mpi": mpi_data["checksum1"],
        "checksum2_seq": seq_data["checksum2"],
        "checksum2_mpi": mpi_data["checksum2"],

        "checksum_correct": checksum_correct,
    }

    if "naive_join_count" in seq_data:
        record["naive_join_count"] = seq_data.get("naive_join_count")
        record["naive_checksum1"] = seq_data.get("naive_checksum1")
        record["naive_checksum2"] = seq_data.get("naive_checksum2")

    for key in PHASE_KEYS:
        record[f"{key}_seq"] = seq_data.get(key)

    for key in PHASE_KEYS:
        record[f"{key}_mpi"] = mpi_data.get(key)

    return record


# MAIN

def main():
    args = parse_args()
    check_executables()

    results_file = Path(args.results_file)
    config_file = Path(args.config_file)

    if args.append and results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    cfg = {
        "run": args.run,
        "workloads": args.workloads,
        "nodes": args.nodes,
        "ranks_per_node": args.ranks_per_node,
        "ps": args.ps,
        "seeds": args.seeds,
        "max_key": args.max_key,
        "hot_partitions": args.hot_partitions,
        "skew_percent": args.skew_percent,
        "correctness_cases": load_correctness_cases(args.correctness_cases_file),
        "strong_ns": args.strong_ns,
        "weak_base_n": args.weak_base_n,
        "timeout_sec": args.timeout_sec,
        "results_file": results_file,
        "config_file": config_file,
    }

    save_json(config_file, {
        "run": cfg["run"],
        "executables": {
            "seq": str(SEQ_EXE),
            "mpi": str(MPI_EXE),
        },
        "workloads": cfg["workloads"],
        "nodes": cfg["nodes"],
        "ranks_per_node": cfg["ranks_per_node"],
        "ps": cfg["ps"],
        "seeds": cfg["seeds"],
        "max_key": cfg["max_key"],
        "hot_partitions": cfg["hot_partitions"],
        "skew_percent": cfg["skew_percent"],
        "correctness_cases": cfg["correctness_cases"],
        "strong_ns": cfg["strong_ns"],
        "weak_base_n": cfg["weak_base_n"],
        "timeout_sec_per_mpi_case": cfg["timeout_sec"],
        "append": args.append,
        "batch_mode": "one srun per node count",
    })

    cases_by_nodes = build_all_mpi_cases(cfg)
    all_cases = [case for cases in cases_by_nodes.values() for case in cases]

    # Run each sequential baseline only once per unique input configuration. 
    # The result is reused for all MPI node counts.

    seq_cache = {}

    print("\n==============================")
    print("SEQUENTIAL BASELINES")
    print("==============================")

    for case in all_cases:
        skey = seq_key(
            case["NR"],
            case["NS"],
            case["P"],
            case["seed"],
            case["workload"],
        )

        if skey in seq_cache:
            continue

        print(
            f"[SEQ][{case['workload']}] "
            f"NR={case['NR']} NS={case['NS']} P={case['P']} seed={case['seed']}"
        )

        seq_cache[skey] = run_seq(
            case["NR"],
            case["NS"],
            case["seed"],
            cfg["max_key"],
            case["P"],
            case["workload"],
            cfg["hot_partitions"],
            cfg["skew_percent"],
            cfg["timeout_sec"],
        )

    # Run MPI batches: exactly one srun for each node count

    mpi_cache = {}

    for nodes in cfg["nodes"]:
        batch_results = run_mpi_batch_for_nodes(
            cases_by_nodes.get(nodes, []),
            nodes,
            cfg["ranks_per_node"],
            cfg["timeout_sec"],
        )
        mpi_cache.update(batch_results)

    # Build final records and compute weak-scaling efficiency. 
    # For weak scaling, efficiency is computed with respect to the one-node MPI run 
    # with the same workload, partition count, and seed.

    weak_one_node_time = {}

    for case in all_cases:
        if case["experiment_type"] == "weak" and int(case["nodes"]) == 1:
            cid = case["case_id"]
            mpi_data = mpi_cache[cid]
            weak_key = (
                case["workload"],
                int(case["P"]),
                int(case["seed"]),
            )
            weak_one_node_time[weak_key] = get_time(mpi_data)

    for case in all_cases:
        skey = seq_key(
            case["NR"],
            case["NS"],
            case["P"],
            case["seed"],
            case["workload"],
        )

        seq_data = seq_cache[skey]
        mpi_data = mpi_cache[case["case_id"]]

        weak_efficiency = None

        if case["experiment_type"] == "weak":
            weak_key = (
                case["workload"],
                int(case["P"]),
                int(case["seed"]),
            )
            one_node_time = weak_one_node_time.get(weak_key)
            mpi_time = get_time(mpi_data)

            if one_node_time is not None and mpi_time and mpi_time > 0:
                weak_efficiency = one_node_time / mpi_time

        record = build_result_record(case, seq_data, mpi_data, weak_efficiency)
        results.append(record)

        if not record["checksum_correct"]:
            raise RuntimeError(
                f"Correctness failed for type={case['experiment_type']}, "
                f"NR={case['NR']}, NS={case['NS']}, P={case['P']}, "
                f"seed={case['seed']}, workload={case['workload']}, "
                f"nodes={case['nodes']}"
            )

        save_json(results_file, results)

    print("\nFinished.")
    print(f"Results saved in: {results_file}")
    print(f"Config saved in:   {config_file}")
    print(f"MPI srun calls used: {sum(1 for n in cfg['nodes'] if cases_by_nodes.get(n))}")


if __name__ == "__main__":
    main()
