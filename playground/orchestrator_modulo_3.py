# This script orchestrates the experimental campaign for Module 3.
# It executes the sequential baseline and the two OpenMP implementations
# on the same inputs, collects their JSON outputs, and stores all results
# in a single file for later analysis and plotting.

import argparse
import subprocess
import json
import os
import tempfile
from pathlib import Path

# PATHS

BUILD_DIR = Path("./build")
SEQ_EXE = BUILD_DIR / "hashjoin_seq"
OMP_FOR_EXE = BUILD_DIR / "hashjoin_openmp_for"
OMP_TASK_EXE = BUILD_DIR / "hashjoin_openmp_task"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "results_modulo3.json"
CONFIG_FILE = RESULTS_DIR / "experiment_config_modulo3.json"

TMP_DIR = RESULTS_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

results = []

# Default experimental parameters.
# The same seeds and partition counts are reused across implementations
# to make the comparison fair and reproducible.

DEFAULT_PS = [1024]
DEFAULT_SEEDS = list(range(10))
DEFAULT_MAX_KEY = 100_000

# Small input cases used to validate correctness.
# These cases include empty relations, single-record relations, and
# small non-trivial datasets. For such sizes, the C++ executables can
# also compare the result against a naive reference join.

DEFAULT_CORRECTNESS_CASES = [
    {"NR": 0, "NS": 0, "P": 4, "threads_list": [1, 2]},
    {"NR": 1, "NS": 1, "P": 4, "threads_list": [1, 2]},
    {"NR": 100, "NS": 100, "P": 8, "threads_list": [1, 2, 4]},
    {"NR": 500, "NS": 500, "P": 16, "threads_list": [1, 2, 4]},
]

DEFAULT_STRONG_NS = [10_000_000]
DEFAULT_STRONG_THREADS = [1, 2, 4, 8, 16, 24, 32]

DEFAULT_WEAK_BASE_N = 200_000
DEFAULT_WEAK_THREADS = [1, 2, 4, 8, 16, 24, 32]

DEFAULT_WORKLOADS = ["uniform", "skewed"]
DEFAULT_HOT_PARTITIONS = 16
DEFAULT_SKEW_PERCENT = 90
DEFAULT_SKEW_PERCENT_LIST = [0, 50, 75, 90, 95, 99]

DEFAULT_SKEW_N = 10_000_000
DEFAULT_SKEW_THREADS = 16

DEFAULT_TIMEOUT_SEC = 1800

IMPLEMENTATION_EXES = {
    "omp_for": OMP_FOR_EXE,
    "omp_task": OMP_TASK_EXE,
}
# Phase timing keys expected from the C++ executables.
# These values are copied into the final result records to allow a breakdown
# of the execution time across partitioning, joining, and accumulation phases.

PHASE_KEYS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Module 3 experiments: sequential baseline, OpenMP for, OpenMP task"
    )

    parser.add_argument(
        "--run",
        nargs="+",
        choices=["correctness", "strong", "weak", "skew_sensitivity"],
        default=["correctness", "strong", "weak"],
        help="Which experiment groups to run"
    )

    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["omp_for", "omp_task"],
        default=["omp_for", "omp_task"],
        help="OpenMP implementations to test"
    )

    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["uniform", "skewed"],
        default=DEFAULT_WORKLOADS,
        help="Input workloads to test"
    )

    parser.add_argument(
        "--ps",
        nargs="+",
        type=int,
        default=DEFAULT_PS,
        help="Partition counts P"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to use"
    )

    parser.add_argument(
        "--max-key",
        type=int,
        default=DEFAULT_MAX_KEY,
        help="max_key parameter"
    )

    parser.add_argument(
        "--hot-partitions",
        type=int,
        default=DEFAULT_HOT_PARTITIONS,
        help="Number of hot partitions for skewed workload"
    )

    parser.add_argument(
        "--skew-percent",
        type=int,
        default=DEFAULT_SKEW_PERCENT,
        help="Percentage of records assigned to hot partitions for skewed workload"
    )

    parser.add_argument(
        "--skew-percent-list",
        nargs="+",
        type=int,
        default=DEFAULT_SKEW_PERCENT_LIST,
        help="Skew percentages used by the skew_sensitivity experiment"
    )

    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Timeout per run in seconds"
    )

    parser.add_argument(
        "--results-file",
        type=str,
        default=str(RESULTS_FILE),
        help="Output results JSON path"
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default=str(CONFIG_FILE),
        help="Output config JSON path"
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing results file instead of overwriting"
    )

    # correctness
    parser.add_argument(
        "--correctness-cases-file",
        type=str,
        default=None,
        help="Optional JSON file for correctness cases"
    )

    # strong
    parser.add_argument(
        "--strong-ns",
        nargs="+",
        type=int,
        default=DEFAULT_STRONG_NS,
        help="Problem sizes N for strong scaling"
    )

    parser.add_argument(
        "--strong-threads",
        nargs="+",
        type=int,
        default=DEFAULT_STRONG_THREADS,
        help="Thread counts for strong scaling"
    )

    # weak
    parser.add_argument(
        "--weak-base-n",
        type=int,
        default=DEFAULT_WEAK_BASE_N,
        help="Base N per thread for weak scaling"
    )

    parser.add_argument(
        "--weak-threads",
        nargs="+",
        type=int,
        default=DEFAULT_WEAK_THREADS,
        help="Thread counts for weak scaling"
    )

    parser.add_argument(
        "--run-seq-for-weak",
        action="store_true",
        help="Also run sequential version for each weak scaling size"
    )

    # skew sensitivity
    parser.add_argument(
        "--skew-n",
        type=int,
        default=DEFAULT_SKEW_N,
        help="Problem size N used by skew_sensitivity"
    )

    parser.add_argument(
        "--skew-threads",
        type=int,
        default=DEFAULT_SKEW_THREADS,
        help="Thread count used by skew_sensitivity"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def check_executables(implementations):
    if not SEQ_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {SEQ_EXE}")

    for impl in implementations:
        exe = IMPLEMENTATION_EXES[impl]
        if not exe.exists():
            raise FileNotFoundError(f"Missing executable for {impl}: {exe}")


def save_results(results_file: Path):
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def save_config(config_file: Path, cfg: dict):
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def load_correctness_cases(path):
    if path is None:
        return DEFAULT_CORRECTNESS_CASES
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Ensures that the number of hot partitions is valid for the selected P.
# For skewed workloads with skew_percent < 100, at least one cold partition
# is preserved so that the dataset actually contains both hot and cold regions.

def normalize_hot_partitions(hot_partitions, p, workload="uniform", skew_percent=90):
    p = int(p)
    hot = max(1, min(int(hot_partitions), p))

    # In a skewed workload with skew_percent < 100 at least a 
    # partition must be cold. So hot_partitions must be < P.
    if workload == "skewed" and int(skew_percent) < 100 and p > 1:
        hot = min(hot, p - 1)

    return hot


def validate_power_of_two(p):
    return p > 0 and (p & (p - 1)) == 0


def parse_stdout(stdout: str):
    parsed = {}

    for line in stdout.strip().splitlines():
        line = line.strip()

        if line.startswith("NR="):
            tokens = line.split()
            for token in tokens:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                value = value.strip()

                if key in ["NR", "NS", "P", "T", "seed", "hot_partitions", "skew_percent"]:
                    try:
                        parsed[key] = int(value)
                    except ValueError:
                        pass
                elif key == "workload":
                    parsed[key] = value

        elif line.startswith("join_count="):
            parsed["join_count"] = int(line.split("=", 1)[1])

        elif line.startswith("checksum1="):
            parsed["checksum1"] = int(line.split("=", 1)[1])

        elif line.startswith("checksum2="):
            parsed["checksum2"] = int(line.split("=", 1)[1])

        elif line.startswith("time_sec="):
            parsed["time_sec"] = float(line.split("=", 1)[1])

        elif line.startswith("time_"):
            key, value = line.split("=", 1)
            try:
                parsed[key] = float(value)
            except ValueError:
                pass

        elif line.startswith("naive_join_count="):
            parsed["naive_join_count"] = int(line.split("=", 1)[1])

        elif line.startswith("naive_checksum1="):
            parsed["naive_checksum1"] = int(line.split("=", 1)[1])

        elif line.startswith("naive_checksum2="):
            parsed["naive_checksum2"] = int(line.split("=", 1)[1])

    return parsed


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_run_data(stdout_data: dict, json_data: dict):
    merged = dict(stdout_data)
    merged.update(json_data)
    return merged

# Runs one executable and asks it to write detailed measurements to a temporary
# JSON file. The standard output is also parsed because some summary values are
# printed there. The two sources are then merged into a single dictionary.
def run_cmd_with_json(cmd_base, timeout_sec: int):
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix="run_", suffix=".json", dir=TMP_DIR)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    cmd = list(cmd_base) + ["-json", str(tmp_path)]

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )

        stdout_data = parse_stdout(completed.stdout)

        if not tmp_path.exists():
            raise RuntimeError(f"Expected JSON output file not found: {tmp_path}")

        json_data = load_json_file(tmp_path)
        return merge_run_data(stdout_data, json_data)

    except subprocess.CalledProcessError as exc:
        print("\nCommand failed:")
        print(" ".join(cmd))
        print("\nSTDOUT:")
        print(exc.stdout)
        print("\nSTDERR:")
        print(exc.stderr)
        raise

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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


def run_seq(nr, ns, seed, max_key, p, workload, hot_partitions, skew_percent, timeout_sec):
    cmd = [str(SEQ_EXE)] + common_args(
        nr, ns, seed, max_key, p, workload, hot_partitions, skew_percent
    )
    return run_cmd_with_json(cmd, timeout_sec)


def run_implementation(implementation, nr, ns, seed, max_key, p, t, workload, hot_partitions, skew_percent, timeout_sec):
    exe = IMPLEMENTATION_EXES[implementation]
    cmd = [str(exe)] + common_args(
        nr, ns, seed, max_key, p, workload, hot_partitions, skew_percent
    ) + [
        "-t", str(t),
    ]
    return run_cmd_with_json(cmd, timeout_sec)


def get_time(run_data):
    return run_data.get("time_total", run_data.get("time_sec"))


def checksums_equal(a, b):
    return (
        a["join_count"] == b["join_count"]
        and a["checksum1"] == b["checksum1"]
        and a["checksum2"] == b["checksum2"]
    )

# Builds the final result entry stored in results_modulo3.json.
# Each record contains the experiment metadata, timings, speedup,
# checksums, and phase-level measurements.

def build_result_record(experiment_type, implementation, nr, ns, p, seed, t, seq, impl_data,
                        max_key, workload, hot_partitions, skew_percent):
    seq_time = get_time(seq) if seq is not None else None
    impl_time = get_time(impl_data)

    checksum_correct = None
    speedup_vs_seq = None

    if seq is not None:
        checksum_correct = checksums_equal(seq, impl_data)
        speedup_vs_seq = seq_time / impl_time if impl_time and impl_time > 0 else 0.0

    record = {
        "experiment_type": experiment_type,
        "implementation": implementation,
        "NR": nr,
        "NS": ns,
        "P": p,
        "seed": seed,
        "max_key": max_key,
        "threads": t,
        "N_per_thread": nr / t if t > 0 else None,
        "workload": workload,
        "hot_partitions": normalize_hot_partitions(hot_partitions, p, workload, skew_percent),
        "skew_percent": skew_percent,

        "time_seq": seq_time,
        "time_impl": impl_time,
        "speedup_vs_seq": speedup_vs_seq,

        "join_count_seq": seq["join_count"] if seq is not None else None,
        "join_count_impl": impl_data["join_count"],
        "checksum1_seq": seq["checksum1"] if seq is not None else None,
        "checksum1_impl": impl_data["checksum1"],
        "checksum2_seq": seq["checksum2"] if seq is not None else None,
        "checksum2_impl": impl_data["checksum2"],

        "checksum_correct": checksum_correct,
    }

    if seq is not None and "naive_join_count" in seq:
        record["naive_join_count"] = seq.get("naive_join_count")
        record["naive_checksum1"] = seq.get("naive_checksum1")
        record["naive_checksum2"] = seq.get("naive_checksum2")

    if seq is not None:
        for k in PHASE_KEYS:
            record[f"{k}_seq"] = seq.get(k)

    for k in PHASE_KEYS:
        record[f"{k}_impl"] = impl_data.get(k)

    return record


# ============================================================
# EXPERIMENTS
# ============================================================

# Runs correctness experiments on small inputs.
# The sequential output is used as reference, and each OpenMP result must match
# the same join_count and checksum values.
def run_correctness_experiments(cfg):
    print("\n==============================")
    print("CORRECTNESS EXPERIMENTS")
    print("==============================")

    for workload in cfg["workloads"]:
        for case in cfg["correctness_cases"]:
            nr = case["NR"]
            ns = case["NS"]
            p = case["P"]
            local_threads = case["threads_list"]

            for seed in cfg["seeds"]:
                print(f"[CORRECTNESS][SEQ][{workload}] NR={nr} NS={ns} P={p} seed={seed}")
                seq = run_seq(
                    nr, ns, seed, cfg["max_key"], p, workload,
                    cfg["hot_partitions"], cfg["skew_percent"], cfg["timeout_sec"]
                )

                for implementation in cfg["implementations"]:
                    for t in local_threads:
                        print(f"[CORRECTNESS][{implementation}][{workload}] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                        impl_data = run_implementation(
                            implementation, nr, ns, seed, cfg["max_key"], p, t, workload,
                            cfg["hot_partitions"], cfg["skew_percent"], cfg["timeout_sec"]
                        )

                        results.append(build_result_record(
                            experiment_type="correctness",
                            implementation=implementation,
                            nr=nr, ns=ns, p=p, seed=seed, t=t,
                            seq=seq, impl_data=impl_data, max_key=cfg["max_key"],
                            workload=workload, hot_partitions=cfg["hot_partitions"],
                            skew_percent=cfg["skew_percent"]
                        ))
                        save_results(cfg["results_file"])

# Runs strong scaling experiments.
# The input size is fixed and the number of threads is varied.
# This is used to evaluate speedup with respect to the sequential baseline.

def run_strong_scaling_experiments(cfg):
    print("\n==============================")
    print("STRONG SCALING EXPERIMENTS")
    print("==============================")

    for workload in cfg["workloads"]:
        for p in cfg["ps"]:
            for n in cfg["strong_ns"]:
                nr = int(n)
                ns = int(n)

                for seed in cfg["seeds"]:
                    print(f"[STRONG][SEQ][{workload}] NR={nr} NS={ns} P={p} seed={seed}")
                    seq = run_seq(
                        nr, ns, seed, cfg["max_key"], p, workload,
                        cfg["hot_partitions"], cfg["skew_percent"], cfg["timeout_sec"]
                    )

                    for implementation in cfg["implementations"]:
                        for t in cfg["strong_threads"]:
                            print(f"[STRONG][{implementation}][{workload}] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                            impl_data = run_implementation(
                                implementation, nr, ns, seed, cfg["max_key"], p, t, workload,
                                cfg["hot_partitions"], cfg["skew_percent"], cfg["timeout_sec"]
                            )

                            results.append(build_result_record(
                                experiment_type="strong",
                                implementation=implementation,
                                nr=nr, ns=ns, p=p, seed=seed, t=t,
                                seq=seq, impl_data=impl_data, max_key=cfg["max_key"],
                                workload=workload, hot_partitions=cfg["hot_partitions"],
                                skew_percent=cfg["skew_percent"]
                            ))
                            save_results(cfg["results_file"])

# Runs weak scaling experiments.
# For each thread count T, the input size is scaled as weak_base_n * T.
# The goal is to evaluate whether execution time remains stable when the
# amount of work per thread is kept approximately constant.

def run_weak_scaling_experiments(cfg):
    print("\n==============================")
    print("WEAK SCALING EXPERIMENTS")
    print("==============================")

    weak_cases = [
        (cfg["weak_base_n"] * t, cfg["weak_base_n"] * t, t)
        for t in cfg["weak_threads"]
    ]

    for workload in cfg["workloads"]:
        for p in cfg["ps"]:
            for seed in cfg["seeds"]:
                for nr, ns, t in weak_cases:
                    seq = None

                    if cfg["run_seq_for_weak"]:
                        print(f"[WEAK][SEQ][{workload}] NR={nr} NS={ns} P={p} seed={seed}")
                        seq = run_seq(
                            nr, ns, seed, cfg["max_key"], p, workload,
                            cfg["hot_partitions"], cfg["skew_percent"], cfg["timeout_sec"]
                        )

                    for implementation in cfg["implementations"]:
                        print(f"[WEAK][{implementation}][{workload}] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                        impl_data = run_implementation(
                            implementation, nr, ns, seed, cfg["max_key"], p, t, workload,
                            cfg["hot_partitions"], cfg["skew_percent"], cfg["timeout_sec"]
                        )

                        results.append(build_result_record(
                            experiment_type="weak",
                            implementation=implementation,
                            nr=nr, ns=ns, p=p, seed=seed, t=t,
                            seq=seq, impl_data=impl_data, max_key=cfg["max_key"],
                            workload=workload, hot_partitions=cfg["hot_partitions"],
                            skew_percent=cfg["skew_percent"]
                        ))
                        save_results(cfg["results_file"])

# Runs skew-sensitivity experiments.
# The input size and thread count are fixed, while the skew percentage varies.
# This highlights how loop-based and task-based OpenMP strategies react to
# increasing load imbalance across partitions.

def run_skew_sensitivity_experiments(cfg):
    print("\n==============================")
    print("SKEW SENSITIVITY EXPERIMENTS")
    print("==============================")

    workload = "skewed"
    nr = int(cfg["skew_n"])
    ns = int(cfg["skew_n"])
    t = int(cfg["skew_threads"])

    for p in cfg["ps"]:
        for skew_percent in cfg["skew_percent_list"]:
            for seed in cfg["seeds"]:
                print(f"[SKEW][SEQ] NR={nr} NS={ns} P={p} seed={seed} skew={skew_percent}")
                seq = run_seq(
                    nr, ns, seed, cfg["max_key"], p, workload,
                    cfg["hot_partitions"], skew_percent, cfg["timeout_sec"]
                )

                for implementation in cfg["implementations"]:
                    print(f"[SKEW][{implementation}] NR={nr} NS={ns} P={p} seed={seed} T={t} skew={skew_percent}")
                    impl_data = run_implementation(
                        implementation, nr, ns, seed, cfg["max_key"], p, t, workload,
                        cfg["hot_partitions"], skew_percent, cfg["timeout_sec"]
                    )

                    results.append(build_result_record(
                        experiment_type="skew_sensitivity",
                        implementation=implementation,
                        nr=nr, ns=ns, p=p, seed=seed, t=t,
                        seq=seq, impl_data=impl_data, max_key=cfg["max_key"],
                        workload=workload, hot_partitions=cfg["hot_partitions"],
                        skew_percent=skew_percent
                    ))
                    save_results(cfg["results_file"])


# MAIN

def main():
    args = parse_args()

    check_executables(args.implementations)

    results_file = Path(args.results_file)
    config_file = Path(args.config_file)

    global results
    if args.append and results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    cfg = {
        "run": args.run,
        "implementations": args.implementations,
        "workloads": args.workloads,
        "ps": args.ps,
        "seeds": args.seeds,
        "max_key": args.max_key,
        "hot_partitions": args.hot_partitions,
        "skew_percent": args.skew_percent,
        "skew_percent_list": args.skew_percent_list,
        "timeout_sec": args.timeout_sec,
        "results_file": results_file,
        "config_file": config_file,
        "correctness_cases": load_correctness_cases(args.correctness_cases_file),
        "strong_ns": args.strong_ns,
        "strong_threads": args.strong_threads,
        "weak_base_n": args.weak_base_n,
        "weak_threads": args.weak_threads,
        "run_seq_for_weak": args.run_seq_for_weak,
        "skew_n": args.skew_n,
        "skew_threads": args.skew_threads,
    }

    save_config(config_file, {
        "run": cfg["run"],
        "executables": {
            "seq": str(SEQ_EXE),
            "omp_for": str(OMP_FOR_EXE),
            "omp_task": str(OMP_TASK_EXE),
        },
        "ps": cfg["ps"],
        "seeds": cfg["seeds"],
        "max_key": cfg["max_key"],
        "timeout_sec": cfg["timeout_sec"],
        "implementations": cfg["implementations"],
        "workloads": cfg["workloads"],
        "hot_partitions": cfg["hot_partitions"],
        "skew_percent": cfg["skew_percent"],
        "skew_percent_list": cfg["skew_percent_list"],
        "correctness_cases": cfg["correctness_cases"],
        "strong_ns": cfg["strong_ns"],
        "strong_threads": cfg["strong_threads"],
        "weak_base_n": cfg["weak_base_n"],
        "weak_threads": cfg["weak_threads"],
        "weak_cases": [
            (cfg["weak_base_n"] * t, cfg["weak_base_n"] * t, t)
            for t in cfg["weak_threads"]
        ],
        "run_seq_for_weak": cfg["run_seq_for_weak"],
        "skew_n": cfg["skew_n"],
        "skew_threads": cfg["skew_threads"],
        "append": args.append,
    })

    if "correctness" in cfg["run"]:
        run_correctness_experiments(cfg)

    if "strong" in cfg["run"]:
        run_strong_scaling_experiments(cfg)

    if "weak" in cfg["run"]:
        run_weak_scaling_experiments(cfg)

    if "skew_sensitivity" in cfg["run"]:
        run_skew_sensitivity_experiments(cfg)

    print("\nFinished.")
    print(f"Results saved in: {results_file}")
    print(f"Config saved in:   {config_file}")


if __name__ == "__main__":
    main()
