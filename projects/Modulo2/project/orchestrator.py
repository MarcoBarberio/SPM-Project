import argparse
import shutil
import subprocess
import json
import os
import tempfile
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BUILD_DIR = Path("./build")
SEQ_EXE = BUILD_DIR / "hashjoin_seq"
PAR_EXE = BUILD_DIR / "hashjoin_par"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "results.json"
CONFIG_FILE = RESULTS_DIR / "experiment_config.json"

TMP_DIR = RESULTS_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

results = []

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_PS = [1024]
DEFAULT_SEEDS = list(range(10))
DEFAULT_MAX_KEY = 100_000

DEFAULT_CORRECTNESS_CASES = [
    {"NR": 0, "NS": 0, "P": 4, "threads_list": [1, 2]},
    {"NR": 1, "NS": 1, "P": 4, "threads_list": [1, 2]},
    {"NR": 100, "NS": 100, "P": 8, "threads_list": [1, 2, 4]},
    {"NR": 500, "NS": 500, "P": 16, "threads_list": [1, 2, 4]},
]

DEFAULT_STRONG_NS = [10_000_000]
DEFAULT_STRONG_THREADS = [1, 2, 4, 8]

DEFAULT_WEAK_BASE_N = 200_000
DEFAULT_WEAK_THREADS = [1, 2, 4, 8]

DEFAULT_TIMEOUT_SEC = 1800


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run Module 2 experiments")

    parser.add_argument(
        "--run",
        nargs="+",
        choices=["correctness", "strong", "weak"],
        default=["correctness", "strong", "weak"],
        help="Which experiment groups to run"
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
        help="Also run sequential version for weak scaling cases"
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def check_executables():
    if not SEQ_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {SEQ_EXE}")
    if not PAR_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {PAR_EXE}")


def save_results(results_file: Path):
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def save_config(config_file: Path, cfg: dict):
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def load_correctness_cases(path):
    if path is None:
        return DEFAULT_CORRECTNESS_CASES
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_stdout(stdout: str):
    parsed = {}

    for line in stdout.strip().splitlines():
        line = line.strip()

        if line.startswith("NR="):
            tokens = line.split()
            for token in tokens:
                if token.startswith("NR="):
                    parsed["NR"] = int(token.split("=")[1])
                elif token.startswith("NS="):
                    parsed["NS"] = int(token.split("=")[1])
                elif token.startswith("P="):
                    parsed["P"] = int(token.split("=")[1])
                elif token.startswith("T="):
                    parsed["T"] = int(token.split("=")[1])
                elif token.startswith("seed="):
                    parsed["seed"] = int(token.split("=")[1])

        elif line.startswith("join_count="):
            parsed["join_count"] = int(line.split("=", 1)[1])

        elif line.startswith("checksum1="):
            parsed["checksum1"] = int(line.split("=", 1)[1])

        elif line.startswith("checksum2="):
            parsed["checksum2"] = int(line.split("=", 1)[1])

        elif line.startswith("time_sec="):
            parsed["time_sec"] = float(line.split("=", 1)[1])

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


def run_cmd_with_json(cmd_base, timeout_sec: int):
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

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run_seq(nr, ns, seed, max_key, p, timeout_sec):
    cmd = [
        str(SEQ_EXE),
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
    ]
    return run_cmd_with_json(cmd, timeout_sec)


def run_par(nr, ns, seed, max_key, p, t, timeout_sec):
    cmd = [
        str(PAR_EXE),
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
        "-t", str(t),
    ]
    return run_cmd_with_json(cmd, timeout_sec)


def build_result_record(experiment_type, nr, ns, p, seed, t, seq, par, max_key):
    seq_time = seq.get("time_total", seq.get("time_sec")) if seq is not None else None
    par_time = par.get("time_total", par.get("time_sec"))

    checksum_correct = None
    speedup = None

    if seq is not None:
        checksum_correct = (
            seq["join_count"] == par["join_count"]
            and seq["checksum1"] == par["checksum1"]
            and seq["checksum2"] == par["checksum2"]
        )
        speedup = seq_time / par_time if par_time and par_time > 0 else 0.0

    record = {
        "experiment_type": experiment_type,
        "NR": nr,
        "NS": ns,
        "P": p,
        "seed": seed,
        "max_key": max_key,
        "threads": t,
        "N_per_thread": nr / t if t > 0 else None,

        "time_seq": seq_time,
        "time_par": par_time,
        "speedup": speedup,

        "join_count_seq": seq["join_count"] if seq is not None else None,
        "join_count_par": par["join_count"],
        "checksum1_seq": seq["checksum1"] if seq is not None else None,
        "checksum1_par": par["checksum1"],
        "checksum2_seq": seq["checksum2"] if seq is not None else None,
        "checksum2_par": par["checksum2"],

        "checksum_correct": checksum_correct,
    }

    if seq is not None and "naive_join_count" in seq:
        record["naive_join_count"] = seq["naive_join_count"]
        record["naive_checksum1"] = seq["naive_checksum1"]
        record["naive_checksum2"] = seq["naive_checksum2"]

    phase_keys = [
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

    if seq is not None:
        for k in phase_keys:
            record[f"{k}_seq"] = seq.get(k)

    for k in phase_keys:
        record[f"{k}_par"] = par.get(k)

    return record


# ============================================================
# EXPERIMENTS
# ============================================================

def run_correctness_experiments(cfg):
    print("\n==============================")
    print("CORRECTNESS EXPERIMENTS")
    print("==============================")

    for case in cfg["correctness_cases"]:
        nr = case["NR"]
        ns = case["NS"]
        p = case["P"]
        local_threads = case["threads_list"]

        for seed in cfg["seeds"]:
            print(f"[CORRECTNESS][SEQ] NR={nr} NS={ns} P={p} seed={seed}")
            seq = run_seq(nr, ns, seed, cfg["max_key"], p, cfg["timeout_sec"])

            for t in local_threads:
                print(f"[CORRECTNESS][PAR] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                par = run_par(nr, ns, seed, cfg["max_key"], p, t, cfg["timeout_sec"])

                results.append(build_result_record(
                    experiment_type="correctness",
                    nr=nr, ns=ns, p=p, seed=seed, t=t,
                    seq=seq, par=par, max_key=cfg["max_key"]
                ))
                save_results(cfg["results_file"])


def run_strong_scaling_experiments(cfg):
    print("\n==============================")
    print("STRONG SCALING EXPERIMENTS")
    print("==============================")

    for p in cfg["ps"]:
        for n in cfg["strong_ns"]:
            nr = int(n)
            ns = int(n)

            for seed in cfg["seeds"]:
                print(f"[STRONG][SEQ] NR={nr} NS={ns} P={p} seed={seed}")
                seq = run_seq(nr, ns, seed, cfg["max_key"], p, cfg["timeout_sec"])

                for t in cfg["strong_threads"]:
                    print(f"[STRONG][PAR] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                    par = run_par(nr, ns, seed, cfg["max_key"], p, t, cfg["timeout_sec"])

                    results.append(build_result_record(
                        experiment_type="strong",
                        nr=nr, ns=ns, p=p, seed=seed, t=t,
                        seq=seq, par=par, max_key=cfg["max_key"]
                    ))
                    save_results(cfg["results_file"])


def run_weak_scaling_experiments(cfg):
    print("\n==============================")
    print("WEAK SCALING EXPERIMENTS")
    print("==============================")

    weak_cases = [(cfg["weak_base_n"] * t, cfg["weak_base_n"] * t, t) for t in cfg["weak_threads"]]

    for p in cfg["ps"]:
        for seed in cfg["seeds"]:
            for nr, ns, t in weak_cases:
                seq = None

                if cfg["run_seq_for_weak"]:
                    print(f"[WEAK][SEQ] NR={nr} NS={ns} P={p} seed={seed}")
                    seq = run_seq(nr, ns, seed, cfg["max_key"], p, cfg["timeout_sec"])

                print(f"[WEAK][PAR] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                par = run_par(nr, ns, seed, cfg["max_key"], p, t, cfg["timeout_sec"])

                results.append(build_result_record(
                    experiment_type="weak",
                    nr=nr, ns=ns, p=p, seed=seed, t=t,
                    seq=seq, par=par, max_key=cfg["max_key"]
                ))
                save_results(cfg["results_file"])


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    check_executables()

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
        "ps": args.ps,
        "seeds": args.seeds,
        "max_key": args.max_key,
        "timeout_sec": args.timeout_sec,
        "results_file": results_file,
        "config_file": config_file,
        "correctness_cases": load_correctness_cases(args.correctness_cases_file),
        "strong_ns": args.strong_ns,
        "strong_threads": args.strong_threads,
        "weak_base_n": args.weak_base_n,
        "weak_threads": args.weak_threads,
        "run_seq_for_weak": args.run_seq_for_weak,
    }

    # salva config leggibile
    save_config(config_file, {
        "run": cfg["run"],
        "ps": cfg["ps"],
        "seeds": cfg["seeds"],
        "max_key": cfg["max_key"],
        "timeout_sec": cfg["timeout_sec"],
        "correctness_cases": cfg["correctness_cases"],
        "strong_ns": cfg["strong_ns"],
        "strong_threads": cfg["strong_threads"],
        "weak_base_n": cfg["weak_base_n"],
        "weak_threads": cfg["weak_threads"],
        "weak_cases": [(cfg["weak_base_n"] * t, cfg["weak_base_n"] * t, t) for t in cfg["weak_threads"]],
        "run_seq_for_weak": cfg["run_seq_for_weak"],
        "append": args.append,
    })

    if "correctness" in cfg["run"]:
        run_correctness_experiments(cfg)

    if "strong" in cfg["run"]:
        run_strong_scaling_experiments(cfg)

    if "weak" in cfg["run"]:
        run_weak_scaling_experiments(cfg)

    print("\nFinished.")
    print(f"Results saved in: {results_file}")
    print(f"Config saved in:   {config_file}")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()