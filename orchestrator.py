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

# cartella per eventuali file temporanei
TMP_DIR = RESULTS_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# ============================================================
# GLOBAL CONFIG
# ============================================================

Ps = [1024]
seeds = list(range(10))
max_key = 100_000

# correctness / debug
RUN_CORRECTNESS = True
correctness_cases = [
    {"NR": 0, "NS": 0, "P": 4, "threads_list": [1, 2]},
    {"NR": 1, "NS": 1, "P": 4, "threads_list": [1, 2]},
    {"NR": 100, "NS": 100, "P": 8, "threads_list": [1, 2, 4]},
    {"NR": 500, "NS": 500, "P": 16, "threads_list": [1, 2, 4]},
]

# strong scaling: problema fisso, variano i thread
RUN_STRONG = True
strong_Ns = [10_000_000]
strong_threads = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# weak scaling: carico per thread costante, quindi N cresce con T
RUN_WEAK = True
weak_base_N = 200_000
weak_threads = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
weak_cases = [(weak_base_N * t, weak_base_N * t, t) for t in weak_threads]

# se True, nel weak scaling esegue anche la sequenziale per lo stesso input
RUN_SEQ_FOR_WEAK = True

TIMEOUT_SEC = 1800  # 30 minuti a run

# ============================================================
# RESULTS
# ============================================================

results = []

# ============================================================
# HELPERS
# ============================================================

def check_executables():
    if not SEQ_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {SEQ_EXE}")
    if not PAR_EXE.exists():
        raise FileNotFoundError(f"Missing executable: {PAR_EXE}")


def save_results():
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def save_config():
    config = {
        "Ps": Ps,
        "seeds": seeds,
        "max_key": max_key,
        "RUN_CORRECTNESS": RUN_CORRECTNESS,
        "correctness_cases": correctness_cases,
        "RUN_STRONG": RUN_STRONG,
        "strong_Ns": strong_Ns,
        "strong_threads": strong_threads,
        "RUN_WEAK": RUN_WEAK,
        "weak_base_N": weak_base_N,
        "weak_threads": weak_threads,
        "weak_cases": weak_cases,
        "RUN_SEQ_FOR_WEAK": RUN_SEQ_FOR_WEAK,
        "TIMEOUT_SEC": TIMEOUT_SEC,
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def parse_stdout(stdout: str):
    """
    Tiene solo i campi eventualmente utili da stdout.
    Il JSON della run è la fonte principale.
    """
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
    """
    Unisce i dati letti dal JSON con quelli eventualmente stampati su stdout.
    Il JSON ha priorità.
    """
    merged = dict(stdout_data)
    merged.update(json_data)
    return merged


def run_cmd_with_json(cmd_base):
    """
    Esegue il comando aggiungendo -json <tmpfile>, legge il json prodotto e lo elimina.
    """
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
            timeout=TIMEOUT_SEC
        )

        stdout_data = parse_stdout(completed.stdout)

        if not tmp_path.exists():
            raise RuntimeError(f"Expected JSON output file not found: {tmp_path}")

        json_data = load_json_file(tmp_path)
        merged = merge_run_data(stdout_data, json_data)
        return merged

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run_seq(nr, ns, seed, max_key, p):
    cmd = [
        str(SEQ_EXE),
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
    ]
    return run_cmd_with_json(cmd)


def run_par(nr, ns, seed, max_key, p, t):
    cmd = [
        str(PAR_EXE),
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
        "-t", str(t),
    ]
    return run_cmd_with_json(cmd)


def build_result_record(
    experiment_type,
    nr,
    ns,
    p,
    seed,
    t,
    seq,
    par,
    max_key,
):
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

    # naive verifier, if present
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

def run_correctness_experiments():
    print("\n==============================")
    print("CORRECTNESS EXPERIMENTS")
    print("==============================")

    for case in correctness_cases:
        nr = case["NR"]
        ns = case["NS"]
        p = case["P"]
        local_threads = case["threads_list"]

        for seed in seeds:
            print(f"[CORRECTNESS][SEQ] NR={nr} NS={ns} P={p} seed={seed}")
            seq = run_seq(nr, ns, seed, max_key, p)

            for t in local_threads:
                print(f"[CORRECTNESS][PAR] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                par = run_par(nr, ns, seed, max_key, p, t)

                record = build_result_record(
                    experiment_type="correctness",
                    nr=nr,
                    ns=ns,
                    p=p,
                    seed=seed,
                    t=t,
                    seq=seq,
                    par=par,
                    max_key=max_key,
                )
                results.append(record)
                save_results()


def run_strong_scaling_experiments():
    print("\n==============================")
    print("STRONG SCALING EXPERIMENTS")
    print("==============================")

    for p in Ps:
        for n in strong_Ns:
            nr = int(n)
            ns = int(n)

            for seed in seeds:
                print(f"[STRONG][SEQ] NR={nr} NS={ns} P={p} seed={seed}")
                seq = run_seq(nr, ns, seed, max_key, p)

                for t in strong_threads:
                    print(f"[STRONG][PAR] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                    par = run_par(nr, ns, seed, max_key, p, t)

                    record = build_result_record(
                        experiment_type="strong",
                        nr=nr,
                        ns=ns,
                        p=p,
                        seed=seed,
                        t=t,
                        seq=seq,
                        par=par,
                        max_key=max_key,
                    )
                    results.append(record)
                    save_results()


def run_weak_scaling_experiments():
    print("\n==============================")
    print("WEAK SCALING EXPERIMENTS")
    print("==============================")

    for p in Ps:
        for seed in seeds:
            for nr, ns, t in weak_cases:
                seq = None

                if RUN_SEQ_FOR_WEAK:
                    print(f"[WEAK][SEQ] NR={nr} NS={ns} P={p} seed={seed}")
                    seq = run_seq(nr, ns, seed, max_key, p)

                print(f"[WEAK][PAR] NR={nr} NS={ns} P={p} seed={seed} T={t}")
                par = run_par(nr, ns, seed, max_key, p, t)

                record = build_result_record(
                    experiment_type="weak",
                    nr=nr,
                    ns=ns,
                    p=p,
                    seed=seed,
                    t=t,
                    seq=seq,
                    par=par,
                    max_key=max_key,
                )
                results.append(record)
                save_results()


# ============================================================
# MAIN
# ============================================================

def main():
    check_executables()
    save_config()

    if RUN_CORRECTNESS:
        run_correctness_experiments()

    if RUN_STRONG:
        run_strong_scaling_experiments()

    if RUN_WEAK:
        run_weak_scaling_experiments()

    print("\nFinished.")
    print(f"Results saved in: {RESULTS_FILE}")
    print(f"Config saved in:   {CONFIG_FILE}")


if __name__ == "__main__":
    main()