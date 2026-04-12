import subprocess
import json
import os

# ============================================================
# Parameters
# ============================================================

Ns = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
seeds = list(range(5))
Ps = [1024]
threads_list = [1, 2, 4, 8,16, 32, 64]
max_key = 100_000

results = []
os.makedirs("results", exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def parse_output(stdout: str):
    """
    Parse the stdout produced by hashjoin_seq / hashjoin_par.
    Expected lines like:
        NR=...
        join_count=...
        checksum1=...
        checksum2=...
        time_sec=...
    """
    parsed = {}

    for line in stdout.strip().splitlines():
        line = line.strip()

        if line.startswith("NR="):
            # Example:
            # NR=10000 NS=10000 P=1024 seed=1 [0, 1000)
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


def run_seq(nr, ns, seed, max_key, p):
    cmd = [
        "./build/hashjoin_seq",
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
    ]

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return parse_output(completed.stdout)


def run_par(nr, ns, seed, max_key, p, t):
    cmd = [
        "./build/hashjoin_par",
        "-nr", str(nr),
        "-ns", str(ns),
        "-seed", str(seed),
        "-max-key", str(max_key),
        "-p", str(p),
        "-t", str(t),
    ]

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return parse_output(completed.stdout)


# ============================================================
# Experiments
# ============================================================

for p in Ps:
    for N in Ns:
        nr = int(N)
        ns = int(N)

        for seed in seeds:
            print(f"Running SEQ  NR={nr}, NS={ns}, P={p}, seed={seed}")
            seq = run_seq(nr, ns, seed, max_key, p)

            seq_time = seq["time_sec"]

            for t in threads_list:
                print(f"Running PAR  NR={nr}, NS={ns}, P={p}, seed={seed}, T={t}")
                par = run_par(nr, ns, seed, max_key, p, t)

                par_time = par["time_sec"]

                checksum_correct = (
                    seq["join_count"] == par["join_count"]
                    and seq["checksum1"] == par["checksum1"]
                    and seq["checksum2"] == par["checksum2"]
                )

                result = {
                    "NR": nr,
                    "NS": ns,
                    "P": p,
                    "seed": seed,
                    "max_key": max_key,
                    "threads": t,

                    "time_seq": seq_time,
                    "time_par": par_time,
                    "speedup": seq_time / par_time if par_time > 0 else 0.0,

                    "join_count_seq": seq["join_count"],
                    "join_count_par": par["join_count"],
                    "checksum1_seq": seq["checksum1"],
                    "checksum1_par": par["checksum1"],
                    "checksum2_seq": seq["checksum2"],
                    "checksum2_par": par["checksum2"],

                    "checksum_correct": checksum_correct
                }

                # If present, also store naive verifier outputs
                if "naive_join_count" in seq:
                    result["naive_join_count"] = seq["naive_join_count"]
                    result["naive_checksum1"] = seq["naive_checksum1"]
                    result["naive_checksum2"] = seq["naive_checksum2"]

                results.append(result)


# ============================================================
# Save JSON
# ============================================================

with open("results/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Finished. Results saved in results/results.json")