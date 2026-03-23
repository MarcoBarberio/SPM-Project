import subprocess
import json
import os

Ns = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 16
warmup = 5

results = []

os.makedirs("results", exist_ok=True)

for N in Ns:
    N = int(N)
    for seed in seeds:
        print(f"Running N={N}, seed={seed}")

        base_out = subprocess.check_output(
            ["./build/main_baseline", str(N), str(k), str(seed), str(warmup), "1"]
        ).decode().strip()

        vec_out = subprocess.check_output(
            ["./build/main_autovec", str(N), str(k), str(seed), str(warmup), "1"]
        ).decode().strip()

        avx_out = subprocess.check_output(
            ["./build/main_avx", str(N), str(k), str(seed), str(warmup), "1"]
        ).decode().strip()

        base_parts = base_out.split(",")
        vec_parts = vec_out.split(",")
        avx_parts = avx_out.split(",")

        base_time = float(base_parts[4])
        vec_time = float(vec_parts[4])
        avx_time = float(avx_parts[4])

        base_checksum = int(base_parts[5])
        vec_checksum = int(vec_parts[5])
        avx_checksum = int(avx_parts[5])

        correct = (base_checksum == vec_checksum == avx_checksum)

        results.append({
            "N": N,
            "seed": seed,
            "k": k,
            "time_baseline": base_time,
            "time_autovec": vec_time,
            "time_avx": avx_time,
            "speedup_autovec": base_time / vec_time,
            "speedup_avx": base_time / avx_time,
            "checksum_baseline": base_checksum,
            "checksum_autovec": vec_checksum,
            "checksum_avx": avx_checksum,
            "correct": correct
        })

with open("results/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Finished. Results saved in results/results.json")