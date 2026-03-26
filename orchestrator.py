import subprocess
import json
import os

Ns = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]
seeds = list(range(11))
k = 16
warmup = 5

results = []
os.makedirs("results", exist_ok=True)

def run_and_read_json(executable, N, k, seed, warmup):
    # Run executable
    subprocess.run(
        [executable, str(N), str(k), str(seed), str(warmup), "--json", "--print"],
        check=True
    )

    filename = f"result_n{N}_k{k}.json"

    # Read JSON
    with open(filename) as f:
        data = json.load(f)

    # Delete JSON file
    os.remove(filename)

    return data


for N in Ns:
    N = int(N)
    for seed in seeds:
        print(f"Running N={N}, seed={seed}")

        base = run_and_read_json("./build/main_baseline", N, k, seed, warmup)
        vec  = run_and_read_json("./build/main_autovec", N, k, seed, warmup)
        avx  = run_and_read_json("./build/main_avx", N, k, seed, warmup)

        base_time = base["time"]
        vec_time  = vec["time"]
        avx_time  = avx["time"]

        base_checksum = base["checksum"]
        vec_checksum  = vec["checksum"]
        avx_checksum  = avx["checksum"]

        # Check checksum
        correct_checksum = (base_checksum == vec_checksum == avx_checksum)

        # Check arrays if N < 500
        correct_array = True
        if N < 500:
            base_map = base.get("mapping", [])
            vec_map  = vec.get("mapping", [])
            avx_map  = avx.get("mapping", [])

            correct_array = (base_map == vec_map == avx_map)

        results.append({
            "N": N,
            "seed": seed,
            "k": k,
            "time_baseline": base_time,
            "time_autovec": vec_time,
            "time_avx": avx_time,
            "speedup_autovec": base_time / vec_time,
            "speedup_avx": base_time / avx_time,
            "checksum_correct": correct_checksum,
            "array_correct": correct_array
        })

# Save final results
with open("results/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Finished. Results saved in results/results.json")