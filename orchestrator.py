### Orchestrator to run all experiments and collect results in a JSON file.
import subprocess
import json
import os

# parameters
Ns = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]
seeds = list(range(11))
k = 16
warmup = 5

results = []
os.makedirs("results", exist_ok=True)

def run_and_read_json(executable, N, k, seed, warmup):
    """Run experiments and collect results in a JSON file"""
    subprocess.run(
        [executable, str(N), str(k), str(seed), str(warmup), "--json", "--print"],
        check=True
    )

    filename = f"result_n{N}_k{k}.json"

    with open(filename) as f:
        data = json.load(f)

    os.remove(filename)

    return data


for N in Ns:
    N = int(N)
    for seed in seeds:
        print(f"Running N={N}, seed={seed}")

        base = run_and_read_json("./build/main_baseline", N, k, seed, warmup)
        vec  = run_and_read_json("./build/main_autovec", N, k, seed, warmup)
        avx  = run_and_read_json("./build/main_avx", N, k, seed, warmup)
        cuda = run_and_read_json("./build/main_cuda", N, k, seed, warmup)

        base_time = base["time"]
        vec_time  = vec["time"]
        avx_time  = avx["time"]
        cuda_time = cuda["time"]

        # CUDA detailed times
        cuda_h2d    = cuda.get("time_h2d", 0)
        cuda_kernel = cuda.get("time_kernel", 0)
        cuda_d2h    = cuda.get("time_d2h", 0)

        # Throughput (Melem/s)
        throughput_base = N / base_time
        throughput_vec  = N / vec_time
        throughput_avx  = N / avx_time
        throughput_cuda_total  = N / cuda_time if cuda_time > 0 else 0
        throughput_cuda_kernel = N / cuda_kernel if cuda_kernel > 0 else 0

        base_checksum = base["checksum"]
        vec_checksum  = vec["checksum"]
        avx_checksum  = avx["checksum"]
        cuda_checksum = cuda["checksum"]

        correct_checksum = (base_checksum == vec_checksum == avx_checksum == cuda_checksum)

        correct_array = True
        if N < 500:
            base_map = base.get("mapping", [])
            vec_map  = vec.get("mapping", [])
            avx_map  = avx.get("mapping", [])
            cuda_map = cuda.get("mapping", [])
            correct_array = (base_map == vec_map == avx_map == cuda_map)

        results.append({
            "N": N,
            "seed": seed,
            "k": k,

            "time_baseline": base_time,
            "time_autovec": vec_time,
            "time_avx": avx_time,
            "time_cuda_total": cuda_time,

            "time_cuda_h2d": cuda_h2d,
            "time_cuda_kernel": cuda_kernel,
            "time_cuda_d2h": cuda_d2h,

            "throughput_baseline": throughput_base,
            "throughput_autovec": throughput_vec,
            "throughput_avx": throughput_avx,
            "throughput_cuda_total": throughput_cuda_total,
            "throughput_cuda_kernel": throughput_cuda_kernel,

            "speedup_autovec": base_time / vec_time,
            "speedup_avx": base_time / avx_time,
            "speedup_cuda_total": base_time / cuda_time,
            "speedup_cuda_kernel": base_time / cuda_kernel if cuda_kernel > 0 else 0,

            "checksum_correct": correct_checksum,
            "array_correct": correct_array
        })

with open("results/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Finished. Results saved in results/results.json")