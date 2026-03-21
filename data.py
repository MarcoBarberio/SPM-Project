import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CREA CARTELLE
# =========================
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# =========================
# PARAMETRI ESPERIMENTO
# =========================
executables = {
    "baseline": "./build/main_novec",
    "autovec": "./build/main_vec",
    # "avx2": "./build/main_avx"  # aggiungerai dopo
}

n_values = [10_000, 100_000, 1_000_000, 10_000_000]
k = 10
seed = 24
warmup = 3
repetitions = 5

results = []

# =========================
# ESECUZIONE BENCHMARK
# =========================
for version, exe in executables.items():
    for n in n_values:
        times = []
        checksums = []

        for r in range(repetitions):
            output = subprocess.check_output(
                [exe, str(n), str(k), str(seed), str(warmup)]
            ).decode("utf-8")

            # Formato: n,k,seed,warmup,time,checksum
            parts = output.strip().split(",")
            time = float(parts[4])
            checksum = int(parts[5])

            times.append(time)
            checksums.append(checksum)

        # Verifica checksum coerenti
        if len(set(checksums)) != 1:
            print(f"WARNING: checksum mismatch for {version}, n={n}")

        mean_time = np.mean(times)
        std_time = np.std(times)

        results.append({
            "version": version,
            "n": n,
            "mean_time": mean_time,
            "std_time": std_time,
            "throughput": n / mean_time
        })

df = pd.DataFrame(results)

# =========================
# SPEEDUP
# =========================
baseline = df[df["version"] == "baseline"][["n", "mean_time"]]
baseline = baseline.rename(columns={"mean_time": "baseline_time"})

df = df.merge(baseline, on="n")
df["speedup"] = df["baseline_time"] / df["mean_time"]

# =========================
# SALVA CSV
# =========================
df.to_csv("results/tables/raw_results.csv", index=False)

# =========================
# GRAFICI
# =========================

# Time vs N
plt.figure()
for version in df["version"].unique():
    sub = df[df["version"] == version]
    plt.errorbar(sub["n"], sub["mean_time"], yerr=sub["std_time"], label=version)

plt.xlabel("N")
plt.ylabel("Time (s)")
plt.xscale("log")
plt.legend()
plt.title("Execution Time vs N")
plt.savefig("results/plots/time_vs_n.png")
plt.close()

# Throughput vs N
plt.figure()
for version in df["version"].unique():
    sub = df[df["version"] == version]
    plt.plot(sub["n"], sub["throughput"], label=version)

plt.xlabel("N")
plt.ylabel("Throughput (elements/s)")
plt.xscale("log")
plt.legend()
plt.title("Throughput vs N")
plt.savefig("results/plots/throughput_vs_n.png")
plt.close()

# Speedup vs N
plt.figure()
for version in df["version"].unique():
    if version == "baseline":
        continue
    sub = df[df["version"] == version]
    plt.plot(sub["n"], sub["speedup"], label=version)

plt.xlabel("N")
plt.ylabel("Speedup")
plt.xscale("log")
plt.legend()
plt.title("Speedup vs N")
plt.savefig("results/plots/speedup_vs_n.png")
plt.close()

# =========================
# TABELLA RIASSUNTIVA
# =========================
summary = df.copy()
summary["time (mean ± std)"] = summary.apply(
    lambda row: f"{row['mean_time']:.6f} ± {row['std_time']:.6f}", axis=1
)
summary["throughput (M elems/s)"] = summary["throughput"].apply(lambda x: f"{x/1e6:.2f}")
summary["speedup"] = summary["speedup"].apply(lambda x: f"{x:.2f}")

summary_table = summary[[
    "version", "n", "time (mean ± std)", "throughput (M elems/s)", "speedup"
]]

print("\n===== SUMMARY TABLE =====")
print(summary_table)

summary_table.to_csv("results/tables/summary_table.csv", index=False)

# =========================
# TABELLA LATEX
# =========================
latex_table = summary_table.to_latex(index=False)
with open("results/tables/summary_table.tex", "w") as f:
    f.write(latex_table)

print("\nLaTeX table saved in results/tables/summary_table.tex")