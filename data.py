import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# =========================
# CARTELLE
# =========================
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# =========================
# LOAD JSON
# =========================
with open("results/results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# =========================
# CALCOLI
# =========================
df["throughput_baseline"] = df["N"] / df["time_baseline"]
df["throughput_autovec"] = df["N"] / df["time_autovec"]
df["throughput_avx"] = df["N"] / df["time_avx"]

df["throughput_cuda_total"] = df["N"] / df["time_cuda_total"]
df["throughput_cuda_kernel"] = df["N"] / df["time_cuda_kernel"]

df["speedup_autovec"] = df["time_baseline"] / df["time_autovec"]
df["speedup_avx"] = df["time_baseline"] / df["time_avx"]
df["speedup_cuda_total"] = df["time_baseline"] / df["time_cuda_total"]
df["speedup_cuda_kernel"] = df["time_baseline"] / df["time_cuda_kernel"]

# =========================
# SALVA RAW CSV
# =========================
df.to_csv("results/tables/raw_results.csv", index=False)

# =========================
# TABELLA RIASSUNTIVA
# =========================
summary = df.groupby("N").agg({
    "time_baseline": ["median", "std"],
    "time_autovec": ["median", "std"],
    "time_avx": ["median", "std"],
    "time_cuda_total": ["median", "std"],
    "time_cuda_kernel": ["median", "std"],
    "time_cuda_h2d": ["median", "std"],
    "time_cuda_d2h": ["median", "std"],
    "speedup_autovec": ["median", "std"],
    "speedup_avx": ["median", "std"],
    "speedup_cuda_total": ["median", "std"],
    "speedup_cuda_kernel": ["median", "std"],
    "throughput_avx": ["median", "std"],
    "throughput_cuda_total": ["median", "std"],
    "throughput_cuda_kernel": ["median", "std"]
}).reset_index()

summary.columns = [
    "N",
    "baseline_median", "baseline_std",
    "autovec_median", "autovec_std",
    "avx_median", "avx_std",
    "cuda_total_median", "cuda_total_std",
    "cuda_kernel_median", "cuda_kernel_std",
    "cuda_h2d_median", "cuda_h2d_std",
    "cuda_d2h_median", "cuda_d2h_std",
    "speedup_autovec_median", "speedup_autovec_std",
    "speedup_avx_median", "speedup_avx_std",
    "speedup_cuda_total_median", "speedup_cuda_total_std",
    "speedup_cuda_kernel_median", "speedup_cuda_kernel_std",
    "throughput_avx_median", "throughput_avx_std",
    "throughput_cuda_total_median", "throughput_cuda_total_std",
    "throughput_cuda_kernel_median", "throughput_cuda_kernel_std"
]

# =========================
# CONVERSIONI UNITÀ
# =========================
# tempi in microsecondi
for col in ["baseline_median","baseline_std",
            "autovec_median","autovec_std",
            "avx_median","avx_std",
            "cuda_total_median","cuda_total_std",
            "cuda_kernel_median","cuda_kernel_std",
            "cuda_h2d_median","cuda_h2d_std",
            "cuda_d2h_median","cuda_d2h_std"]:
    summary[col] *= 1e6

# throughput in milioni elem/s
for col in ["throughput_avx_median","throughput_avx_std",
            "throughput_cuda_total_median","throughput_cuda_total_std",
            "throughput_cuda_kernel_median","throughput_cuda_kernel_std"]:
    summary[col] /= 1e6

# =========================
# FORMATTAZIONE
# =========================
def fmt_mean_std(mean, std, decimals=2, unit=""):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f} {unit}".strip()

def fmt_mean(mean, decimals=2, suffix=""):
    return f"{mean:.{decimals}f}{suffix}"

# =========================
# TABELLA CPU
# =========================
formatted_cpu = pd.DataFrame({
    "N": summary["N"].astype(int),
    "baseline": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["baseline_median"], summary["baseline_std"])],
    "autovec": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["autovec_median"], summary["autovec_std"])],
    "AVX2": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["avx_median"], summary["avx_std"])],
    "speedup autovec": [fmt_mean(m, suffix="x") for m in summary["speedup_autovec_median"]],
    "speedup AVX2": [fmt_mean(m, suffix="x") for m in summary["speedup_avx_median"]],
    "throughput AVX2": [fmt_mean_std(m, s, unit="Melem/s") for m, s in zip(summary["throughput_avx_median"], summary["throughput_avx_std"])]
})

formatted_cpu.to_csv("results/tables/summary_cpu.csv", index=False)

# =========================
# TABELLA CUDA
# =========================
formatted_cuda = pd.DataFrame({
    "N": summary["N"].astype(int),
    "baseline": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["baseline_median"], summary["baseline_std"])],
    "CUDA total": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["cuda_total_median"], summary["cuda_total_std"])],
    "CUDA kernel": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["cuda_kernel_median"], summary["cuda_kernel_std"])],
    "H2D": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["cuda_h2d_median"], summary["cuda_h2d_std"])],
    "D2H": [fmt_mean_std(m, s, unit="µs") for m, s in zip(summary["cuda_d2h_median"], summary["cuda_d2h_std"])],
    "speedup CUDA total": [fmt_mean(m, suffix="x") for m in summary["speedup_cuda_total_median"]],
    "speedup CUDA kernel": [fmt_mean(m, suffix="x") for m in summary["speedup_cuda_kernel_median"]],
    "throughput CUDA total": [fmt_mean_std(m, s, unit="Melem/s") for m, s in zip(summary["throughput_cuda_total_median"], summary["throughput_cuda_total_std"])],
    "throughput CUDA kernel": [fmt_mean_std(m, s, unit="Melem/s") for m, s in zip(summary["throughput_cuda_kernel_median"], summary["throughput_cuda_kernel_std"])]
})

formatted_cuda.to_csv("results/tables/summary_cuda.csv", index=False)
# =========================
# TABELLE LATEX
# =========================
latex_cpu = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
""" + "\n" + formatted_cpu.to_latex(index=False, escape=True) + r"""%
}
\caption{Performance comparison between baseline, autovectorized and AVX2 implementations. Median execution time over multiple runs is reported.}
\label{tab:cpu-performance}
\end{table}
"""

with open("results/tables/summary_cpu.tex", "w", encoding="utf-8") as f:
    f.write(latex_cpu)

latex_cuda = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
""" + "\n" + formatted_cuda.to_latex(index=False, escape=True) + r"""%
}
\caption{CUDA performance breakdown. Total time includes host-device transfers.}
\label{tab:cuda-performance}
\end{table}
"""

with open("results/tables/summary_cuda.tex", "w", encoding="utf-8") as f:
    f.write(latex_cuda)

with open("results/tables/summary_cuda.tex", "w", encoding="utf-8") as f:
    f.write(latex_cuda)

# =========================
# GRAFICI
# =========================

# Time vs N
plt.figure()
plt.errorbar(summary["N"], summary["baseline_median"], yerr=summary["baseline_std"], label="baseline")
plt.errorbar(summary["N"], summary["autovec_median"], yerr=summary["autovec_std"], label="autovec")
plt.errorbar(summary["N"], summary["avx_median"], yerr=summary["avx_std"], label="AVX2")
plt.errorbar(summary["N"], summary["cuda_total_median"], yerr=summary["cuda_total_std"], label="CUDA total")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Time (µs)")
plt.title("Execution Time vs N")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/time_vs_n.png")
plt.close()

# Throughput vs N
plt.figure()
plt.plot(summary["N"], summary["throughput_avx_median"], label="AVX2")
plt.plot(summary["N"], summary["throughput_cuda_kernel_median"], label="CUDA kernel")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Throughput (Melem/s)")
plt.title("Throughput vs N")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/throughput_vs_n.png")
plt.close()

# Speedup vs N
plt.figure()
plt.plot(summary["N"], summary["speedup_avx_median"], label="AVX2")
plt.plot(summary["N"], summary["speedup_cuda_total_median"], label="CUDA total")
plt.plot(summary["N"], summary["speedup_cuda_kernel_median"], label="CUDA kernel")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Speedup")
plt.title("Speedup vs N")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/speedup_vs_n.png")
plt.close()

# CUDA total vs kernel
plt.figure()
plt.plot(summary["N"], summary["baseline_median"], label="CPU baseline")
plt.plot(summary["N"], summary["cuda_total_median"], label="CUDA total")
plt.plot(summary["N"], summary["cuda_kernel_median"], label="CUDA kernel")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Time (µs)")
plt.title("CUDA Total vs Kernel vs CPU")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/cuda_total_vs_kernel.png")
plt.close()

# CUDA breakdown
plt.figure()
plt.plot(summary["N"], summary["cuda_h2d_median"], label="H2D")
plt.plot(summary["N"], summary["cuda_kernel_median"], label="Kernel")
plt.plot(summary["N"], summary["cuda_d2h_median"], label="D2H")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Time (µs)")
plt.title("CUDA Time Breakdown")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/cuda_breakdown.png")
plt.close()

print("Analysis complete. Tables and plots saved in results/")