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

df["speedup_autovec"] = df["time_baseline"] / df["time_autovec"]
df["speedup_avx"] = df["time_baseline"] / df["time_avx"]

# =========================
# SALVA RAW CSV
# =========================
df.to_csv("results/tables/raw_results.csv", index=False)

# =========================
# TABELLA RIASSUNTIVA (MEDIANA)
# =========================
summary = df.groupby("N").agg({
    "time_baseline": ["median", "std"],
    "time_autovec": ["median", "std"],
    "time_avx": ["median", "std"],
    "speedup_autovec": ["median", "std"],
    "speedup_avx": ["median", "std"],
    "throughput_autovec": ["median", "std"],
    "throughput_avx": ["median", "std"]
}).reset_index()

summary.columns = [
    "N",
    "baseline_median", "baseline_std",
    "autovec_median", "autovec_std",
    "avx_median", "avx_std",
    "speedup_autovec_median", "speedup_autovec_std",
    "speedup_avx_median", "speedup_avx_std",
    "throughput_autovec_median", "throughput_autovec_std",
    "throughput_avx_median", "throughput_avx_std"
]

# =========================
# CONVERSIONI UNITÀ
# =========================
# tempi in microsecondi
for col in ["baseline_median", "baseline_std",
            "autovec_median", "autovec_std",
            "avx_median", "avx_std"]:
    summary[col] *= 1e6

# throughput in milioni elementi/s
for col in ["throughput_autovec_median", "throughput_autovec_std",
            "throughput_avx_median", "throughput_avx_std"]:
    summary[col] /= 1e6

# =========================
# FUNZIONI FORMATTAZIONE
# =========================
def fmt_mean_std(mean, std, decimals=2, unit=""):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f} {unit}".strip()

def fmt_mean(mean, decimals=2, suffix=""):
    return f"{mean:.{decimals}f}{suffix}"

# =========================
# TABELLA FORMATTA
# =========================
formatted = pd.DataFrame({
    "N": summary["N"].astype(int),
    "baseline": [
        fmt_mean_std(m, s, decimals=2, unit="µs")
        for m, s in zip(summary["baseline_median"], summary["baseline_std"])
    ],
    "autovec": [
        fmt_mean_std(m, s, decimals=2, unit="µs")
        for m, s in zip(summary["autovec_median"], summary["autovec_std"])
    ],
    "AVX2": [
        fmt_mean_std(m, s, decimals=2, unit="µs")
        for m, s in zip(summary["avx_median"], summary["avx_std"])
    ],
    "speedup autovec": [
        fmt_mean(m, decimals=2, suffix="x")
        for m in summary["speedup_autovec_median"]
    ],
    "speedup AVX2": [
        fmt_mean(m, decimals=2, suffix="x")
        for m in summary["speedup_avx_median"]
    ],
    "throughput autovec": [
        fmt_mean_std(m, s, decimals=2, unit="Melem/s")
        for m, s in zip(summary["throughput_autovec_median"], summary["throughput_autovec_std"])
    ],
    "throughput AVX2": [
        fmt_mean_std(m, s, decimals=2, unit="Melem/s")
        for m, s in zip(summary["throughput_avx_median"], summary["throughput_avx_std"])
    ]
})

# =========================
# SALVA TABELLA CSV
# =========================
formatted.to_csv("results/tables/summary_table.csv", index=False)

# =========================
# TABELLA LATEX
# =========================
tabular_only = formatted.to_latex(
    index=False,
    escape=True,
    column_format="rlllllll"
)

latex_table = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
""" + "\n" + tabular_only + r"""%
}
\caption{Performance comparison between baseline, autovectorized, and AVX2 implementations. Median execution time over multiple runs is reported.}
\label{tab:performance-comparison}
\end{table}
"""

with open("results/tables/summary_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

# =========================
# GRAFICO TIME VS N
# =========================
plt.figure()
plt.errorbar(summary["N"], summary["baseline_median"], yerr=summary["baseline_std"], label="baseline")
plt.errorbar(summary["N"], summary["autovec_median"], yerr=summary["autovec_std"], label="autovec")
plt.errorbar(summary["N"], summary["avx_median"], yerr=summary["avx_std"], label="AVX2")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Time (µs)")
plt.title("Execution Time vs N")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/time_vs_n.png")
plt.close()

# =========================
# GRAFICO THROUGHPUT VS N
# =========================
plt.figure()
plt.plot(summary["N"], summary["throughput_autovec_median"], label="Autovec")
plt.plot(summary["N"], summary["throughput_avx_median"], label="AVX2")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Throughput (Melem/s)")
plt.title("Throughput vs N")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/throughput_vs_n.png")
plt.close()

# =========================
# GRAFICO SPEEDUP VS N
# =========================
plt.figure()
plt.plot(summary["N"], summary["speedup_autovec_median"], label="Autovec")
plt.plot(summary["N"], summary["speedup_avx_median"], label="AVX2")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Speedup")
plt.title("Speedup vs N")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/speedup_vs_n.png")
plt.close()

# =========================
# GRAFICO SPEEDUP VS PROBLEM SIZE (NUOVO)
# =========================
plt.figure()
plt.plot(summary["N"], summary["speedup_autovec_median"], marker='o', label="Autovec Speedup")
plt.plot(summary["N"], summary["speedup_avx_median"], marker='o', label="AVX2 Speedup")
plt.xscale("log")
plt.xlabel("Problem size (N)")
plt.ylabel("Speedup")
plt.title("Speedup vs Problem Size")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/speedup_vs_size.png")
plt.close()

print("Analysis complete. Tables and plots saved in results/")