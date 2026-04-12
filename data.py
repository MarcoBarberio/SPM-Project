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
# CONTROLLI / NORMALIZZAZIONE
# =========================
required_cols = [
    "NR", "NS", "P", "seed", "max_key", "threads",
    "time_seq", "time_par", "speedup", "checksum_correct"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results.json: {missing}")

# chiave di dimensione problema per comodità
# assumiamo spesso NR == NS, ma teniamo comunque entrambe
df["N"] = df["NR"]

# throughput join/s
df["throughput_seq"] = df["NR"] / df["time_seq"]
df["throughput_par"] = df["NR"] / df["time_par"]

# weak scaling efficiency rispetto a T=1
# la calcoliamo dopo in una tabella dedicata

# =========================
# SALVA RAW CSV
# =========================
df.to_csv("results/tables/raw_results.csv", index=False)

# =========================
# FORMATTAZIONE
# =========================
def fmt_mean_std(mean, std, decimals=2, unit=""):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f} {unit}".strip()

def fmt_mean(mean, decimals=2, suffix=""):
    return f"{mean:.{decimals}f}{suffix}"

# =========================
# TABELLA 1: SEQ VS PAR PER (N, threads)
# =========================
summary_by_n_t = df.groupby(["N", "threads"]).agg({
    "time_seq": ["median", "std"],
    "time_par": ["median", "std"],
    "speedup": ["median", "std"],
    "throughput_seq": ["median", "std"],
    "throughput_par": ["median", "std"],
    "checksum_correct": ["min"]
}).reset_index()

summary_by_n_t.columns = [
    "N", "threads",
    "time_seq_median", "time_seq_std",
    "time_par_median", "time_par_std",
    "speedup_median", "speedup_std",
    "throughput_seq_median", "throughput_seq_std",
    "throughput_par_median", "throughput_par_std",
    "checksum_correct"
]

# conversioni unità
for col in [
    "time_seq_median", "time_seq_std",
    "time_par_median", "time_par_std"
]:
    summary_by_n_t[col] *= 1e6   # microsecondi

for col in [
    "throughput_seq_median", "throughput_seq_std",
    "throughput_par_median", "throughput_par_std"
]:
    summary_by_n_t[col] /= 1e6   # milioni di record/s

formatted_seq_par = pd.DataFrame({
    "N": summary_by_n_t["N"].astype(int),
    "threads": summary_by_n_t["threads"].astype(int),
    "seq time": [
        fmt_mean_std(m, s, unit="µs")
        for m, s in zip(summary_by_n_t["time_seq_median"], summary_by_n_t["time_seq_std"])
    ],
    "par time": [
        fmt_mean_std(m, s, unit="µs")
        for m, s in zip(summary_by_n_t["time_par_median"], summary_by_n_t["time_par_std"])
    ],
    "speedup": [
        fmt_mean_std(m, s)
        for m, s in zip(summary_by_n_t["speedup_median"], summary_by_n_t["speedup_std"])
    ],
    "seq throughput": [
        fmt_mean_std(m, s, unit="Mrec/s")
        for m, s in zip(summary_by_n_t["throughput_seq_median"], summary_by_n_t["throughput_seq_std"])
    ],
    "par throughput": [
        fmt_mean_std(m, s, unit="Mrec/s")
        for m, s in zip(summary_by_n_t["throughput_par_median"], summary_by_n_t["throughput_par_std"])
    ],
    "checksum correct": summary_by_n_t["checksum_correct"].astype(bool)
})

formatted_seq_par.to_csv("results/tables/summary_seq_par.csv", index=False)

latex_seq_par = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
""" + "\n" + formatted_seq_par.to_latex(index=False, escape=True) + r"""%
}
\caption{Sequential vs parallel partitioned hash join. Median and standard deviation over multiple runs.}
\label{tab:seq-par-performance}
\end{table}
"""

with open("results/tables/summary_seq_par.tex", "w", encoding="utf-8") as f:
    f.write(latex_seq_par)

# =========================
# TABELLA 2: STRONG SCALING
# =========================
# Per ogni N, variamo solo i thread
strong = df.groupby(["N", "threads"]).agg({
    "time_par": ["median", "std"],
    "speedup": ["median", "std"],
    "checksum_correct": ["min"]
}).reset_index()

strong.columns = [
    "N", "threads",
    "time_par_median", "time_par_std",
    "speedup_median", "speedup_std",
    "checksum_correct"
]

strong["efficiency_median"] = strong["speedup_median"] / strong["threads"]

strong["time_par_median"] *= 1e6
strong["time_par_std"] *= 1e6

formatted_strong = pd.DataFrame({
    "N": strong["N"].astype(int),
    "threads": strong["threads"].astype(int),
    "par time": [
        fmt_mean_std(m, s, unit="µs")
        for m, s in zip(strong["time_par_median"], strong["time_par_std"])
    ],
    "speedup": [
        fmt_mean_std(m, s)
        for m, s in zip(strong["speedup_median"], strong["speedup_std"])
    ],
    "efficiency": [
        fmt_mean(m, suffix="")
        for m in strong["efficiency_median"]
    ],
    "checksum correct": strong["checksum_correct"].astype(bool)
})

formatted_strong.to_csv("results/tables/summary_strong_scaling.csv", index=False)

latex_strong = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
""" + "\n" + formatted_strong.to_latex(index=False, escape=True) + r"""%
}
\caption{Strong scaling results for the parallel partitioned hash join.}
\label{tab:strong-scaling}
\end{table}
"""

with open("results/tables/summary_strong_scaling.tex", "w", encoding="utf-8") as f:
    f.write(latex_strong)

# =========================
# TABELLA 3: WEAK SCALING
# =========================
# Assumiamo che i dati weak scaling abbiano N che cresce con i thread.
# Calcoliamo l'efficienza weak rispetto al caso T=1 con stesso carico per thread.
weak_base = df[df["threads"] == 1].groupby("N").agg({
    "time_par": "median"
}).reset_index().rename(columns={"time_par": "base_time_t1"})

weak = df.groupby(["N", "threads"]).agg({
    "time_par": ["median", "std"],
    "checksum_correct": ["min"]
}).reset_index()

weak.columns = [
    "N", "threads",
    "time_par_median", "time_par_std",
    "checksum_correct"
]

# per stimare weak scaling, prendiamo come baseline il più piccolo N con T=1
if not weak_base.empty:
    reference_time = weak_base.sort_values("N").iloc[0]["base_time_t1"]
    weak["weak_efficiency"] = reference_time / weak["time_par_median"]
else:
    weak["weak_efficiency"] = np.nan

weak["time_par_median"] *= 1e6
weak["time_par_std"] *= 1e6

formatted_weak = pd.DataFrame({
    "N": weak["N"].astype(int),
    "threads": weak["threads"].astype(int),
    "par time": [
        fmt_mean_std(m, s, unit="µs")
        for m, s in zip(weak["time_par_median"], weak["time_par_std"])
    ],
    "weak efficiency": [
        fmt_mean(m) if pd.notna(m) else "nan"
        for m in weak["weak_efficiency"]
    ],
    "checksum correct": weak["checksum_correct"].astype(bool)
})

formatted_weak.to_csv("results/tables/summary_weak_scaling.csv", index=False)

latex_weak = r"""\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
""" + "\n" + formatted_weak.to_latex(index=False, escape=True) + r"""%
}
\caption{Weak scaling results for the parallel partitioned hash join.}
\label{tab:weak-scaling}
\end{table}
"""

with open("results/tables/summary_weak_scaling.tex", "w", encoding="utf-8") as f:
    f.write(latex_weak)

# =========================
# GRAFICI
# =========================

# 1) Time seq vs par al variare di N, per ogni T
for t in sorted(df["threads"].unique()):
    tmp = summary_by_n_t[summary_by_n_t["threads"] == t].sort_values("N")
    plt.figure()
    plt.errorbar(tmp["N"], tmp["time_seq_median"], yerr=tmp["time_seq_std"], label="seq")
    plt.errorbar(tmp["N"], tmp["time_par_median"], yerr=tmp["time_par_std"], label=f"par T={t}")
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Time (µs)")
    plt.title(f"Sequential vs Parallel Time (T={t})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/plots/time_seq_vs_par_t{t}.png")
    plt.close()

# 2) Speedup vs threads per ogni N
for n in sorted(df["N"].unique()):
    tmp = strong[strong["N"] == n].sort_values("threads")
    plt.figure()
    plt.errorbar(tmp["threads"], tmp["speedup_median"], yerr=tmp["speedup_std"], label=f"N={n}")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title(f"Strong Scaling Speedup (N={n})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/plots/speedup_vs_threads_n{int(n)}.png")
    plt.close()

# 3) Efficiency vs threads per ogni N
for n in sorted(df["N"].unique()):
    tmp = strong[strong["N"] == n].sort_values("threads")
    plt.figure()
    plt.plot(tmp["threads"], tmp["efficiency_median"], marker="o", label=f"N={n}")
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.title(f"Strong Scaling Efficiency (N={n})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/plots/efficiency_vs_threads_n{int(n)}.png")
    plt.close()

# 4) Throughput par vs N per ogni T
for t in sorted(summary_by_n_t["threads"].unique()):
    tmp = summary_by_n_t[summary_by_n_t["threads"] == t].sort_values("N")
    plt.figure()
    plt.errorbar(tmp["N"], tmp["throughput_par_median"], yerr=tmp["throughput_par_std"], label=f"T={t}")
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Throughput (Mrec/s)")
    plt.title(f"Parallel Throughput vs N (T={t})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/plots/throughput_vs_n_t{t}.png")
    plt.close()

# 5) Weak scaling: time vs threads
tmp = weak.sort_values("threads")
plt.figure()
plt.errorbar(tmp["threads"], tmp["time_par_median"], yerr=tmp["time_par_std"], label="weak scaling")
plt.xlabel("Threads")
plt.ylabel("Time (µs)")
plt.title("Weak Scaling Time")
plt.grid(True)
plt.legend()
plt.savefig("results/plots/weak_scaling_time.png")
plt.close()

# 6) Weak scaling efficiency vs threads
tmp = weak.sort_values("threads")
plt.figure()
plt.plot(tmp["threads"], tmp["weak_efficiency"], marker="o", label="weak efficiency")
plt.xlabel("Threads")
plt.ylabel("Weak Scaling Efficiency")
plt.title("Weak Scaling Efficiency")
plt.grid(True)
plt.legend()
plt.savefig("results/plots/weak_scaling_efficiency.png")
plt.close()

print("Analysis complete. Tables and plots saved in results/")