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
# TABELLA RIASSUNTIVA
# =========================
summary = df.groupby("N").agg({
    "time_baseline": ["mean", "std"],
    "time_autovec": ["mean", "std"],
    "time_avx": ["mean", "std"],
    "speedup_autovec": "mean",
    "speedup_avx": "mean",
    "throughput_avx": "mean"
}).reset_index()

summary.columns = [
    "N",
    "baseline_mean", "baseline_std",
    "autovec_mean", "autovec_std",
    "avx_mean", "avx_std",
    "speedup_autovec",
    "speedup_avx",
    "throughput_avx"
]

summary.to_csv("results/tables/summary_table.csv", index=False)

# =========================
# TABELLA LATEX
# =========================
latex_table = summary.to_latex(index=False, float_format="%.4f")
with open("results/tables/summary_table.tex", "w") as f:
    f.write(latex_table)

# =========================
# GRAFICO TIME VS N
# =========================
plt.figure()
plt.errorbar(summary["N"], summary["baseline_mean"], yerr=summary["baseline_std"], label="baseline")
plt.errorbar(summary["N"], summary["autovec_mean"], yerr=summary["autovec_std"], label="autovec")
plt.errorbar(summary["N"], summary["avx_mean"], yerr=summary["avx_std"], label="avx")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.title("Execution Time vs N")
plt.legend()
plt.savefig("results/plots/time_vs_n.png")
plt.close()

# =========================
# GRAFICO THROUGHPUT VS N
# =========================
plt.figure()
plt.plot(summary["N"], summary["throughput_avx"], label="AVX")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Throughput (elements/s)")
plt.title("Throughput vs N")
plt.legend()
plt.savefig("results/plots/throughput_vs_n.png")
plt.close()

# =========================
# GRAFICO SPEEDUP VS N
# =========================
plt.figure()
plt.plot(summary["N"], summary["speedup_autovec"], label="Autovec")
plt.plot(summary["N"], summary["speedup_avx"], label="AVX")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Speedup")
plt.title("Speedup vs N")
plt.legend()
plt.savefig("results/plots/speedup_vs_n.png")
plt.close()

print("Analysis complete. Tables and plots saved in results/")