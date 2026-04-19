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
# CONTROLLI
# =========================
required_cols = [
    "NR", "NS", "P", "seed", "max_key", "threads",
    "time_seq", "time_par"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results.json: {missing}")

df["N"] = df["NR"]

# =========================
# FILTRI DATASET
# =========================
if "experiment_type" in df.columns:
    strong_df = df[df["experiment_type"] == "strong"].copy()
    weak_all_df = df[df["experiment_type"] == "weak"].copy()
else:
    strong_df = df.copy()
    weak_all_df = df.copy()

# =========================
# STRONG SCALING SOLO N = 10.000.000
# =========================

# =========================
# COSTRUZIONE DATI WEAK SCALING
# =========================
weak_agg = weak_all_df.groupby(["N", "threads"]).agg(
    time_par_median=("time_par", "median"),
    time_par_std=("time_par", "std"),
    checksum_correct=("checksum_correct", "min")
).reset_index()

available_pairs = set(zip(weak_agg["N"], weak_agg["threads"]))
unique_threads = sorted(weak_agg["threads"].unique())
unique_N = sorted(weak_agg["N"].unique())

weak_rows = []

for base_N in unique_N:
    candidate = []
    ok = True
    for t in unique_threads:
        needed_N = base_N * t
        if (needed_N, t) not in available_pairs:
            ok = False
            break
        row = weak_agg[(weak_agg["N"] == needed_N) & (weak_agg["threads"] == t)].iloc[0]
        candidate.append(row)
    if ok:
        weak_rows = candidate
        break

if weak_rows:
    weak_df = pd.DataFrame(weak_rows).copy().reset_index(drop=True)

    base_time = weak_df[weak_df["threads"] == 1]["time_par_median"].iloc[0]
    weak_df["weak_efficiency"] = base_time / weak_df["time_par_median"]

    weak_df.to_csv("results/tables/weak_scaling_selected.csv", index=False)

    # =========================
    # GRAFICO 3: WEAK SCALING TIME
    # =========================
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        weak_df["threads"],
        weak_df["time_par_median"],
        yerr=weak_df["time_par_std"].fillna(0),
        marker="o",
        capsize=4,
        label="Measured"
    )

    plt.axhline(base_time, linestyle="--", label="Ideal constant time")

    plt.xticks(list(weak_df["threads"]))
    plt.xlabel("Threads")
    plt.ylabel("Time (s)")
    plt.title("Weak Scaling: Time vs Threads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/weak_scaling_time.png", dpi=200)
    plt.close()

    # =========================
    # GRAFICO 4: WEAK SCALING EFFICIENCY
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(
        weak_df["threads"],
        weak_df["weak_efficiency"],
        marker="o",
        label="Weak scaling efficiency"
    )
    plt.axhline(1.0, linestyle="--", label="Ideal efficiency")

    plt.xticks(list(weak_df["threads"]))
    plt.xlabel("Threads")
    plt.ylabel("Weak Scaling Efficiency")
    plt.title("Weak Scaling: Efficiency vs Threads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/weak_scaling_efficiency.png", dpi=200)
    plt.close()

else:
    print("No valid weak scaling subset found.")
    print("To evaluate weak scaling correctly, you need runs where N grows proportionally with threads.")

# =========================
# ISTOGRAMMA STACKED DELLE FASI
# =========================
phase_cols_seq = [
    "time_histogram_R_seq",
    "time_prefix_R_seq",
    "time_scatter_R_seq",
    "time_histogram_S_seq",
    "time_prefix_S_seq",
    "time_scatter_S_seq",
    "time_join_seq",
    "time_accumulation_seq",
    "time_total_seq",
]

phase_cols_par = [
    "time_histogram_R_par",
    "time_prefix_R_par",
    "time_scatter_R_par",
    "time_histogram_S_par",
    "time_prefix_S_par",
    "time_scatter_S_par",
    "time_join_par",
    "time_accumulation_par",
    "time_total_par",
]

has_phase_data = all(col in strong_df.columns for col in phase_cols_seq + phase_cols_par)

if has_phase_data:
    phase_rows = []

    # riga sequenziale: basta una sola, facciamo mediana sui seed
    seq_row = {
        "label": "seq",
        "Histogram R": strong_df["time_histogram_R_seq"].median(),
        "Prefix R": strong_df["time_prefix_R_seq"].median(),
        "Scatter R": strong_df["time_scatter_R_seq"].median(),
        "Histogram S": strong_df["time_histogram_S_seq"].median(),
        "Prefix S": strong_df["time_prefix_S_seq"].median(),
        "Scatter S": strong_df["time_scatter_S_seq"].median(),
        "Join Local": strong_df["time_join_seq"].median(),
        "Accumulation": strong_df["time_accumulation_seq"].median(),
        "Total": strong_df["time_total_seq"].median(),
    }
    phase_rows.append(seq_row)

    # righe parallele: una per ogni numero di thread
    for t in sorted(strong_df["threads"].unique()):
        tmp = strong_df[strong_df["threads"] == t]

        row = {
            "label": str(int(t)),
            "Histogram R": tmp["time_histogram_R_par"].median(),
            "Prefix R": tmp["time_prefix_R_par"].median(),
            "Scatter R": tmp["time_scatter_R_par"].median(),
            "Histogram S": tmp["time_histogram_S_par"].median(),
            "Prefix S": tmp["time_prefix_S_par"].median(),
            "Scatter S": tmp["time_scatter_S_par"].median(),
            "Join Local": tmp["time_join_par"].median(),
            "Accumulation": tmp["time_accumulation_par"].median(),
            "Total": tmp["time_total_par"].median(),
        }
        phase_rows.append(row)

    phase_df = pd.DataFrame(phase_rows)

    phase_names = [
        "Histogram R",
        "Prefix R",
        "Scatter R",
        "Histogram S",
        "Prefix S",
        "Scatter S",
        "Join Local",
        "Accumulation",
    ]

    phase_df["Measured Sum"] = phase_df[phase_names].sum(axis=1)
    phase_df["Other Overhead"] = (phase_df["Total"] - phase_df["Measured Sum"]).clip(lower=0.0)

    # conversione in ms
    for col in phase_names + ["Other Overhead", "Total"]:
        phase_df[col] = phase_df[col] * 1000.0

    phase_df.to_csv("results/tables/phase_breakdown.csv", index=False)

    colors = {
        "Histogram R":   "#4E79A7",  # blue
        "Prefix R":      "#A0CBE8",  # light blue
        "Scatter R":     "#2F5D8A",  # deep steel blue

        "Histogram S":   "#59A14F",  # green
        "Prefix S":      "#8CD17D",  # light green
        "Scatter S":     "#2E7D32",  # dark green

        "Join Local":    "#E15759",  # coral red
        "Accumulation":  "#F28E2B",  # orange
        "Other Overhead":"#9D9D9D",  # neutral gray
    }

    plt.figure(figsize=(11, 6))
    bottom = np.zeros(len(phase_df))

    plt.ylabel("Time (ms)")
    plt.xlabel("Series / Threads")
    plt.title("Phase Breakdown — Execution Time")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("results/plots/phase_breakdown.png", dpi=200)
    plt.close()

else:
    print("Phase timing columns not found. Skipping phase breakdown histogram.")