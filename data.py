import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_INPUT_CANDIDATES = [
    Path("results/results_modulo3.json"),
    Path("results_modulo3.json"),
]

DEFAULT_PLOTS_DIR = Path("results/plots_modulo3")
DEFAULT_TABLES_DIR = Path("results/tables_modulo3")


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Module 3 plots for OpenMP for/task hash join experiments."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to results_modulo3.json. Default: results/results_modulo3.json, then ./results_modulo3.json."
    )

    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory where PNG plots will be saved."
    )

    parser.add_argument(
        "--tables-dir",
        type=str,
        default=str(DEFAULT_TABLES_DIR),
        help="Directory where CSV summary tables will be saved."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Resolution of generated figures."
    )

    parser.add_argument(
        "--skew-thread",
        type=int,
        default=None,
        help="Thread count used for the skew-level comparison. Default: max available thread count."
    )

    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Show standard deviation error bars over seeds."
    )

    return parser.parse_args()


# ============================================================
# IO / VALIDATION
# ============================================================

def resolve_input_path(path_arg):
    if path_arg is not None:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find results_modulo3.json. Tried: "
        + ", ".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    )


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    required_cols = [
        "experiment_type",
        "implementation",
        "NR",
        "NS",
        "P",
        "seed",
        "threads",
        "workload",
        "time_impl",
        "join_count_impl",
        "checksum1_impl",
        "checksum2_impl",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in results file: {missing}")

    df["N"] = df["NR"]

    for col in ["threads", "NR", "NS", "P", "seed", "N"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["time_seq", "time_impl", "speedup_vs_seq", "skew_percent", "N_per_thread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def check_correctness(df):
    # Weak scaling may have checksum_correct = None because the sequential baseline was not run.
    if "checksum_correct" not in df.columns:
        return

    comparable = df[df["checksum_correct"].notna()].copy()
    if comparable.empty:
        print("No rows with checksum_correct available.")
        return

    bad = comparable[comparable["checksum_correct"] != True]
    if not bad.empty:
        out_cols = [
            "experiment_type", "implementation", "workload", "NR", "NS", "P", "seed", "threads",
            "join_count_seq", "join_count_impl", "checksum1_seq", "checksum1_impl",
            "checksum2_seq", "checksum2_impl",
        ]
        available = [c for c in out_cols if c in bad.columns]
        raise ValueError(
            "Some checksum comparisons failed. First failing rows:\n"
            + bad[available].head(10).to_string(index=False)
        )

    print(f"Correctness check passed for {len(comparable)} comparable rows.")


# ============================================================
# AGGREGATION HELPERS
# ============================================================

def aggregate_strong_speedup(df):
    strong = df[df["experiment_type"] == "strong"].copy()

    if strong.empty:
        return pd.DataFrame()

    if "speedup_vs_seq" not in strong.columns:
        if {"time_seq", "time_impl"}.issubset(strong.columns):
            strong["speedup_vs_seq"] = strong["time_seq"] / strong["time_impl"]
        else:
            raise ValueError("Strong scaling data needs either speedup_vs_seq or both time_seq and time_impl.")

    strong = strong[strong["speedup_vs_seq"].notna()].copy()

    agg = strong.groupby(
        ["workload", "implementation", "threads"],
        as_index=False
    ).agg(
        speedup_median=("speedup_vs_seq", "median"),
        speedup_mean=("speedup_vs_seq", "mean"),
        speedup_std=("speedup_vs_seq", "std"),
        time_impl_median=("time_impl", "median"),
        time_seq_median=("time_seq", "median"),
        seeds=("seed", "nunique"),
        N=("N", "median"),
        P=("P", "median"),
    )

    return agg.sort_values(["workload", "implementation", "threads"])


def aggregate_weak_efficiency(df):
    weak = df[df["experiment_type"] == "weak"].copy()

    if weak.empty:
        return pd.DataFrame()

    weak = weak[weak["time_impl"].notna()].copy()

    time_agg = weak.groupby(
        ["workload", "implementation", "threads"],
        as_index=False
    ).agg(
        time_impl_median=("time_impl", "median"),
        time_impl_mean=("time_impl", "mean"),
        time_impl_std=("time_impl", "std"),
        N=("N", "median"),
        N_per_thread=("N_per_thread", "median") if "N_per_thread" in weak.columns else ("N", "median"),
        seeds=("seed", "nunique"),
        P=("P", "median"),
    )

    rows = []

    for (workload, implementation), group in time_agg.groupby(["workload", "implementation"]):
        group = group.sort_values("threads").copy()

        base = group[group["threads"] == group["threads"].min()]
        if base.empty:
            continue

        base_threads = int(base["threads"].iloc[0])
        base_time = float(base["time_impl_median"].iloc[0])

        # Weak scaling efficiency:
        # ideal weak scaling means constant execution time.
        # If the smallest run is T=1, this is time(1) / time(T).
        group["base_threads"] = base_threads
        group["base_time_impl_median"] = base_time
        group["weak_efficiency"] = base_time / group["time_impl_median"]
        rows.append(group)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["workload", "implementation", "threads"])


def aggregate_skew_level_speedup(df, selected_thread=None):
    # Prefer explicit skew_sensitivity experiments. If absent, fall back to strong rows,
    # but a meaningful skew-level plot requires more than one skew_percent value.
    if "skew_percent" not in df.columns:
        return pd.DataFrame(), None

    skew = df[df["experiment_type"] == "skew_sensitivity"].copy()
    if skew.empty:
        skew = df[df["experiment_type"] == "strong"].copy()

    if "speedup_vs_seq" not in skew.columns:
        return pd.DataFrame(), None

    skew = skew[(skew["workload"] == "skewed") & skew["speedup_vs_seq"].notna()].copy()

    if skew.empty:
        return pd.DataFrame(), None

    if selected_thread is None:
        selected_thread = int(skew["threads"].max())

    skew = skew[skew["threads"] == selected_thread].copy()

    if skew["skew_percent"].nunique() < 2:
        return pd.DataFrame(), selected_thread

    agg = skew.groupby(
        ["skew_percent", "implementation", "threads"],
        as_index=False
    ).agg(
        speedup_median=("speedup_vs_seq", "median"),
        speedup_mean=("speedup_vs_seq", "mean"),
        speedup_std=("speedup_vs_seq", "std"),
        time_impl_median=("time_impl", "median"),
        seeds=("seed", "nunique"),
        N=("N", "median"),
        P=("P", "median"),
    )

    return agg.sort_values(["implementation", "skew_percent"]), selected_thread


def build_phase_breakdown_df(df, workload, implementation):
    strong = df[
        (df["experiment_type"] == "strong")
        & (df["workload"] == workload)
        & (df["implementation"] == implementation)
    ].copy()

    if strong.empty:
        return pd.DataFrame()

    phase_map_seq = {
        "Histogram R": "time_histogram_R_seq",
        "Prefix R": "time_prefix_R_seq",
        "Scatter R": "time_scatter_R_seq",
        "Histogram S": "time_histogram_S_seq",
        "Prefix S": "time_prefix_S_seq",
        "Scatter S": "time_scatter_S_seq",
        "Join Local": "time_join_seq",
        "Accumulation": "time_accumulation_seq",
        "Total": "time_total_seq",
    }

    phase_map_impl = {
        "Histogram R": "time_histogram_R_impl",
        "Prefix R": "time_prefix_R_impl",
        "Scatter R": "time_scatter_R_impl",
        "Histogram S": "time_histogram_S_impl",
        "Prefix S": "time_prefix_S_impl",
        "Scatter S": "time_scatter_S_impl",
        "Join Local": "time_join_impl",
        "Accumulation": "time_accumulation_impl",
        "Total": "time_total_impl",
    }

    required = list(phase_map_seq.values()) + list(phase_map_impl.values())
    missing = [c for c in required if c not in strong.columns]
    if missing:
        print(f"Phase timing columns missing for {implementation}/{workload}: {missing}")
        return pd.DataFrame()

    rows = []

    # Sequential reference row.
    seq_row = {"label": "seq", "threads": 0}
    for name, col in phase_map_seq.items():
        seq_row[name] = strong[col].median()
    rows.append(seq_row)

    # Parallel/OpenMP rows.
    for t in sorted(strong["threads"].dropna().unique()):
        tmp = strong[strong["threads"] == t]
        row = {"label": str(int(t)), "threads": int(t)}
        for name, col in phase_map_impl.items():
            row[name] = tmp[col].median()
        rows.append(row)

    phase_df = pd.DataFrame(rows)

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

    # Convert seconds to milliseconds.
    for col in phase_names + ["Other Overhead", "Total"]:
        phase_df[col] = phase_df[col] * 1000.0

    return phase_df


# ============================================================
# PLOTTING HELPERS
# ============================================================

def setup_axis(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)


def get_thread_ticks(df):
    return sorted(int(t) for t in df["threads"].dropna().unique())


def apply_thread_ticks(ax, df):
    ticks = get_thread_ticks(df)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def savefig(path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Saved: {path}")


def plot_strong_by_workload(strong_agg, plots_dir, dpi, show_std):
    if strong_agg.empty:
        print("No strong scaling data found. Skipping strong speedup plots.")
        return

    workloads = sorted(strong_agg["workload"].dropna().unique())

    for workload in workloads:
        tmp = strong_agg[strong_agg["workload"] == workload].copy()
        if tmp.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        threads = get_thread_ticks(tmp)
        max_thread = max(threads)

        ax.plot(threads, threads, linestyle="--", label="Ideal speedup")

        for implementation in sorted(tmp["implementation"].unique()):
            g = tmp[tmp["implementation"] == implementation].sort_values("threads")
            yerr = g["speedup_std"].fillna(0) if show_std else None
            ax.errorbar(
                g["threads"],
                g["speedup_median"],
                yerr=yerr,
                marker="o",
                capsize=4 if show_std else 0,
                label=implementation,
            )

        setup_axis(
            ax,
            xlabel="Threads",
            ylabel="Speedup vs sequential",
            title=f"Strong Scaling Speedup — {workload.capitalize()} Workload",
        )
        apply_thread_ticks(ax, tmp)
        ax.set_xlim(left=0.5, right=max_thread + 1)
        ax.legend()

        savefig(plots_dir / f"strong_speedup_{workload}.png", dpi)


def plot_strong_combined(strong_agg, plots_dir, dpi, show_std):
    if strong_agg.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    threads = get_thread_ticks(strong_agg)
    ax.plot(threads, threads, linestyle="--", label="Ideal speedup")

    for (workload, implementation), g in strong_agg.groupby(["workload", "implementation"]):
        g = g.sort_values("threads")
        yerr = g["speedup_std"].fillna(0) if show_std else None
        ax.errorbar(
            g["threads"],
            g["speedup_median"],
            yerr=yerr,
            marker="o",
            capsize=4 if show_std else 0,
            label=f"{implementation} — {workload}",
        )

    setup_axis(
        ax,
        xlabel="Threads",
        ylabel="Speedup vs sequential",
        title="Strong Scaling Speedup — OpenMP for vs OpenMP task",
    )
    apply_thread_ticks(ax, strong_agg)
    ax.legend()

    savefig(plots_dir / "strong_speedup_combined.png", dpi)


def plot_workload_impact(strong_agg, plots_dir, dpi, show_std):
    if strong_agg.empty:
        return

    for implementation in sorted(strong_agg["implementation"].unique()):
        tmp = strong_agg[strong_agg["implementation"] == implementation].copy()
        if tmp["workload"].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        threads = get_thread_ticks(tmp)
        ax.plot(threads, threads, linestyle="--", label="Ideal speedup")

        for workload in sorted(tmp["workload"].unique()):
            g = tmp[tmp["workload"] == workload].sort_values("threads")
            yerr = g["speedup_std"].fillna(0) if show_std else None
            ax.errorbar(
                g["threads"],
                g["speedup_median"],
                yerr=yerr,
                marker="o",
                capsize=4 if show_std else 0,
                label=workload,
            )

        setup_axis(
            ax,
            xlabel="Threads",
            ylabel="Speedup vs sequential",
            title=f"Workload Impact on Speedup — {implementation}",
        )
        apply_thread_ticks(ax, tmp)
        ax.legend()

        savefig(plots_dir / f"workload_impact_speedup_{implementation}.png", dpi)


def plot_weak_efficiency(weak_agg, plots_dir, dpi, show_std):
    if weak_agg.empty:
        print("No weak scaling data found. Skipping weak efficiency plots.")
        return

    workloads = sorted(weak_agg["workload"].dropna().unique())

    for workload in workloads:
        tmp = weak_agg[weak_agg["workload"] == workload].copy()
        if tmp.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(1.0, linestyle="--", label="Ideal efficiency")

        for implementation in sorted(tmp["implementation"].unique()):
            g = tmp[tmp["implementation"] == implementation].sort_values("threads")

            if show_std:
                yerr = (
                    g["weak_efficiency"]
                    * (g["time_impl_std"].fillna(0) / g["time_impl_median"])
                ).fillna(0)
            else:
                yerr = None

            ax.errorbar(
                g["threads"],
                g["weak_efficiency"],
                yerr=yerr,
                marker="o",
                capsize=4 if show_std else 0,
                label=implementation,
            )

        setup_axis(
            ax,
            xlabel="Threads",
            ylabel="Weak scaling efficiency",
            title=f"Weak Scaling Efficiency — {workload.capitalize()} Workload",
        )
        apply_thread_ticks(ax, tmp)
        ax.legend()

        savefig(plots_dir / f"weak_efficiency_{workload}.png", dpi)

    # Combined version.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, linestyle="--", label="Ideal efficiency")

    for (workload, implementation), g in weak_agg.groupby(["workload", "implementation"]):
        g = g.sort_values("threads")
        ax.plot(
            g["threads"],
            g["weak_efficiency"],
            marker="o",
            label=f"{implementation} — {workload}",
        )

    setup_axis(
        ax,
        xlabel="Threads",
        ylabel="Weak scaling efficiency",
        title="Weak Scaling Efficiency — OpenMP for vs OpenMP task",
    )
    apply_thread_ticks(ax, weak_agg)
    ax.legend()

    savefig(plots_dir / "weak_efficiency_combined.png", dpi)


def plot_phase_breakdowns(df, plots_dir, tables_dir, dpi):
    workloads = sorted(df["workload"].dropna().unique())
    implementations = sorted(df["implementation"].dropna().unique())

    phase_names = [
        "Histogram R",
        "Prefix R",
        "Scatter R",
        "Histogram S",
        "Prefix S",
        "Scatter S",
        "Join Local",
        "Accumulation",
        "Other Overhead",
    ]

    colors = {
        "Histogram R": "#4E79A7",
        "Prefix R": "#A0CBE8",
        "Scatter R": "#2F5D8A",
        "Histogram S": "#59A14F",
        "Prefix S": "#8CD17D",
        "Scatter S": "#2E7D32",
        "Join Local": "#E15759",
        "Accumulation": "#F28E2B",
        "Other Overhead": "#9D9D9D",
    }

    any_plot = False

    for workload in workloads:
        for implementation in implementations:
            phase_df = build_phase_breakdown_df(df, workload, implementation)

            if phase_df.empty:
                continue

            any_plot = True

            table_path = tables_dir / f"phase_breakdown_{workload}_{implementation}.csv"
            phase_df.to_csv(table_path, index=False)
            print(f"Saved: {table_path}")

            fig, ax = plt.subplots(figsize=(11, 6))
            bottom = np.zeros(len(phase_df))

            for phase in phase_names:
                values = phase_df[phase].to_numpy()
                ax.bar(
                    phase_df["label"],
                    values,
                    bottom=bottom,
                    label=phase,
                    color=colors.get(phase),
                )
                bottom += values

            setup_axis(
                ax,
                xlabel="Series / Threads",
                ylabel="Time (ms)",
                title=f"Phase Breakdown — {implementation} — {workload.capitalize()} Workload",
            )
            ax.legend(loc="upper right", frameon=True, fontsize=8)
            savefig(plots_dir / f"phase_breakdown_{workload}_{implementation}.png", dpi)

    if not any_plot:
        print("No phase timing data found. Skipping phase breakdown plots.")


def plot_skew_level_speedup(skew_agg, selected_thread, plots_dir, dpi, show_std):
    if skew_agg.empty:
        print(
            "No skew-level speedup plot generated: the results contain fewer than two skew_percent values.\n"
            "To create this plot, run the orchestrator with the skew_sensitivity experiment, e.g.\n"
            "  python3 orchestrator_modulo_3.py --run skew_sensitivity --seeds 0 1 2 3 4 5 6 7 8 9 --append"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for implementation in sorted(skew_agg["implementation"].unique()):
        g = skew_agg[skew_agg["implementation"] == implementation].sort_values("skew_percent")
        yerr = g["speedup_std"].fillna(0) if show_std else None
        ax.errorbar(
            g["skew_percent"],
            g["speedup_median"],
            yerr=yerr,
            marker="o",
            capsize=4 if show_std else 0,
            label=implementation,
        )

    setup_axis(
        ax,
        xlabel="Skew percentage assigned to hot partitions (%)",
        ylabel="Speedup vs sequential",
        title=f"Speedup vs Skew Level — T={selected_thread}",
    )
    ax.set_xticks(sorted(skew_agg["skew_percent"].unique()))
    ax.legend()

    savefig(plots_dir / "skew_level_speedup.png", dpi)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    input_path = resolve_input_path(args.input)
    plots_dir = Path(args.plots_dir)
    tables_dir = Path(args.tables_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    df = load_results(input_path)

    check_correctness(df)

    # Strong scaling speedup tables and plots.
    strong_agg = aggregate_strong_speedup(df)
    if not strong_agg.empty:
        strong_path = tables_dir / "strong_speedup_summary.csv"
        strong_agg.to_csv(strong_path, index=False)
        print(f"Saved: {strong_path}")

    plot_strong_by_workload(strong_agg, plots_dir, args.dpi, args.show_std)
    plot_strong_combined(strong_agg, plots_dir, args.dpi, args.show_std)
    plot_workload_impact(strong_agg, plots_dir, args.dpi, args.show_std)

    # Weak scaling efficiency.
    weak_agg = aggregate_weak_efficiency(df)
    if not weak_agg.empty:
        weak_path = tables_dir / "weak_efficiency_summary.csv"
        weak_agg.to_csv(weak_path, index=False)
        print(f"Saved: {weak_path}")

    plot_weak_efficiency(weak_agg, plots_dir, args.dpi, args.show_std)

    # Phase breakdown stacked bars.
    plot_phase_breakdowns(df, plots_dir, tables_dir, args.dpi)

    # Optional skew-level speedup plot.
    skew_agg, selected_thread = aggregate_skew_level_speedup(df, args.skew_thread)
    if not skew_agg.empty:
        skew_path = tables_dir / "skew_level_speedup_summary.csv"
        skew_agg.to_csv(skew_path, index=False)
        print(f"Saved: {skew_path}")

    plot_skew_level_speedup(skew_agg, selected_thread, plots_dir, args.dpi, args.show_std)

    print("\nDone.")
    print(f"Plots directory:  {plots_dir}")
    print(f"Tables directory: {tables_dir}")


if __name__ == "__main__":
    main()