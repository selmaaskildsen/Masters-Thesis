"""
nmpc_solver_performance_summary.py

Aggregate solver-performance results from the previous NMPC simulations.

The script searches for saved .npz log files in the result folders from:
- NMPC reference case
- Speed sensitivity
- Tire-model mismatch
- Parameter sensitivity

It creates:
    results_nmpc_solver_performance/
        solver_time_representative_timeseries.png
        solver_time_distribution.png
        solver_performance_summary.png
        solver_performance_metrics.csv
        solver_performance_latex_rows.txt

Run from the folder that contains the result directories, e.g.:
    python nmpc_solver_performance_summary.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTROL_PERIOD_MS = 100.0

OUT_DIR = Path("results_nmpc_solver_performance")

NPZ_FILES = [
    Path("results_nmpc_reference_case/nmpc_reference_logs.npz"),
    Path("results_nmpc_speed_sensitivity/speed_sensitivity_logs.npz"),
    Path("results_nmpc_tire_mismatch/tire_mismatch_logs.npz"),
    Path("results_nmpc_parameter_sensitivity/parameter_sensitivity_logs.npz"),
]


def clean_label(prefix: str) -> str:
    label = prefix

    replacements = {
        "nmpc_dugoff_": "",
        "_log": "",
        "_ctrl_": "/",
        "_plant": "",
        "linear": "Linear",
        "dugoff": "Dugoff",
        "pacejka": "Pacejka",
        "speed_2ms": "Speed 2 m/s",
        "speed_3ms": "Speed 3 m/s",
        "speed_5ms": "Speed 5 m/s",
        "speed_7ms": "Speed 7 m/s",
        "speed_10ms": "Speed 10 m/s",
        "reference": "Reference",
        "nominal": "Nominal",
        "reduced_friction": "Reduced friction",
        "low_friction": "Low friction",
        "reduced_stiffness": "Reduced stiffness",
        "low_stiffness": "Low stiffness",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    label = label.replace("__", "_")
    label = label.replace("_", " ")
    label = " ".join(label.split())
    return label.strip()


def scenario_group_from_file(npz_path: Path) -> str:
    parent = npz_path.parent.name

    if "reference" in parent:
        return "NMPC reference"
    if "speed" in parent:
        return "Speed sensitivity"
    if "tire_mismatch" in parent:
        return "Tire-model mismatch"
    if "parameter" in parent:
        return "Parameter sensitivity"

    return parent


def extract_cases_from_npz(npz_path: Path) -> list[dict]:
    if not npz_path.exists():
        print(f"[WARN] Missing file: {npz_path}")
        return []

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    solve_keys = [k for k in keys if k.endswith("solve_time_log")]

    cases = []

    for solve_key in solve_keys:
        prefix = solve_key[: -len("_solve_time_log")]

        solve_time = np.asarray(data[solve_key], dtype=float)
        solve_time = solve_time[np.isfinite(solve_time)]

        if solve_time.size == 0:
            continue

        fail_key = prefix + "_hard_fail_log"
        iter_key = prefix + "_iter_log"
        soft_key = prefix + "_soft_fail_log"
        t_key = prefix + "_t_log"
        s_key = prefix + "_s_log"

        hard_failures = 0
        soft_warnings = 0
        max_iter = np.nan
        mean_iter = np.nan

        if fail_key in data:
            hard_failures = int(np.nansum(np.asarray(data[fail_key], dtype=float)))

        if soft_key in data:
            soft_warnings = int(np.nansum(np.asarray(data[soft_key], dtype=float)))

        if iter_key in data:
            iter_log = np.asarray(data[iter_key], dtype=float)
            iter_log = iter_log[np.isfinite(iter_log)]
            if iter_log.size > 0:
                max_iter = float(np.nanmax(iter_log))
                mean_iter = float(np.nanmean(iter_log))

        if t_key in data:
            x_axis = np.asarray(data[t_key], dtype=float)
            x_name = "time_s"
        elif s_key in data:
            x_axis = np.asarray(data[s_key], dtype=float)
            x_name = "path_progress_m"
        else:
            x_axis = np.arange(len(solve_time), dtype=float)
            x_name = "step"

        n = min(len(x_axis), len(solve_time))
        x_axis = x_axis[:n]
        solve_time = solve_time[:n]

        case = {
            "group": scenario_group_from_file(npz_path),
            "prefix": prefix,
            "label": clean_label(prefix),
            "mean_solve_ms": float(np.nanmean(solve_time)),
            "median_solve_ms": float(np.nanmedian(solve_time)),
            "p95_solve_ms": float(np.nanpercentile(solve_time, 95)),
            "max_solve_ms": float(np.nanmax(solve_time)),
            "std_solve_ms": float(np.nanstd(solve_time)),
            "max_ipopt_iter": max_iter,
            "mean_ipopt_iter": mean_iter,
            "hard_failures": hard_failures,
            "soft_warnings": soft_warnings,
            "num_samples": int(solve_time.size),
            "real_time": "Yes" if (float(np.nanpercentile(solve_time, 95)) < CONTROL_PERIOD_MS and hard_failures == 0) else "No/limited",
            "x_axis": x_axis,
            "x_name": x_name,
            "solve_time_log": solve_time,
        }

        cases.append(case)

    return cases


def aggregate_by_group(cases: list[dict]) -> list[dict]:
    groups = []
    for group_name in ["NMPC reference", "Speed sensitivity", "Tire-model mismatch", "Parameter sensitivity"]:
        group_cases = [c for c in cases if c["group"] == group_name]
        if not group_cases:
            continue

        all_solve = np.concatenate([c["solve_time_log"] for c in group_cases])
        fails = int(sum(c["hard_failures"] for c in group_cases))
        soft = int(sum(c["soft_warnings"] for c in group_cases))
        max_iter = np.nanmax([c["max_ipopt_iter"] for c in group_cases])

        groups.append(
            {
                "scenario": group_name,
                "mean_solve_ms": float(np.nanmean(all_solve)),
                "median_solve_ms": float(np.nanmedian(all_solve)),
                "p95_solve_ms": float(np.nanpercentile(all_solve, 95)),
                "max_solve_ms": float(np.nanmax(all_solve)),
                "max_ipopt_iter": float(max_iter),
                "hard_failures": fails,
                "soft_warnings": soft,
                "num_samples": int(all_solve.size),
                "real_time": "Yes" if (float(np.nanpercentile(all_solve, 95)) < CONTROL_PERIOD_MS and fails == 0) else "No/limited",
            }
        )

    return groups


def write_tables(group_rows: list[dict], case_rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "solver_performance_metrics.csv"

    fieldnames = [
        "scenario",
        "mean_solve_ms",
        "median_solve_ms",
        "p95_solve_ms",
        "max_solve_ms",
        "max_ipopt_iter",
        "hard_failures",
        "soft_warnings",
        "num_samples",
        "real_time",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(group_rows)

    lines = []
    for row in group_rows:
        lines.append(
            f"{row['scenario']} & "
            f"{row['mean_solve_ms']:.2f} & "
            f"{row['median_solve_ms']:.2f} & "
            f"{row['p95_solve_ms']:.2f} & "
            f"{row['max_solve_ms']:.2f} & "
            f"{int(row['hard_failures'])} & "
            f"{row['real_time']} \\\\"
        )

    (out_dir / "solver_performance_latex_rows.txt").write_text("\n".join(lines) + "\n")

    # Detailed per-case CSV can be useful for checking outliers.
    detailed_path = out_dir / "solver_performance_cases.csv"
    detailed_fieldnames = [
        "group",
        "label",
        "mean_solve_ms",
        "median_solve_ms",
        "p95_solve_ms",
        "max_solve_ms",
        "max_ipopt_iter",
        "hard_failures",
        "soft_warnings",
        "num_samples",
        "real_time",
    ]

    with detailed_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({k: row[k] for k in detailed_fieldnames})


def make_plots(cases: list[dict], group_rows: list[dict], out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Representative time-series:
    # Prefer one nominal/reference case and the two worst speed cases if available.
    preferred_tokens = [
        "reference",
        "speed_2ms",
        "speed_3ms",
        "speed_7ms",
        "low_friction",
    ]

    representative = []
    used = set()

    for token in preferred_tokens:
        for c in cases:
            if token in c["prefix"] and c["prefix"] not in used:
                representative.append(c)
                used.add(c["prefix"])
                break

    if len(representative) == 0:
        representative = cases[: min(5, len(cases))]

    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    for c in representative:
        x = c["x_axis"]
        y = c["solve_time_log"]
        n = min(len(x), len(y))
        ax.plot(x[:n], y[:n], linewidth=1.6, label=f"{c['group']}: {c['label']}")

    ax.axhline(CONTROL_PERIOD_MS, linestyle="--", linewidth=1.6, label="Control period")
    ax.set_xlabel("Time [s] / simulation coordinate")
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("NMPC solve time for representative simulation scenarios")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "solver_time_representative_timeseries.png", dpi=300)

    # Distribution boxplot for scenario groups.
    group_names = [row["scenario"] for row in group_rows]
    grouped_data = []
    for name in group_names:
        group_cases = [c for c in cases if c["group"] == name]
        grouped_data.append(np.concatenate([c["solve_time_log"] for c in group_cases]))

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.boxplot(grouped_data, labels=group_names, showfliers=True)
    ax.axhline(CONTROL_PERIOD_MS, linestyle="--", linewidth=1.6, label="Control period")
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("Distribution of NMPC solve times across scenario groups")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "solver_time_distribution.png", dpi=300)

    # Summary metrics.
    x = np.arange(len(group_rows))
    labels = [row["scenario"] for row in group_rows]

    mean_solve = np.array([row["mean_solve_ms"] for row in group_rows])
    p95_solve = np.array([row["p95_solve_ms"] for row in group_rows])
    max_solve = np.array([row["max_solve_ms"] for row in group_rows])
    failures = np.array([row["hard_failures"] for row in group_rows])

    fig, axs = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)

    axs[0].plot(x, mean_solve, marker="o", linewidth=1.8, label="Mean solve time")
    axs[0].plot(x, p95_solve, marker="o", linewidth=1.8, label="95th percentile")
    axs[0].plot(x, max_solve, marker="o", linewidth=1.8, label="Max solve time")
    axs[0].axhline(CONTROL_PERIOD_MS, linestyle="--", linewidth=1.6, label="Control period")
    axs[0].set_ylabel("Solve time [ms]")

    axs[1].bar(x, failures, width=0.55, label="Hard failures")
    axs[1].set_ylabel("Failures [-]")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=20, ha="right")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("NMPC solver-performance summary")
    fig.tight_layout()
    fig.savefig(out_dir / "solver_performance_summary.png", dpi=300)

    plt.close("all")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    for npz_path in NPZ_FILES:
        cases.extend(extract_cases_from_npz(npz_path))

    if not cases:
        raise RuntimeError(
            "No solver logs found. Run the NMPC result scripts first, then run this aggregator."
        )

    group_rows = aggregate_by_group(cases)

    write_tables(group_rows, cases, OUT_DIR)
    make_plots(cases, group_rows, OUT_DIR)

    print("\n=== Solver performance aggregation completed ===")
    print(f"Output directory: {OUT_DIR}")
    print("\nScenario-level LaTeX rows:")
    print((OUT_DIR / "solver_performance_latex_rows.txt").read_text().strip())

    print("\nDetailed worst cases by max solve time:")
    worst = sorted(cases, key=lambda c: c["max_solve_ms"], reverse=True)[:8]
    for c in worst:
        print(
            f"{c['group']:24s} | {c['label']:30s} | "
            f"mean={c['mean_solve_ms']:.2f} ms | "
            f"p95={c['p95_solve_ms']:.2f} ms | "
            f"max={c['max_solve_ms']:.2f} ms | "
            f"fails={c['hard_failures']}"
        )


if __name__ == "__main__":
    main()
