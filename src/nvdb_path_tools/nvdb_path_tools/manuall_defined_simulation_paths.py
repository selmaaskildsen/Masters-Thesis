#!/usr/bin/env python3
"""
Compute and plot manually defined simulation paths for Table 4.2.

Outputs:
    fig_manual_reference_paths.png
    fig_manual_path_curvature_speed.png
    manual_path_metrics.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pathlib import Path


# ============================================================
# Path definitions
# ============================================================

def build_mild_waypoints() -> np.ndarray:
    """
    Smooth mild path used as the nominal baseline reference path.
    """
    x = np.linspace(0.0, 90.0, 60)
    y = 4.5 * (1.0 - np.cos(np.pi * x / 90.0))

    return np.column_stack([x, y])


def build_sharp_waypoints() -> np.ndarray:
    """
    Sharper path used for the curvature stress scenario.
    """
    x = np.array([0, 10, 20, 30, 40, 50, 55, 60, 65, 75, 90, 110], dtype=float)
    y = np.array([0,  0,  0,  2,  8, 18, 28, 35, 38, 36, 30, 25], dtype=float)

    return np.column_stack([x, y])


# ============================================================
# Parameters
# ============================================================

SAMPLE_SPACING_M = 0.1

V_NOMINAL = 7.0       # [m/s]
V_MIN = 2.0           # [m/s]
V_MAX = 10.0          # [m/s]
A_Y_MAX = 1.8         # [m/s^2]
KAPPA_EPS = 1e-4


# ============================================================
# Utility functions
# ============================================================

def remove_duplicate_neighbors(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if points.shape[0] < 2:
        return points

    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.hstack(([True], ds > eps))
    return points[keep]


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(ds)))


def build_spline_path(waypoints: np.ndarray, sample_spacing: float = SAMPLE_SPACING_M):
    waypoints = remove_duplicate_neighbors(np.asarray(waypoints, dtype=float))

    s_raw = cumulative_arc_length(waypoints)

    if len(s_raw) < 3:
        raise ValueError("Need at least three unique waypoints for cubic spline.")

    if not np.all(np.diff(s_raw) > 0):
        raise ValueError("Arc-length parameter is not strictly increasing.")

    xs = CubicSpline(s_raw, waypoints[:, 0])
    ys = CubicSpline(s_raw, waypoints[:, 1])

    s = np.arange(0.0, float(s_raw[-1]), sample_spacing)
    if s.size == 0 or s[-1] < s_raw[-1]:
        s = np.append(s, float(s_raw[-1]))

    x = xs(s)
    y = ys(s)

    dx = xs(s, 1)
    dy = ys(s, 1)
    ddx = xs(s, 2)
    ddy = ys(s, 2)

    psi = np.unwrap(np.arctan2(dy, dx))

    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-9)
    kappa = (dx * ddy - dy * ddx) / denom

    return {
        "s": s,
        "x": x,
        "y": y,
        "psi": psi,
        "kappa": kappa,
        "waypoints": waypoints,
        "length": float(s[-1]),
    }


def curvature_aware_speed_profile(kappa: np.ndarray) -> np.ndarray:
    abs_kappa = np.maximum(np.abs(kappa), KAPPA_EPS)
    v_curve = np.sqrt(A_Y_MAX / abs_kappa)
    v_profile = np.minimum(v_curve, V_NOMINAL)
    return np.clip(v_profile, V_MIN, V_MAX)


def compute_metrics(path_data, v_profile):
    s = path_data["s"]
    kappa = path_data["kappa"]

    return {
        "length_m": float(s[-1]),
        "mean_abs_kappa_1pm": float(np.mean(np.abs(kappa))),
        "p95_abs_kappa_1pm": float(np.percentile(np.abs(kappa), 95)),
        "max_abs_kappa_1pm": float(np.max(np.abs(kappa))),
        "min_speed_mps": float(np.min(v_profile)),
        "mean_speed_mps": float(np.mean(v_profile)),
        "max_speed_mps": float(np.max(v_profile)),
    }


# ============================================================
# Plotting
# ============================================================

def plot_single_reference_path(path_data, output_file, title, label):
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        path_data["x"],
        path_data["y"],
        linewidth=2.0,
        label=label,
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_manual_paths_separately(mild_data, sharp_data):
    plot_single_reference_path(
        mild_data,
        "fig_mild_reference_path.png",
        "Mild manually defined reference path",
        "Mild path",
    )

    plot_single_reference_path(
        sharp_data,
        "fig_sharp_reference_path.png",
        "Sharp manually defined reference path",
        "Sharp path",
    )


def plot_single_curvature_and_speed(path_data, speed_profile, output_file, title):
    s = path_data["s"]
    kappa = path_data["kappa"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color_kappa = "tab:blue"
    color_speed = "tab:orange"

    line1 = ax1.plot(
        s,
        kappa,
        color=color_kappa,
        linewidth=1.7,
        label=r"$\kappa(s)$",
    )
    ax1.set_xlabel("Path progress $s$ [m]")
    ax1.set_ylabel(r"Curvature $\kappa$ [1/m]", color=color_kappa)
    ax1.tick_params(axis="y", labelcolor=color_kappa)
    ax1.grid(True)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        s,
        speed_profile,
        color=color_speed,
        linewidth=1.7,
        label=r"$v_x(s)$",
    )
    ax2.set_ylabel(r"Reference speed $v_x$ [m/s]", color=color_speed)
    ax2.tick_params(axis="y", labelcolor=color_speed)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def save_metrics_csv(metrics_dict, output_file):
    lines = [
        "path,length_m,mean_abs_kappa_1pm,p95_abs_kappa_1pm,"
        "max_abs_kappa_1pm,min_speed_mps,mean_speed_mps,max_speed_mps"
    ]

    for name, metrics in metrics_dict.items():
        lines.append(
            f"{name},"
            f"{metrics['length_m']:.6g},"
            f"{metrics['mean_abs_kappa_1pm']:.6g},"
            f"{metrics['p95_abs_kappa_1pm']:.6g},"
            f"{metrics['max_abs_kappa_1pm']:.6g},"
            f"{metrics['min_speed_mps']:.6g},"
            f"{metrics['mean_speed_mps']:.6g},"
            f"{metrics['max_speed_mps']:.6g}"
        )

    Path(output_file).write_text("\n".join(lines), encoding="utf-8")


def print_latex_table_values(metrics_dict):
    print("\nValues for Table 4.2:")
    print("-" * 90)
    print(
        f"{'Path':<12} "
        f"{'Length [m]':>12} "
        f"{'Mean |k|':>12} "
        f"{'P95 |k|':>12} "
        f"{'Max |k|':>12} "
        f"{'Min speed':>12} "
        f"{'Mean speed':>12}"
    )
    print("-" * 90)

    for name, metrics in metrics_dict.items():
        print(
            f"{name:<12} "
            f"{metrics['length_m']:>12.2f} "
            f"{metrics['mean_abs_kappa_1pm']:>12.4f} "
            f"{metrics['p95_abs_kappa_1pm']:>12.4f} "
            f"{metrics['max_abs_kappa_1pm']:>12.4f} "
            f"{metrics['min_speed_mps']:>12.2f} "
            f"{metrics['mean_speed_mps']:>12.2f}"
        )

    print("-" * 90)

    print("\nLaTeX table rows:")
    print("-" * 90)

    mild = metrics_dict["Mild path"]
    sharp = metrics_dict["Sharp path"]

    print(
        "Mild path & Smooth cosine-shaped waypoint path & "
        f"{mild['length_m']:.0f} & "
        f"{mild['max_abs_kappa_1pm']:.3f} & "
        f"{mild['min_speed_mps']:.2f} & "
        "Baseline tracking and nominal comparison \\\\"
    )

    print(
        "Sharp path & High-curvature manually defined waypoint path & "
        f"{sharp['length_m']:.0f} & "
        f"{sharp['max_abs_kappa_1pm']:.3f} & "
        f"{sharp['min_speed_mps']:.2f} & "
        "Curvature, speed, mismatch, and sensitivity studies \\\\"
    )


# ============================================================
# Main
# ============================================================

def main():
    mild_waypoints = build_mild_waypoints()
    sharp_waypoints = build_sharp_waypoints()

    mild_data = build_spline_path(mild_waypoints)
    sharp_data = build_spline_path(sharp_waypoints)

    mild_speed = curvature_aware_speed_profile(mild_data["kappa"])
    sharp_speed = curvature_aware_speed_profile(sharp_data["kappa"])

    mild_metrics = compute_metrics(mild_data, mild_speed)
    sharp_metrics = compute_metrics(sharp_data, sharp_speed)

    metrics_dict = {
        "Mild path": mild_metrics,
        "Sharp path": sharp_metrics,
    }

    plot_manual_paths_separately(
        mild_data,
        sharp_data,
    )

    plot_single_curvature_and_speed(
        mild_data,
        mild_speed,
        "fig_mild_curvature_speed.png",
        "Curvature and speed profile for mild path",
    )

    plot_single_curvature_and_speed(
        sharp_data,
        sharp_speed,
        "fig_sharp_curvature_speed.png",
        "Curvature and speed profile for sharp path",
    )

    save_metrics_csv(
        metrics_dict,
        "manual_path_metrics.csv",
    )

    print_latex_table_values(metrics_dict)

    print("\nSaved files:")
    print("  fig_mild_reference_path.png")
    print("  fig_sharp_reference_path.png")
    print("  fig_manual_path_curvature_speed.png")
    print("  manual_path_metrics.csv")


if __name__ == "__main__":
    main()