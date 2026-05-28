#!/usr/bin/env python3
"""
Plot NVDB-based route geometry, curvature, and curvature-aware speed profile.

Run:
    python3 plot_nvdb_route_quality.py

Output:
    fig_nvdb_xy.png
    fig_nvdb_curvature_speed.png
    fig_nvdb_heading_steering.png
    nvdb_route_metrics.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString


# ============================================================
# Route settings
# ============================================================

NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
X_CLIENT = "ntnu-masteroppgave-nmpc"
SRID = 5973

# Use ranges if NVDB accepts them. These correspond to the point order you gave.
ROUTE_SEGMENTS = [
    "3201 KV1548 K S1D1 m1-985",
    "3201 KV1531 K S1D1 m551-1541",
    "3201 KV1535 K S2D1 m1-619",
]

ROUTE_REVERSE = [
    True,   # desired direction: m985 -> m1
    True,   # desired direction: m1541 -> m551
    False,  # desired direction: m1 -> m619
]

SAMPLE_SPACING_M = 0.5
JOIN_TOLERANCE_M = 1.0
ROUTE_JOIN_MAX_DIST_M = 15.0

# Speed-profile parameters
V_NOMINAL = 7.0      # [m/s]
V_MIN = 2.0          # [m/s]
V_MAX = 10.0         # [m/s]
A_Y_MAX = 1.8        # [m/s^2]
KAPPA_EPS = 1e-4

# Vehicle parameter for simple feedforward steering estimate
WHEELBASE = 2.72     # [m], or use lf + lr if you want 1.23 + 1.49


# ============================================================
# NVDB helpers
# ============================================================

JsonDict = Dict[str, Any]


def fetch_nvdb_segments_for_reference(vegref: str) -> List[JsonDict]:
    endpoint = f"{NVDB_BASE_URL}/vegnett/api/v4/veglenkesekvenser/segmentert"

    headers = {
        "X-Client": X_CLIENT,
        "Accept": "application/json",
    }

    params: Dict[str, Any] = {
        "srid": SRID,
        "antall": 1000,
        "inkluderAntall": "false",
        "vegsystemreferanse": [vegref],
    }

    return paginate_nvdb(endpoint, headers=headers, params=params)


def paginate_nvdb(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
) -> List[JsonDict]:
    out: List[JsonDict] = []
    next_url: Optional[str] = url
    first = True

    for _ in range(200):
        if next_url is None:
            break

        if first:
            response = requests.get(
                next_url,
                headers=headers,
                params=params,
                timeout=(5.0, 30.0),
            )
            first = False
        else:
            response = requests.get(
                next_url,
                headers=headers,
                timeout=(5.0, 30.0),
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"NVDB request failed for {url}: "
                f"HTTP {response.status_code}, {response.text[:300]}"
            )

        data = response.json()

        if isinstance(data, dict):
            objects = data.get("objekter", [])
            if isinstance(objects, list):
                out.extend([obj for obj in objects if isinstance(obj, dict)])

            next_url = None
            metadata = data.get("metadata", {})
            if isinstance(metadata, dict):
                next_info = metadata.get("neste", {})
                if isinstance(next_info, dict):
                    href = next_info.get("href")
                    if isinstance(href, str) and href and href != response.url:
                        next_url = href

        elif isinstance(data, list):
            out.extend([obj for obj in data if isinstance(obj, dict)])
            next_url = None

        else:
            next_url = None

    print(f"NVDB: {len(out)} segments fetched.")
    return out


def extract_wkt(segment: JsonDict) -> str:
    geometry = segment.get("geometri")
    if isinstance(geometry, dict):
        wkt_str = geometry.get("wkt")
        return wkt_str if isinstance(wkt_str, str) else ""
    if isinstance(geometry, str):
        return geometry
    return ""


def wkt_to_xy(wkt_str: str) -> Optional[np.ndarray]:
    try:
        geometry = shapely_wkt.loads(wkt_str)
    except Exception:
        return None

    if not hasattr(geometry, "coords"):
        return None

    coords = np.asarray(list(geometry.coords), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return None

    return coords[:, :2].copy()


def segment_sort_key(segment: JsonDict) -> float:
    vsr = segment.get("vegsystemreferanse", {})
    if isinstance(vsr, dict):
        strekning = vsr.get("strekning", {})
        if isinstance(strekning, dict):
            fra_meter = strekning.get("fra_meter")
            til_meter = strekning.get("til_meter")
            if isinstance(fra_meter, (int, float)) and isinstance(til_meter, (int, float)):
                return float(min(fra_meter, til_meter))

    startposisjon = segment.get("startposisjon")
    if isinstance(startposisjon, (int, float)):
        return float(startposisjon)

    return 0.0


def remove_duplicate_neighbors(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if points.shape[0] < 2:
        return points

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.hstack(([True], distances > eps))
    return points[keep]


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    ds = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(ds)))


def parse_segments_to_polylines(
    segments: Sequence[JsonDict],
) -> List[Tuple[float, np.ndarray]]:
    polylines: List[Tuple[float, np.ndarray]] = []

    for segment in segments:
        wkt_str = extract_wkt(segment)
        if not wkt_str:
            continue

        points = wkt_to_xy(wkt_str)
        if points is None or points.shape[0] < 2:
            continue

        points = remove_duplicate_neighbors(points)
        if points.shape[0] < 2:
            continue

        polylines.append((segment_sort_key(segment), points))

    return sorted(polylines, key=lambda item: item[0])


def polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def are_polylines_overlapping(
    points_a: np.ndarray,
    points_b: np.ndarray,
    overlap_tol: float = 1.0,
    overlap_ratio_thresh: float = 0.7,
) -> bool:
    if points_a.shape[0] < 2 or points_b.shape[0] < 2:
        return False

    line_a = LineString(points_a)
    line_b = LineString(points_b)

    if line_a.length <= 1e-6 or line_b.length <= 1e-6:
        return False

    inter_len = line_a.buffer(overlap_tol, cap_style=2).intersection(line_b).length
    min_len = min(line_a.length, line_b.length)
    return inter_len / min_len >= overlap_ratio_thresh


def filter_overlapping_polylines(
    polylines: Sequence[Tuple[float, np.ndarray]],
) -> List[Tuple[float, np.ndarray]]:
    kept: List[Tuple[float, np.ndarray]] = []

    for sort_key, points in polylines:
        duplicate = False
        for _, kept_points in kept:
            if are_polylines_overlapping(points, kept_points):
                duplicate = True
                break

        if not duplicate:
            kept.append((sort_key, points))

    return kept


def sort_and_merge_polylines(
    polylines: Sequence[Tuple[float, np.ndarray]],
    join_tol: float,
) -> np.ndarray:
    """
    Simplified robust merge:
    Sort by meter position and connect nearest endpoint if needed.
    """
    polylines = filter_overlapping_polylines(polylines)

    if not polylines:
        return np.zeros((0, 2), dtype=float)

    merged = polylines[0][1].copy()

    for _, candidate in polylines[1:]:
        current_end = merged[-1]

        d_start = np.linalg.norm(candidate[0] - current_end)
        d_end = np.linalg.norm(candidate[-1] - current_end)

        if d_end < d_start:
            candidate = candidate[::-1].copy()
            join_dist = d_end
        else:
            join_dist = d_start

        if join_dist > join_tol:
            print(f"Warning: local segment join distance = {join_dist:.2f} m")

        if np.allclose(merged[-1], candidate[0], atol=1e-6):
            merged = np.vstack((merged, candidate[1:]))
        else:
            merged = np.vstack((merged, candidate))

    return remove_duplicate_neighbors(merged)


def build_polyline_for_reference(vegref: str) -> np.ndarray:
    segments = fetch_nvdb_segments_for_reference(vegref)
    polylines = parse_segments_to_polylines(segments)

    if not polylines:
        raise RuntimeError(f"No valid geometry found for {vegref}")

    merged = sort_and_merge_polylines(polylines, join_tol=JOIN_TOLERANCE_M)

    if merged.shape[0] < 2:
        raise RuntimeError(f"Too few points after merge for {vegref}")

    return merged


def stitch_route_parts(
    route_parts: Sequence[np.ndarray],
    route_reverse: Sequence[bool],
) -> np.ndarray:
    oriented_parts: List[np.ndarray] = []

    for points, reverse in zip(route_parts, route_reverse):
        oriented_parts.append(points[::-1].copy() if reverse else points.copy())

    full = oriented_parts[0]

    for i, part in enumerate(oriented_parts[1:], start=2):
        d_start = np.linalg.norm(part[0] - full[-1])
        d_end = np.linalg.norm(part[-1] - full[-1])

        if d_end < d_start:
            part = part[::-1].copy()
            join_dist = d_end
        else:
            join_dist = d_start

        print(f"Route join distance to part {i}: {join_dist:.2f} m")

        if join_dist > ROUTE_JOIN_MAX_DIST_M:
            print(
                f"Warning: route join distance is large ({join_dist:.2f} m). "
                "Check route order and reverse flags."
            )

        if np.allclose(full[-1], part[0], atol=1e-6):
            full = np.vstack((full, part[1:]))
        else:
            full = np.vstack((full, part))

    return remove_duplicate_neighbors(full)


# ============================================================
# Path quantities
# ============================================================

def build_spline_path(points_global: np.ndarray):
    points_global = remove_duplicate_neighbors(points_global)

    s_raw = cumulative_arc_length(points_global)

    if not np.all(np.diff(s_raw) > 0.0):
        raise RuntimeError("Arc length is not strictly increasing.")

    xs = CubicSpline(s_raw, points_global[:, 0])
    ys = CubicSpline(s_raw, points_global[:, 1])

    s_samples = np.arange(0.0, float(s_raw[-1]), SAMPLE_SPACING_M)
    if s_samples.size == 0 or s_samples[-1] < s_raw[-1]:
        s_samples = np.append(s_samples, float(s_raw[-1]))

    x_global = xs(s_samples)
    y_global = ys(s_samples)

    dx = xs(s_samples, 1)
    dy = ys(s_samples, 1)
    ddx = xs(s_samples, 2)
    ddy = ys(s_samples, 2)

    origin = np.array([x_global[0], y_global[0]])
    x_local = x_global - origin[0]
    y_local = y_global - origin[1]

    psi = np.unwrap(np.arctan2(dy, dx))

    denom = np.power(dx * dx + dy * dy, 1.5)
    kappa = (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9)

    return {
        "s": s_samples,
        "x": x_local,
        "y": y_local,
        "psi": psi,
        "kappa": kappa,
        "x_global": x_global,
        "y_global": y_global,
        "origin": origin,
        "s_raw": s_raw,
        "points_global": points_global,
    }


def curvature_aware_speed_profile(kappa: np.ndarray) -> np.ndarray:
    abs_kappa = np.maximum(np.abs(kappa), KAPPA_EPS)
    v_curve = np.sqrt(A_Y_MAX / abs_kappa)
    v_profile = np.minimum(v_curve, V_NOMINAL)
    return np.clip(v_profile, V_MIN, V_MAX)


def compute_metrics(path_data: Dict[str, np.ndarray], v_profile: np.ndarray) -> Dict[str, float]:
    s = path_data["s"]
    kappa = path_data["kappa"]

    return {
        "path_length_m": float(s[-1]),
        "mean_abs_kappa_1pm": float(np.mean(np.abs(kappa))),
        "p95_abs_kappa_1pm": float(np.percentile(np.abs(kappa), 95)),
        "max_abs_kappa_1pm": float(np.max(np.abs(kappa))),
        "min_speed_mps": float(np.min(v_profile)),
        "max_speed_mps": float(np.max(v_profile)),
        "mean_speed_mps": float(np.mean(v_profile)),
    }


# ============================================================
# Plotting
# ============================================================

def save_route_metrics(metrics: Dict[str, float], output_file: str) -> None:
    lines = ["metric,value"]
    for key, value in metrics.items():
        lines.append(f"{key},{value:.8g}")

    Path(output_file).write_text("\n".join(lines), encoding="utf-8")

def compute_route_part_boundaries(route_parts, route_reverse):
    """
    Returns approximate s-positions where each route part ends,
    after applying the same orientation logic as route stitching.
    """
    oriented_parts = []

    for points, reverse in zip(route_parts, route_reverse):
        part = points[::-1].copy() if reverse else points.copy()
        oriented_parts.append(part)

    boundaries = []
    labels = []

    full = oriented_parts[0]
    current_length = polyline_length(full)

    boundaries.append(current_length)
    labels.append("end part 1")

    for i, part in enumerate(oriented_parts[1:], start=2):
        d_start = np.linalg.norm(part[0] - full[-1])
        d_end = np.linalg.norm(part[-1] - full[-1])

        if d_end < d_start:
            part = part[::-1].copy()
            join_dist = d_end
        else:
            join_dist = d_start

        if np.allclose(full[-1], part[0], atol=1e-6):
            full = np.vstack((full, part[1:]))
        else:
            full = np.vstack((full, part))

        full = remove_duplicate_neighbors(full)
        current_length = polyline_length(full)

        boundaries.append(current_length)
        labels.append(f"end part {i}")

        print(
            f"Boundary after part {i}: s = {current_length:.2f} m, "
            f"join distance = {join_dist:.2f} m"
        )

    return boundaries[:-1], labels[:-1]


def plot_xy(path_data: Dict[str, np.ndarray], output_file: str) -> None:
    x = path_data["x"]
    y = path_data["y"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=2.0, label="Spline path")
    plt.scatter(x[0], y[0], marker="o", label="Start")
    plt.scatter(x[-1], y[-1], marker="x", label="End")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("NVDB-based route in local ENU frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_curvature_speed(
    path_data: Dict[str, np.ndarray],
    v_profile: np.ndarray,
    output_file: str,
    route_boundaries_s=None,
    route_boundary_labels=None,
) -> None:
    s = path_data["s"]
    kappa = path_data["kappa"]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    color_kappa = "tab:blue"
    color_speed = "tab:orange"

    line1 = ax1.plot(
        s,
        kappa,
        color=color_kappa,
        linewidth=1.5,
        label=r"$\kappa(s)$",
    )
    ax1.set_xlabel("Path progress $s$ [m]")
    ax1.set_ylabel(r"Curvature $\kappa$ [1/m]", color=color_kappa)
    ax1.tick_params(axis="y", labelcolor=color_kappa)
    ax1.grid(True)

    if route_boundaries_s is not None:
        for i, s_join in enumerate(route_boundaries_s):
            ax1.axvline(
                s_join,
                color="black",
                linewidth=1.0,
                alpha=0.7,
            )
            label = (
                route_boundary_labels[i]
                if route_boundary_labels is not None and i < len(route_boundary_labels)
                else f"join {i+1}"
            )
            ax1.text(
                s_join,
                ax1.get_ylim()[1],
                label,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
            )

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        s,
        v_profile,
        color=color_speed,
        linewidth=1.5,
        label=r"$v_x(s)$",
    )
    ax2.set_ylabel(r"Reference speed $v_x$ [m/s]", color=color_speed)
    ax2.tick_params(axis="y", labelcolor=color_speed)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Curvature and curvature-aware speed profile")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_heading_steering(
    path_data: Dict[str, np.ndarray],
    output_file: str,
    route_boundaries_s=None,
    route_boundary_labels=None,
) -> None:
    s = path_data["s"]
    psi = path_data["psi"]
    kappa = path_data["kappa"]

    delta_ff = np.arctan(WHEELBASE * kappa)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    color_heading = "tab:green"
    color_delta = "tab:red"

    line1 = ax1.plot(
        s,
        np.rad2deg(psi),
        color=color_heading,
        linewidth=1.5,
        label=r"$\psi(s)$",
    )
    ax1.set_xlabel("Path progress $s$ [m]")
    ax1.set_ylabel("Heading [deg]", color=color_heading)
    ax1.tick_params(axis="y", labelcolor=color_heading)
    ax1.grid(True)

    if route_boundaries_s is not None:
        for i, s_join in enumerate(route_boundaries_s):
            ax1.axvline(
                s_join,
                color="black",
                linewidth=1.0,
                alpha=0.7,
            )
            label = (
                route_boundary_labels[i]
                if route_boundary_labels is not None and i < len(route_boundary_labels)
                else f"join {i+1}"
            )
            ax1.text(
                s_join,
                ax1.get_ylim()[1],
                label,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
            )

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        s,
        np.rad2deg(delta_ff),
        color=color_delta,
        linewidth=1.5,
        label=r"$\delta_\mathrm{ff}(s)$",
    )
    ax2.set_ylabel("Feedforward steering estimate [deg]", color=color_delta)
    ax2.tick_params(axis="y", labelcolor=color_delta)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Path heading and curvature-based steering estimate")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_curvature_histogram(
    path_data: Dict[str, np.ndarray],
    output_file: str,
) -> None:
    kappa_abs = np.abs(path_data["kappa"])

    plt.figure(figsize=(7, 4))
    plt.hist(kappa_abs, bins=40)
    plt.grid(True)
    plt.xlabel(r"$|\kappa|$ [1/m]")
    plt.ylabel("Number of samples")
    plt.title("Distribution of absolute path curvature")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def print_spike_diagnostics(path_data, route_boundaries_s, threshold=0.08):
    s = path_data["s"]
    kappa = path_data["kappa"]

    spike_idx = np.where(np.abs(kappa) > threshold)[0]

    if len(spike_idx) == 0:
        print(f"\nNo curvature spikes above {threshold:.3f} 1/m.")
        return

    print(f"\nCurvature spikes above {threshold:.3f} 1/m:")

    for idx in spike_idx:
        s_spike = s[idx]
        k_spike = kappa[idx]

        if route_boundaries_s is not None and len(route_boundaries_s) > 0:
            distances = [abs(s_spike - b) for b in route_boundaries_s]
            nearest_idx = int(np.argmin(distances))
            nearest_dist = distances[nearest_idx]
            nearest_boundary = route_boundaries_s[nearest_idx]

            print(
                f"  s = {s_spike:.2f} m, "
                f"kappa = {k_spike:.4f} 1/m, "
                f"nearest route join = {nearest_boundary:.2f} m "
                f"(distance {nearest_dist:.2f} m)"
            )
        else:
            print(
                f"  s = {s_spike:.2f} m, "
                f"kappa = {k_spike:.4f} 1/m"
            )    


# ============================================================
# Main
# ============================================================

def main() -> None:
    route_parts = []

    for vegref in ROUTE_SEGMENTS:
        print(f"\nFetching: {vegref}")
        part = build_polyline_for_reference(vegref)
        print(f"  points: {part.shape[0]}, length: {polyline_length(part):.1f} m")
        route_parts.append(part)

    route_boundaries_s, route_boundary_labels = compute_route_part_boundaries(
        route_parts,
        ROUTE_REVERSE,
    )

    route_global = stitch_route_parts(route_parts, ROUTE_REVERSE)

    print(f"\nFull route points: {route_global.shape[0]}")
    print(f"Approx. raw route length: {polyline_length(route_global):.1f} m")

    path_data = build_spline_path(route_global)
    v_profile = curvature_aware_speed_profile(path_data["kappa"])

    metrics = compute_metrics(path_data, v_profile)

    print("\nRoute metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4g}")

    save_route_metrics(metrics, "nvdb_route_metrics.csv")

    plot_xy(path_data, "fig_nvdb_xy.png")

    plot_curvature_speed(
        path_data,
        v_profile,
        "fig_nvdb_curvature_speed.png",
        route_boundaries_s=route_boundaries_s,
        route_boundary_labels=route_boundary_labels,
    )

    plot_heading_steering(
        path_data,
        "fig_nvdb_heading_steering.png",
        route_boundaries_s=route_boundaries_s,
        route_boundary_labels=route_boundary_labels,
    )

    plot_curvature_histogram(
        path_data,
        "fig_nvdb_curvature_histogram.png",
    )

    print_spike_diagnostics(
        path_data,
        route_boundaries_s,
        threshold=0.08,
    )

    print("\nSaved:")
    print("  fig_nvdb_xy.png")
    print("  fig_nvdb_curvature_speed.png")
    print("  fig_nvdb_heading_steering.png")
    print("  fig_nvdb_curvature_histogram.png")
    print("  nvdb_route_metrics.csv")

if __name__ == "__main__":
    main()