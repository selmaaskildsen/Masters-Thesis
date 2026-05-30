"""
nmpc_tire_mismatch_result.py

Tire-model mismatch simulation for the results chapter:
- Uses the same sharp path and constant speed for all cases
- Compares matched and mismatched controller/plant tire models
- Pacejka is used only in the plant
- Saves comparison plots, logs, CSV metrics, and LaTeX table rows

Run:
    python nmpc_tire_mismatch_result.py

Outputs:
    results_nmpc_tire_mismatch/
        tire_mismatch_trajectories.png
        tire_mismatch_errors_steering_vs_s.png
        tire_mismatch_tire_response_vs_s.png
        tire_mismatch_solver_summary.png
        tire_mismatch_metrics.csv
        tire_mismatch_latex_rows.txt
        tire_mismatch_logs.npz
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nmpc_reference_case_result import (
    VehicleParams,
    ControllerParams,
    ModelParams,
    SpeedProfileParams,
    SimulationParams,
    SplinePath,
    NMPCController,
    create_sharp_waypoints,
    preview_path_data,
    simulate_step,
    compute_metrics,
    smooth_signal,
)


# Same path, speed, vehicle parameters and controller tuning for all cases.
VX_REF = 7.0
PLOT_CASES = {
    "Linear/Linear",
    "Dugoff/Dugoff",
    "Linear/Pacejka",
    "Dugoff/Pacejka",
}
CASE_COLORS = {
    "Linear/Linear": "C0",
    "Dugoff/Dugoff": "C1",
    "Linear/Pacejka": "C2",
    "Dugoff/Pacejka": "C3",
}


def make_constant_speed_profile(path: SplinePath, vx: float) -> np.ndarray:
    return np.full_like(path.s, float(vx), dtype=float)


def safe_tan_alpha(alpha: float, limit: float = 20.0) -> float:
    return float(np.clip(np.tan(alpha), -limit, limit))


def tire_response_numeric(
    alpha_f: float,
    alpha_r: float,
    delta: float,
    vx: float,
    veh: VehicleParams,
    model: ModelParams,
    ax: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Compute approximate plant tire forces for logging.

    Returns:
        Fyf, Fyr, Fxf, Fxr

    This is used for plotting/interpretation only. The actual plant simulation
    is still performed by simulate_step from the NMPC reference implementation.
    """
    g = veh.g
    m = veh.m
    L = veh.L

    Fzf = m * g * veh.lr / L
    Fzr = m * g * veh.lf / L

    Fx_total = m * ax
    if Fx_total >= 0.0:
        front_frac = 0.6
    else:
        front_frac = 0.7

    Fxf = front_frac * Fx_total
    Fxr = (1.0 - front_frac) * Fx_total

    model_name = model.tire_model.lower()

    if model_name == "linear":
        Fyf = veh.Cf * alpha_f
        Fyr = veh.Cr * alpha_r

    elif model_name == "dugoff":
        def dugoff_force(alpha: float, C_alpha: float, Fz: float, Fx: float) -> float:
            tan_alpha = safe_tan_alpha(alpha)
            Fy_lin = C_alpha * tan_alpha
            denom = 2.0 * np.sqrt(Fx**2 + Fy_lin**2) + 1e-9
            lam = veh.mu * Fz / denom

            if lam < 1.0:
                f_lam = lam * (2.0 - lam)
            else:
                f_lam = 1.0

            return float(Fy_lin * f_lam)

        Fyf = dugoff_force(alpha_f, veh.Cf, Fzf, Fxf)
        Fyr = dugoff_force(alpha_r, veh.Cr, Fzr, Fxr)

    elif model_name == "pacejka":
        # Same style as the plant implementation used in the reference code:
        # use D = mu Fz and choose B so the initial slope matches C_alpha.
        C_shape = 1.30
        E_shape = -1.60

        def pacejka_force(alpha: float, C_alpha: float, Fz: float) -> float:
            D = veh.mu * Fz
            alpha_deg = np.rad2deg(alpha)
            B = C_alpha / (C_shape * D * (180.0 / np.pi) + 1e-9)
            return float(D * np.sin(C_shape * np.arctan(B * alpha_deg - E_shape * (B * alpha_deg - np.arctan(B * alpha_deg)))))

        Fyf = pacejka_force(alpha_f, veh.Cf, Fzf)
        Fyr = pacejka_force(alpha_r, veh.Cr, Fzr)

    else:
        raise ValueError(f"Unsupported tire model for logging: {model.tire_model}")

    return Fyf, Fyr, Fxf, Fxr


def compute_slip_and_force_logs(
    ey_log: np.ndarray,
    epsi_log: np.ndarray,
    vy_log: np.ndarray,
    r_log: np.ndarray,
    delta_log: np.ndarray,
    vx_log: np.ndarray,
    ax_log: np.ndarray,
    veh_plant: VehicleParams,
    model_plant: ModelParams,
) -> dict:
    alpha_f_log = []
    alpha_r_log = []
    Fyf_log = []
    Fyr_log = []
    Fxf_log = []
    Fxr_log = []

    for vy, r, delta, vx, ax in zip(vy_log, r_log, delta_log, vx_log, ax_log):
        vx_safe = max(abs(float(vx)), 0.1)

        alpha_f = float(delta - np.arctan2(vy + veh_plant.lf * r, vx_safe))
        alpha_r = float(-np.arctan2(vy - veh_plant.lr * r, vx_safe))

        Fyf, Fyr, Fxf, Fxr = tire_response_numeric(
            alpha_f=alpha_f,
            alpha_r=alpha_r,
            delta=delta,
            vx=vx_safe,
            veh=veh_plant,
            model=model_plant,
            ax=float(ax),
        )

        alpha_f_log.append(alpha_f)
        alpha_r_log.append(alpha_r)
        Fyf_log.append(Fyf)
        Fyr_log.append(Fyr)
        Fxf_log.append(Fxf)
        Fxr_log.append(Fxr)

    return {
        "alpha_f_log": np.asarray(alpha_f_log),
        "alpha_r_log": np.asarray(alpha_r_log),
        "Fyf_log": np.asarray(Fyf_log),
        "Fyr_log": np.asarray(Fyr_log),
        "Fxf_log": np.asarray(Fxf_log),
        "Fxr_log": np.asarray(Fxr_log),
    }


def run_mismatch_case(
    tire_model_ctrl: str,
    tire_model_plant: str,
    case_label: str,
) -> tuple[dict, dict]:
    stiffness_scale_ctrl = 1.00
    stiffness_scale_plant = 1.00
    mu_ctrl = 0.80
    mu_plant = 0.80

    ctrl_par = ControllerParams()
    sim_par = SimulationParams()
    speed_par = SpeedProfileParams(
        vx_nominal=VX_REF,
        vx_min=VX_REF,
        vx_max=VX_REF,
        ay_max=1.5,
    )

    model_ctrl = ModelParams(tire_model=tire_model_ctrl)
    model_plant = ModelParams(tire_model=tire_model_plant)

    veh_ctrl = VehicleParams(
        m=1757.0,
        Iz=3100.0,
        lf=1.23,
        lr=1.49,
        Cf=125000.0 * stiffness_scale_ctrl,
        Cr=118000.0 * stiffness_scale_ctrl,
        mu=mu_ctrl,
    )

    veh_plant = VehicleParams(
        m=1757.0,
        Iz=3100.0,
        lf=1.23,
        lr=1.49,
        Cf=125000.0 * stiffness_scale_plant,
        Cr=118000.0 * stiffness_scale_plant,
        mu=mu_plant,
    )

    path_name = "sharp"
    x_wp, y_wp = create_sharp_waypoints()
    path = SplinePath(x_wp, y_wp, ds=0.5)

    speed_profile = make_constant_speed_profile(path, VX_REF)
    ax_profile = np.zeros_like(speed_profile)

    controller = NMPCController(path, veh_ctrl, ctrl_par, model_ctrl)

    x = np.array([1.0, np.deg2rad(2.0), 0.0, 0.0, 0.0], dtype=float)
    s_current = 0.0

    t_log = []
    x_log = []
    y_log = []
    ey_log = []
    epsi_log = []
    vy_log = []
    r_log = []
    delta_log = []
    ddelta_log = []
    s_log = []
    vx_log = []
    ax_log = []
    kappa_log = []
    solve_time_log = []
    iter_log = []
    hard_fail_log = []
    soft_fail_log = []
    status_log = []

    steps = int(sim_par.sim_time / sim_par.dt)
    terminal_margin = 3.0

    for k in range(steps):
        _, kappa_preview, vx_preview, ax_preview = preview_path_data(
            path, speed_profile, ax_profile, x, s_current, ctrl_par, model_ctrl
        )

        hard_failed = 0
        soft_failed = 0

        try:
            _, U_opt, solve_time, iter_count, return_status, soft_failure = controller.solve(
                x, kappa_preview, vx_preview, ax_preview
            )
            u = U_opt[0]

            if soft_failure:
                soft_failed = 1
                print(
                    f"[SOFT WARNING] {case_label}, step {k}: "
                    f"IPOPT status = {return_status}, using solution."
                )

        except RuntimeError as err:
            print(f"\n[HARD WARNING] {case_label}, solver failed at step {k}: {err}")
            print("State at hard failure:")
            print(f"e_y     = {x[0]:.3f} m")
            print(f"e_psi   = {np.rad2deg(x[1]):.3f} deg")
            print(f"v_y     = {x[2]:.3f} m/s")
            print(f"r       = {x[3]:.3f} rad/s")
            print(f"delta   = {np.rad2deg(x[4]):.3f} deg")
            print(f"s       = {s_current:.3f} m")
            print(f"kappa   = {path.interp_kappa(s_current):.5f} 1/m")

            hard_failed = 1
            return_status = "HARD_FAILURE"

            # Safe fallback: steer back toward zero if a hard failure occurs.
            u = np.array([-x[4] / sim_par.dt])
            u[0] = np.clip(
                u[0],
                -np.deg2rad(ctrl_par.delta_rate_max_deg_s),
                np.deg2rad(ctrl_par.delta_rate_max_deg_s),
            )

            solve_time = np.nan
            iter_count = np.nan

        x, s_current, vx_now, ax_now = simulate_step(
            x,
            s_current,
            u,
            path,
            speed_profile,
            ax_profile,
            veh_plant,
            sim_par,
            model_plant,
        )

        xg, yg, _ = path.global_from_error_state(s_current, x[0], x[1])

        t = k * sim_par.dt

        t_log.append(t)
        x_log.append(xg)
        y_log.append(yg)
        ey_log.append(x[0])
        epsi_log.append(x[1])
        vy_log.append(x[2])
        r_log.append(x[3])
        delta_log.append(x[4])
        ddelta_log.append(float(u[0]))
        s_log.append(s_current)
        vx_log.append(vx_now)
        ax_log.append(ax_now)
        kappa_log.append(path.interp_kappa(s_current))
        solve_time_log.append(solve_time)
        iter_log.append(iter_count)
        hard_fail_log.append(hard_failed)
        soft_fail_log.append(soft_failed)
        status_log.append(return_status)

        if s_current >= path.length - terminal_margin:
            break

    t_log = np.asarray(t_log)
    x_log = np.asarray(x_log)
    y_log = np.asarray(y_log)
    ey_log = np.asarray(ey_log)
    epsi_log = np.asarray(epsi_log)
    vy_log = np.asarray(vy_log)
    r_log = np.asarray(r_log)
    delta_log = np.asarray(delta_log)
    ddelta_log = np.asarray(ddelta_log)
    s_log = np.asarray(s_log)
    vx_log = np.asarray(vx_log)
    ax_log = np.asarray(ax_log)
    kappa_log = np.asarray(kappa_log)
    solve_time_log = np.asarray(solve_time_log)
    iter_log = np.asarray(iter_log)
    hard_fail_log = np.asarray(hard_fail_log)
    soft_fail_log = np.asarray(soft_fail_log)
    status_log = np.asarray(status_log, dtype=object)

    tire_logs = compute_slip_and_force_logs(
        ey_log=ey_log,
        epsi_log=epsi_log,
        vy_log=vy_log,
        r_log=r_log,
        delta_log=delta_log,
        vx_log=vx_log,
        ax_log=ax_log,
        veh_plant=veh_plant,
        model_plant=model_plant,
    )

    metrics = compute_metrics(
        t_log=t_log,
        ey_log=ey_log,
        epsi_log=epsi_log,
        delta_log=delta_log,
        ddelta_log=ddelta_log,
        kappa_log=kappa_log,
        vx_log=vx_log,
        solve_time_log=solve_time_log,
        iter_log=iter_log,
        hard_fail_log=hard_fail_log,
        soft_fail_log=soft_fail_log,
        s_log=s_log,
        path=path,
    )

    scenario_info = {
        "scenario": f"tire_mismatch_{tire_model_ctrl}_ctrl_{tire_model_plant}_plant",
        "case_label": case_label,
        "controller": f"NMPC {tire_model_ctrl.capitalize()}",
        "plant": f"{tire_model_plant.capitalize()} tire model",
        "path": path_name,
        "vx_nominal_m_s": VX_REF,
        "initial_cross_track_error_m": 1.0,
        "initial_heading_error_deg": 2.0,
        "path_length_m": path.length,
        "max_abs_kappa_1_m": float(np.max(np.abs(path.kappa))),
        "dt_s": sim_par.dt,
        "N": ctrl_par.N,
        "tire_model_ctrl": model_ctrl.tire_model,
        "tire_model_plant": model_plant.tire_model,
        "mu_ctrl": veh_ctrl.mu,
        "mu_plant": veh_plant.mu,
        "stiffness_scale_ctrl": stiffness_scale_ctrl,
        "stiffness_scale_plant": stiffness_scale_plant,
    }

    summary = {
        **scenario_info,
        **metrics,
        "max_abs_alpha_f_deg": float(np.nanmax(np.abs(np.rad2deg(tire_logs["alpha_f_log"])))),
        "max_abs_alpha_r_deg": float(np.nanmax(np.abs(np.rad2deg(tire_logs["alpha_r_log"])))),
        "max_abs_Fyf_N": float(np.nanmax(np.abs(tire_logs["Fyf_log"]))),
        "max_abs_Fyr_N": float(np.nanmax(np.abs(tire_logs["Fyr_log"]))),
    }

    logs = {
        "t_log": t_log,
        "x_log": x_log,
        "y_log": y_log,
        "ey_log": ey_log,
        "epsi_log": epsi_log,
        "vy_log": vy_log,
        "r_log": r_log,
        "delta_log": delta_log,
        "ddelta_log": ddelta_log,
        "s_log": s_log,
        "vx_log": vx_log,
        "ax_log": ax_log,
        "kappa_log": kappa_log,
        "kappa_plot": smooth_signal(kappa_log, window=21, polyorder=3),
        "solve_time_log": solve_time_log,
        "iter_log": iter_log,
        "hard_fail_log": hard_fail_log,
        "soft_fail_log": soft_fail_log,
        "status_log": status_log,
        "path_s": path.s,
        "path_x": path.x,
        "path_y": path.y,
        "path_kappa": path.kappa,
        "path_speed_profile": speed_profile,
        "x_wp": x_wp,
        "y_wp": y_wp,
        **tire_logs,
    }

    return summary, logs


def save_metrics(results: list[tuple[dict, dict]], out_dir: Path) -> None:
    metric_rows = [summary for summary, _ in results]

    csv_path = out_dir / "tire_mismatch_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = []
    for summary, _ in results:
        ctrl = summary["tire_model_ctrl"].capitalize()
        plant = summary["tire_model_plant"].capitalize()
        line = (
            f"{ctrl} & "
            f"{plant} & "
            f"{summary['rms_ey_m']:.3f} & "
            f"{summary['max_abs_ey_m']:.3f} & "
            f"{summary['rms_epsi_deg']:.2f} & "
            f"{summary['max_abs_delta_deg']:.2f} & "
            f"{summary['avg_solve_ms']:.2f} & "
            f"{summary['hard_failures']} \\\\"
        )
        lines.append(line)

    (out_dir / "tire_mismatch_latex_rows.txt").write_text("\n".join(lines) + "\n")


def save_npz_logs(results: list[tuple[dict, dict]], out_dir: Path) -> None:
    npz_data = {}

    for summary, logs in results:
        ctrl = summary["tire_model_ctrl"]
        plant = summary["tire_model_plant"]
        prefix = f"{ctrl}_ctrl_{plant}_plant"

        for key, value in logs.items():
            npz_data[f"{prefix}_{key}"] = value

        for key, value in summary.items():
            if isinstance(value, (int, float, bool, np.number)):
                npz_data[f"{prefix}_metric_{key}"] = value

    np.savez(out_dir / "tire_mismatch_logs.npz", **npz_data)


def make_comparison_plots(results: list[tuple[dict, dict]], out_dir: Path) -> None:
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

    plot_results = [
        (summary, logs)
        for summary, logs in results
        if summary["case_label"] in PLOT_CASES
    ]

    # ------------------------------------------------------------
    # Plot 1: trajectories
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    first_logs = plot_results[0][1]
    ax.plot(first_logs["path_x"], first_logs["path_y"], "--", linewidth=2.4, label="Reference path")

    for summary, logs in plot_results:
        label = summary["case_label"]
        color = CASE_COLORS[label]
        ax.plot(logs["x_log"], logs["y_log"], linewidth=1.9, label=label, color=color)

    ax.scatter([first_logs["x_log"][0]], [first_logs["y_log"][0]], marker="o", s=70, label="Start")
    ax.scatter([first_logs["path_x"][-1]], [first_logs["path_y"][-1]], marker="x", s=110, label="Goal")

    x_all = [first_logs["path_x"]]
    y_all = [first_logs["path_y"]]
    for _, logs in plot_results:
        x_all.append(logs["x_log"])
        y_all.append(logs["y_log"])

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    x_range = np.max(x_all) - np.min(x_all)
    y_range = np.max(y_all) - np.min(y_all)

    x_margin = 0.04 * max(1.0, x_range)
    y_margin = 0.80 * max(1.0, y_range)

    ax.set_xlim(np.min(x_all) - x_margin, np.max(x_all) + x_margin)
    ax.set_ylim(np.min(y_all) - y_margin, np.max(y_all) + y_margin)
    ax.set_aspect("auto")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("NMPC tire-model mismatch: trajectory comparison")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "tire_mismatch_trajectories.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 2: tracking errors and steering response versus s
    # ------------------------------------------------------------

    fig, axs = plt.subplots(4, 1, figsize=(8.8, 9.8), sharex=True)

    for summary, logs in plot_results:
        label = summary["case_label"]
        color = CASE_COLORS[label]
        s = logs["s_log"]

        axs[0].plot(s, logs["ey_log"], linewidth=1.8, label=label, color=color)
        axs[1].plot(s, np.rad2deg(logs["epsi_log"]), linewidth=1.8, label=label, color=color)
        axs[2].plot(s, np.rad2deg(logs["delta_log"]), linewidth=1.8, label=label, color=color)
        axs[3].plot(s, np.rad2deg(logs["ddelta_log"]), linewidth=1.8, label=label, color=color)

    axs[0].set_ylabel(r"$e_y$ [m]")
    axs[1].set_ylabel(r"$e_\psi$ [deg]")
    axs[2].set_ylabel(r"$\delta$ [deg]")
    axs[3].set_ylabel(r"$\dot{\delta}$ [deg/s]")
    axs[3].set_xlabel(r"Path progress $s$ [m]")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("Tire-model mismatch: tracking errors and steering response")
    fig.tight_layout()
    fig.savefig(out_dir / "tire_mismatch_errors_steering_vs_s.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 3: tire response versus s
    # ------------------------------------------------------------

    fig, axs = plt.subplots(4, 1, figsize=(8.8, 9.8), sharex=True)

    for summary, logs in plot_results:
        label = summary["case_label"]
        color = CASE_COLORS[label]
        s = logs["s_log"]

        axs[0].plot(s, np.rad2deg(logs["alpha_f_log"]), linewidth=1.8, label=label, color=color)
        axs[1].plot(s, np.rad2deg(logs["alpha_r_log"]), linewidth=1.8, label=label, color=color)
        axs[2].plot(s, logs["Fyf_log"], linewidth=1.8, label=label, color=color)
        axs[3].plot(s, logs["Fyr_log"], linewidth=1.8, label=label, color=color)

    axs[0].set_ylabel(r"$\alpha_f$ [deg]")
    axs[1].set_ylabel(r"$\alpha_r$ [deg]")
    axs[2].set_ylabel(r"$F_{yf}$ [N]")
    axs[3].set_ylabel(r"$F_{yr}$ [N]")
    axs[3].set_xlabel(r"Path progress $s$ [m]")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("Tire-model mismatch: slip angles and lateral tire forces")
    fig.tight_layout()
    fig.savefig(out_dir / "tire_mismatch_tire_response_vs_s.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 4: solver summary
    # ------------------------------------------------------------

    labels = [summary["case_label"] for summary, _ in results]
    x = np.arange(len(labels))
    bar_colors = [CASE_COLORS[label] for label in labels]

    avg_solve_ms = np.array([summary["avg_solve_ms"] for summary, _ in results], dtype=float)
    max_solve_ms = np.array([summary["max_solve_ms"] for summary, _ in results], dtype=float)
    max_iter = np.array([summary["max_ipopt_iter"] for summary, _ in results], dtype=float)
    hard_failures = np.array([summary["hard_failures"] for summary, _ in results], dtype=float)

    fig, axs = plt.subplots(4, 1, figsize=(9.5, 8.8), sharex=True)

    axs[0].plot(x, avg_solve_ms, marker="o", linewidth=1.8, label="Mean solve time")
    axs[0].plot(x, max_solve_ms, marker="o", linewidth=1.8, label="Max solve time")
    axs[0].axhline(100.0, linestyle="--", linewidth=1.5, label="Control period")
    axs[0].set_ylabel("Solve time [ms]")

    axs[1].plot(x, max_iter, marker="o", linewidth=1.8, label="Max IPOPT iterations")
    axs[1].set_ylabel("Iterations [-]")

    axs[2].bar(x, hard_failures, width=0.55, label="Hard failures", color=bar_colors)
    axs[2].set_ylabel("Failures [-]")

    axs[3].bar(
        x,
        np.array([summary["soft_warnings"] for summary, _ in results], dtype=float),
        width=0.55,
        label="Soft warnings",
        color=bar_colors
    )
    axs[3].set_ylabel("Warnings [-]")

    axs[3].set_xticks(x)
    axs[3].set_xticklabels(labels, rotation=20, ha="right")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("NMPC tire-model mismatch: solver summary")
    fig.tight_layout()
    fig.savefig(out_dir / "tire_mismatch_solver_summary.png", dpi=300)

    plt.close("all")


def main() -> None:
    out_dir = Path("results_nmpc_tire_mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("linear", "linear", "Linear/Linear"),
        ("dugoff", "dugoff", "Dugoff/Dugoff"),
        ("linear", "pacejka", "Linear/Pacejka"),
        ("dugoff", "pacejka", "Dugoff/Pacejka"),
    ]

    results = []

    for tire_model_ctrl, tire_model_plant, case_label in cases:
        print(f"\n=== Running tire mismatch case: {case_label} ===")
        summary, logs = run_mismatch_case(
            tire_model_ctrl=tire_model_ctrl,
            tire_model_plant=tire_model_plant,
            case_label=case_label,
        )
        results.append((summary, logs))

        print(f"RMS e_y:             {summary['rms_ey_m']:.4f} m")
        print(f"Max |e_y|:           {summary['max_abs_ey_m']:.4f} m")
        print(f"P95 |e_y|:           {summary['p95_abs_ey_m']:.4f} m")
        print(f"RMS e_psi:           {summary['rms_epsi_deg']:.3f} deg")
        print(f"Max |delta|:         {summary['max_abs_delta_deg']:.2f} deg")
        print(f"Max |alpha_f|:       {summary['max_abs_alpha_f_deg']:.2f} deg")
        print(f"Max |alpha_r|:       {summary['max_abs_alpha_r_deg']:.2f} deg")
        print(f"Max |Fyf|:           {summary['max_abs_Fyf_N']:.1f} N")
        print(f"Max |Fyr|:           {summary['max_abs_Fyr_N']:.1f} N")
        print(f"Avg solve time:      {summary['avg_solve_ms']:.2f} ms")
        print(f"Max solve time:      {summary['max_solve_ms']:.2f} ms")
        print(f"Avg IPOPT iter:      {summary['avg_ipopt_iter']:.2f}")
        print(f"Hard failures:       {summary['hard_failures']}")
        print(f"Soft warnings:       {summary['soft_warnings']}")
        print(f"Completion:          {summary['completion_pct']:.2f} %")

    save_metrics(results, out_dir)
    save_npz_logs(results, out_dir)
    make_comparison_plots(results, out_dir)

    print("\n=== Tire-model mismatch completed ===")
    print(f"Output directory: {out_dir}")
    print("\nLaTeX rows:")
    print((out_dir / "tire_mismatch_latex_rows.txt").read_text().strip())


if __name__ == "__main__":
    main()
