"""
nmpc_speed_sensitivity_result.py

Speed-sensitivity simulation for the results chapter:
- Uses the NMPC Dugoff controller and matched Dugoff plant
- Uses the same sharp reference path for all cases
- Compares constant longitudinal speeds: 2, 3, 5, 7, and 10 m/s
- Saves comparison plots, logs, CSV metrics, and LaTeX table rows

Run:
    python nmpc_speed_sensitivity_result.py

Outputs:
    results_nmpc_speed_sensitivity/
        speed_trajectories_comparison.png
        speed_errors_steering_vs_s.png
        speed_dynamic_response_vs_s.png
        speed_solver_summary.png
        speed_sensitivity_metrics.csv
        speed_sensitivity_latex_rows.txt
        speed_sensitivity_logs.npz
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_SPEED_CASES = {2.0, 7.0, 10.0}

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


def make_constant_speed_profile(path: SplinePath, vx: float) -> np.ndarray:
    return np.full_like(path.s, float(vx), dtype=float)


def run_speed_case(vx_ref: float) -> tuple[dict, dict]:
    tire_model_ctrl = "dugoff"
    tire_model_plant = "dugoff"

    stiffness_scale_ctrl = 1.00
    stiffness_scale_plant = 1.00
    mu_ctrl = 0.80
    mu_plant = 0.80

    ctrl_par = ControllerParams()
    # Longer simulation time is needed for the 2 m/s case to reach the end of the path.
    sim_par = SimulationParams(sim_time=80.0)
    speed_par = SpeedProfileParams(
        vx_nominal=float(vx_ref),
        vx_min=float(vx_ref),
        vx_max=float(vx_ref),
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

    speed_profile = make_constant_speed_profile(path, vx_ref)
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
        s_preview, kappa_preview, vx_preview, ax_preview = preview_path_data(
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
                    f"[SOFT WARNING] v={vx_ref:.1f} m/s, step {k}: "
                    f"IPOPT status = {return_status}, using solution."
                )

        except RuntimeError as err:
            print(f"\n[HARD WARNING] v={vx_ref:.1f} m/s, solver failed at step {k}: {err}")
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

    t_log = np.array(t_log)
    x_log = np.array(x_log)
    y_log = np.array(y_log)
    ey_log = np.array(ey_log)
    epsi_log = np.array(epsi_log)
    vy_log = np.array(vy_log)
    r_log = np.array(r_log)
    delta_log = np.array(delta_log)
    ddelta_log = np.array(ddelta_log)
    s_log = np.array(s_log)
    vx_log = np.array(vx_log)
    ax_log = np.array(ax_log)
    kappa_log = np.array(kappa_log)
    solve_time_log = np.array(solve_time_log)
    iter_log = np.array(iter_log)
    hard_fail_log = np.array(hard_fail_log)
    soft_fail_log = np.array(soft_fail_log)
    status_log = np.array(status_log, dtype=object)

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
        "scenario": f"nmpc_dugoff_speed_{vx_ref:.0f}ms",
        "controller": "NMPC Dugoff",
        "plant": "Dugoff tire model",
        "path": path_name,
        "vx_nominal_m_s": float(vx_ref),
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
    }

    return summary, logs


def save_metrics(results: list[tuple[dict, dict]], out_dir: Path) -> None:
    metric_rows = [summary for summary, _ in results]

    csv_path = out_dir / "speed_sensitivity_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = []
    for summary, _ in results:
        line = (
            f"{summary['vx_nominal_m_s']:.0f} & "
            f"{summary['rms_ey_m']:.3f} & "
            f"{summary['max_abs_ey_m']:.3f} & "
            f"{summary['rms_epsi_deg']:.2f} & "
            f"{summary['max_abs_delta_deg']:.2f} & "
            f"{summary['max_abs_ddelta_deg_s']:.2f} & "
            f"{summary['avg_solve_ms']:.2f} & "
            f"{summary['hard_failures']} \\\\"
        )
        lines.append(line)

    (out_dir / "speed_sensitivity_latex_rows.txt").write_text("\n".join(lines) + "\n")


def save_npz_logs(results: list[tuple[dict, dict]], out_dir: Path) -> None:
    npz_data = {}

    for summary, logs in results:
        prefix = f"v{int(round(summary['vx_nominal_m_s']))}"
        for key, value in logs.items():
            npz_data[f"{prefix}_{key}"] = value
        for key, value in summary.items():
            if isinstance(value, (int, float, bool, np.number)):
                npz_data[f"{prefix}_metric_{key}"] = value

    np.savez(out_dir / "speed_sensitivity_logs.npz", **npz_data)


def make_comparison_plots(results: list[tuple[dict, dict]], out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Only show representative cases in the plots to keep the figures readable.
    # All speed cases are still included in the CSV file and LaTeX table rows.
    plot_results = [
        (summary, logs)
        for summary, logs in results
        if float(summary["vx_nominal_m_s"]) in PLOT_SPEED_CASES
    ]

    # ------------------------------------------------------------
    # Plot 1: trajectories
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    first_logs = plot_results[0][1]
    ax.plot(first_logs["path_x"], first_logs["path_y"], "--", linewidth=2.4, label="Reference path")

    for summary, logs in plot_results:
        label = rf"$v_x={summary['vx_nominal_m_s']:.0f}$ m/s"
        ax.plot(logs["x_log"], logs["y_log"], linewidth=1.9, label=label)

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

    # Keep visual style consistent with previous trajectory figures.
    ax.set_aspect("auto")

    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("NMPC Dugoff speed sensitivity: representative trajectories")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "speed_trajectories_comparison.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 2: tracking errors and steering response versus path progress
    # ------------------------------------------------------------

    fig, axs = plt.subplots(4, 1, figsize=(8.8, 9.8), sharex=True)

    for summary, logs in plot_results:
        label = rf"{summary['vx_nominal_m_s']:.0f} m/s"
        s = logs["s_log"]

        axs[0].plot(s, logs["ey_log"], linewidth=1.8, label=label)
        axs[1].plot(s, np.rad2deg(logs["epsi_log"]), linewidth=1.8, label=label)
        axs[2].plot(s, np.rad2deg(logs["delta_log"]), linewidth=1.8, label=label)
        axs[3].plot(s, np.rad2deg(logs["ddelta_log"]), linewidth=1.8, label=label)

    axs[0].set_ylabel(r"$e_y$ [m]")
    axs[1].set_ylabel(r"$e_\psi$ [deg]")
    axs[2].set_ylabel(r"$\delta$ [deg]")
    axs[3].set_ylabel(r"$\dot{\delta}$ [deg/s]")
    axs[3].set_xlabel(r"Path progress $s$ [m]")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("Representative speed cases: tracking errors and steering response versus path progress")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_errors_steering_vs_s.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 3: dynamic response versus path progress
    # ------------------------------------------------------------

    fig, axs = plt.subplots(3, 1, figsize=(8.8, 7.6), sharex=True)

    for summary, logs in plot_results:
        label = rf"{summary['vx_nominal_m_s']:.0f} m/s"
        s = logs["s_log"]
        r_ref = logs["vx_log"] * logs["kappa_plot"]

        axs[0].plot(s, logs["kappa_plot"], linewidth=1.8, label=label)
        axs[1].plot(s, logs["vx_log"], linewidth=1.8, label=label)

        # Use one legend entry per speed in the yaw-rate subplot.
        # The dashed reference uses the same default color as the corresponding yaw-rate curve.
        yaw_line, = axs[2].plot(s, logs["r_log"], linewidth=1.8, label=label)
        axs[2].plot(
            s,
            r_ref,
            "--",
            linewidth=1.4,
            color=yaw_line.get_color(),
            label="_nolegend_",
        )

    axs[0].set_ylabel(r"$\kappa$ [1/m]")
    axs[1].set_ylabel(r"$v_x$ [m/s]")
    axs[2].set_ylabel("Yaw rate [rad/s]")
    axs[2].set_xlabel(r"Path progress $s$ [m]")

    for ax in axs:
        ax.grid(True)

    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    axs[2].legend(
        loc="lower right",
        title="Solid: $r$\nDashed: $v_x\\kappa$",
        title_fontsize=9,
        fontsize=9,
    )

    fig.suptitle("Representative speed cases: dynamic response versus path progress")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_dynamic_response_vs_s.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 4: solver summary versus speed
    # ------------------------------------------------------------

    speeds = np.array([summary["vx_nominal_m_s"] for summary, _ in results], dtype=float)
    avg_solve_ms = np.array([summary["avg_solve_ms"] for summary, _ in results], dtype=float)
    max_solve_ms = np.array([summary["max_solve_ms"] for summary, _ in results], dtype=float)
    max_iter = np.array([summary["max_ipopt_iter"] for summary, _ in results], dtype=float)
    hard_failures = np.array([summary["hard_failures"] for summary, _ in results], dtype=float)
    soft_warnings = np.array([summary["soft_warnings"] for summary, _ in results], dtype=float)

    fig, axs = plt.subplots(4, 1, figsize=(8.8, 8.6), sharex=True)

    axs[0].plot(speeds, avg_solve_ms, marker="o", linewidth=1.8, label="Mean solve time")
    axs[0].plot(speeds, max_solve_ms, marker="o", linewidth=1.8, label="Max solve time")
    axs[0].axhline(100.0, linestyle="--", linewidth=1.5, label="Control period")
    axs[0].set_ylabel("Solve time [ms]")

    axs[1].plot(speeds, max_iter, marker="o", linewidth=1.8, label="Max IPOPT iterations")
    axs[1].set_ylabel("Iterations [-]")

    axs[2].bar(speeds, hard_failures, width=0.45, label="Hard failures")
    axs[2].set_ylabel("Hard failures [-]")

    axs[3].bar(speeds, soft_warnings, width=0.45, label="Soft warnings")
    axs[3].set_ylabel("Soft warnings [-]")
    axs[3].set_xlabel(r"Longitudinal speed $v_x$ [m/s]")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("NMPC Dugoff speed sensitivity: solver summary")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_solver_summary.png", dpi=300)

    plt.close("all")


def main() -> None:
    out_dir = Path("results_nmpc_speed_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    speed_cases = [2.0, 3.0, 5.0, 7.0, 10.0]

    results = []

    for vx in speed_cases:
        print(f"\n=== Running NMPC speed sensitivity case: v_x = {vx:.1f} m/s ===")
        summary, logs = run_speed_case(vx_ref=vx)
        results.append((summary, logs))

        print(f"RMS e_y:          {summary['rms_ey_m']:.4f} m")
        print(f"Max |e_y|:        {summary['max_abs_ey_m']:.4f} m")
        print(f"RMS e_psi:        {summary['rms_epsi_deg']:.3f} deg")
        print(f"Max |delta|:      {summary['max_abs_delta_deg']:.2f} deg")
        print(f"Max |delta_dot|:  {summary['max_abs_ddelta_deg_s']:.2f} deg/s")
        print(f"Avg solve time:   {summary['avg_solve_ms']:.2f} ms")
        print(f"Max solve time:   {summary['max_solve_ms']:.2f} ms")
        print(f"Avg IPOPT iter:   {summary['avg_ipopt_iter']:.2f}")
        print(f"Hard failures:    {summary['hard_failures']}")
        print(f"Soft warnings:    {summary['soft_warnings']}")
        print(f"Completion:       {summary['completion_pct']:.2f} %")

    save_metrics(results, out_dir)
    save_npz_logs(results, out_dir)
    make_comparison_plots(results, out_dir)

    print("\n=== NMPC speed sensitivity completed ===")
    print("Representative plot cases: 2, 7, and 10 m/s")
    print("CSV and LaTeX rows include all cases: 2, 3, 5, 7, and 10 m/s")
    print(f"Output directory: {out_dir}")
    print("\nLaTeX rows:")
    print((out_dir / "speed_sensitivity_latex_rows.txt").read_text().strip())


if __name__ == "__main__":
    main()
