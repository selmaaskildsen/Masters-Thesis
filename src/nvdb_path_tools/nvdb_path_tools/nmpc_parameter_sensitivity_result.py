"""
nmpc_parameter_sensitivity_result.py

Friction and cornering-stiffness sensitivity simulation for the results chapter:
- Uses the same sharp path, constant speed, and NMPC Dugoff controller for all cases
- Keeps the controller parameters nominal
- Varies only the plant tire parameters:
    * Nominal
    * Reduced friction: mu = 0.75 mu0
    * Low friction:     mu = 0.50 mu0
    * Reduced stiffness: Cf, Cr = 0.75 nominal
    * Low stiffness:     Cf, Cr = 0.50 nominal
- Saves comparison plots, logs, CSV metrics, and LaTeX table rows

Run:
    python nmpc_parameter_sensitivity_result.py

Outputs:
    results_nmpc_parameter_sensitivity/
        parameter_sensitivity_trajectories.png
        friction_sensitivity_errors_steering_vs_s.png
        stiffness_sensitivity_errors_steering_vs_s.png
        parameter_sensitivity_summary.png
        parameter_sensitivity_metrics.csv
        parameter_sensitivity_latex_rows.txt
        parameter_sensitivity_logs.npz
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


VX_REF = 7.0
MU0 = 0.80
CF0 = 125000.0
CR0 = 118000.0


def make_constant_speed_profile(path: SplinePath, vx: float) -> np.ndarray:
    return np.full_like(path.s, float(vx), dtype=float)


def run_parameter_case(
    case_label: str,
    plant_mu_scale: float,
    plant_stiffness_scale: float,
) -> tuple[dict, dict]:
    """
    Run one sensitivity case.

    The controller is kept nominal in all cases:
        NMPC Dugoff, mu=0.8, Cf=125000, Cr=118000.

    Only the plant tire parameters are changed.
    """
    tire_model_ctrl = "dugoff"
    tire_model_plant = "dugoff"

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
        Cf=CF0,
        Cr=CR0,
        mu=MU0,
    )

    veh_plant = VehicleParams(
        m=1757.0,
        Iz=3100.0,
        lf=1.23,
        lr=1.49,
        Cf=CF0 * plant_stiffness_scale,
        Cr=CR0 * plant_stiffness_scale,
        mu=MU0 * plant_mu_scale,
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
        "scenario": f"parameter_sensitivity_{case_label.lower().replace(' ', '_').replace('-', '_')}",
        "case_label": case_label,
        "controller": "NMPC Dugoff",
        "plant": "Dugoff tire model",
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
        "mu_scale_plant": plant_mu_scale,
        "stiffness_scale_ctrl": 1.0,
        "stiffness_scale_plant": plant_stiffness_scale,
        "Cf_ctrl_N_rad": veh_ctrl.Cf,
        "Cr_ctrl_N_rad": veh_ctrl.Cr,
        "Cf_plant_N_rad": veh_plant.Cf,
        "Cr_plant_N_rad": veh_plant.Cr,
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

    csv_path = out_dir / "parameter_sensitivity_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = []
    for summary, _ in results:
        line = (
            f"{summary['case_label']} & "
            f"{summary['mu_plant']:.2f} & "
            f"{summary['stiffness_scale_plant']:.2f} & "
            f"{summary['rms_ey_m']:.3f} & "
            f"{summary['max_abs_ey_m']:.3f} & "
            f"{summary['rms_epsi_deg']:.2f} & "
            f"{summary['max_abs_delta_deg']:.2f} & "
            f"{summary['avg_solve_ms']:.2f} & "
            f"{summary['hard_failures']} \\\\"
        )
        lines.append(line)

    (out_dir / "parameter_sensitivity_latex_rows.txt").write_text("\n".join(lines) + "\n")


def save_npz_logs(results: list[tuple[dict, dict]], out_dir: Path) -> None:
    npz_data = {}

    for summary, logs in results:
        prefix = summary["case_label"].lower().replace(" ", "_").replace("-", "_")
        for key, value in logs.items():
            npz_data[f"{prefix}_{key}"] = value
        for key, value in summary.items():
            if isinstance(value, (int, float, bool, np.number)):
                npz_data[f"{prefix}_metric_{key}"] = value

    np.savez(out_dir / "parameter_sensitivity_logs.npz", **npz_data)


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

    result_by_label = {summary["case_label"]: (summary, logs) for summary, logs in results}

    friction_results = [
        result_by_label["Nominal"],
        result_by_label["Reduced friction"],
        result_by_label["Low friction"],
    ]

    stiffness_results = [
        result_by_label["Nominal"],
        result_by_label["Reduced stiffness"],
        result_by_label["Low stiffness"],
    ]

    # ------------------------------------------------------------
    # Plot 1: trajectories for all cases
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    first_logs = results[0][1]
    ax.plot(first_logs["path_x"], first_logs["path_y"], "--", linewidth=2.4, label="Reference path")

    for summary, logs in results:
        ax.plot(logs["x_log"], logs["y_log"], linewidth=1.8, label=summary["case_label"])

    ax.scatter([first_logs["x_log"][0]], [first_logs["y_log"][0]], marker="o", s=70, label="Start")
    ax.scatter([first_logs["path_x"][-1]], [first_logs["path_y"][-1]], marker="x", s=110, label="Goal")

    x_all = [first_logs["path_x"]]
    y_all = [first_logs["path_y"]]
    for _, logs in results:
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
    ax.set_title("NMPC parameter sensitivity: trajectory comparison")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "parameter_sensitivity_trajectories.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 2: friction sensitivity
    # ------------------------------------------------------------

    fig, axs = plt.subplots(4, 1, figsize=(8.8, 9.8), sharex=True)

    for summary, logs in friction_results:
        label = summary["case_label"]
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

    fig.suptitle("Friction sensitivity: tracking errors and steering response")
    fig.tight_layout()
    fig.savefig(out_dir / "friction_sensitivity_errors_steering_vs_s.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 3: cornering-stiffness sensitivity
    # ------------------------------------------------------------

    fig, axs = plt.subplots(4, 1, figsize=(8.8, 9.8), sharex=True)

    for summary, logs in stiffness_results:
        label = summary["case_label"]
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

    fig.suptitle("Cornering-stiffness sensitivity: tracking errors and steering response")
    fig.tight_layout()
    fig.savefig(out_dir / "stiffness_sensitivity_errors_steering_vs_s.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 4: summary metrics
    # ------------------------------------------------------------

    labels = [summary["case_label"] for summary, _ in results]
    x = np.arange(len(labels))

    rms_ey = np.array([summary["rms_ey_m"] for summary, _ in results], dtype=float)
    p95_ey = np.array([summary["p95_abs_ey_m"] for summary, _ in results], dtype=float)
    max_delta = np.array([summary["max_abs_delta_deg"] for summary, _ in results], dtype=float)
    iadc = np.array([summary["iadc_deg"] for summary, _ in results], dtype=float)
    avg_solve = np.array([summary["avg_solve_ms"] for summary, _ in results], dtype=float)
    max_solve = np.array([summary["max_solve_ms"] for summary, _ in results], dtype=float)
    hard_failures = np.array([summary["hard_failures"] for summary, _ in results], dtype=float)

    fig, axs = plt.subplots(4, 1, figsize=(9.8, 9.5), sharex=True)

    axs[0].plot(x, rms_ey, marker="o", linewidth=1.8, label=r"RMS $e_y$")
    axs[0].plot(x, p95_ey, marker="o", linewidth=1.8, label=r"P95 $|e_y|$")
    axs[0].set_ylabel("Error [m]")

    axs[1].plot(x, max_delta, marker="o", linewidth=1.8, label=r"Max $|\delta|$")
    axs[1].plot(x, iadc, marker="o", linewidth=1.8, label="IADC")
    axs[1].set_ylabel("Steering [deg]")

    axs[2].plot(x, avg_solve, marker="o", linewidth=1.8, label="Mean solve time")
    axs[2].plot(x, max_solve, marker="o", linewidth=1.8, label="Max solve time")
    axs[2].axhline(100.0, linestyle="--", linewidth=1.5, label="Control period")
    axs[2].set_ylabel("Solve time [ms]")

    axs[3].bar(x, hard_failures, width=0.55, label="Hard failures")
    axs[3].set_ylabel("Failures [-]")

    axs[3].set_xticks(x)
    axs[3].set_xticklabels(labels, rotation=20, ha="right")

    for ax in axs:
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("NMPC parameter sensitivity: summary metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_sensitivity_summary.png", dpi=300)

    plt.close("all")


def main() -> None:
    out_dir = Path("results_nmpc_parameter_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("Nominal", 1.00, 1.00),
        ("Reduced friction", 0.75, 1.00),
        ("Low friction", 0.50, 1.00),
        ("Reduced stiffness", 1.00, 0.75),
        ("Low stiffness", 1.00, 0.50),
    ]

    results = []

    for case_label, plant_mu_scale, plant_stiffness_scale in cases:
        print(f"\n=== Running parameter sensitivity case: {case_label} ===")
        summary, logs = run_parameter_case(
            case_label=case_label,
            plant_mu_scale=plant_mu_scale,
            plant_stiffness_scale=plant_stiffness_scale,
        )
        results.append((summary, logs))

        print(f"Plant mu:            {summary['mu_plant']:.3f}")
        print(f"Plant stiffness:     {summary['stiffness_scale_plant']:.2f}")
        print(f"RMS e_y:             {summary['rms_ey_m']:.4f} m")
        print(f"Max |e_y|:           {summary['max_abs_ey_m']:.4f} m")
        print(f"P95 |e_y|:           {summary['p95_abs_ey_m']:.4f} m")
        print(f"RMS e_psi:           {summary['rms_epsi_deg']:.3f} deg")
        print(f"Max |delta|:         {summary['max_abs_delta_deg']:.2f} deg")
        print(f"IADC:                {summary['iadc_deg']:.2f} deg")
        print(f"Avg solve time:      {summary['avg_solve_ms']:.2f} ms")
        print(f"Max solve time:      {summary['max_solve_ms']:.2f} ms")
        print(f"Avg IPOPT iter:      {summary['avg_ipopt_iter']:.2f}")
        print(f"Max IPOPT iter:      {summary['max_ipopt_iter']:.0f}")
        print(f"Hard failures:       {summary['hard_failures']}")
        print(f"Soft warnings:       {summary['soft_warnings']}")
        print(f"Completion:          {summary['completion_pct']:.2f} %")

    save_metrics(results, out_dir)
    save_npz_logs(results, out_dir)
    make_comparison_plots(results, out_dir)

    print("\n=== Parameter sensitivity completed ===")
    print(f"Output directory: {out_dir}")
    print("\nLaTeX rows:")
    print((out_dir / "parameter_sensitivity_latex_rows.txt").read_text().strip())


if __name__ == "__main__":
    main()
