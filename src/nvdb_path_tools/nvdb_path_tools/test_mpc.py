import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
from scipy.signal import cont2discrete


# ============================================================
# Parameter containers
# ============================================================

@dataclass
class VehicleParams:
    m: float
    Iz: float
    lf: float
    lr: float
    Cf: float
    Cr: float


@dataclass
class SimParams:
    dt: float
    t_end: float
    vx: float
    use_rk4: bool


@dataclass
class MPCParams:
    N: int
    q_ey: float
    q_epsi: float
    q_vy: float
    q_r: float
    q_delta: float
    r_ddelta: float
    delta_max_deg: float
    delta_rate_max_deg_s: float
    use_curvature_feedforward: bool = True


@dataclass
class PathParams:
    ds: float


# ============================================================
# Utility
# ============================================================

def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def cumulative_arclength(waypoints: np.ndarray) -> np.ndarray:
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(waypoints))
    s[1:] = np.cumsum(seg_lengths)
    return s


# ============================================================
# Path generation from waypoints using cubic splines
# ============================================================

def generate_spline_path_from_waypoints(waypoints: np.ndarray, ds: float = 0.5) -> dict:
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints must have shape (N, 2)")

    if len(waypoints) < 3:
        raise ValueError("Need at least 3 waypoints for cubic spline interpolation.")

    s_wp = cumulative_arclength(waypoints)
    total_length = s_wp[-1]

    cs_x = CubicSpline(s_wp, waypoints[:, 0], bc_type="natural")
    cs_y = CubicSpline(s_wp, waypoints[:, 1], bc_type="natural")

    s_path = np.arange(0.0, total_length + ds, ds)

    x_path = cs_x(s_path)
    y_path = cs_y(s_path)

    dx_ds = cs_x(s_path, 1)
    dy_ds = cs_y(s_path, 1)
    ddx_ds2 = cs_x(s_path, 2)
    ddy_ds2 = cs_y(s_path, 2)

    psi_path = np.arctan2(dy_ds, dx_ds)

    denom = (dx_ds**2 + dy_ds**2) ** 1.5
    denom = np.maximum(denom, 1e-9)

    kappa_path = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom

    return {
        "x": x_path,
        "y": y_path,
        "psi": psi_path,
        "kappa": kappa_path,
        "s": s_path,
        "length": total_length
    }


# ============================================================
# Closest-point and path error logic
# ============================================================

def find_closest_path_index(
    Xg: float,
    Yg: float,
    path: dict,
    last_idx: int,
    window_back: int = 10,
    window_forward: int = 120
) -> int:
    n = len(path["x"])
    i0 = max(0, last_idx - window_back)
    i1 = min(n, last_idx + window_forward)

    if i1 <= i0:
        return last_idx

    dx = path["x"][i0:i1] - Xg
    dy = path["y"][i0:i1] - Yg
    dist2 = dx**2 + dy**2

    idx_local = np.argmin(dist2)
    return i0 + idx_local


def compute_path_errors(Xg: float, Yg: float, psi_g: float, path: dict, idx: int):
    xr = path["x"][idx]
    yr = path["y"][idx]
    psi_r = path["psi"][idx]

    dx = Xg - xr
    dy = Yg - yr

    nx = -np.sin(psi_r)
    ny = np.cos(psi_r)

    ey = dx * nx + dy * ny
    epsi = wrap_angle(psi_g - psi_r)

    return ey, epsi


def build_curvature_preview(path: dict, closest_idx: int, horizon: int, step_distance: float) -> np.ndarray:
    preview = np.zeros(horizon)

    if len(path["s"]) < 2:
        return preview

    ds_path = path["s"][1] - path["s"][0]
    ds_path = max(ds_path, 1e-6)

    for k in range(horizon):
        idx = closest_idx + int(round((k * step_distance) / ds_path))
        idx = min(idx, len(path["kappa"]) - 1)
        preview[k] = path["kappa"][idx]

    return preview


# ============================================================
# Tire model
# ============================================================

def slip_angles(vy: float, r: float, delta: float, vx: float, p: VehicleParams):
    vx_safe = max(vx, 0.5)
    alpha_f = (vy + p.lf * r) / vx_safe - delta
    alpha_r = (vy - p.lr * r) / vx_safe
    return alpha_f, alpha_r


def tire_forces_linear(vy: float, r: float, delta: float, vx: float, p: VehicleParams):
    alpha_f, alpha_r = slip_angles(vy, r, delta, vx, p)
    Fyf = -p.Cf * alpha_f
    Fyr = -p.Cr * alpha_r
    return Fyf, Fyr, alpha_f, alpha_r


# ============================================================
# Nonlinear plant model
# State: [e_y, e_psi, v_y, r, delta]
# Input: delta_dot
# ============================================================

def continuous_dynamics_error_model(
    x: np.ndarray,
    delta_dot_cmd: float,
    kappa: float,
    vx: float,
    p: VehicleParams
):
    ey, epsi, vy, r, delta = x

    Fyf, Fyr, alpha_f, alpha_r = tire_forces_linear(vy, r, delta, vx, p)

    ey_dot = vy + vx * epsi
    epsi_dot = r - vx * kappa
    vy_dot = (Fyf + Fyr) / p.m - vx * r
    r_dot = (p.lf * Fyf - p.lr * Fyr) / p.Iz
    delta_dot = delta_dot_cmd

    x_dot = np.array([ey_dot, epsi_dot, vy_dot, r_dot, delta_dot], dtype=float)

    debug = {
        "Fyf": Fyf,
        "Fyr": Fyr,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "delta": delta
    }
    return x_dot, debug


def euler_step_error_model(
    x: np.ndarray,
    delta_dot_cmd: float,
    kappa: float,
    vx: float,
    p: VehicleParams,
    dt: float
):
    x_dot, debug = continuous_dynamics_error_model(x, delta_dot_cmd, kappa, vx, p)
    return x + dt * x_dot, debug


def rk4_step_error_model(
    x: np.ndarray,
    delta_dot_cmd: float,
    kappa: float,
    vx: float,
    p: VehicleParams,
    dt: float
):
    k1, _ = continuous_dynamics_error_model(x, delta_dot_cmd, kappa, vx, p)
    k2, _ = continuous_dynamics_error_model(x + 0.5 * dt * k1, delta_dot_cmd, kappa, vx, p)
    k3, _ = continuous_dynamics_error_model(x + 0.5 * dt * k2, delta_dot_cmd, kappa, vx, p)
    k4, debug = continuous_dynamics_error_model(x + dt * k3, delta_dot_cmd, kappa, vx, p)

    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next, debug


# ============================================================
# Linearized model for MPC
# State: [e_y, e_psi, v_y, r, delta]
# Input: delta_dot
# ============================================================

def linear_state_space_matrices(vx: float, p: VehicleParams):
    vx_safe = max(vx, 0.5)

    A = np.array([
        [0.0, vx_safe, 1.0, 0.0, 0.0],
        [0.0, 0.0,     0.0, 1.0, 0.0],
        [0.0, 0.0,
         -(p.Cf + p.Cr) / (p.m * vx_safe),
         -((p.Cf * p.lf - p.Cr * p.lr) / (p.m * vx_safe) + vx_safe),
         p.Cf / p.m],
        [0.0, 0.0,
         -(p.Cf * p.lf - p.Cr * p.lr) / (p.Iz * vx_safe),
         -(p.Cf * p.lf**2 + p.Cr * p.lr**2) / (p.Iz * vx_safe),
         p.Cf * p.lf / p.Iz],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=float)

    B = np.array([
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0]
    ], dtype=float)

    E = np.array([
        [0.0],
        [-vx_safe],
        [0.0],
        [0.0],
        [0.0]
    ], dtype=float)

    return A, B, E


def discretize_zoh(A: np.ndarray, B: np.ndarray, E: np.ndarray, dt: float):
    B_aug = np.hstack([B, E])  # inputs: [delta_dot, kappa]
    C = np.eye(A.shape[0])
    D = np.zeros((A.shape[0], B_aug.shape[1]))

    Ad, B_aug_d, _, _, _ = cont2discrete((A, B_aug, C, D), dt, method="zoh")
    Bd = B_aug_d[:, [0]]
    Ed = B_aug_d[:, [1]]
    return Ad, Bd, Ed


# ============================================================
# Feedforward steering from curvature
# ============================================================

def curvature_feedforward(kappa: float, p: VehicleParams) -> float:
    return (p.lf + p.lr) * kappa


# ============================================================
# MPC solver
# ============================================================

def solve_mpc(
    x0: np.ndarray,
    kappa_preview: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    Ed: np.ndarray,
    mpc: MPCParams,
    vehicle: VehicleParams
):
    nx = Ad.shape[0]
    nu = Bd.shape[1]
    N = mpc.N

    Q = np.diag([mpc.q_ey, mpc.q_epsi, mpc.q_vy, mpc.q_r, mpc.q_delta])
    Qf = Q.copy()
    R = np.array([[mpc.r_ddelta]])

    delta_max = np.deg2rad(mpc.delta_max_deg)
    delta_rate_max = np.deg2rad(mpc.delta_rate_max_deg_s)

    x = cp.Variable((nx, N + 1))
    u = cp.Variable((nu, N))   # delta_dot

    cost = 0.0
    constraints = [x[:, 0] == x0]

    for k in range(N):
        kappa_k = float(kappa_preview[k])

        delta_ff = curvature_feedforward(kappa_k, vehicle) if mpc.use_curvature_feedforward else 0.0

        constraints += [
            x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k] + Ed.flatten() * kappa_k
        ]

        constraints += [
            x[4, k] <= delta_max,
            x[4, k] >= -delta_max
        ]

        constraints += [
            u[:, k] <= delta_rate_max,
            u[:, k] >= -delta_rate_max
        ]

        cost += cp.quad_form(x[:, k], Q)
        cost += cp.quad_form(u[:, k], R)

        # Encourage steering toward feedforward value
        cost += 5.0 * cp.sum_squares(x[4, k] - delta_ff)

    constraints += [
        x[4, N] <= delta_max,
        x[4, N] >= -delta_max
    ]

    cost += cp.quad_form(x[:, N], Qf)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(
        solver=cp.OSQP,
        warm_start=True,
        verbose=False,
        max_iter=50000,
        eps_abs=1e-5,
        eps_rel=1e-5
    )

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"MPC solve failed. Status: {problem.status}")

    u_opt = np.array(u.value).reshape(nu, N)
    x_opt = np.array(x.value)

    return float(u_opt[0, 0]), u_opt, x_opt, problem.status


# ============================================================
# Metrics
# ============================================================

def compute_metrics(results: dict) -> dict:
    t = results["time"]
    X = results["state"]

    ey = X[:, 0]
    epsi = X[:, 1]
    delta = results["delta"]
    kappa = results["kappa"]

    if len(t) < 2:
        return {
            "IAE_CTE": 0.0,
            "IAE_Heading_deg_s": 0.0,
            "IADC_deg": 0.0,
            "Max_delta_deg": np.rad2deg(np.max(np.abs(delta))) if len(delta) > 0 else 0.0,
            "CNXTE_simple": 0.0
        }

    dt = np.mean(np.diff(t))

    iae_cte = np.sum(np.abs(ey)) * dt
    iae_heading = np.sum(np.abs(np.rad2deg(epsi))) * dt
    iadc = np.sum(np.abs(np.diff(np.rad2deg(delta))))
    max_delta_deg = np.max(np.abs(np.rad2deg(delta)))
    cnxte = np.mean(np.abs(ey) * (1.0 + 10.0 * np.abs(kappa)))

    return {
        "IAE_CTE": iae_cte,
        "IAE_Heading_deg_s": iae_heading,
        "IADC_deg": iadc,
        "Max_delta_deg": max_delta_deg,
        "CNXTE_simple": cnxte
    }


# ============================================================
# Closed-loop simulation
# ============================================================

def simulate_closed_loop(
    vehicle: VehicleParams,
    sim: SimParams,
    mpc: MPCParams,
    path: dict
):
    # Constant-speed baseline
    vx = sim.vx

    A, B, E = linear_state_space_matrices(vx, vehicle)
    Ad, Bd, Ed = discretize_zoh(A, B, E, sim.dt)

    n_steps = int(sim.t_end / sim.dt) + 1
    time = np.linspace(0.0, sim.t_end, n_steps)

    Xg, Yg, psi_g = path["x"][0], path["y"][0], path["psi"][0]
    vy, r, delta = 0.0, 0.0, 0.0
    closest_idx = 0

    X_state = np.zeros((n_steps, 5))
    delta_log = np.zeros(n_steps)
    delta_dot_log = np.zeros(n_steps)
    delta_ff_log = np.zeros(n_steps)
    kappa_log = np.zeros(n_steps)
    Fyf_log = np.zeros(n_steps)
    Fyr_log = np.zeros(n_steps)
    af_log = np.zeros(n_steps)
    ar_log = np.zeros(n_steps)
    status_log = []

    Xg_log = np.zeros(n_steps)
    Yg_log = np.zeros(n_steps)
    psi_g_log = np.zeros(n_steps)

    Xclosest_log = np.zeros(n_steps)
    Yclosest_log = np.zeros(n_steps)
    idx_log = np.zeros(n_steps, dtype=int)

    step_distance = vx * sim.dt
    delta_max = np.deg2rad(mpc.delta_max_deg)

    for i, _ in enumerate(time):
        closest_idx = find_closest_path_index(Xg, Yg, path, last_idx=closest_idx)

        ey, epsi = compute_path_errors(Xg, Yg, psi_g, path, closest_idx)
        x_mpc = np.array([ey, epsi, vy, r, delta], dtype=float)

        kappa_preview = build_curvature_preview(
            path=path,
            closest_idx=closest_idx,
            horizon=mpc.N,
            step_distance=step_distance
        )

        delta_dot_cmd, _, _, status = solve_mpc(
            x0=x_mpc,
            kappa_preview=kappa_preview,
            Ad=Ad,
            Bd=Bd,
            Ed=Ed,
            mpc=mpc,
            vehicle=vehicle
        )
        status_log.append(status)

        kappa_now = path["kappa"][closest_idx]
        delta_ff_now = curvature_feedforward(kappa_now, vehicle) if mpc.use_curvature_feedforward else 0.0

        x_dyn = np.array([ey, epsi, vy, r, delta], dtype=float)
        if sim.use_rk4:
            x_next, debug = rk4_step_error_model(
                x_dyn, delta_dot_cmd, kappa_now, vx, vehicle, sim.dt
            )
        else:
            x_next, debug = euler_step_error_model(
                x_dyn, delta_dot_cmd, kappa_now, vx, vehicle, sim.dt
            )

        vy = x_next[2]
        r = x_next[3]
        delta = np.clip(x_next[4], -delta_max, delta_max)

        # Global kinematics with constant longitudinal speed
        Xg_dot = vx * np.cos(psi_g) - vy * np.sin(psi_g)
        Yg_dot = vx * np.sin(psi_g) + vy * np.cos(psi_g)
        psi_g_dot = r

        Xg += sim.dt * Xg_dot
        Yg += sim.dt * Yg_dot
        psi_g = wrap_angle(psi_g + sim.dt * psi_g_dot)

        X_state[i, :] = np.array([ey, epsi, vy, r, delta])
        delta_log[i] = delta
        delta_dot_log[i] = delta_dot_cmd
        delta_ff_log[i] = delta_ff_now
        kappa_log[i] = kappa_now
        Fyf_log[i] = debug["Fyf"]
        Fyr_log[i] = debug["Fyr"]
        af_log[i] = debug["alpha_f"]
        ar_log[i] = debug["alpha_r"]

        Xg_log[i] = Xg
        Yg_log[i] = Yg
        psi_g_log[i] = psi_g

        Xclosest_log[i] = path["x"][closest_idx]
        Yclosest_log[i] = path["y"][closest_idx]
        idx_log[i] = closest_idx

        if closest_idx >= len(path["x"]) - 2:
            X_state = X_state[:i + 1]
            delta_log = delta_log[:i + 1]
            delta_dot_log = delta_dot_log[:i + 1]
            delta_ff_log = delta_ff_log[:i + 1]
            kappa_log = kappa_log[:i + 1]
            Fyf_log = Fyf_log[:i + 1]
            Fyr_log = Fyr_log[:i + 1]
            af_log = af_log[:i + 1]
            ar_log = ar_log[:i + 1]
            Xg_log = Xg_log[:i + 1]
            Yg_log = Yg_log[:i + 1]
            psi_g_log = psi_g_log[:i + 1]
            Xclosest_log = Xclosest_log[:i + 1]
            Yclosest_log = Yclosest_log[:i + 1]
            idx_log = idx_log[:i + 1]
            time = time[:i + 1]
            break

    results = {
        "time": time,
        "state": X_state,
        "delta": delta_log,
        "delta_dot": delta_dot_log,
        "delta_ff": delta_ff_log,
        "kappa": kappa_log,
        "Fyf": Fyf_log,
        "Fyr": Fyr_log,
        "alpha_f": af_log,
        "alpha_r": ar_log,
        "X_global": Xg_log,
        "Y_global": Yg_log,
        "psi_global": psi_g_log,
        "X_ref": path["x"],
        "Y_ref": path["y"],
        "psi_ref": path["psi"],
        "kappa_ref": path["kappa"],
        "s_ref": path["s"],
        "X_closest": Xclosest_log,
        "Y_closest": Yclosest_log,
        "closest_idx": idx_log,
        "solver_statuses": status_log,
        "vx_constant": vx
    }

    results["metrics"] = compute_metrics(results)
    return results


# ============================================================
# Plotting
# ============================================================

def plot_path(results, waypoints=None):
    plt.figure(figsize=(10, 6))
    plt.plot(results["X_ref"], results["Y_ref"], "--", linewidth=2, label="Spline path")
    plt.plot(results["X_global"], results["Y_global"], linewidth=2, label="Vehicle path")
    plt.plot(results["X_closest"], results["Y_closest"], ":", linewidth=1.2, label="Closest path points")

    if waypoints is not None:
        plt.plot(waypoints[:, 0], waypoints[:, 1], "o", label="Waypoints")

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Desired spline path vs vehicle trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()


def plot_reference_path_properties(path):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(path["s"], np.rad2deg(path["psi"]))
    plt.ylabel("Path heading [deg]")
    plt.title("Reference path properties")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(path["s"], path["kappa"])
    plt.xlabel("Arc length s [m]")
    plt.ylabel("Curvature [1/m]")
    plt.grid(True)

    plt.tight_layout()


def plot_signals(results, title_prefix="Dynamic bicycle MPC"):
    t = results["time"]
    X = results["state"]

    ey = X[:, 0]
    epsi = X[:, 1]
    vy = X[:, 2]
    r = X[:, 3]
    delta = X[:, 4]

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, ey)
    plt.ylabel(r"$e_y$ [m]")
    plt.title(f"{title_prefix} - Tracking errors")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, np.rad2deg(epsi))
    plt.xlabel("Time [s]")
    plt.ylabel(r"$e_\psi$ [deg]")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, vy)
    plt.ylabel(r"$v_y$ [m/s]")
    plt.title(f"{title_prefix} - Vehicle lateral states")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, np.rad2deg(r))
    plt.xlabel("Time [s]")
    plt.ylabel(r"$r$ [deg/s]")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, np.rad2deg(delta), label="delta")
    plt.plot(t, np.rad2deg(results["delta_ff"]), "--", label="delta_ff")
    plt.ylabel("Steering [deg]")
    plt.title(f"{title_prefix} - Steering")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, np.rad2deg(results["delta_dot"]))
    plt.ylabel(r"$\dot{\delta}$ [deg/s]")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, np.full_like(t, results["vx_constant"]))
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, np.rad2deg(results["alpha_f"]), label=r"$\alpha_f$")
    plt.plot(t, np.rad2deg(results["alpha_r"]), label=r"$\alpha_r$")
    plt.ylabel("Slip angles [deg]")
    plt.title(f"{title_prefix} - Tire slip angles")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, results["Fyf"], label=r"$F_{yf}$")
    plt.plot(t, results["Fyr"], label=r"$F_{yr}$")
    plt.xlabel("Time [s]")
    plt.ylabel("Lateral force [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, results["kappa"])
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\kappa$ [1/m]")
    plt.title(f"{title_prefix} - Experienced path curvature")
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    vehicle = VehicleParams(
        m=1757.0,
        Iz=3100.0,
        lf=1.23,
        lr=1.49,
        Cf=125000.0,
        Cr=118000.0
    )

    sim = SimParams(
        dt=0.03,
        t_end=18.0,
        vx=7.0,          # constant-speed baseline
        use_rk4=True
    )

    mpc = MPCParams(
        N=60,
        q_ey=1200.0,
        q_epsi=400.0,
        q_vy=5.0,
        q_r=5.0,
        q_delta=1.0,
        r_ddelta=0.2,
        delta_max_deg=27.0,
        delta_rate_max_deg_s=90.0,
        use_curvature_feedforward=True
    )

    path_params = PathParams(
        ds=0.25
    )

    waypoints = np.array([
        [0.0, 0.0],
        [5.0, 2.5],
        [10.0, 10.0],
        [15.0, 10.5],
        [20.0, 5.0],
        [25.0, 2.5],
        [30.0, 0.0],
        [35.0, -2.5],
        [40.0, 0.0]
    ])

    path = generate_spline_path_from_waypoints(waypoints, ds=path_params.ds)

    print("Generated spline path with length:", path["s"][-1])
    print("Max curvature:", np.max(np.abs(path["kappa"])))
    print("Constant speed:", sim.vx, "m/s")

    results = simulate_closed_loop(vehicle, sim, mpc, path)

    print("\nMetrics:")
    for k, v in results["metrics"].items():
        print(f"{k}: {v:.4f}")

    unique_statuses = sorted(set(results["solver_statuses"]))
    print("\nSolver statuses encountered:", unique_statuses)

    plot_reference_path_properties(path)
    plot_path(results, waypoints=waypoints)
    plot_signals(results)
    plt.show()