import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline


# ============================================================
# Utility functions
# ============================================================

def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    return np.unwrap(angles)


# ============================================================
# Parameter containers
# ============================================================

@dataclass
class VehicleParams:
    m: float = 1757.0
    Iz: float = 3100.0
    lf: float = 1.23
    lr: float = 1.49
    Cf: float = 125000.0
    Cr: float = 118000.0
    mu: float = 0.80
    g: float = 9.81

    @property
    def L(self) -> float:
        return self.lf + self.lr


@dataclass
class ControllerParams:
    N: int = 18
    dt: float = 0.1

    q_ey: float = 80.0
    q_epsi: float = 50.0
    q_vy: float = 5.0
    q_r: float = 3.0
    q_delta: float = 0.8
    r_ddelta: float = 3.0
    r_ddelta_smooth: float = 2.0

    terminal_factor_ey: float = 2.0
    terminal_factor_epsi: float = 2.0
    terminal_factor_vy: float = 1.0
    terminal_factor_r: float = 1.0
    terminal_factor_delta: float = 1.0

    delta_max_deg: float = 30.0
    delta_rate_max_deg_s: float = 40.0

    ey_max: float = 8.0
    epsi_max_deg: float = 90.0
    vy_max: float = 6.0
    r_max: float = 1.5

    ipopt_max_iter: int = 1200
    ipopt_tol: float = 1e-4


@dataclass
class ModelParams:
    denom_eps: float = 1e-3
    vx_eps: float = 0.5

    # Fixed for ROS controller: Dugoff
    tire_model: str = "dugoff"

    traction_front_frac: float = 0.6
    braking_front_frac: float = 0.7


@dataclass
class SpeedProfileParams:
    vx_nominal: float = 8.0
    vx_min: float = 4.0
    vx_max: float = 10.0
    ay_max: float = 1.5


# ============================================================
# Path representation
# ============================================================

class SplinePath:
    def __init__(self, x_wp, y_wp, ds=0.5):
        self.x_wp = np.asarray(x_wp, dtype=float)
        self.y_wp = np.asarray(y_wp, dtype=float)

        dx = np.diff(self.x_wp)
        dy = np.diff(self.y_wp)
        s_wp = np.concatenate(([0.0], np.cumsum(np.sqrt(dx**2 + dy**2))))

        if np.any(np.diff(s_wp) <= 0.0):
            raise ValueError("Waypoints must not contain duplicate consecutive points.")

        self.s_wp = s_wp
        self.length = float(s_wp[-1])

        self.cs_x = CubicSpline(s_wp, self.x_wp, bc_type="natural")
        self.cs_y = CubicSpline(s_wp, self.y_wp, bc_type="natural")

        self.s = np.arange(0.0, self.length, ds)
        if len(self.s) == 0 or self.s[-1] < self.length:
            self.s = np.append(self.s, self.length)

        self.x = self.cs_x(self.s)
        self.y = self.cs_y(self.s)

        dx_ds = self.cs_x(self.s, 1)
        dy_ds = self.cs_y(self.s, 1)
        ddx_ds = self.cs_x(self.s, 2)
        ddy_ds = self.cs_y(self.s, 2)

        self.psi = unwrap_angles(np.arctan2(dy_ds, dx_ds))

        denom = (dx_ds**2 + dy_ds**2) ** 1.5 + 1e-9
        self.kappa = (dx_ds * ddy_ds - dy_ds * ddx_ds) / denom

    def interp_x(self, s_val):
        s_val = np.clip(s_val, self.s[0], self.s[-1])
        return float(np.interp(s_val, self.s, self.x))

    def interp_y(self, s_val):
        s_val = np.clip(s_val, self.s[0], self.s[-1])
        return float(np.interp(s_val, self.s, self.y))

    def interp_psi(self, s_val):
        s_val = np.clip(s_val, self.s[0], self.s[-1])
        return float(np.interp(s_val, self.s, self.psi))

    def interp_kappa(self, s_val):
        s_val = np.clip(s_val, self.s[0], self.s[-1])
        return float(np.interp(s_val, self.s, self.kappa))


# ============================================================
# Speed and preview
# ============================================================

def build_curvature_aware_speed_profile(path: SplinePath, speed_par: SpeedProfileParams):
    kappa_abs = np.abs(path.kappa)
    v_curve = np.sqrt(speed_par.ay_max / np.maximum(kappa_abs, 1e-4))
    v_profile = np.minimum(v_curve, speed_par.vx_nominal)
    v_profile = np.clip(v_profile, speed_par.vx_min, speed_par.vx_max)
    return v_profile


def estimate_sdot(ey, epsi, vy, vx, kappa, model_par: ModelParams):
    denom = 1.0 - kappa * ey

    if abs(denom) < model_par.denom_eps:
        denom = np.sign(denom + 1e-9) * model_par.denom_eps

    sdot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / denom
    return max(0.2, float(sdot))


def preview_path_data(path, speed_profile, ax_profile, x, s_current, ctrl_par, model_par):
    ey, epsi, vy = x[0], x[1], x[2]

    vx0 = float(np.interp(np.clip(s_current, path.s[0], path.s[-1]), path.s, speed_profile))
    kappa0 = path.interp_kappa(s_current)

    sdot0 = estimate_sdot(ey, epsi, vy, vx0, kappa0, model_par)

    s_preview = s_current + sdot0 * ctrl_par.dt * np.arange(ctrl_par.N + 1)
    s_preview = np.clip(s_preview, path.s[0], path.s[-1])

    kappa_preview = np.interp(s_preview, path.s, path.kappa)
    vx_preview = np.interp(s_preview, path.s, speed_profile)
    ax_preview = np.interp(s_preview, path.s, ax_profile)

    return s_preview, kappa_preview, vx_preview, ax_preview


# ============================================================
# Tire model
# ============================================================

def split_longitudinal_force(Fx_total, model_par: ModelParams):
    front_frac = ca.if_else(
        Fx_total >= 0.0,
        model_par.traction_front_frac,
        model_par.braking_front_frac,
    )

    return front_frac * Fx_total, (1.0 - front_frac) * Fx_total


def linear_lateral_force(alpha, C_alpha):
    return C_alpha * alpha


def dugoff_lateral_force(alpha, Fz, C_alpha, mu, Fx=0.0):
    eps = 1e-6

    tan_alpha = ca.tan(alpha)
    tan_alpha = ca.fmin(ca.fmax(tan_alpha, -20.0), 20.0)

    denom = 2.0 * ca.sqrt(Fx**2 + (C_alpha * tan_alpha)**2 + eps)
    lam = mu * Fz / (denom + eps)

    f_lam = ca.if_else(lam < 1.0, lam * (2.0 - lam), 1.0)

    return C_alpha * tan_alpha * f_lam


def lateral_force_casadi(alpha, Fz, C_alpha, mu, Fx, model_par: ModelParams):
    if model_par.tire_model != "dugoff":
        raise ValueError(
            f"ROS controller is configured for Dugoff only, but got tire_model='{model_par.tire_model}'."
        )

    return dugoff_lateral_force(alpha, Fz, C_alpha, mu, Fx)


# ============================================================
# Dynamic bicycle model in path-relative error states
# x = [ey, epsi, vy, r, delta]
# u = [delta_dot]
# ============================================================

def continuous_dynamics_casadi(x, u, kappa, vx, ax, veh, model_par):
    ey = x[0]
    epsi = x[1]
    vy = x[2]
    r = x[3]
    delta = x[4]
    ddelta = u[0]

    vx = ca.fmax(vx, model_par.vx_eps)

    m = veh.m
    Iz = veh.Iz
    lf = veh.lf
    lr = veh.lr
    Cf = veh.Cf
    Cr = veh.Cr
    mu = veh.mu
    g0 = veh.g

    Fzf = m * g0 * lr / (lf + lr)
    Fzr = m * g0 * lf / (lf + lr)

    Fx_total = m * ax
    Fx_f, Fx_r = split_longitudinal_force(Fx_total, model_par)

    alpha_f = delta - ca.atan2(vy + lf * r, vx)
    alpha_r = -ca.atan2(vy - lr * r, vx)

    Fyf = lateral_force_casadi(alpha_f, Fzf, Cf, mu, Fx_f, model_par)
    Fyr = lateral_force_casadi(alpha_r, Fzr, Cr, mu, Fx_r, model_par)

    denom = 1.0 - kappa * ey
    denom = ca.if_else(
        ca.fabs(denom) < model_par.denom_eps,
        model_par.denom_eps * ca.sign(denom + 1e-9),
        denom,
    )

    sdot = (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / denom

    ey_dot = vy * ca.cos(epsi) + vx * ca.sin(epsi)
    epsi_dot = r - kappa * sdot
    vy_dot = (Fyf * ca.cos(delta) + Fyr) / m - vx * r
    r_dot = (lf * Fyf * ca.cos(delta) - lr * Fyr) / Iz
    delta_dot = ddelta

    return ca.vertcat(ey_dot, epsi_dot, vy_dot, r_dot, delta_dot)


def rk4_step_casadi(x, u, kappa, vx, ax, veh, ctrl_par, model_par):
    dt = ctrl_par.dt

    k1 = continuous_dynamics_casadi(x, u, kappa, vx, ax, veh, model_par)
    k2 = continuous_dynamics_casadi(x + 0.5 * dt * k1, u, kappa, vx, ax, veh, model_par)
    k3 = continuous_dynamics_casadi(x + 0.5 * dt * k2, u, kappa, vx, ax, veh, model_par)
    k4 = continuous_dynamics_casadi(x + dt * k3, u, kappa, vx, ax, veh, model_par)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================
# NMPC Controller
# ============================================================

class NMPCController:
    def __init__(self, path, veh_ctrl, ctrl_par, model_ctrl):
        self.path = path
        self.veh = veh_ctrl
        self.ctrl_par = ctrl_par
        self.model_par = model_ctrl

        self.delta_max = np.deg2rad(ctrl_par.delta_max_deg)
        self.ddelta_max = np.deg2rad(ctrl_par.delta_rate_max_deg_s)

        self.nx = 5
        self.nu = 1
        self.N = ctrl_par.N

        self.last_sol = None
        self.last_u0 = np.array([0.0], dtype=float)

        self._build_solver()

    def _build_solver(self):
        N = self.N
        nx = self.nx
        L = self.veh.L

        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", 1, N)

        P = ca.SX.sym("P", nx + 3 * (N + 1))

        cost = 0
        g = []

        g.append(X[:, 0] - P[0:nx])

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            kappa_k = P[nx + k]
            vx_k = P[nx + (N + 1) + k]
            ax_k = P[nx + 2 * (N + 1) + k]

            x_next = rk4_step_casadi(
                xk,
                uk,
                kappa_k,
                vx_k,
                ax_k,
                self.veh,
                self.ctrl_par,
                self.model_par,
            )

            g.append(X[:, k + 1] - x_next)

            ey = xk[0]
            epsi = xk[1]
            vy = xk[2]
            r = xk[3]
            delta = xk[4]
            ddelta = uk[0]

            r_ref = vx_k * kappa_k
            delta_ref = ca.atan(L * kappa_k)

            cost += self.ctrl_par.q_ey * ey**2
            cost += self.ctrl_par.q_epsi * epsi**2
            cost += self.ctrl_par.q_vy * vy**2
            cost += self.ctrl_par.q_r * (r - r_ref)**2
            cost += self.ctrl_par.q_delta * (delta - delta_ref)**2
            cost += self.ctrl_par.r_ddelta * ddelta**2

            if k > 0:
                cost += self.ctrl_par.r_ddelta_smooth * (U[0, k] - U[0, k - 1])**2

        xN = X[:, N]
        kappa_N = P[nx + N]
        vx_N = P[nx + (N + 1) + N]

        r_ref_N = vx_N * kappa_N
        delta_ref_N = ca.atan(L * kappa_N)

        cost += self.ctrl_par.terminal_factor_ey * self.ctrl_par.q_ey * xN[0]**2
        cost += self.ctrl_par.terminal_factor_epsi * self.ctrl_par.q_epsi * xN[1]**2
        cost += self.ctrl_par.terminal_factor_vy * self.ctrl_par.q_vy * xN[2]**2
        cost += self.ctrl_par.terminal_factor_r * self.ctrl_par.q_r * (xN[3] - r_ref_N)**2
        cost += self.ctrl_par.terminal_factor_delta * self.ctrl_par.q_delta * (xN[4] - delta_ref_N)**2

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)

        nlp = {
            "x": opt_vars,
            "f": cost,
            "g": g,
            "p": P,
        }

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.ctrl_par.ipopt_max_iter,
            "ipopt.tol": self.ctrl_par.ipopt_tol,
            "ipopt.acceptable_tol": 5e-2,
            "ipopt.acceptable_iter": 3,
            "ipopt.mu_strategy": "adaptive",
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        ey_max = self.ctrl_par.ey_max
        epsi_max = np.deg2rad(self.ctrl_par.epsi_max_deg)
        vy_max = self.ctrl_par.vy_max
        r_max = self.ctrl_par.r_max

        lbx = []
        ubx = []

        for _ in range(N + 1):
            lbx += [-ey_max, -epsi_max, -vy_max, -r_max, -self.delta_max]
            ubx += [ey_max, epsi_max, vy_max, r_max, self.delta_max]

        for _ in range(N):
            lbx += [-self.ddelta_max]
            ubx += [self.ddelta_max]

        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)
        self.lbg = np.zeros(g.shape[0], dtype=float)
        self.ubg = np.zeros(g.shape[0], dtype=float)

    def _make_initial_guess(self, x0, kappa_preview, vx_preview):
        nX = self.nx * (self.N + 1)
        L = self.veh.L

        if self.last_sol is not None:
            X_prev = self.last_sol[:nX].reshape((self.N + 1, self.nx))
            U_prev = self.last_sol[nX:].reshape((self.N, self.nu))

            X_guess = np.zeros_like(X_prev)
            U_guess = np.zeros_like(U_prev)

            X_guess[:-1] = X_prev[1:]
            U_guess[:-1] = U_prev[1:]

            X_guess[0] = x0

            kappa_last = float(kappa_preview[-1])
            vx_last = float(vx_preview[-1])

            X_guess[-1] = np.array([
                0.0,
                0.0,
                0.0,
                vx_last * kappa_last,
                np.arctan(L * kappa_last),
            ])

            U_guess[-1] = U_prev[-1]

            return np.concatenate([X_guess.reshape(-1), U_guess.reshape(-1)])

        X_guess = np.zeros((self.N + 1, self.nx))
        U_guess = np.zeros((self.N, self.nu))

        ey0, epsi0, _, _, _ = x0
        decay = np.linspace(1.0, 0.0, self.N + 1)

        delta_refs = np.zeros(self.N + 1)

        for k in range(self.N + 1):
            kappa_k = float(kappa_preview[k])
            vx_k = float(vx_preview[k])

            X_guess[k, 0] = decay[k] * ey0
            X_guess[k, 1] = decay[k] * epsi0
            X_guess[k, 2] = 0.0
            X_guess[k, 3] = vx_k * kappa_k
            X_guess[k, 4] = np.arctan(L * kappa_k)

            delta_refs[k] = X_guess[k, 4]

        X_guess[0] = x0

        for k in range(self.N):
            U_guess[k, 0] = np.clip(
                (delta_refs[k + 1] - delta_refs[k]) / self.ctrl_par.dt,
                -self.ddelta_max,
                self.ddelta_max,
            )

        return np.concatenate([X_guess.reshape(-1), U_guess.reshape(-1)])

    def solve(self, x0, kappa_preview, vx_preview, ax_preview):
        p = np.concatenate([x0, kappa_preview, vx_preview, ax_preview])
        w0 = self._make_initial_guess(x0, kappa_preview, vx_preview)

        t0 = time.perf_counter()

        sol = self.solver(
            x0=w0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p,
        )

        solve_time = time.perf_counter() - t0
        stats = self.solver.stats()

        return_status = stats.get("return_status", "UNKNOWN")
        success = bool(stats.get("success", False))

        acceptable_statuses = [
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
        ]

        if (not success) and (return_status not in acceptable_statuses):
            raise RuntimeError(f"NMPC failed. IPOPT status: {return_status}")

        w_opt = np.array(sol["x"]).flatten()
        self.last_sol = w_opt.copy()

        nX = self.nx * (self.N + 1)
        X_opt = w_opt[:nX].reshape((self.N + 1, self.nx))
        U_opt = w_opt[nX:].reshape((self.N, self.nu))

        self.last_u0 = U_opt[0].copy()

        soft_failure = (not success) and (return_status in acceptable_statuses)

        return X_opt, U_opt, solve_time, stats.get("iter_count", np.nan), return_status, soft_failure