import time
import requests
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import csv
import os
import traceback
from shapely import wkt
from scipy.interpolate import CubicSpline
from dataclasses import dataclass


# ============================================================
# Utility functions
# ============================================================

def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    return np.unwrap(angles)


def safe_trapezoid(y, x):
    if len(y) < 2 or len(x) < 2:
        return np.nan
    return np.trapezoid(y, x)


def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)


def safe_nanmax(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmax(x)


def safe_nanpercentile(x, p):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanpercentile(x, p)


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

    # Controller: "linear" or "dugoff"
    # Plant:      "linear", "dugoff", or "pacejka"
    tire_model: str = "dugoff"

    traction_front_frac: float = 0.6
    braking_front_frac: float = 0.7

    # Simplified Pacejka shape parameters.
    # IMPORTANT:
    # D is not fixed anymore. It is computed as D = mu * Fz.
    # B is computed automatically so that the initial slope equals C_alpha.
    pacejka_Cf_shape: float = 1.219
    pacejka_Ef: float = -1.02

    pacejka_Cr_shape: float = 1.207
    pacejka_Er: float = -0.65


@dataclass
class SpeedProfileParams:
    vx_nominal: float = 7.0
    vx_min: float = 3.0
    vx_max: float = 10.0
    ay_max: float = 1.5


@dataclass
class SimulationParams:
    dt: float = 0.1
    sim_time: float = 80.0


# ============================================================
# NVDB path generation
# ============================================================

def sort_points_from_start(points, start_point):
    points = np.array(points, dtype=float)
    sorted_points = [np.array(start_point, dtype=float)]
    remaining = points.copy()

    d = np.linalg.norm(remaining - start_point, axis=1)
    remaining = np.delete(remaining, np.argmin(d), axis=0)

    while len(remaining) > 0:
        last = sorted_points[-1]
        d = np.linalg.norm(remaining - last, axis=1)
        idx = np.argmin(d)
        sorted_points.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)

    return np.array(sorted_points)


def fetch_nvdb_waypoints(
    vegsystemreferanse="KV1253",
    kommune="3201",
    max_points=300,
):
    url = "https://nvdbapiles.atlas.vegvesen.no/vegnett/api/v4/veglenkesekvenser"

    params = {
        "vegsystemreferanse": vegsystemreferanse,
        "kommune": kommune,
    }

    headers = {
        "X-Client": "ntnu-masteroppgave-nmpc"
    }

    resp = requests.get(url, params=params, headers=headers, timeout=25)
    resp.raise_for_status()
    data = resp.json()

    if "objekter" not in data or len(data["objekter"]) == 0:
        raise ValueError("Ingen NVDB-objekter funnet.")

    alle_segmenter = []

    for item in data["objekter"]:
        for veglenke in item.get("veglenker", []):
            geom = veglenke.get("geometri", {}).get("wkt", "")

            if not geom or not geom.startswith("LINESTRING"):
                continue

            startpos = veglenke.get("startposisjon", 0.0)
            sluttpos = veglenke.get("sluttposisjon", 0.0)
            retning = veglenke.get("retning", "").upper()
            reversert = veglenke.get("reversert", False)

            overlap = False
            for s_start, s_slutt, _ in alle_segmenter:
                if not (sluttpos <= s_start or startpos >= s_slutt):
                    overlap = True
                    break

            if overlap:
                continue

            line = wkt.loads(geom)
            x, y = line.xy
            segment = list(zip(x, y))

            if retning == "MOT" or reversert:
                segment = segment[::-1]

            alle_segmenter.append((startpos, sluttpos, segment))

    if len(alle_segmenter) == 0:
        raise ValueError("Ingen gyldige LINESTRING-segmenter funnet fra NVDB.")

    alle_segmenter.sort(key=lambda s: s[0])

    startpunkt = alle_segmenter[0][2][0]
    alle_punkter = [p for _, _, seg in alle_segmenter for p in seg]

    sorted_pts_global = sort_points_from_start(alle_punkter, startpunkt)

    x = sorted_pts_global[:, 0].copy()
    y = sorted_pts_global[:, 1].copy()

    x -= startpunkt[0]
    y -= startpunkt[1]

    dist = np.zeros(len(x))
    for i in range(1, len(x)):
        dist[i] = dist[i - 1] + np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])

    mask = np.diff(dist, prepend=-1.0) > 1e-6
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        raise ValueError("For få gyldige NVDB-punkter til spline.")

    if max_points is not None and len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]
        y = y[idx]

    return x, y


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

    def global_from_error_state(self, s_val, ey, epsi):
        x_ref = self.interp_x(s_val)
        y_ref = self.interp_y(s_val)
        psi_ref = self.interp_psi(s_val)

        x = x_ref - ey * np.sin(psi_ref)
        y = y_ref + ey * np.cos(psi_ref)
        psi = wrap_angle(psi_ref + epsi)

        return x, y, psi


# ============================================================
# Speed profile
# ============================================================

def build_curvature_aware_speed_profile(path: SplinePath, speed_par: SpeedProfileParams):
    kappa_abs = np.abs(path.kappa)
    v_curve = np.sqrt(speed_par.ay_max / np.maximum(kappa_abs, 1e-4))
    v_profile = np.minimum(v_curve, speed_par.vx_nominal)
    v_profile = np.clip(v_profile, speed_par.vx_min, speed_par.vx_max)
    return v_profile


def build_longitudinal_acc_profile(path: SplinePath, speed_profile: np.ndarray):
    if len(path.s) < 3:
        return np.zeros_like(speed_profile)

    dv_ds = np.gradient(speed_profile, path.s, edge_order=2)
    ax_profile = speed_profile * dv_ds
    return ax_profile


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
# Tire models
# ============================================================

def split_longitudinal_force(Fx_total, model_par: ModelParams):
    if isinstance(Fx_total, (float, np.floating, int)):
        front_frac = model_par.traction_front_frac if Fx_total >= 0.0 else model_par.braking_front_frac
        return front_frac * Fx_total, (1.0 - front_frac) * Fx_total

    front_frac = ca.if_else(
        Fx_total >= 0.0,
        model_par.traction_front_frac,
        model_par.braking_front_frac
    )
    return front_frac * Fx_total, (1.0 - front_frac) * Fx_total


def linear_lateral_force(alpha, C_alpha):
    return C_alpha * alpha


def linear_lateral_force_np(alpha, C_alpha):
    return C_alpha * alpha


def dugoff_lateral_force(alpha, Fz, C_alpha, mu, Fx=0.0):
    eps = 1e-6

    tan_alpha = ca.tan(alpha)
    tan_alpha = ca.fmin(ca.fmax(tan_alpha, -20.0), 20.0)

    denom = 2.0 * ca.sqrt(Fx**2 + (C_alpha * tan_alpha)**2 + eps)
    lam = mu * Fz / (denom + eps)
    f_lam = ca.if_else(lam < 1.0, lam * (2.0 - lam), 1.0)

    return C_alpha * tan_alpha * f_lam


def dugoff_lateral_force_np(alpha, Fz, C_alpha, mu, Fx=0.0):
    eps = 1e-6

    tan_alpha = np.tan(alpha)
    tan_alpha = np.clip(tan_alpha, -20.0, 20.0)

    denom = 2.0 * np.sqrt(Fx**2 + (C_alpha * tan_alpha)**2 + eps)
    lam = mu * Fz / (denom + eps)
    f_lam = lam * (2.0 - lam) if lam < 1.0 else 1.0

    return C_alpha * tan_alpha * f_lam


def pacejka_lateral_force_np(alpha_rad, Fz, C_alpha, mu, C_shape, E):
    """
    Simplified pure lateral Pacejka Magic Formula.

    The classical form is
        Fy = D sin(C atan(B alpha - E(B alpha - atan(B alpha))))

    Here:
        D = mu * Fz

    B is computed so that the initial slope around alpha = 0 equals C_alpha.
    Since alpha is internally converted to degrees, B has unit 1/deg:
        C_alpha [N/rad] = B[1/deg] * C * D[N] * 180/pi
        B = C_alpha / (C * D * 180/pi)
    """

    alpha_deg = np.rad2deg(alpha_rad)

    D = mu * Fz
    if D <= 1e-6:
        return 0.0

    B = C_alpha / (C_shape * D * (180.0 / np.pi))

    return D * np.sin(
        C_shape * np.arctan(
            B * alpha_deg
            - E * (B * alpha_deg - np.arctan(B * alpha_deg))
        )
    )


def lateral_force_casadi(alpha, Fz, C_alpha, mu, Fx, model_par: ModelParams):
    if model_par.tire_model == "linear":
        return linear_lateral_force(alpha, C_alpha)

    if model_par.tire_model == "dugoff":
        return dugoff_lateral_force(alpha, Fz, C_alpha, mu, Fx)

    raise ValueError(
        f"Controller tire model '{model_par.tire_model}' is not supported. "
        "Use 'linear' or 'dugoff' for the controller."
    )


def lateral_force_np(alpha, Fz, C_alpha, mu, Fx, model_par: ModelParams, axle="front"):
    if model_par.tire_model == "linear":
        return linear_lateral_force_np(alpha, C_alpha)

    if model_par.tire_model == "dugoff":
        return dugoff_lateral_force_np(alpha, Fz, C_alpha, mu, Fx)

    if model_par.tire_model == "pacejka":
        if axle == "front":
            return pacejka_lateral_force_np(
                alpha_rad=alpha,
                Fz=Fz,
                C_alpha=C_alpha,
                mu=mu,
                C_shape=model_par.pacejka_Cf_shape,
                E=model_par.pacejka_Ef,
            )

        if axle == "rear":
            return pacejka_lateral_force_np(
                alpha_rad=alpha,
                Fz=Fz,
                C_alpha=C_alpha,
                mu=mu,
                C_shape=model_par.pacejka_Cr_shape,
                E=model_par.pacejka_Er,
            )

        raise ValueError(f"Unknown axle: {axle}")

    raise ValueError(f"Unknown plant tire model: {model_par.tire_model}")


# ============================================================
# Continuous dynamics
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
        denom
    )

    sdot = (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / denom

    ey_dot = vy * ca.cos(epsi) + vx * ca.sin(epsi)
    epsi_dot = r - kappa * sdot
    vy_dot = (Fyf * ca.cos(delta) + Fyr) / m - vx * r
    r_dot = (lf * Fyf * ca.cos(delta) - lr * Fyr) / Iz
    delta_dot = ddelta

    return ca.vertcat(ey_dot, epsi_dot, vy_dot, r_dot, delta_dot)


def continuous_dynamics_np(x, u, kappa, vx, ax, veh, model_par):
    ey, epsi, vy, r, delta = x
    ddelta = float(u[0])

    vx = max(float(vx), model_par.vx_eps)

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

    Fx_total = m * float(ax)
    Fx_f, Fx_r = split_longitudinal_force(Fx_total, model_par)

    alpha_f = delta - np.arctan2(vy + lf * r, vx)
    alpha_r = -np.arctan2(vy - lr * r, vx)

    Fyf = lateral_force_np(alpha_f, Fzf, Cf, mu, Fx_f, model_par, axle="front")
    Fyr = lateral_force_np(alpha_r, Fzr, Cr, mu, Fx_r, model_par, axle="rear")

    denom = 1.0 - kappa * ey
    if abs(denom) < model_par.denom_eps:
        denom = np.sign(denom + 1e-9) * model_par.denom_eps

    sdot = (vx * np.cos(epsi) - vy * np.sin(epsi)) / denom

    ey_dot = vy * np.cos(epsi) + vx * np.sin(epsi)
    epsi_dot = r - kappa * sdot
    vy_dot = (Fyf * np.cos(delta) + Fyr) / m - vx * r
    r_dot = (lf * Fyf * np.cos(delta) - lr * Fyr) / Iz
    delta_dot = ddelta

    dx = np.array([ey_dot, epsi_dot, vy_dot, r_dot, delta_dot], dtype=float)
    return dx, float(sdot)


# ============================================================
# RK4
# ============================================================

def rk4_step_casadi(x, u, kappa, vx, ax, veh, ctrl_par, model_par):
    dt = ctrl_par.dt

    k1 = continuous_dynamics_casadi(x, u, kappa, vx, ax, veh, model_par)
    k2 = continuous_dynamics_casadi(x + 0.5 * dt * k1, u, kappa, vx, ax, veh, model_par)
    k3 = continuous_dynamics_casadi(x + 0.5 * dt * k2, u, kappa, vx, ax, veh, model_par)
    k4 = continuous_dynamics_casadi(x + dt * k3, u, kappa, vx, ax, veh, model_par)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_step_np(x, u, kappa, vx, ax, veh, sim_par, model_par):
    dt = sim_par.dt

    k1, sdot1 = continuous_dynamics_np(x, u, kappa, vx, ax, veh, model_par)
    k2, sdot2 = continuous_dynamics_np(x + 0.5 * dt * k1, u, kappa, vx, ax, veh, model_par)
    k3, sdot3 = continuous_dynamics_np(x + 0.5 * dt * k2, u, kappa, vx, ax, veh, model_par)
    k4, sdot4 = continuous_dynamics_np(x + dt * k3, u, kappa, vx, ax, veh, model_par)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    x_next[1] = wrap_angle(x_next[1])

    sdot = (sdot1 + 2*sdot2 + 2*sdot3 + sdot4) / 6.0

    return x_next, float(sdot)


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
                xk, uk, kappa_k, vx_k, ax_k,
                self.veh, self.ctrl_par, self.model_par
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

        nlp = {"x": opt_vars, "f": cost, "g": g, "p": P}

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
            ubx += [ ey_max,  epsi_max,  vy_max,  r_max,  self.delta_max]

        for _ in range(N):
            lbx += [-self.ddelta_max]
            ubx += [ self.ddelta_max]

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
                np.arctan(L * kappa_last)
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
            X_guess[k, 4] = np.arctan(self.veh.L * kappa_k)

            delta_refs[k] = X_guess[k, 4]

        X_guess[0] = x0

        for k in range(self.N):
            U_guess[k, 0] = np.clip(
                (delta_refs[k + 1] - delta_refs[k]) / self.ctrl_par.dt,
                -self.ddelta_max,
                self.ddelta_max
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
            p=p
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


# ============================================================
# Simulation
# ============================================================

def simulate_step(x, s_current, u, path, speed_profile, ax_profile, veh_plant, sim_par, model_plant):
    s_current = np.clip(s_current, path.s[0], path.s[-1])

    kappa = path.interp_kappa(s_current)
    vx = float(np.interp(s_current, path.s, speed_profile))
    ax = float(np.interp(s_current, path.s, ax_profile))

    x_next, sdot = rk4_step_np(x, u, kappa, vx, ax, veh_plant, sim_par, model_plant)

    s_next = s_current + sim_par.dt * max(0.0, sdot)
    s_next = np.clip(s_next, path.s[0], path.s[-1])

    return x_next, s_next, vx, ax


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    t_log,
    ey_log,
    epsi_log,
    delta_log,
    ddelta_log,
    kappa_log,
    vx_log,
    solve_time_log,
    iter_log,
    hard_fail_log,
    soft_fail_log,
    s_log,
    path,
):
    abs_ey = np.abs(ey_log)
    abs_epsi = np.abs(epsi_log)
    abs_delta = np.abs(delta_log)
    abs_ddelta = np.abs(ddelta_log)
    abs_kappa = np.abs(kappa_log)

    duration = t_log[-1] - t_log[0] if len(t_log) > 1 else 0.0
    progress = s_log[-1] if len(s_log) > 0 else 0.0
    completion_pct = 100.0 * progress / path.length if path.length > 0 else np.nan

    valid_solve = solve_time_log[~np.isnan(solve_time_log)]
    valid_iters = iter_log[~np.isnan(iter_log)]

    # Same CNXTE definition as in the project thesis:
    # pointwise CNXTE = |cross-track error| * |curvature|
    cnxte_signal = abs_ey * abs_kappa

    metrics = {
        "rms_ey_m": np.sqrt(safe_nanmean(ey_log**2)),
        "max_abs_ey_m": safe_nanmax(abs_ey),
        "p95_abs_ey_m": safe_nanpercentile(abs_ey, 95),
        "iae_ey_m_s": safe_trapezoid(abs_ey, t_log),

        "rms_epsi_deg": np.rad2deg(np.sqrt(safe_nanmean(epsi_log**2))),
        "max_abs_epsi_deg": np.rad2deg(safe_nanmax(abs_epsi)),
        "iae_epsi_rad_s": safe_trapezoid(abs_epsi, t_log),
        "iae_epsi_deg_s": np.rad2deg(safe_trapezoid(abs_epsi, t_log)),

        "max_abs_delta_deg": np.rad2deg(safe_nanmax(abs_delta)),
        "rms_delta_deg": np.rad2deg(np.sqrt(safe_nanmean(delta_log**2))),
        "max_abs_ddelta_deg_s": np.rad2deg(safe_nanmax(abs_ddelta)),
        "rms_ddelta_deg_s": np.rad2deg(np.sqrt(safe_nanmean(ddelta_log**2))),
        "iadc_rad": safe_trapezoid(abs_ddelta, t_log),
        "iadc_deg": np.rad2deg(safe_trapezoid(abs_ddelta, t_log)),

        # CNXTE exactly as in the project thesis: mean, 95th percentile and max
        "mean_cnxte": safe_nanmean(cnxte_signal),
        "p95_cnxte": safe_nanpercentile(cnxte_signal, 95),
        "max_cnxte": safe_nanmax(cnxte_signal),

        "mean_abs_kappa": safe_nanmean(abs_kappa),
        "max_abs_kappa": safe_nanmax(abs_kappa),

        "mean_vx_m_s": safe_nanmean(vx_log),
        "min_vx_m_s": np.nanmin(vx_log) if len(vx_log) > 0 else np.nan,
        "max_vx_m_s": np.nanmax(vx_log) if len(vx_log) > 0 else np.nan,
        "duration_s": duration,
        "progress_m": progress,
        "completion_pct": completion_pct,

        "avg_solve_ms": 1000.0 * np.mean(valid_solve) if len(valid_solve) > 0 else np.nan,
        "max_solve_ms": 1000.0 * np.max(valid_solve) if len(valid_solve) > 0 else np.nan,
        "avg_ipopt_iter": np.mean(valid_iters) if len(valid_iters) > 0 else np.nan,
        "max_ipopt_iter": np.max(valid_iters) if len(valid_iters) > 0 else np.nan,
        "hard_failures": int(np.sum(hard_fail_log)),
        "soft_warnings": int(np.sum(soft_fail_log)),
    }

    return metrics


def print_metrics_table(metrics):
    print("\n" + "=" * 80)
    print("NMPC PERFORMANCE METRICS")
    print("=" * 80)

    print("Tracking performance")
    print(f"  RMS e_y:                  {metrics['rms_ey_m']:10.4f} m")
    print(f"  Max |e_y|:                {metrics['max_abs_ey_m']:10.4f} m")
    print(f"  95% |e_y|:                {metrics['p95_abs_ey_m']:10.4f} m")
    print(f"  IAE e_y:                  {metrics['iae_ey_m_s']:10.4f} m s")
    print(f"  RMS e_psi:                {metrics['rms_epsi_deg']:10.4f} deg")
    print(f"  Max |e_psi|:              {metrics['max_abs_epsi_deg']:10.4f} deg")
    print(f"  IAE e_psi:                {metrics['iae_epsi_rad_s']:10.4f} rad s")
    print(f"  IAE e_psi:                {metrics['iae_epsi_deg_s']:10.4f} deg s")

    print("\nControl effort")
    print(f"  Max |delta|:              {metrics['max_abs_delta_deg']:10.4f} deg")
    print(f"  RMS delta:                {metrics['rms_delta_deg']:10.4f} deg")
    print(f"  Max |delta_dot|:          {metrics['max_abs_ddelta_deg_s']:10.4f} deg/s")
    print(f"  RMS delta_dot:            {metrics['rms_ddelta_deg_s']:10.4f} deg/s")
    print(f"  IADC:                     {metrics['iadc_rad']:10.4f} rad")
    print(f"  IADC:                     {metrics['iadc_deg']:10.4f} deg")

    print("\nCurvature-aware metrics")
    print(f"  Mean CNXTE:               {metrics['mean_cnxte']:10.6f}")
    print(f"  95% CNXTE:                {metrics['p95_cnxte']:10.6f}")
    print(f"  Max CNXTE:                {metrics['max_cnxte']:10.6f}")
    print(f"  Mean |kappa|:             {metrics['mean_abs_kappa']:10.6f} 1/m")
    print(f"  Max |kappa|:              {metrics['max_abs_kappa']:10.6f} 1/m")

    print("\nSpeed and progress")
    print(f"  Mean v_x:                 {metrics['mean_vx_m_s']:10.4f} m/s")
    print(f"  Min v_x:                  {metrics['min_vx_m_s']:10.4f} m/s")
    print(f"  Max v_x:                  {metrics['max_vx_m_s']:10.4f} m/s")
    print(f"  Duration:                 {metrics['duration_s']:10.4f} s")
    print(f"  Progress:                 {metrics['progress_m']:10.4f} m")
    print(f"  Completion:               {metrics['completion_pct']:10.2f} %")

    print("\nSolver performance")
    print(f"  Avg solve time:           {metrics['avg_solve_ms']:10.4f} ms")
    print(f"  Max solve time:           {metrics['max_solve_ms']:10.4f} ms")
    print(f"  Avg IPOPT iterations:     {metrics['avg_ipopt_iter']:10.4f}")
    print(f"  Max IPOPT iterations:     {metrics['max_ipopt_iter']:10.4f}")
    print(f"  Hard solver failures:     {metrics['hard_failures']:10d}")
    print(f"  Soft solver warnings:     {metrics['soft_warnings']:10d}")

def print_csv_and_latex_rows(metrics):
    headers = [
        "RMS ey [m]",
        "Max |ey| [m]",
        "P95 |ey| [m]",
        "IAE ey [m s]",
        "RMS epsi [deg]",
        "IAE epsi [rad s]",
        "Max |delta| [deg]",
        "Max |ddelta| [deg/s]",
        "IADC [rad]",
        "Mean CNXTE",
        "P95 CNXTE",
        "Max CNXTE",
        "Avg solve [ms]",
        "Max solve [ms]",
        "Hard fails",
        "Soft warns",
    ]

    values = [
        f"{metrics['rms_ey_m']:.4f}",
        f"{metrics['max_abs_ey_m']:.4f}",
        f"{metrics['p95_abs_ey_m']:.4f}",
        f"{metrics['iae_ey_m_s']:.4f}",
        f"{metrics['rms_epsi_deg']:.3f}",
        f"{metrics['iae_epsi_rad_s']:.4f}",
        f"{metrics['max_abs_delta_deg']:.2f}",
        f"{metrics['max_abs_ddelta_deg_s']:.2f}",
        f"{metrics['iadc_rad']:.4f}",
        f"{metrics['mean_cnxte']:.6f}",
        f"{metrics['p95_cnxte']:.6f}",
        f"{metrics['max_cnxte']:.6f}",
        f"{metrics['avg_solve_ms']:.2f}",
        f"{metrics['max_solve_ms']:.2f}",
        str(metrics["hard_failures"]),
        str(metrics["soft_warnings"]),
    ]

    print("\n" + "=" * 80)
    print("CSV HEADER")
    print("=" * 80)
    print(",".join(headers))

    print("\n" + "=" * 80)
    print("CSV ROW")
    print("=" * 80)
    print(",".join(values))

    print("\n" + "=" * 80)
    print("LATEX TABLE ROW")
    print("=" * 80)

    latex_values = [
        f"{metrics['rms_ey_m']:.4f}",
        f"{metrics['max_abs_ey_m']:.4f}",
        f"{metrics['p95_abs_ey_m']:.4f}",
        f"{metrics['rms_epsi_deg']:.3f}",
        f"{metrics['iae_epsi_rad_s']:.4f}",
        f"{metrics['max_abs_ddelta_deg_s']:.2f}",
        f"{metrics['iadc_rad']:.4f}",
        f"{metrics['mean_cnxte']:.6f}",
        f"{metrics['p95_cnxte']:.6f}",
        f"{metrics['max_cnxte']:.6f}",
        f"{metrics['avg_solve_ms']:.2f}",
        f"{metrics['max_solve_ms']:.2f}",
        str(metrics["soft_warnings"]),
    ]

    print(" & ".join(latex_values) + r" \\")


# ============================================================
# Pacejka parameter reporting helper
# ============================================================

def compute_pacejka_report_values(veh: VehicleParams, model_par: ModelParams):
    Fzf = veh.m * veh.g * veh.lr / veh.L
    Fzr = veh.m * veh.g * veh.lf / veh.L

    Df = veh.mu * Fzf
    Dr = veh.mu * Fzr

    Bf = veh.Cf / (model_par.pacejka_Cf_shape * Df * (180.0 / np.pi)) if Df > 1e-6 else np.nan
    Br = veh.Cr / (model_par.pacejka_Cr_shape * Dr * (180.0 / np.pi)) if Dr > 1e-6 else np.nan

    return Fzf, Fzr, Df, Dr, Bf, Br


# ============================================================
# Experiment runner
# ============================================================

@dataclass
class ExperimentConfig:
    name: str
    group: str

    path_type: str = "nvdb"
    synthetic_path: str = ""

    vegsystemreferanse: str = "KV1253"
    kommune: str = "3201"

    tire_model_ctrl: str = "dugoff"
    tire_model_plant: str = "pacejka"

    stiffness_scale_ctrl: float = 1.0
    stiffness_scale_plant: float = 1.0

    mu_ctrl: float = 0.8
    mu_plant: float = 0.8

    vx_nominal: float = 7.0
    vx_min: float = 3.0
    vx_max: float = 10.0
    ay_max: float = 1.5

    N: int = 18
    dt: float = 0.1
    sim_time: float = 80.0

    use_longitudinal_acc_profile: bool = False
    max_points: int = 300


def create_synthetic_waypoints(path_name):
    if path_name == "straight":
        x = np.linspace(0.0, 120.0, 13)
        y = np.zeros_like(x)
        return x, y

    if path_name == "mild":
        x = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120], dtype=float)
        y = np.array([0, 0, 3, 8, 10, 8, 3, 0, 0], dtype=float)
        return x, y

    if path_name == "sharp":
        x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 85, 100], dtype=float)
        y = np.array([0, 0, 2, 12, 28, 38, 35, 25, 18, 18], dtype=float)
        return x, y

    if path_name == "s_curve":
        x = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120], dtype=float)
        y = np.array([0, 0, 8, 15, 8, 0, -8, -15, -8], dtype=float)
        return x, y

    if path_name == "constant_radius":
        R = 35.0
        theta = np.linspace(0.0, np.pi / 2.0, 20)
        x = R * np.sin(theta)
        y = R * (1.0 - np.cos(theta))
        return x, y

    raise ValueError(f"Unknown synthetic path: {path_name}")


def build_experiment_list():
    experiments = []

    for path_name in ["straight", "constant_radius", "mild"]:
        experiments.append(ExperimentConfig(
            name=f"verification_{path_name}_dugoff_dugoff_v7_mu08",
            group="verification",
            path_type="synthetic",
            synthetic_path=path_name,
            tire_model_ctrl="dugoff",
            tire_model_plant="dugoff",
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=18,
        ))

    for path_name in ["mild", "sharp", "s_curve"]:
        experiments.append(ExperimentConfig(
            name=f"geometry_{path_name}_dugoff_pacejka_v7_mu08",
            group="geometry",
            path_type="synthetic",
            synthetic_path=path_name,
            tire_model_ctrl="dugoff",
            tire_model_plant="pacejka",
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=18,
        ))

    for road in ["KV1253", "KV1167"]:
        experiments.append(ExperimentConfig(
            name=f"geometry_{road}_dugoff_pacejka_v7_mu08",
            group="geometry",
            path_type="nvdb",
            vegsystemreferanse=road,
            kommune="3201",
            tire_model_ctrl="dugoff",
            tire_model_plant="pacejka",
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=18,
        ))

    tire_tests = [
        ("linear", "linear"),
        ("dugoff", "dugoff"),
        ("linear", "pacejka"),
        ("dugoff", "pacejka"),
    ]

    for ctrl_model, plant_model in tire_tests:
        experiments.append(ExperimentConfig(
            name=f"tiremodel_KV1167_{ctrl_model}_{plant_model}_v7_mu08",
            group="tire_model",
            path_type="nvdb",
            vegsystemreferanse="KV1167",
            kommune="3201",
            tire_model_ctrl=ctrl_model,
            tire_model_plant=plant_model,
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=18,
        ))

    for vx in [4.0, 6.0, 8.0, 10.0]:
        experiments.append(ExperimentConfig(
            name=f"speed_KV1167_dugoff_pacejka_v{vx:.0f}_mu08",
            group="speed",
            path_type="nvdb",
            vegsystemreferanse="KV1167",
            kommune="3201",
            tire_model_ctrl="dugoff",
            tire_model_plant="pacejka",
            vx_nominal=vx,
            vx_min=3.0,
            vx_max=max(10.0, vx),
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=18,
        ))

    for mu_plant in [0.8, 0.6, 0.4, 0.3]:
        experiments.append(ExperimentConfig(
            name=f"friction_KV1167_dugoff_pacejka_v7_muplant{mu_plant:.1f}",
            group="friction",
            path_type="nvdb",
            vegsystemreferanse="KV1167",
            kommune="3201",
            tire_model_ctrl="dugoff",
            tire_model_plant="pacejka",
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=mu_plant,
            N=18,
        ))

    for scale in [0.7, 1.0, 1.3]:
        experiments.append(ExperimentConfig(
            name=f"stiffness_KV1167_dugoff_pacejka_v7_scaleplant{scale:.1f}",
            group="stiffness",
            path_type="nvdb",
            vegsystemreferanse="KV1167",
            kommune="3201",
            tire_model_ctrl="dugoff",
            tire_model_plant="pacejka",
            stiffness_scale_ctrl=1.0,
            stiffness_scale_plant=scale,
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=18,
        ))

    for N in [12, 18, 25]:
        experiments.append(ExperimentConfig(
            name=f"solver_KV1167_dugoff_pacejka_N{N}",
            group="solver",
            path_type="nvdb",
            vegsystemreferanse="KV1167",
            kommune="3201",
            tire_model_ctrl="dugoff",
            tire_model_plant="pacejka",
            vx_nominal=7.0,
            mu_ctrl=0.8,
            mu_plant=0.8,
            N=N,
        ))

    return experiments


def build_path_for_experiment(cfg, path_cache):
    if cfg.path_type == "synthetic":
        x_wp, y_wp = create_synthetic_waypoints(cfg.synthetic_path)
        return x_wp, y_wp, f"SYNTHETIC_{cfg.synthetic_path}"

    if cfg.path_type == "nvdb":
        cache_key = (cfg.vegsystemreferanse, cfg.kommune, cfg.max_points)

        if cache_key not in path_cache:
            path_cache[cache_key] = fetch_nvdb_waypoints(
                vegsystemreferanse=cfg.vegsystemreferanse,
                kommune=cfg.kommune,
                max_points=cfg.max_points,
            )

        x_wp, y_wp = path_cache[cache_key]
        return x_wp, y_wp, f"NVDB_{cfg.vegsystemreferanse}"

    raise ValueError(f"Unknown path_type: {cfg.path_type}")


def run_experiment(cfg, path_cache):
    ctrl_par = ControllerParams(N=cfg.N, dt=cfg.dt)
    sim_par = SimulationParams(dt=cfg.dt, sim_time=cfg.sim_time)

    speed_par = SpeedProfileParams(
        vx_nominal=cfg.vx_nominal,
        vx_min=cfg.vx_min,
        vx_max=cfg.vx_max,
        ay_max=cfg.ay_max,
    )

    model_ctrl = ModelParams(tire_model=cfg.tire_model_ctrl)
    model_plant = ModelParams(tire_model=cfg.tire_model_plant)

    veh_ctrl = VehicleParams(
        Cf=125000.0 * cfg.stiffness_scale_ctrl,
        Cr=118000.0 * cfg.stiffness_scale_ctrl,
        mu=cfg.mu_ctrl,
    )

    veh_plant = VehicleParams(
        Cf=125000.0 * cfg.stiffness_scale_plant,
        Cr=118000.0 * cfg.stiffness_scale_plant,
        mu=cfg.mu_plant,
    )

    x_wp, y_wp, path_name = build_path_for_experiment(cfg, path_cache)

    path = SplinePath(x_wp, y_wp, ds=0.5)
    speed_profile = build_curvature_aware_speed_profile(path, speed_par)

    if cfg.use_longitudinal_acc_profile:
        ax_profile = build_longitudinal_acc_profile(path, speed_profile)
    else:
        ax_profile = np.zeros_like(speed_profile)

    controller = NMPCController(path, veh_ctrl, ctrl_par, model_ctrl)

    x = np.array([0.0, np.deg2rad(2.0), 0.0, 0.0, 0.0], dtype=float)
    s_current = 0.0

    t_log = []
    ey_log = []
    epsi_log = []
    vy_log = []
    r_log = []
    delta_log = []
    ddelta_log = []
    s_log = []
    vx_log = []
    kappa_log = []
    solve_time_log = []
    iter_log = []
    hard_fail_log = []
    soft_fail_log = []

    steps = int(sim_par.sim_time / sim_par.dt)
    terminal_margin = 3.0

    for k in range(steps):
        s_preview, kappa_preview, vx_preview, ax_preview = preview_path_data(
            path, speed_profile, ax_profile, x, s_current, ctrl_par, model_ctrl
        )

        hard_failed = 0
        soft_failed = 0

        try:
            _, U_opt, solve_time, iter_count, _, soft_failure = controller.solve(
                x, kappa_preview, vx_preview, ax_preview
            )
            u = U_opt[0]

            if soft_failure:
                soft_failed = 1

        except RuntimeError:
            hard_failed = 1
            u = np.array([-x[4] / sim_par.dt])
            u[0] = np.clip(u[0], -controller.ddelta_max, controller.ddelta_max)
            solve_time = np.nan
            iter_count = np.nan

        x, s_current, vx_now, _ = simulate_step(
            x, s_current, u, path, speed_profile, ax_profile,
            veh_plant, sim_par, model_plant
        )

        t = k * sim_par.dt

        t_log.append(t)
        ey_log.append(x[0])
        epsi_log.append(x[1])
        vy_log.append(x[2])
        r_log.append(x[3])
        delta_log.append(x[4])
        ddelta_log.append(float(u[0]))
        s_log.append(s_current)
        vx_log.append(vx_now)
        kappa_log.append(path.interp_kappa(s_current))
        solve_time_log.append(solve_time)
        iter_log.append(iter_count)
        hard_fail_log.append(hard_failed)
        soft_fail_log.append(soft_failed)

        if s_current >= path.length - terminal_margin:
            break

    t_log = np.array(t_log)
    ey_log = np.array(ey_log)
    epsi_log = np.array(epsi_log)
    vy_log = np.array(vy_log)
    r_log = np.array(r_log)
    delta_log = np.array(delta_log)
    ddelta_log = np.array(ddelta_log)
    s_log = np.array(s_log)
    vx_log = np.array(vx_log)
    kappa_log = np.array(kappa_log)
    solve_time_log = np.array(solve_time_log)
    iter_log = np.array(iter_log)
    hard_fail_log = np.array(hard_fail_log)
    soft_fail_log = np.array(soft_fail_log)

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

    ay_demand = vx_log**2 * np.abs(kappa_log)

    metrics["max_abs_vy_m_s"] = safe_nanmax(np.abs(vy_log))
    metrics["max_abs_r_rad_s"] = safe_nanmax(np.abs(r_log))
    metrics["mean_ay_demand_m_s2"] = safe_nanmean(ay_demand)
    metrics["max_ay_demand_m_s2"] = safe_nanmax(ay_demand)
    metrics["mu_g_plant_m_s2"] = cfg.mu_plant * 9.81
    metrics["max_ay_over_mu_g"] = metrics["max_ay_demand_m_s2"] / metrics["mu_g_plant_m_s2"]

    info = {
        "experiment": cfg.name,
        "group": cfg.group,
        "path_name": path_name,
        "path_type": cfg.path_type,
        "synthetic_path": cfg.synthetic_path,
        "vegsystemreferanse": cfg.vegsystemreferanse,
        "kommune": cfg.kommune,
        "path_length_m": path.length,
        "num_waypoints": len(x_wp),
        "path_max_abs_kappa": float(np.max(np.abs(path.kappa))),
        "path_mean_abs_kappa": float(np.mean(np.abs(path.kappa))),
        "ctrl_model": cfg.tire_model_ctrl,
        "plant_model": cfg.tire_model_plant,
        "mu_ctrl": cfg.mu_ctrl,
        "mu_plant": cfg.mu_plant,
        "stiffness_scale_ctrl": cfg.stiffness_scale_ctrl,
        "stiffness_scale_plant": cfg.stiffness_scale_plant,
        "vx_nominal": cfg.vx_nominal,
        "vx_min": cfg.vx_min,
        "vx_max": cfg.vx_max,
        "ay_max": cfg.ay_max,
        "N": cfg.N,
        "dt": cfg.dt,
        "sim_time": cfg.sim_time,
        "failed": False,
        "error": "",
    }

    return {**info, **metrics}


def save_results_to_csv(rows, filename):
    if len(rows) == 0:
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "nmpc_experiment_results.csv")

    experiments = build_experiment_list()
    path_cache = {}
    rows = []

    print("\n" + "=" * 80)
    print("RUNNING NMPC EXPERIMENT BATCH")
    print("=" * 80)
    print(f"Number of experiments: {len(experiments)}")
    print(f"Output CSV: {output_csv}")

    batch_start = time.perf_counter()

    for i, cfg in enumerate(experiments, start=1):
        print("\n" + "-" * 80)
        print(f"[{i}/{len(experiments)}] Running: {cfg.name}")
        print("-" * 80)

        t0 = time.perf_counter()

        try:
            row = run_experiment(cfg, path_cache)
            row["wall_time_s"] = time.perf_counter() - t0
            rows.append(row)

            print(f"Finished: {cfg.name}")
            print(f"  RMS e_y:        {row['rms_ey_m']:.4f} m")
            print(f"  Max |e_y|:      {row['max_abs_ey_m']:.4f} m")
            print(f"  Completion:     {row['completion_pct']:.2f} %")
            print(f"  Avg solve time: {row['avg_solve_ms']:.2f} ms")
            print(f"  Hard failures:  {row['hard_failures']}")
            print(f"  Soft warnings:  {row['soft_warnings']}")

        except Exception as e:
            error_text = traceback.format_exc()

            print(f"FAILED: {cfg.name}")
            print(str(e))

            rows.append({
                "experiment": cfg.name,
                "group": cfg.group,
                "path_type": cfg.path_type,
                "synthetic_path": cfg.synthetic_path,
                "vegsystemreferanse": cfg.vegsystemreferanse,
                "kommune": cfg.kommune,
                "ctrl_model": cfg.tire_model_ctrl,
                "plant_model": cfg.tire_model_plant,
                "mu_ctrl": cfg.mu_ctrl,
                "mu_plant": cfg.mu_plant,
                "stiffness_scale_ctrl": cfg.stiffness_scale_ctrl,
                "stiffness_scale_plant": cfg.stiffness_scale_plant,
                "vx_nominal": cfg.vx_nominal,
                "vx_min": cfg.vx_min,
                "vx_max": cfg.vx_max,
                "ay_max": cfg.ay_max,
                "N": cfg.N,
                "dt": cfg.dt,
                "sim_time": cfg.sim_time,
                "failed": True,
                "error": str(e),
                "traceback": error_text,
                "wall_time_s": time.perf_counter() - t0,
            })

        save_results_to_csv(rows, output_csv)

    total_time = time.perf_counter() - batch_start

    print("\n" + "=" * 80)
    print("BATCH FINISHED")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Saved results to: {output_csv}")
    print(f"Total wall time: {total_time:.1f} s")


if __name__ == "__main__":
    main()