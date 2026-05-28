"""
linear_mpc_curvature_stress_result.py

Curvature-stress simulation for the results chapter:
- Same controller and plant setup as the linear MPC baseline
- Linear MPC with curvature preview
- Dynamic bicycle error-state model: x = [e_y, e_psi, v_y, r, delta]
- Control input: u = delta_dot
- Linear tire model in both controller and plant
- Compares mild and sharp reference paths at the same nominal speed
- Saves comparison plots, logs, CSV metrics, and LaTeX table rows

Run:
    python linear_mpc_curvature_stress_result.py

Outputs:
    results_curvature_stress_linear_mpc/
        mild_trajectory_linear_mpc.png
        mild_errors_steering_linear_mpc.png
        mild_dynamic_response_linear_mpc.png
        mild_solver_performance_linear_mpc.png
        sharp_trajectory_linear_mpc.png
        sharp_errors_steering_linear_mpc.png
        sharp_dynamic_response_linear_mpc.png
        sharp_solver_performance_linear_mpc.png
        curvature_metrics.csv
        curvature_latex_rows.txt
        curvature_logs.npz
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.signal import savgol_filter


# ============================================================
# Utility functions
# ============================================================

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def safe_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2 or len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x))


def safe_percentile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanpercentile(x, p))


def c2d_zoh_two_inputs(
    A: np.ndarray,
    B: np.ndarray,
    E: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize xdot = A x + B u + E w using zero-order hold."""
    n = A.shape[0]
    m1 = B.shape[1]
    m2 = E.shape[1]
    U = np.hstack([B, E])

    M = np.block(
        [
            [A, U],
            [np.zeros((m1 + m2, n)), np.zeros((m1 + m2, m1 + m2))],
        ]
    )

    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Ud = Md[:n, n : n + m1 + m2]
    Bd = Ud[:, :m1]
    Ed = Ud[:, m1:]
    return Ad, Bd, Ed


def smooth_and_clip_preview(
    kappa_preview: np.ndarray,
    window: int = 5,
    kappa_abs_max: float = 0.20,
) -> np.ndarray:
    """Smooth and clip curvature preview to reduce sensitivity to spline artifacts."""
    kp = np.asarray(kappa_preview, dtype=float).copy()

    if window > 1:
        kernel = np.ones(window, dtype=float) / float(window)
        kp = np.convolve(kp, kernel, mode="same")

    return np.clip(kp, -kappa_abs_max, kappa_abs_max)


def smooth_signal(y: np.ndarray, window: int = 101, polyorder: int = 3) -> np.ndarray:
    """
    Smooth signal for plotting only using a Savitzky-Golay filter.

    This is only used for visualization, not for controller computations.
    """
    y = np.asarray(y, dtype=float)

    if len(y) < 5:
        return y.copy()

    window = int(window)
    if window % 2 == 0:
        window += 1

    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1

    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1

    if window > len(y) or window <= polyorder:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


# ============================================================
# Path representation
# ============================================================

@dataclass
class SplinePath:
    s: np.ndarray
    sx: CubicSpline
    sy: CubicSpline
    s_max: float

    @staticmethod
    def from_waypoints(xy: np.ndarray) -> "SplinePath":
        dx = np.diff(xy[:, 0])
        dy = np.diff(xy[:, 1])
        ds = np.sqrt(dx * dx + dy * dy)
        s = np.concatenate([[0.0], np.cumsum(ds)])

        eps = 1e-6
        for i in range(1, len(s)):
            if s[i] <= s[i - 1]:
                s[i] = s[i - 1] + eps

        sx = CubicSpline(s, xy[:, 0], bc_type="not-a-knot")
        sy = CubicSpline(s, xy[:, 1], bc_type="not-a-knot")
        return SplinePath(s=s, sx=sx, sy=sy, s_max=float(s[-1]))

    def eval_xy(self, s_val: float) -> Tuple[float, float]:
        sc = float(np.clip(s_val, 0.0, self.s_max))
        return float(self.sx(sc)), float(self.sy(sc))

    def d1(self, s_val: float) -> Tuple[float, float]:
        sc = float(np.clip(s_val, 0.0, self.s_max))
        return float(self.sx(sc, 1)), float(self.sy(sc, 1))

    def d2(self, s_val: float) -> Tuple[float, float]:
        sc = float(np.clip(s_val, 0.0, self.s_max))
        return float(self.sx(sc, 2)), float(self.sy(sc, 2))

    def heading(self, s_val: float) -> float:
        dx, dy = self.d1(s_val)
        return float(np.arctan2(dy, dx))

    def curvature(self, s_val: float) -> float:
        dx, dy = self.d1(s_val)
        ddx, ddy = self.d2(s_val)
        denom = (dx * dx + dy * dy) ** 1.5

        if denom < 1e-9:
            return 0.0

        return float((dx * ddy - dy * ddx) / denom)

    def project_s(self, x: float, y: float, s_guess: Optional[float] = None) -> float:
        """Closest-point projection using local search around previous s."""
        s0 = 0.0 if s_guess is None else float(np.clip(s_guess, 0.0, self.s_max))

        window = 30.0
        ds = 0.5
        s_lo = max(0.0, s0 - window)
        s_hi = min(self.s_max, s0 + window)
        ss = np.arange(s_lo, s_hi + ds, ds)

        xs = self.sx(ss)
        ys = self.sy(ss)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        s_best = float(ss[int(np.argmin(d2))])

        a = max(0.0, s_best - ds)
        b = min(self.s_max, s_best + ds)

        for _ in range(10):
            s1 = a + (b - a) / 3.0
            s2 = b - (b - a) / 3.0

            x1, y1 = self.eval_xy(s1)
            x2, y2 = self.eval_xy(s2)

            f1 = (x1 - x) ** 2 + (y1 - y) ** 2
            f2 = (x2 - x) ** 2 + (y2 - y) ** 2

            if f1 < f2:
                b = s2
            else:
                a = s1

        return float(0.5 * (a + b))


# ============================================================
# Vehicle model and plant
# ============================================================

@dataclass
class VehicleParams:
    m: float = 1757.0
    Iz: float = 3100.0
    lf: float = 1.23
    lr: float = 1.49
    Cf: float = 125000.0
    Cr: float = 118000.0
    g: float = 9.81
    delta_max: float = np.deg2rad(30.0)
    delta_dot_max: float = np.deg2rad(90.0)

    @property
    def L(self) -> float:
        return self.lf + self.lr


@dataclass
class GlobalState:
    x: float
    y: float
    psi: float
    vx: float
    vy: float
    r: float
    delta: float


def slip_angles(
    vx: float,
    vy: float,
    r: float,
    delta: float,
    lf: float,
    lr: float,
) -> Tuple[float, float]:
    vx_eff = max(0.2, abs(vx)) * np.sign(vx if vx != 0 else 1.0)

    alpha_f = delta - (vy + lf * r) / vx_eff
    alpha_r = -(vy - lr * r) / vx_eff

    return float(alpha_f), float(alpha_r)


def tire_forces_linear(
    alpha_f: float,
    alpha_r: float,
    p: VehicleParams,
) -> Tuple[float, float]:
    return float(p.Cf * alpha_f), float(p.Cr * alpha_r)


def plant_derivatives(s: GlobalState, u: float, p: VehicleParams) -> GlobalState:
    af, ar = slip_angles(s.vx, s.vy, s.r, s.delta, p.lf, p.lr)
    Fyf, Fyr = tire_forces_linear(af, ar, p)

    vy_dot = (Fyf + Fyr) / p.m - s.vx * s.r
    r_dot = (p.lf * Fyf - p.lr * Fyr) / p.Iz
    delta_dot = u

    x_dot = s.vx * np.cos(s.psi) - s.vy * np.sin(s.psi)
    y_dot = s.vx * np.sin(s.psi) + s.vy * np.cos(s.psi)
    psi_dot = s.r

    return GlobalState(x_dot, y_dot, psi_dot, 0.0, vy_dot, r_dot, delta_dot)


def rk4_step(s: GlobalState, u: float, dt: float, p: VehicleParams) -> GlobalState:
    def add(a: GlobalState, b: GlobalState, k: float) -> GlobalState:
        return GlobalState(
            a.x + k * b.x,
            a.y + k * b.y,
            a.psi + k * b.psi,
            a.vx,
            a.vy + k * b.vy,
            a.r + k * b.r,
            a.delta + k * b.delta,
        )

    k1 = plant_derivatives(s, u, p)
    k2 = plant_derivatives(add(s, k1, dt / 2.0), u, p)
    k3 = plant_derivatives(add(s, k2, dt / 2.0), u, p)
    k4 = plant_derivatives(add(s, k3, dt), u, p)

    sn = GlobalState(
        s.x + dt * (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6.0,
        s.y + dt * (k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6.0,
        wrap_angle(s.psi + dt * (k1.psi + 2 * k2.psi + 2 * k3.psi + k4.psi) / 6.0),
        s.vx,
        s.vy + dt * (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy) / 6.0,
        s.r + dt * (k1.r + 2 * k2.r + 2 * k3.r + k4.r) / 6.0,
        s.delta + dt * (k1.delta + 2 * k2.delta + 2 * k3.delta + k4.delta) / 6.0,
    )

    sn.delta = float(np.clip(sn.delta, -p.delta_max, p.delta_max))
    return sn


# ============================================================
# Error-state conversion and preview
# ============================================================

def error_state(
    path: SplinePath,
    s_guess: float,
    s: GlobalState,
) -> Tuple[np.ndarray, float, float]:
    s0 = path.project_s(s.x, s.y, s_guess=s_guess)

    xr, yr = path.eval_xy(s0)
    psi_r = path.heading(s0)
    kappa = path.curvature(s0)

    n = np.array([-np.sin(psi_r), np.cos(psi_r)])
    e = np.array([s.x - xr, s.y - yr])

    ey = float(n @ e)
    epsi = wrap_angle(s.psi - psi_r)

    xerr = np.array([ey, epsi, s.vy, s.r, s.delta], dtype=float)
    return xerr, s0, kappa


def steer_ff(L: float, kappa: float) -> float:
    return float(np.arctan(L * kappa))


def build_curvature_preview(
    path: SplinePath,
    s0: float,
    vx: float,
    Ts: float,
    N: int,
    smooth_window: int = 5,
    kappa_abs_max: float = 0.20,
) -> np.ndarray:
    kappa_preview = np.zeros(N, dtype=float)

    for i in range(N):
        si = min(path.s_max, s0 + vx * (i + 1) * Ts)
        kappa_preview[i] = path.curvature(si)

    return smooth_and_clip_preview(
        kappa_preview,
        window=smooth_window,
        kappa_abs_max=kappa_abs_max,
    )


# ============================================================
# Linear MPC model and OSQP solver
# ============================================================

def continuous_matrices(
    vx: float,
    p: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xdot = A x + B u + E kappa for x=[ey,epsi,vy,r,delta]."""
    vx_safe = max(0.2, abs(vx)) * np.sign(vx if vx != 0 else 1.0)

    A = np.zeros((5, 5), dtype=float)
    B = np.zeros((5, 1), dtype=float)
    E = np.zeros((5, 1), dtype=float)

    A[0, 1] = vx_safe
    A[0, 2] = 1.0

    A[1, 3] = 1.0
    E[1, 0] = -vx_safe

    A[2, 2] = -(p.Cf + p.Cr) / (p.m * vx_safe)
    A[2, 3] = -(p.Cf * p.lf - p.Cr * p.lr) / (p.m * vx_safe) - vx_safe
    A[2, 4] = p.Cf / p.m

    A[3, 2] = -(p.Cf * p.lf - p.Cr * p.lr) / (p.Iz * vx_safe)
    A[3, 3] = -(p.Cf * p.lf**2 + p.Cr * p.lr**2) / (p.Iz * vx_safe)
    A[3, 4] = p.Cf * p.lf / p.Iz

    B[4, 0] = 1.0

    return A, B, E


@dataclass
class MPCConfig:
    Ts: float = 0.05
    horizon_s: float = 2.5
    w_ey: float = 40.0
    w_epsi: float = 20.0
    w_vy: float = 0.5
    w_r: float = 2.0
    w_delta: float = 0.5
    w_udot: float = 0.2
    terminal_mult: float = 10.0

    @property
    def N(self) -> int:
        return int(round(self.horizon_s / self.Ts))


class OSQPMPC:
    """Sparse linear MPC with dynamics constraints and input/state bounds."""

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Ed: np.ndarray,
        p: VehicleParams,
        mpc: MPCConfig,
    ):
        self.Ad = Ad
        self.Bd = Bd
        self.Ed = Ed
        self.p = p
        self.mpc = mpc

        self.n = 5
        self.m = 1
        self.N = mpc.N

        self.nz = self.N * self.n + self.N * self.m
        self.neq = self.N * self.n
        self.nineq = self.N + self.N
        self.ncon = self.neq + self.nineq

        self.P = self._build_P()
        self.A = self._build_A()
        self.q = np.zeros(self.nz, dtype=float)
        self.l = np.zeros(self.ncon, dtype=float)
        self.u = np.zeros(self.ncon, dtype=float)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.P,
            q=self.q,
            A=self.A,
            l=self.l,
            u=self.u,
            warm_start=True,
            polish=True,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=4000,
        )

    def _idx_x(self, k: int) -> int:
        return k * self.n

    def _idx_u(self, k: int) -> int:
        return self.N * self.n + k * self.m

    def _build_P(self) -> sp.csc_matrix:
        Q = np.diag(
            [
                self.mpc.w_ey,
                self.mpc.w_epsi,
                self.mpc.w_vy,
                self.mpc.w_r,
                self.mpc.w_delta,
            ]
        )
        QN = self.mpc.terminal_mult * Q
        R = np.array([[self.mpc.w_udot]])

        Px = sp.block_diag([Q] * (self.N - 1) + [QN], format="csc")
        Pu = sp.block_diag([R] * self.N, format="csc")

        return sp.block_diag([Px, Pu], format="csc")

    def _build_A(self) -> sp.csc_matrix:
        rows, cols, data = [], [], []

        for k in range(self.N):
            r0 = k * self.n

            cxkp1 = self._idx_x(k)
            for i in range(self.n):
                rows.append(r0 + i)
                cols.append(cxkp1 + i)
                data.append(1.0)

            if k >= 1:
                cxk = self._idx_x(k - 1)
                for i in range(self.n):
                    for j in range(self.n):
                        rows.append(r0 + i)
                        cols.append(cxk + j)
                        data.append(-self.Ad[i, j])

            cuk = self._idx_u(k)
            for i in range(self.n):
                rows.append(r0 + i)
                cols.append(cuk)
                data.append(-self.Bd[i, 0])

        for k in range(self.N):
            r = self.neq + k
            cxk = self._idx_x(k)
            rows.append(r)
            cols.append(cxk + 4)
            data.append(1.0)

        for k in range(self.N):
            r = self.neq + self.N + k
            cuk = self._idx_u(k)
            rows.append(r)
            cols.append(cuk)
            data.append(1.0)

        return sp.csc_matrix((data, (rows, cols)), shape=(self.ncon, self.nz))

    def solve(
        self,
        x0: np.ndarray,
        kappa_preview: np.ndarray,
        vx: float,
        L: float,
    ) -> Tuple[float, float, int, str]:
        assert x0.shape == (5,)
        assert kappa_preview.shape == (self.N,)

        self.q[:] = 0.0

        for k in range(self.N):
            ix = self._idx_x(k)

            r_ref = vx * float(kappa_preview[k])
            d_ref = steer_ff(L, float(kappa_preview[k]))

            self.q[ix + 3] += -2.0 * self.mpc.w_r * r_ref
            self.q[ix + 4] += -2.0 * self.mpc.w_delta * d_ref

        beq = np.zeros(self.neq, dtype=float)
        beq[0:5] = self.Ad @ x0 + (self.Ed[:, 0] * float(kappa_preview[0]))

        for k in range(1, self.N):
            beq[k * 5 : (k + 1) * 5] = self.Ed[:, 0] * float(kappa_preview[k])

        self.l[: self.neq] = beq
        self.u[: self.neq] = beq

        dmax = self.p.delta_max
        umax = self.p.delta_dot_max

        self.l[self.neq : self.neq + self.N] = -dmax
        self.u[self.neq : self.neq + self.N] = dmax

        self.l[self.neq + self.N :] = -umax
        self.u[self.neq + self.N :] = umax

        self.prob.update(q=self.q, l=self.l, u=self.u)

        t0 = time.perf_counter()
        res = self.prob.solve()
        solve_time = time.perf_counter() - t0

        status = res.info.status
        iterations = int(res.info.iter)

        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP failed: {status}")

        z = res.x
        u0 = float(z[self._idx_u(0)])

        return float(np.clip(u0, -umax, umax)), solve_time, iterations, status


# ============================================================
# Scenario setup
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


@dataclass
class ScenarioConfig:
    name: str
    path_name: str
    vx: float = 7.0
    initial_cross_track_error: float = 1.0
    initial_heading_error_deg: float = 2.0
    sim_time: float = 50.0
    goal_radius: float = 2.0
    s_end_margin: float = 3.0
    output_dir: str = "results_curvature_stress_linear_mpc"
    show_plots: bool = False


# ============================================================
# Simulation and result export
# ============================================================

def run_case(cfg: ScenarioConfig) -> tuple[dict, dict]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = VehicleParams()
    mcfg = MPCConfig(Ts=0.05, horizon_s=2.5)

    if cfg.path_name == "mild":
        waypoints = build_mild_waypoints()
    elif cfg.path_name == "sharp":
        waypoints = build_sharp_waypoints()
    else:
        raise ValueError(f"Unknown path_name: {cfg.path_name}")

    path = SplinePath.from_waypoints(waypoints)

    vx = cfg.vx

    A, B, E = continuous_matrices(vx, p)
    Ad, Bd, Ed = c2d_zoh_two_inputs(A, B, E, mcfg.Ts)

    mpc = OSQPMPC(Ad, Bd, Ed, p, mcfg)

    x0_ref, y0_ref = path.eval_xy(0.0)
    psi0_ref = path.heading(0.0)

    ey0 = cfg.initial_cross_track_error
    epsi0 = np.deg2rad(cfg.initial_heading_error_deg)

    x0 = x0_ref - ey0 * np.sin(psi0_ref)
    y0 = y0_ref + ey0 * np.cos(psi0_ref)
    psi0 = wrap_angle(psi0_ref + epsi0)

    state = GlobalState(
        x=x0,
        y=y0,
        psi=psi0,
        vx=vx,
        vy=0.0,
        r=0.0,
        delta=0.0,
    )

    steps = int(cfg.sim_time / mcfg.Ts)
    goal_x, goal_y = waypoints[-1]

    T = np.zeros(steps)
    Xg = np.zeros((steps, 7))
    Xe = np.zeros((steps, 5))
    U = np.zeros(steps)
    K = np.zeros(steps)
    Rref = np.zeros(steps)
    Slog = np.zeros(steps)
    solve_time = np.full(steps, np.nan)
    osqp_iter = np.full(steps, np.nan)
    solve_fail = np.zeros(steps, dtype=int)

    s_guess = 0.0
    last_valid_idx = steps - 1
    stopped_on_goal = False

    for k in range(steps):
        t = k * mcfg.Ts
        T[k] = t

        xerr, s0, kappa = error_state(path, s_guess, state)
        s_guess = s0

        kappa_preview = build_curvature_preview(
            path=path,
            s0=s0,
            vx=vx,
            Ts=mcfg.Ts,
            N=mcfg.N,
            smooth_window=5,
            kappa_abs_max=0.20,
        )

        try:
            u0, stime, iters, _ = mpc.solve(xerr, kappa_preview, vx=vx, L=p.L)
            solve_time[k] = stime
            osqp_iter[k] = iters
        except RuntimeError:
            u0 = 0.0
            solve_fail[k] = 1

        state = rk4_step(state, u0, mcfg.Ts, p)

        Xe[k, :] = xerr
        U[k] = u0
        K[k] = kappa
        Rref[k] = vx * kappa
        Slog[k] = s0
        Xg[k, :] = np.array(
            [
                state.x,
                state.y,
                state.psi,
                state.vx,
                state.vy,
                state.r,
                state.delta,
            ]
        )

        dist_to_goal = np.hypot(state.x - goal_x, state.y - goal_y)

        if dist_to_goal <= cfg.goal_radius and s0 >= path.s_max - cfg.s_end_margin:
            last_valid_idx = k
            stopped_on_goal = True
            break

    T = T[: last_valid_idx + 1]
    Xg = Xg[: last_valid_idx + 1, :]
    Xe = Xe[: last_valid_idx + 1, :]
    U = U[: last_valid_idx + 1]
    K = K[: last_valid_idx + 1]
    Rref = Rref[: last_valid_idx + 1]
    Slog = Slog[: last_valid_idx + 1]
    solve_time = solve_time[: last_valid_idx + 1]
    osqp_iter = osqp_iter[: last_valid_idx + 1]
    solve_fail = solve_fail[: last_valid_idx + 1]

    ey = Xe[:, 0]
    epsi = Xe[:, 1]
    delta = Xe[:, 4]
    ddelta = U
    r = Xg[:, 5]
    vx_log = Xg[:, 3]

    valid_solve = solve_time[~np.isnan(solve_time)]
    valid_iter = osqp_iter[~np.isnan(osqp_iter)]

    s_plot = np.linspace(0.0, path.s_max, 1600)
    x_ref = path.sx(s_plot)
    y_ref = path.sy(s_plot)

    kappa_ref_raw = np.array([path.curvature(sv) for sv in s_plot])
    kappa_ref_plot = smooth_signal(kappa_ref_raw, window=151, polyorder=3)

    kappa_time_plot = np.interp(Slog, s_plot, kappa_ref_plot)
    r_ref_plot = vx_log * kappa_time_plot

    metrics = {
        "scenario": cfg.name,
        "controller": "Linear MPC (OSQP)",
        "plant": "Linear tire model",
        "path": cfg.path_name,
        "vx_nominal_m_s": vx,
        "initial_cross_track_error_m": cfg.initial_cross_track_error,
        "initial_heading_error_deg": cfg.initial_heading_error_deg,
        "path_length_m": path.s_max,
        "max_abs_kappa_1_m": float(np.max(np.abs(kappa_ref_raw))),
        "rms_ey_m": float(np.sqrt(np.mean(ey**2))),
        "max_abs_ey_m": float(np.max(np.abs(ey))),
        "p95_abs_ey_m": safe_percentile(np.abs(ey), 95),
        "iae_ey_m_s": safe_trapezoid(np.abs(ey), T),
        "rms_epsi_deg": float(np.rad2deg(np.sqrt(np.mean(epsi**2)))),
        "max_abs_epsi_deg": float(np.rad2deg(np.max(np.abs(epsi)))),
        "iae_epsi_deg_s": float(np.rad2deg(safe_trapezoid(np.abs(epsi), T))),
        "max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(delta)))),
        "rms_delta_deg": float(np.rad2deg(np.sqrt(np.mean(delta**2)))),
        "max_abs_ddelta_deg_s": float(np.rad2deg(np.max(np.abs(ddelta)))),
        "rms_ddelta_deg_s": float(np.rad2deg(np.sqrt(np.mean(ddelta**2)))),
        "iadc_deg": float(np.rad2deg(safe_trapezoid(np.abs(ddelta), T))),
        "mean_solve_ms": float(1000.0 * np.mean(valid_solve)) if len(valid_solve) else float("nan"),
        "median_solve_ms": float(1000.0 * np.median(valid_solve)) if len(valid_solve) else float("nan"),
        "p95_solve_ms": float(1000.0 * np.percentile(valid_solve, 95)) if len(valid_solve) else float("nan"),
        "max_solve_ms": float(1000.0 * np.max(valid_solve)) if len(valid_solve) else float("nan"),
        "mean_osqp_iter": float(np.mean(valid_iter)) if len(valid_iter) else float("nan"),
        "max_osqp_iter": float(np.max(valid_iter)) if len(valid_iter) else float("nan"),
        "hard_failures": int(np.sum(solve_fail)),
        "duration_s": float(T[-1] - T[0]) if len(T) > 1 else 0.0,
        "stopped_on_goal": bool(stopped_on_goal),
        "final_distance_to_goal_m": float(np.hypot(Xg[-1, 0] - goal_x, Xg[-1, 1] - goal_y)),
    }

    logs = {
        "T": T,
        "Xg": Xg,
        "Xe": Xe,
        "U": U,
        "K": K,
        "K_plot": kappa_time_plot,
        "Rref": Rref,
        "Rref_plot": r_ref_plot,
        "Slog": Slog,
        "solve_time": solve_time,
        "osqp_iter": osqp_iter,
        "solve_fail": solve_fail,
        "s_plot": s_plot,
        "x_ref": x_ref,
        "y_ref": y_ref,
        "kappa_ref": kappa_ref_raw,
        "kappa_ref_plot": kappa_ref_plot,
        "waypoints": waypoints,
    }

    return metrics, logs


def save_metrics(results: list[tuple[ScenarioConfig, dict, dict]], out_dir: Path) -> None:
    metric_rows = [m for _, m, _ in results]

    csv_path = out_dir / "curvature_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = []
    for _, metrics, _ in results:
        line = (
            f"{metrics['path']} & "
            f"{metrics['max_abs_kappa_1_m']:.4f} & "
            f"{metrics['rms_ey_m']:.3f} & "
            f"{metrics['max_abs_ey_m']:.3f} & "
            f"{metrics['rms_epsi_deg']:.2f} & "
            f"{metrics['max_abs_delta_deg']:.2f} & "
            f"{metrics['max_abs_ddelta_deg_s']:.2f} & "
            f"{metrics['mean_solve_ms']:.2f} & "
            f"{metrics['hard_failures']} \\\\"
        )
        lines.append(line)

    (out_dir / "curvature_latex_rows.txt").write_text("\n".join(lines) + "\n")


def save_npz_logs(results: list[tuple[ScenarioConfig, dict, dict]], out_dir: Path) -> None:
    npz_data = {}
    for cfg, metrics, logs in results:
        prefix = cfg.path_name
        for key, value in logs.items():
            npz_data[f"{prefix}_{key}"] = value
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, np.number)):
                npz_data[f"{prefix}_metric_{key}"] = value

    np.savez(out_dir / "curvature_logs.npz", **npz_data)


def make_case_plots(results: list[tuple[ScenarioConfig, dict, dict]], out_dir: Path) -> None:
    """
    Make one set of baseline-style plots for each curvature scenario.

    The plot layout intentionally matches the baseline test:
    1. trajectory
    2. tracking errors and steering response
    3. dynamic response
    4. solver performance
    """
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

    for cfg, metrics, logs in results:
        label = cfg.path_name
        title_label = cfg.path_name.capitalize()

        T = logs["T"]
        Xg = logs["Xg"]
        Xe = logs["Xe"]
        U = logs["U"]
        kappa_time_plot = logs["K_plot"]
        r_ref_plot = logs["Rref_plot"]
        solve_time = logs["solve_time"]
        osqp_iter = logs["osqp_iter"]

        ey = Xe[:, 0]
        epsi = Xe[:, 1]
        delta = Xe[:, 4]
        ddelta = U
        r = Xg[:, 5]
        vx_log = Xg[:, 3]

        x_ref = logs["x_ref"]
        y_ref = logs["y_ref"]
        goal_x = logs["waypoints"][-1, 0]
        goal_y = logs["waypoints"][-1, 1]

        # ========================================================
        # Plot 1: trajectory
        # ========================================================

        fig, ax = plt.subplots(figsize=(10.5, 6.8))

        ax.plot(x_ref, y_ref, "--", linewidth=2.4, label="Reference path")
        ax.plot(Xg[:, 0], Xg[:, 1], linewidth=2.2, label="Vehicle trajectory")
        ax.scatter([Xg[0, 0]], [Xg[0, 1]], marker="o", s=70, label="Start")
        ax.scatter([goal_x], [goal_y], marker="x", s=110, label="Goal")

        x_all = np.concatenate([x_ref, Xg[:, 0]])
        y_all = np.concatenate([y_ref, Xg[:, 1]])

        x_range = np.max(x_all) - np.min(x_all)
        y_range = np.max(y_all) - np.min(y_all)

        x_margin = 0.04 * max(1.0, x_range)
        y_margin = 0.80 * max(1.0, y_range)

        ax.set_xlim(np.min(x_all) - x_margin, np.max(x_all) + x_margin)
        ax.set_ylim(np.min(y_all) - y_margin, np.max(y_all) + y_margin)

        # Do not force equal aspect. The paths are long relative to lateral variation.
        ax.set_aspect("auto")

        ax.grid(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{title_label} path trajectory: Linear MPC with linear tire model")
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_trajectory_linear_mpc.png", dpi=300)

        # ========================================================
        # Plot 2: tracking errors and steering response
        # ========================================================

        fig, axs = plt.subplots(4, 1, figsize=(8.5, 9.5), sharex=True)

        axs[0].plot(T, ey, linewidth=1.8)
        axs[0].set_ylabel(r"$e_y$ [m]")
        axs[0].grid(True)

        axs[1].plot(T, np.rad2deg(epsi), linewidth=1.8)
        axs[1].set_ylabel(r"$e_\psi$ [deg]")
        axs[1].grid(True)

        axs[2].plot(T, np.rad2deg(delta), linewidth=1.8)
        axs[2].set_ylabel(r"$\delta$ [deg]")
        axs[2].grid(True)

        axs[3].plot(T, np.rad2deg(ddelta), linewidth=1.8)
        axs[3].set_ylabel(r"$\dot{\delta}$ [deg/s]")
        axs[3].set_xlabel("Time [s]")
        axs[3].grid(True)

        fig.suptitle(f"{title_label} path tracking errors and steering response")
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_errors_steering_linear_mpc.png", dpi=300)

        # ========================================================
        # Plot 3: dynamic response
        # ========================================================

        fig, axs = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)

        axs[0].plot(T, kappa_time_plot, linewidth=1.8, label=r"Smoothed $\kappa$")
        axs[0].set_ylabel(r"$\kappa$ [1/m]")
        axs[0].grid(True)
        axs[0].legend(loc="best")

        axs[1].plot(T, vx_log, linewidth=1.8)
        axs[1].set_ylabel(r"$v_x$ [m/s]")
        axs[1].grid(True)

        axs[2].plot(T, r, linewidth=1.8, label=r"$r$")
        axs[2].plot(T, r_ref_plot, "--", linewidth=1.8, label=r"$v_x \kappa$")
        axs[2].set_ylabel("Yaw rate [rad/s]")
        axs[2].set_xlabel("Time [s]")
        axs[2].grid(True)
        axs[2].legend(loc="best")

        fig.suptitle(f"{title_label} path dynamic response")
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_dynamic_response_linear_mpc.png", dpi=300)

        # ========================================================
        # Plot 4: solver performance
        # ========================================================

        fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True)

        axs[0].plot(T, 1000.0 * solve_time, linewidth=1.8, label="OSQP solve time")
        axs[0].axhline(50.0, linestyle="--", linewidth=1.5, label="Control period")
        axs[0].set_ylabel("Solve time [ms]")
        axs[0].grid(True)
        axs[0].legend(loc="best")

        axs[1].step(T, osqp_iter, where="post", linewidth=1.8)
        axs[1].set_ylabel("OSQP iterations")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(True)

        fig.suptitle(f"{title_label} path solver performance")
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_solver_performance_linear_mpc.png", dpi=300)

    plt.close("all")



def main() -> None:
    out_dir = Path("results_curvature_stress_linear_mpc")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ScenarioConfig(
            name="linear_mpc_mild_path_7ms",
            path_name="mild",
            output_dir=str(out_dir),
        ),
        ScenarioConfig(
            name="linear_mpc_sharp_path_7ms",
            path_name="sharp",
            output_dir=str(out_dir),
        ),
    ]

    results = []
    for cfg in cases:
        print(f"\n=== Running {cfg.name} ===")
        metrics, logs = run_case(cfg)
        results.append((cfg, metrics, logs))

        print(f"Path:             {metrics['path']}")
        print(f"Max |kappa|:      {metrics['max_abs_kappa_1_m']:.4f} 1/m")
        print(f"RMS e_y:          {metrics['rms_ey_m']:.3f} m")
        print(f"Max |e_y|:        {metrics['max_abs_ey_m']:.3f} m")
        print(f"RMS e_psi:        {metrics['rms_epsi_deg']:.2f} deg")
        print(f"Max |delta|:      {metrics['max_abs_delta_deg']:.2f} deg")
        print(f"Max |delta_dot|:  {metrics['max_abs_ddelta_deg_s']:.2f} deg/s")
        print(f"Mean solve time:  {metrics['mean_solve_ms']:.2f} ms")
        print(f"Max solve time:   {metrics['max_solve_ms']:.2f} ms")
        print(f"Hard failures:    {metrics['hard_failures']}")
        print(f"Stopped on goal:  {metrics['stopped_on_goal']}")

    save_metrics(results, out_dir)
    save_npz_logs(results, out_dir)
    make_case_plots(results, out_dir)

    print("\n=== Curvature stress scenario completed ===")
    print(f"Output directory: {out_dir}")
    print("\nLaTeX rows:")
    print((out_dir / "curvature_latex_rows.txt").read_text().strip())


if __name__ == "__main__":
    main()
