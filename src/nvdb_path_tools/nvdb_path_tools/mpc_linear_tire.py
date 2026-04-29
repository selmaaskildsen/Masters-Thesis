"""
linear_mpc_curvature_preview_baseline.py

Baseline linear MPC for banefølging:
- Error-state dynamisk sykkelmodell: x = [e_y, e_psi, v_y, r, delta], u = delta_dot
- Lineær dekkmodell: F_y = C_alpha * alpha (akseleffektivt)
- Kurvatur-preview som eksogent signal: x_{k+1} = Ad x_k + Bd u_k + Ed kappa_k
- QP løses med OSQP (P, q, A, l, u)
- Plant simuleres i globale koordinater med RK4 og lineær dekkmodell
- Simuleringen stopper når bilen kommer innenfor et område rundt siste punkt
- Plot + metrikker

Merk:
- Denne baseline-versjonen bruker modellmatriser linearisert ved valgt konstant fart.
- Den er derfor best beskrevet som en lineær MPC med curvature preview.
- Neste steg kan være å oppdatere modellmatrisene online med aktuell fart.

Kjør:
  pip install numpy scipy matplotlib osqp
  python linear_mpc_curvature_preview_baseline.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
import osqp
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def c2d_zoh_two_inputs(
    A: np.ndarray,
    B: np.ndarray,
    E: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretize xdot = A x + B u + E w with ZOH using augmented matrix exponential.
    Returns Ad, Bd, Ed such that x_{k+1} = Ad x_k + Bd u_k + Ed w_k
    """
    n = A.shape[0]
    m1 = B.shape[1]
    m2 = E.shape[1]
    U = np.hstack([B, E])

    M = np.block([
        [A, U],
        [np.zeros((m1 + m2, n)), np.zeros((m1 + m2, m1 + m2))]
    ])

    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Ud = Md[:n, n:n + m1 + m2]
    Bd = Ud[:, :m1]
    Ed = Ud[:, m1:]
    return Ad, Bd, Ed


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def smooth_and_clip_preview(
    kappa_preview: np.ndarray,
    window: int = 5,
    kappa_abs_max: float = 0.20
) -> np.ndarray:
    """
    Smooth and clip curvature preview to reduce sensitivity to spline artifacts.
    """
    kp = np.asarray(kappa_preview, dtype=float).copy()

    if window > 1:
        kernel = np.ones(window, dtype=float) / float(window)
        kp = np.convolve(kp, kernel, mode="same")

    kp = np.clip(kp, -kappa_abs_max, kappa_abs_max)
    return kp


# -----------------------------
# Path: x(s), y(s) cubic splines
# -----------------------------
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

    def eval_xy(self, s: float) -> Tuple[float, float]:
        sc = float(np.clip(s, 0.0, self.s_max))
        return float(self.sx(sc)), float(self.sy(sc))

    def d1(self, s: float) -> Tuple[float, float]:
        sc = float(np.clip(s, 0.0, self.s_max))
        return float(self.sx(sc, 1)), float(self.sy(sc, 1))

    def d2(self, s: float) -> Tuple[float, float]:
        sc = float(np.clip(s, 0.0, self.s_max))
        return float(self.sx(sc, 2)), float(self.sy(sc, 2))

    def heading(self, s: float) -> float:
        dx, dy = self.d1(s)
        return float(np.arctan2(dy, dx))

    def curvature(self, s: float) -> float:
        dx, dy = self.d1(s)
        ddx, ddy = self.d2(s)
        denom = (dx * dx + dy * dy) ** 1.5
        if denom < 1e-9:
            return 0.0
        return float((dx * ddy - dy * ddx) / denom)

    def project_s(self, x: float, y: float, s_guess: Optional[float] = None) -> float:
        """
        Baseline projection: coarse sampling around guess + local refine.
        """
        if s_guess is None:
            s0 = 0.0
        else:
            s0 = float(np.clip(s_guess, 0.0, self.s_max))

        window = 30.0
        ds = 0.5
        s_lo = max(0.0, s0 - window)
        s_hi = min(self.s_max, s0 + window)
        ss = np.arange(s_lo, s_hi + ds, ds)

        xs = self.sx(ss)
        ys = self.sy(ss)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        i = int(np.argmin(d2))
        s_best = float(ss[i])

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


# -----------------------------
# Vehicle + linear tire
# -----------------------------
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
    lr: float
) -> Tuple[float, float]:
    vx_eff = max(0.2, abs(vx)) * np.sign(vx)
    alpha_f = delta - (vy + lf * r) / vx_eff
    alpha_r = -(vy - lr * r) / vx_eff
    return float(alpha_f), float(alpha_r)


def tire_forces_linear(alpha_f: float, alpha_r: float, p: VehicleParams) -> Tuple[float, float]:
    Fyf = p.Cf * alpha_f
    Fyr = p.Cr * alpha_r
    return float(Fyf), float(Fyr)


def plant_derivatives(s: GlobalState, u: float, p: VehicleParams) -> GlobalState:
    """
    Plant in global coords + lateral/yaw dyn in body frame.
    u = delta_dot
    """
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
            a.delta + k * b.delta
        )

    k1 = plant_derivatives(s, u, p)
    k2 = plant_derivatives(add(s, k1, dt / 2), u, p)
    k3 = plant_derivatives(add(s, k2, dt / 2), u, p)
    k4 = plant_derivatives(add(s, k3, dt), u, p)

    sn = GlobalState(
        s.x + dt * (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6,
        s.y + dt * (k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6,
        wrap_angle(s.psi + dt * (k1.psi + 2 * k2.psi + 2 * k3.psi + k4.psi) / 6),
        s.vx,
        s.vy + dt * (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy) / 6,
        s.r + dt * (k1.r + 2 * k2.r + 2 * k3.r + k4.r) / 6,
        s.delta + dt * (k1.delta + 2 * k2.delta + 2 * k3.delta + k4.delta) / 6
    )
    sn.delta = float(np.clip(sn.delta, -p.delta_max, p.delta_max))
    return sn


# -----------------------------
# Error-state from global pose
# -----------------------------
def error_state(path: SplinePath, s_guess: float, s: GlobalState) -> Tuple[np.ndarray, float, float]:
    """
    Controller state:
      x = [e_y, e_psi, v_y, r, delta]

    where:
      e_y   = cross-track error
      e_psi = heading error
      v_y   = lateral velocity
      r     = yaw rate
      delta = steering angle
    """
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
    kappa_abs_max: float = 0.20
) -> np.ndarray:
    """
    Build curvature preview along the path based on current progress and speed.
    Preview is smoothed and clipped for robustness.
    """
    kappa_preview = np.zeros(N, dtype=float)
    for i in range(N):
        si = min(path.s_max, s0 + vx * (i + 1) * Ts)
        kappa_preview[i] = path.curvature(si)

    kappa_preview = smooth_and_clip_preview(
        kappa_preview,
        window=smooth_window,
        kappa_abs_max=kappa_abs_max
    )
    return kappa_preview


# -----------------------------
# Baseline linear discrete model for MPC
# -----------------------------
def continuous_matrices(vx: float, p: VehicleParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x = [ey, epsi, vy, r, delta], u = delta_dot, w = kappa
    xdot = A x + B u + E w
    """
    vx_safe = max(0.2, abs(vx)) * np.sign(vx)

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


# -----------------------------
# OSQP MPC (baseline: constant Ad,Bd,Ed for chosen speed)
# -----------------------------
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
    """
    z = [x1..xN, u0..u_{N-1}]
    constraints:
      x1 - Bd*u0 = Ad*x0 + Ed*kappa0
      x_{k+1} - Ad*xk - Bd*uk = Ed*kappak
      bounds on delta (state idx 4) and u=delta_dot
    """
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray, Ed: np.ndarray,
                 p: VehicleParams, mpc: MPCConfig):
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
            max_iter=4000
        )

    def _idx_x(self, k: int) -> int:
        return k * self.n

    def _idx_u(self, k: int) -> int:
        return self.N * self.n + k * self.m

    def _build_P(self) -> sp.csc_matrix:
        Q = np.diag([
            self.mpc.w_ey,
            self.mpc.w_epsi,
            self.mpc.w_vy,
            self.mpc.w_r,
            self.mpc.w_delta
        ])
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

    def solve(self, x0: np.ndarray, kappa_preview: np.ndarray, vx: float, L: float) -> float:
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
            beq[k * 5:(k + 1) * 5] = self.Ed[:, 0] * float(kappa_preview[k])

        self.l[:self.neq] = beq
        self.u[:self.neq] = beq

        dmax = self.p.delta_max
        umax = self.p.delta_dot_max
        self.l[self.neq:self.neq + self.N] = -dmax
        self.u[self.neq:self.neq + self.N] = dmax
        self.l[self.neq + self.N:] = -umax
        self.u[self.neq + self.N:] = umax

        self.prob.update(q=self.q, l=self.l, u=self.u)
        res = self.prob.solve()
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP failed: {res.info.status}")

        z = res.x
        u0 = float(z[self._idx_u(0)])
        return float(np.clip(u0, -umax, umax))


# -----------------------------
# Demo run
# -----------------------------
def build_waypoints() -> np.ndarray:
    return np.array([
        [0.0, 0.0],
        [10.0, 4.0],
        [20.0, 0.0],
        [30.0, 4.0],
        [40.0, 0.0],
        [50.0, 4.0],
        [60.0, 0.0],
        [70.0, 4.0],
    ], dtype=float)


def main():
    p = VehicleParams()
    mcfg = MPCConfig(Ts=0.05, horizon_s=2.5)

    waypoints = build_waypoints()
    path = SplinePath.from_waypoints(waypoints)

    # Keep speed setting here so you can change it yourself later
    vx = 7.0

    A, B, E = continuous_matrices(vx, p)
    Ad, Bd, Ed = c2d_zoh_two_inputs(A, B, E, mcfg.Ts)

    mpc = OSQPMPC(Ad, Bd, Ed, p, mcfg)

    s = GlobalState(
        x=0.0,
        y=5.0,
        psi=np.deg2rad(2.0),
        vx=vx,
        vy=0.0,
        r=0.0,
        delta=0.0
    )

    T_end = 50.0
    steps = int(T_end / mcfg.Ts)

    goal_x, goal_y = waypoints[-1]
    goal_radius = 2.0
    s_end_margin = 3.0

    T = np.zeros(steps)
    Xg = np.zeros((steps, 7))
    Xe = np.zeros((steps, 5))
    U = np.zeros(steps)
    K = np.zeros(steps)
    solve_fail = np.zeros(steps, dtype=int)

    s_guess = 0.0
    last_valid_idx = steps - 1
    stopped_on_goal = False

    for k in range(steps):
        t = k * mcfg.Ts
        T[k] = t

        xerr, s0, kappa = error_state(path, s_guess, s)
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
            u0 = mpc.solve(xerr, kappa_preview, vx=vx, L=p.L)
        except RuntimeError:
            u0 = 0.0
            solve_fail[k] = 1

        s = rk4_step(s, u0, mcfg.Ts, p)

        Xe[k, :] = xerr
        U[k] = u0
        K[k] = kappa
        Xg[k, :] = np.array([s.x, s.y, s.psi, s.vx, s.vy, s.r, s.delta])

        dist_to_goal = np.hypot(s.x - goal_x, s.y - goal_y)
        if dist_to_goal <= goal_radius and s0 >= path.s_max - s_end_margin:
            last_valid_idx = k
            stopped_on_goal = True
            print(f"Stopping simulation at t = {t:.2f} s")
            print(f"Distance to goal = {dist_to_goal:.2f} m")
            break
    else:
        last_valid_idx = steps - 1

    T = T[:last_valid_idx + 1]
    Xg = Xg[:last_valid_idx + 1, :]
    Xe = Xe[:last_valid_idx + 1, :]
    U = U[:last_valid_idx + 1]
    K = K[:last_valid_idx + 1]
    solve_fail = solve_fail[:last_valid_idx + 1]

    ey = Xe[:, 0]
    epsi = Xe[:, 1]
    rms_ey = float(np.sqrt(np.mean(ey**2)))
    max_ey = float(np.max(np.abs(ey)))
    rms_epsi = float(np.sqrt(np.mean(epsi**2)))
    effort_delta = trapezoid(Xe[:, 4]**2, T)
    effort_deltadot = trapezoid(U**2, T)
    final_dist_to_goal = float(np.hypot(Xg[-1, 0] - goal_x, Xg[-1, 1] - goal_y))

    num_failures = int(np.sum(solve_fail))
    fallback_ratio = float(num_failures / max(1, len(solve_fail)))

    print("=== Baseline metrikker ===")
    print(f"RMS e_y: {rms_ey:.3f} m, max|e_y|: {max_ey:.3f} m")
    print(f"RMS e_psi: {np.rad2deg(rms_epsi):.2f} deg")
    print(f"Effort ∫delta^2 dt: {effort_delta:.3f}")
    print(f"Effort ∫delta_dot^2 dt: {effort_deltadot:.3f}")
    print(f"Simulert tid: {T[-1]:.2f} s")
    print(f"Sluttavstand til mål: {final_dist_to_goal:.2f} m")
    print(f"Stoppet på målområde: {stopped_on_goal}")
    print(f"Antall solverfeil/fallback: {num_failures}")
    print(f"Andel fallback: {100.0 * fallback_ratio:.2f} %")

    s_plot = np.linspace(0, path.s_max, 800)
    x_ref = path.sx(s_plot)
    y_ref = path.sy(s_plot)

    plt.figure()
    plt.plot(x_ref, y_ref, label="Referanse")
    plt.plot(Xg[:, 0], Xg[:, 1], label="Spor")
    plt.scatter([goal_x], [goal_y], marker="x", s=100, label="Sluttpunkt")
    circle = plt.Circle((goal_x, goal_y), goal_radius, fill=False, linestyle="--")
    plt.gca().add_patch(circle)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Bane vs spor (baseline, lineær dekkmodell)")
    plt.legend()

    plt.figure()
    plt.plot(T, Xe[:, 0])
    plt.grid(True)
    plt.title("e_y(t)")
    plt.xlabel("t [s]")
    plt.ylabel("m")

    plt.figure()
    plt.plot(T, np.rad2deg(Xe[:, 1]))
    plt.grid(True)
    plt.title("e_psi(t)")
    plt.xlabel("t [s]")
    plt.ylabel("deg")

    plt.figure()
    plt.plot(T, np.rad2deg(Xe[:, 4]), label="delta")
    plt.grid(True)
    plt.title("delta(t)")
    plt.xlabel("t [s]")
    plt.ylabel("deg")
    plt.legend()

    plt.figure()
    plt.plot(T, np.rad2deg(U), label="delta_dot")
    plt.grid(True)
    plt.title("delta_dot(t)")
    plt.xlabel("t [s]")
    plt.ylabel("deg/s")
    plt.legend()

    plt.figure()
    plt.step(T, solve_fail, where="post")
    plt.grid(True)
    plt.title("Solverfeil / fallback")
    plt.xlabel("t [s]")
    plt.ylabel("1 = fallback")

    plt.show()


if __name__ == "__main__":
    main()