import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from scipy.interpolate import CubicSpline
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ----------------------------
# 1) Geometri / hjelpefunksjoner
# ----------------------------
def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------
# 2) Lag bane x(s), y(s)
# ----------------------------
class ParametricPath2D:
    """
    Path parameterized as x(s), y(s) using a cubic spline through waypoints.
    NOTE: s here is a "chord-length-like" parameter based on waypoint distances.
    This satisfies the regularity requirement p'(s) != 0, but is not guaranteed
    to be exact arc length of the spline.
    """
    def __init__(self, waypoints_xy: np.ndarray):
        waypoints_xy = np.asarray(waypoints_xy, dtype=float)
        if waypoints_xy.ndim != 2 or waypoints_xy.shape[1] != 2:
            raise ValueError("waypoints_xy må være av shape (N,2)")

        self.wp = waypoints_xy
        ds = np.linalg.norm(np.diff(self.wp, axis=0), axis=1)
        if np.any(ds < 1e-12):
            raise ValueError("Waypoints inneholder duplikater/for korte segmenter.")

        s_wp = np.concatenate(([0.0], np.cumsum(ds)))
        self.s_wp = s_wp
        self.L = float(s_wp[-1])

        if not SCIPY_OK:
            raise ImportError("Dette skriptet trenger scipy (CubicSpline). Installer: pip install scipy")

        self.spline_x = CubicSpline(s_wp, self.wp[:, 0], bc_type="natural")
        self.spline_y = CubicSpline(s_wp, self.wp[:, 1], bc_type="natural")
        self.spline_dx = self.spline_x.derivative(1)
        self.spline_dy = self.spline_y.derivative(1)

    def pos(self, s: float) -> np.ndarray:
        s = float(s)
        return np.array([self.spline_x(s), self.spline_y(s)], dtype=float)

    def dpos_ds(self, s: float) -> np.ndarray:
        s = float(s)
        return np.array([self.spline_dx(s), self.spline_dy(s)], dtype=float)

    def tangent_heading(self, s: float) -> float:
        dp = self.dpos_ds(s)
        n = np.linalg.norm(dp)
        if n < 1e-12:
            return 0.0
        return float(np.arctan2(dp[1], dp[0]))


# ----------------------------
# 3) Full PF i tråd med Breivik & Fossen (ideal particle + path particle)
# ----------------------------
def simulate_breivik_fossen_pf(
    path: ParametricPath2D,
    s_p0: float,
    ax0: float,
    ay0: float,
    Ud: float = 2.0,          # desired speed of ideal particle
    Delta_e: float = 5.0,     # lookahead parameter (>0)
    gamma: float = 2.0,       # along-track gain (>0)
    dt: float = 0.02,
    T: float = 20.0,
    keep_on_path: bool = True,
    # optional: heading dynamics (set k_chi = None for instantaneous chi=chi_d)
    k_chi: float | None = None,
    chi0: float = 0.0,
):
    """
    Closed-loop Guidance-Based Path Following (2D) consistent with Breivik & Fossen:

    Given:
      - Desired path p_p(s_p) = [x(s_p), y(s_p)]
      - Actual (ideal) particle p_a and heading chi

    Compute:
      eps = R_p^T (p_a - p_p) = [s_along, e]^T
      chi_r = atan2(-e, Delta_e)
      chi_d = chi_p + chi_r
      U_p   = Ud*cos(chi_r) + gamma*s_along
      s_p_dot = U_p / ||p'(s_p)||

    Update:
      - Ideal particle kinematics:
          if k_chi is None: chi <- chi_d instantly
              p_a_dot = Ud [cos(chi_d), sin(chi_d)]
          else: first-order heading:
              chi_dot = k_chi * wrap(chi_d - chi)
              p_a_dot = Ud [cos(chi), sin(chi)]
      - Path particle parameter:
          s_p <- s_p + s_p_dot * dt
    """

    if gamma <= 0:
        raise ValueError("gamma må være > 0")
    if Ud <= 0:
        raise ValueError("Ud må være > 0")
    if Delta_e <= 0:
        raise ValueError("Delta_e må være > 0")

    N = int(np.ceil(T / dt))
    s_p = float(s_p0)
    ax, ay = float(ax0), float(ay0)
    chi = float(chi0)

    hist = {
        "t": np.zeros(N),
        "s_p": np.zeros(N),
        "ppx": np.zeros(N),
        "ppy": np.zeros(N),
        "s_along": np.zeros(N),
        "e": np.zeros(N),
        "Up": np.zeros(N),
        "chi_p": np.zeros(N),
        "chi_r": np.zeros(N),
        "chi_d": np.zeros(N),
        "ax": np.zeros(N),
        "ay": np.zeros(N),
        "chi": np.zeros(N),
    }

    for k in range(N):
        t = k * dt

        # Clamp path parameter
        if keep_on_path:
            s_p = clamp(s_p, 0.0, path.L)

        # Path particle quantities
        p_p = path.pos(s_p)
        chi_p = path.tangent_heading(s_p)

        # Errors in PATH frame
        R_p = rot2(chi_p)
        p_a = np.array([ax, ay], dtype=float)
        eps = R_p.T @ (p_a - p_p)
        s_along = float(eps[0])
        e = float(eps[1])

        # Guidance law (lookahead)
        chi_r = float(np.arctan2(-e, Delta_e))
        chi_d = wrap_angle(float(chi_p + chi_r))

        # Path particle speed (virtual input)
        Up = float(Ud * np.cos(chi_r) + gamma * s_along)

        # Path parameter update (general parameterization)
        dp = path.dpos_ds(s_p)
        speed_scale = max(np.linalg.norm(dp), 1e-12)
        s_p_dot = Up / speed_scale

        # --- Update ACTUAL (ideal) particle ---
        if k_chi is None:
            # instantaneous heading
            chi = chi_d
            ax += Ud * np.cos(chi) * dt
            ay += Ud * np.sin(chi) * dt
        else:
            # first-order heading dynamics
            chi_dot = k_chi * wrap_angle(chi_d - chi)
            chi = wrap_angle(chi + chi_dot * dt)
            ax += Ud * np.cos(chi) * dt
            ay += Ud * np.sin(chi) * dt

        # --- Update path particle parameter ---
        s_p = s_p + s_p_dot * dt

        # Log
        hist["t"][k] = t
        hist["s_p"][k] = s_p
        hist["ppx"][k] = p_p[0]
        hist["ppy"][k] = p_p[1]
        hist["s_along"][k] = s_along
        hist["e"][k] = e
        hist["Up"][k] = Up
        hist["chi_p"][k] = chi_p
        hist["chi_r"][k] = chi_r
        hist["chi_d"][k] = chi_d
        hist["ax"][k] = ax
        hist["ay"][k] = ay
        hist["chi"][k] = chi

    return hist


# ----------------------------
# 4) Animasjon + plots
# ----------------------------
if __name__ == "__main__":
    waypoints = np.array([
        [0, 0],
        [20, 0],
        [20, 5],
        [0, 5],
        [0, 0],
    ], dtype=float)

    path = ParametricPath2D(waypoints_xy=waypoints)

    # Start path-partikkel
    s_p0 = path.L * 0.10

    # Start actual (ideal) particle
    ax0, ay0 = 5.0, -2.0

    # Breivik & Fossen parametre
    Ud = 2.0
    Delta_e = 5.0
    gamma = 2.0

    dt = 0.02
    T = 20.0

    # Set k_chi=None for ideal instantaneous heading tracking (matches ideal particle assumption)
    # Or set e.g. k_chi=3.0 for a more realistic heading response
    hist = simulate_breivik_fossen_pf(
        path=path,
        s_p0=s_p0,
        ax0=ax0,
        ay0=ay0,
        Ud=Ud,
        Delta_e=Delta_e,
        gamma=gamma,
        dt=dt,
        T=T,
        keep_on_path=True,
        k_chi=None,   # try also: 3.0
        chi0=np.deg2rad(0.0),
    )

    # Background path for plotting
    s_grid = np.linspace(0, path.L, 500)
    p_grid = np.array([path.pos(s) for s in s_grid])

    # --- Figure with 2x2 axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_xy = axs[0, 0]
    ax_s  = axs[0, 1]
    ax_e  = axs[1, 0]
    ax_u  = axs[1, 1]

    for ax in axs.flat:
        ax.set_box_aspect(1)

    # XY plot
    ax_xy.plot(p_grid[:, 0], p_grid[:, 1], label="Bane x(s), y(s)")
    ax_xy.plot(waypoints[:, 0], waypoints[:, 1], "o", label="Waypoints")

    actual_dot, = ax_xy.plot([], [], "rx", markersize=10, label="Actual (ideal) particle")
    actual_trail, = ax_xy.plot([], [], "r:", linewidth=1.0, label="Actual trail")

    path_dot, = ax_xy.plot([], [], "ko", markersize=7, label="Path particle")
    trail_line, = ax_xy.plot([], [], "--", linewidth=1.5, label="Path trail")

    status_text = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes,
                             va="top", ha="left")

    ax_xy.grid(True)
    ax_xy.legend()
    ax_xy.set_title("Guidance-Based Path Following (Breivik & Fossen, 2D)")

    # Fixed viewing window based on path extents
    x_min, x_max = np.min(p_grid[:, 0]) - 5, np.max(p_grid[:, 0]) + 5
    y_min, y_max = np.min(p_grid[:, 1]) - 5, np.max(p_grid[:, 1]) + 5
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)

    # Time axis
    t_all = hist["t"]

    def setup_time_axis(ax, y, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_xlim(t_all[0], t_all[-1])
        ymin_, ymax_ = np.min(y), np.max(y)
        pad_ = 0.1 * (ymax_ - ymin_ + 1e-9)
        ax.set_ylim(ymin_ - pad_, ymax_ + pad_)
        line, = ax.plot([], [], linewidth=2)
        v = ax.axvline(t_all[0])
        return line, v

    s_line, v_s = setup_time_axis(ax_s, hist["s_along"], "Along-track error s_along(t)", "s_along [m]")
    e_line, v_e = setup_time_axis(ax_e, hist["e"], "Cross-track error e(t)", "e [m]")
    u_line, v_u = setup_time_axis(ax_u, hist["Up"], "Up(t) = Ud cos(chi_r) + gamma s_along", "Up [m/s]")

    # Animation frames
    stride = 2
    frames = np.arange(0, len(t_all), stride)

    def init():
        path_dot.set_data([], [])
        trail_line.set_data([], [])
        actual_dot.set_data([], [])
        actual_trail.set_data([], [])
        status_text.set_text("")

        s_line.set_data([], [])
        e_line.set_data([], [])
        u_line.set_data([], [])

        v_s.set_xdata([t_all[0], t_all[0]])
        v_e.set_xdata([t_all[0], t_all[0]])
        v_u.set_xdata([t_all[0], t_all[0]])

        return (path_dot, trail_line, actual_dot, actual_trail, status_text,
                s_line, e_line, u_line, v_s, v_e, v_u)

    def update(i):
        k = frames[i]
        tk = t_all[k]

        # XY: path particle
        xk, yk = hist["ppx"][k], hist["ppy"][k]
        path_dot.set_data([xk], [yk])
        trail_line.set_data(hist["ppx"][:k+1], hist["ppy"][:k+1])

        # XY: actual particle
        axk, ayk = hist["ax"][k], hist["ay"][k]
        actual_dot.set_data([axk], [ayk])
        actual_trail.set_data(hist["ax"][:k+1], hist["ay"][:k+1])

        status_text.set_text(
            f"t = {tk:.2f} s\n"
            f"s_along = {hist['s_along'][k]:.2f} m\n"
            f"e = {hist['e'][k]:.2f} m\n"
            f"chi_r = {hist['chi_r'][k]:.2f} rad\n"
            f"Up = {hist['Up'][k]:.2f} m/s"
        )

        # Time plots
        s_line.set_data(t_all[:k+1], hist["s_along"][:k+1])
        e_line.set_data(t_all[:k+1], hist["e"][:k+1])
        u_line.set_data(t_all[:k+1], hist["Up"][:k+1])

        v_s.set_xdata([tk, tk])
        v_e.set_xdata([tk, tk])
        v_u.set_xdata([tk, tk])

        return (path_dot, trail_line, actual_dot, actual_trail, status_text,
                s_line, e_line, u_line, v_s, v_e, v_u)

    ani = FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=True, interval=1000 * dt * stride
    )

    plt.show()
