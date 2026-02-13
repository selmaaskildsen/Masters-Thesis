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


# ----------------------------
# 2) Lag bane x(s), y(s)
# ----------------------------
class ParametricPath2D:
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
        dp = dp / n
        return float(np.arctan2(dp[1], dp[0]))


# ----------------------------
# 3) Online simulering: p_actual(t) beveger seg fram/tilbake i x
#    + Breivik & Fossen kontrollov for path-partikkelen
# ----------------------------
def simulate_with_moving_actual_x(
    path: ParametricPath2D,
    s0: float,
    x0: float,
    y0: float,
    A: float = 3.0,          # amplitude [m]
    f: float = 0.2,          # frekvens [Hz]
    Ud: float = 2.0,         # "desired speed"
    Delta_e: float = 5.0,    # lookahead parameter
    gamma: float = 10.0,
    dt: float = 0.02,
    T: float = 20.0,
    keep_on_path: bool = True
):
    """
    Faktisk partikkel (ekstern test):
        x_a(t) = x0 + A*sin(2*pi*f*t)
        y_a(t) = y0

    Breivik & Fossen (2D) path particle:
        eps = R_p^T (p - p_p) = [s, e]
        chi_r = atan2(-e, Delta_e)
        Up = Ud*cos(chi_r) + gamma*s
        s_p_dot = Up / ||p'(s_p)||
    """
    if gamma <= 0:
        raise ValueError("gamma må være > 0")
    if Ud <= 0:
        raise ValueError("Ud må være > 0")
    if Delta_e <= 0:
        raise ValueError("Delta_e må være > 0")
    if f < 0:
        raise ValueError("f må være >= 0")

    N = int(np.ceil(T / dt))
    s_p = float(s0)

    hist = {
        "t": np.zeros(N),
        "s_p": np.zeros(N),
        "ppx": np.zeros(N),
        "ppy": np.zeros(N),
        "s_err": np.zeros(N),
        "e_err": np.zeros(N),
        "Up": np.zeros(N),
        "xp": np.zeros(N),
        "chi_r": np.zeros(N),
        "chi_d": np.zeros(N),
        "ax": np.zeros(N),
        "ay": np.zeros(N),
    }

    for k in range(N):
        t = k * dt

        # Faktisk partikkelposisjon (fram/tilbake i x)
        ax = x0 + A * np.sin(2.0 * np.pi * f * t)
        ay = y0
        p_actual = np.array([ax, ay], dtype=float)

        # Path-partikkel (clamp på banen)
        if keep_on_path:
            s_p = clamp(s_p, 0.0, path.L)

        pp = path.pos(s_p)
        xp = path.tangent_heading(s_p)

        # Feil i PATH-rammen
        Rp = rot2(xp)
        eps = Rp.T @ (p_actual - pp)
        s_err, e_err = float(eps[0]), float(eps[1])

        # --- Breivik & Fossen guidance ---
        chi_r = float(np.arctan2(-e_err, Delta_e))
        chi_d = float(xp + chi_r)

        # Path particle speed (kontrolloven)
        Up = float(Ud * np.cos(chi_r) + gamma * s_err)

        # Parameteroppdatering
        dp = path.dpos_ds(s_p)
        speed_scale = max(np.linalg.norm(dp), 1e-12)
        s_p_dot = Up / speed_scale

        # Integrer
        s_p = s_p + s_p_dot * dt

        # Logg
        hist["t"][k] = t
        hist["s_p"][k] = s_p
        hist["ppx"][k] = pp[0]
        hist["ppy"][k] = pp[1]
        hist["s_err"][k] = s_err
        hist["e_err"][k] = e_err
        hist["Up"][k] = Up
        hist["xp"][k] = xp
        hist["chi_r"][k] = chi_r
        hist["chi_d"][k] = chi_d
        hist["ax"][k] = ax
        hist["ay"][k] = ay

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
    s0 = path.L * 0

    # Faktisk partikkel sin x-oscillasjon
    x0, y0 = 10.0, 2.0
    A = 4.0
    f = 0.15

    # Breivik & Fossen parametre
    Ud = 2.0
    Delta_e = 5.0
    gamma = 10.0   # <- endret til 50

    dt = 0.02
    T = 20.0

    hist = simulate_with_moving_actual_x(
        path=path,
        s0=s0,
        x0=x0,
        y0=y0,
        A=A,
        f=f,
        Ud=Ud,
        Delta_e=Delta_e,
        gamma=gamma,
        dt=dt,
        T=T,
        keep_on_path=True
    )

    # Bane for bakgrunn
    s_grid = np.linspace(0, path.L, 500)
    p_grid = np.array([path.pos(s) for s in s_grid])

    # --- Figur med 4 akser (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_xy = axs[0, 0]
    ax_s  = axs[0, 1]
    ax_e  = axs[1, 0]
    ax_u  = axs[1, 1]

    # Gjør alle fire "vinduer" like store
    for ax in axs.flat:
        ax.set_box_aspect(1)

    # XY-plot
    ax_xy.plot(p_grid[:, 0], p_grid[:, 1], label="Bane x(s), y(s)")
    ax_xy.plot(waypoints[:, 0], waypoints[:, 1], "o", label="Waypoints")

    actual_dot, = ax_xy.plot([], [], "rx", markersize=10, label="Faktisk partikkel")
    actual_trail, = ax_xy.plot([], [], "r:", linewidth=1.0, label="Faktisk spor")

    path_dot, = ax_xy.plot([], [], "ko", markersize=7, label="Path-partikkel")
    trail_line, = ax_xy.plot([], [], "--", linewidth=1.5, label="Path spor")

    status_text = ax_xy.text(0.02, 0.98, "", transform=ax_xy.transAxes,
                             va="top", ha="left")

    ax_xy.grid(True)
    ax_xy.legend()
    ax_xy.set_title("XY (Breivik & Fossen Up + lookahead)")

    # Fast XY-visningsvindu (IKKE avhengig av path). Basert på test-signal (x0, y0, A) + margin.
    margin_x = 10.0
    margin_y = 10.0
    X_MIN = x0 - abs(A) - margin_x
    X_MAX = x0 + abs(A) + margin_x
    Y_MIN = y0 - margin_y
    Y_MAX = y0 + margin_y

    ax_xy.set_xlim(X_MIN, X_MAX)
    ax_xy.set_ylim(Y_MIN, Y_MAX)

    # --- Tidsakse for plots
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

    s_line, v_s = setup_time_axis(ax_s, hist["s_err"], "Along-track error s(t)", "s [m]")
    e_line, v_e = setup_time_axis(ax_e, hist["e_err"], "Cross-track error e(t)", "e [m]")
    u_line, v_u = setup_time_axis(ax_u, hist["Up"], "Up(t) = Ud cos(chi_r) + gamma s", "Up [m/s]")

    # Frames / stride
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
            f"s = {hist['s_err'][k]:.2f} m\n"
            f"e = {hist['e_err'][k]:.2f} m\n"
            f"chi_r = {hist['chi_r'][k]:.2f} rad\n"
            f"Up = {hist['Up'][k]:.2f} m/s"
        )

        # time plots
        s_line.set_data(t_all[:k+1], hist["s_err"][:k+1])
        e_line.set_data(t_all[:k+1], hist["e_err"][:k+1])
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
