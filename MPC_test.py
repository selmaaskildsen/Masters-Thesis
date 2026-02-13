import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


# -----------------------------
# Parametre
# -----------------------------
class Params:
    def __init__(self):
        self.m = 1800.0
        self.Iz = 2600.0
        self.lf = 1.2
        self.lr = 1.6
        self.g = 9.81

        self.mu = 0.9
        self.Caf = 90000.0
        self.Car = 90000.0

        # Simplified Pacejka params
        self.Bf, self.Cf, self.Ef = 8.0, 1.3, 0.0
        self.Br, self.Cr, self.Er = 8.0, 1.3, 0.0

        self.dt = 0.05
        self.N = 30

        self.delta_max = np.deg2rad(30)
        self.ax_min, self.ax_max = -3.0, 2.0

        # "rate limit" as penalty (hard constraint variant comes later)
        self.deltadot_max = np.deg2rad(250)  # rad/s (used for scaling/tuning)
        self.vx_eps = 0.5


# -----------------------------
# Dekkmodeller
# -----------------------------
def Fy_dugoff(alpha, Fz, Ca, mu):
    Fy0 = Ca * alpha
    absFy0 = ca.fabs(Fy0) + 1e-8
    lam = (mu * Fz) / (2.0 * absFy0)
    f = ca.if_else(lam < 1.0, (2.0 - lam) * lam, 1.0)
    Fy = f * Fy0
    Fy = ca.fmax(ca.fmin(Fy, mu * Fz), -mu * Fz)
    return Fy

def Fy_pacejka(alpha, Fz, mu, B, C, E):
    D = mu * Fz
    return D * ca.sin(C * ca.atan(B*alpha - E*(B*alpha - ca.atan(B*alpha))))


# -----------------------------
# Dynamikk (symbolsk)
# -----------------------------
def make_dynamics(p: Params, tire_model="dugoff"):
    X = ca.SX.sym("X", 6)  # [x,y,psi,vx,vy,r]
    U = ca.SX.sym("U", 2)  # [delta, ax]
    x, y, psi, vx, vy, r = X[0], X[1], X[2], X[3], X[4], X[5]
    delta, ax = U[0], U[1]

    L = p.lf + p.lr
    Fzf = p.m * p.g * (p.lr / L)
    Fzr = p.m * p.g * (p.lf / L)

    vx_safe = ca.fmax(vx, p.vx_eps)
    alpha_f = delta - ca.atan2(vy + p.lf*r, vx_safe)
    alpha_r = -ca.atan2(vy - p.lr*r, vx_safe)

    if tire_model == "dugoff":
        Fyf = Fy_dugoff(alpha_f, Fzf, p.Caf, p.mu)
        Fyr = Fy_dugoff(alpha_r, Fzr, p.Car, p.mu)
    elif tire_model == "pacejka":
        Fyf = Fy_pacejka(alpha_f, Fzf, p.mu, p.Bf, p.Cf, p.Ef)
        Fyr = Fy_pacejka(alpha_r, Fzr, p.mu, p.Br, p.Cr, p.Er)
    else:
        raise ValueError("tire_model must be 'dugoff' or 'pacejka'")

    # enkel long. kraft: Fxr = m*ax
    Fxr = p.m * ax
    Fxf = 0.0

    # kinematikk
    xdot = vx*ca.cos(psi) - vy*ca.sin(psi)
    ydot = vx*ca.sin(psi) + vy*ca.cos(psi)
    psidot = r

    # dynamikk (body frame)
    vxdot = (Fxf*ca.cos(delta) - Fyf*ca.sin(delta) + Fxr)/p.m + r*vy
    vydot = (Fyf*ca.cos(delta) + Fxf*ca.sin(delta) + Fyr)/p.m - r*vx
    rdot  = (p.lf*(Fyf*ca.cos(delta) + Fxf*ca.sin(delta)) - p.lr*Fyr)/p.Iz

    Xdot = ca.vertcat(xdot, ydot, psidot, vxdot, vydot, rdot)
    return ca.Function("f", [X, U], [Xdot])

def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u)
    k4 = f(x + dt*k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Referanse: sirkel (kan byttes til spline/NVDB senere)
# ref[k] = [x_ref, y_ref, psi_ref, vx_ref]
# -----------------------------
def circle_reference(x0, N, dt, R=50.0, vx_ref=8.0):
    # Vi lager en reference med konstant krumning 1/R.
    # For enkelhet antar vi s0 ut fra x0 (grovt), og lager framover langs sirkel.
    # Dette er bare for test. Senere bytter vi dette til spline-basert sampling.
    ref = np.zeros((N+1, 4))
    # start vinkel basert på posisjon
    theta0 = np.arctan2(x0[1], x0[0] + R)  # litt arbitrary, ok for demo
    ds = vx_ref * dt
    for k in range(N+1):
        theta = theta0 + (ds*k)/R
        x = R*np.cos(theta) - R
        y = R*np.sin(theta)
        psi = theta + np.pi/2
        ref[k] = [x, y, psi, vx_ref]
    return ref


# -----------------------------
# NMPC bygg: error-states kost
# -----------------------------
def build_nmpc(p: Params, tire_model="dugoff"):
    f = make_dynamics(p, tire_model=tire_model)
    nx, nu, N, dt = 6, 2, p.N, p.dt

    X = ca.SX.sym("X", nx, N+1)
    U = ca.SX.sym("U", nu, N)

    # Parameter: initial state + ref (N+1)*4
    nref = 4
    P = ca.SX.sym("P", nx + (N+1)*nref)

    x0 = P[0:nx]
    ref_flat = P[nx:]

    def ref_k(k):
        b = k*nref
        return ref_flat[b:b+nref]

    # Vekter (startverdier)
    Qey   = 80.0
    Qepsi = 20.0
    Qevx  = 2.0
    Qvy   = 2.0
    Qr    = 2.0

    Rdelta = 5.0
    Rax    = 0.5

    Sddelta = 200.0
    Sdax    = 2.0

    obj = 0
    g = []
    g.append(X[:,0] - x0)

    for k in range(N):
        xr, yr, psir, vxr = ref_k(k)[0], ref_k(k)[1], ref_k(k)[2], ref_k(k)[3]

        dx = X[0,k] - xr
        dy = X[1,k] - yr

        # Rotate into path frame to get ey (cross-track) and ex (lag)
        ex_path =  ca.cos(psir)*dx + ca.sin(psir)*dy
        ey_path = -ca.sin(psir)*dx + ca.cos(psir)*dy

        epsi = ca.atan2(ca.sin(X[2,k] - psir), ca.cos(X[2,k] - psir))
        evx  = X[3,k] - vxr

        # State penalty on error + "stability" states
        obj += Qey*(ey_path**2) + Qepsi*(epsi**2) + Qevx*(evx**2)
        obj += Qvy*(X[4,k]**2) + Qr*(X[5,k]**2)

        # Input penalty
        obj += Rdelta*(U[0,k]**2) + Rax*(U[1,k]**2)

        # Delta-input penalty
        if k > 0:
            ddelta = (U[0,k] - U[0,k-1])
            dax    = (U[1,k] - U[1,k-1])
            obj += Sddelta*(ddelta**2) + Sdax*(dax**2)

        # Dynamics constraint
        Xn = rk4(f, X[:,k], U[:,k], dt)
        g.append(X[:,k+1] - Xn)

    # Terminal cost (sterkere)
    xr, yr, psir, vxr = ref_k(N)[0], ref_k(N)[1], ref_k(N)[2], ref_k(N)[3]
    dx = X[0,N] - xr
    dy = X[1,N] - yr
    eyN = -ca.sin(psir)*dx + ca.cos(psir)*dy
    epsiN = ca.atan2(ca.sin(X[2,N] - psir), ca.cos(X[2,N] - psir))
    evxN = X[3,N] - vxr
    obj += 2*(Qey*(eyN**2) + Qepsi*(epsiN**2) + Qevx*(evxN**2))

    OPT = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g)
    nlp = {"f": obj, "x": OPT, "g": g, "p": P}

    # Bounds
    nX = nx*(N+1)
    nU = nu*N
    lbx = -np.inf*np.ones(nX+nU)
    ubx =  np.inf*np.ones(nX+nU)

    # input bounds
    off = nX
    for k in range(N):
        lbx[off + k*nu + 0] = -p.delta_max
        ubx[off + k*nu + 0] =  p.delta_max
        lbx[off + k*nu + 1] = p.ax_min
        ubx[off + k*nu + 1] = p.ax_max

    lbg = np.zeros(g.shape[0])
    ubg = np.zeros(g.shape[0])

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 200, "ipopt.tol": 1e-4}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return solver, (nx, nu, N), (lbx, ubx, lbg, ubg)


def pack_P(x0, ref):
    return np.concatenate([x0, ref.reshape(-1)])


def unpack(sol, dims):
    nx, nu, N = dims
    w = sol["x"].full().squeeze()
    nX = nx*(N+1)
    X = w[:nX].reshape((nx, N+1), order="F")
    U = w[nX:].reshape((nu, N), order="F")
    return X, U


# -----------------------------
# Simuler lukket sløyfe
# -----------------------------
def simulate_closed_loop(tire_model="dugoff", T=12.0):
    p = Params()
    solver, dims, bnds = build_nmpc(p, tire_model=tire_model)
    nx, nu, N = dims
    lbx, ubx, lbg, ubg = bnds

    steps = int(T/p.dt)

    # init state: litt off path
    x = np.array([0.0, 3.0, 0.3, 8.0, 0.0, 0.0])

    X_log = [x.copy()]
    U_log = []

    # warm start
    X_guess = np.tile(x.reshape(-1,1), (1, N+1))
    U_guess = np.zeros((nu, N))

    for t in range(steps):
        ref = circle_reference(x, N, p.dt, R=50.0, vx_ref=8.0)
        P = pack_P(x, ref)

        w0 = np.concatenate([X_guess.reshape(-1, order="F"), U_guess.reshape(-1, order="F")])

        sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=P)
        Xopt, Uopt = unpack(sol, dims)

        u = Uopt[:,0]
        U_log.append(u.copy())

        # step simulation with same RK4 model (truth-model = same as pred for now)
        f = make_dynamics(p, tire_model=tire_model)
        x_next = rk4(f, ca.DM(x), ca.DM(u), p.dt).full().squeeze()
        x = x_next
        X_log.append(x.copy())

        # shift warm-start
        X_guess = np.hstack([Xopt[:,1:], Xopt[:,-1:]])
        U_guess = np.hstack([Uopt[:,1:], Uopt[:,-1:]])

    return np.array(X_log), np.array(U_log), p


if __name__ == "__main__":
    X, U, p = simulate_closed_loop(tire_model="dugoff", T=15.0)

    plt.figure()
    plt.plot(X[:,0], X[:,1], label="trajectory")
    plt.axis("equal")
    plt.title("NMPC trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()

    plt.figure()
    plt.plot(U[:,0], label="delta [rad]")
    plt.plot(U[:,1], label="ax [m/s^2]")
    plt.title("Control inputs")
    plt.xlabel("step")
    plt.legend()

    plt.show()
