# stranglib/integrators.py
import numpy as np
from numpy.linalg import norm
from .system import m, e, H_physical
from scipy.integrate import solve_ivp

def simulate_analytic(q0, p0, t_span, t_eval, E_func):
    """
    Integra las ecuaciones de Hamilton en 1D usando solve_ivp (SciPy).

    Parámetros
    ----------
    q0, p0 : float
        Condiciones iniciales de posición y momento.
    t_span : (t_ini, t_fin)
        Intervalo de integración.
    t_eval : array_like
        Tiempos en los que se evalúa la solución.
    E_func : callable
        Campo eléctrico E(t).

    Devuelve
    --------
    traj  : ndarray
        q(t) evaluado en t_eval.
    mom   : ndarray
        p(t) evaluado en t_eval.
    times : ndarray
        Copia de t_eval (o sol.t).
    Hs    : ndarray
        Energía física H(q,p,t) evaluada en cada tiempo.
    """
    def hamilton_eqs(t, X):
        q, p = X
        E = E_func(t)
        return [p / m, e * E]

    sol = solve_ivp(
        hamilton_eqs,
        t_span,
        [q0, p0],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    t = sol.t
    q = sol.y[0]
    p = sol.y[1]

    # Energía física en cada tiempo
    Hs = np.array([H_physical(qi, pi, ti, E_func) for qi, pi, ti in zip(q, p, t)])

    return q, p, t, Hs

def strang_step(q, p, t, Eext, dt, E_func, E_dot_func):
    """
    Un paso del integrador de Strang en el espacio extendido (q, p, t, Eext).
    """
    # Flujo cinético (medio paso)
    q = q + 0.5 * dt * (p / m)
    t = t + 0.5 * dt * 0.5

    # Flujo del campo (paso completo en p y Eext)
    t_mid = t + 0.5 * dt * 0.5
    p = p + dt * e * E_func(t_mid)
    Eext = Eext + dt * e * q * E_dot_func(t_mid)
    t = t + dt * 0.5

    # Flujo cinético (medio paso final)
    q = q + 0.5 * dt * (p / m)
    t = t + 0.5 * dt * 0.5

    return q, p, t, Eext


def rk4_step(q, p, t, dt, E_func):
    """
    Un paso de RK4 para el sistema (q, p, t):
        dq/dt = p/m
        dp/dt = e E(t)
    """
    def f_q(p):
        return p / m

    def f_p(t):
        return e * E_func(t)

    k1q = f_q(p)
    k1p = f_p(t)

    k2q = f_q(p + 0.5 * dt * k1p)
    k2p = f_p(t + 0.5 * dt)

    k3q = f_q(p + 0.5 * dt * k2p)
    k3p = f_p(t + 0.5 * dt)

    k4q = f_q(p + dt * k3p)
    k4p = f_p(t + dt)

    q_new = q + dt * (k1q + 2*k2q + 2*k3q + k4q) / 6
    p_new = p + dt * (k1p + 2*k2p + 2*k3p + k4p) / 6

    return q_new, p_new, t + dt


def simulate_strang(q0, p0, t0, E0, dt, steps, E_func, E_dot_func):
    """
    Integra usando Strang en el espacio extendido.

    Devuelve:
        traj   : array de q(t)
        mom    : array de p(t)
        times  : array de tiempos t
        Hs     : energía física H(q,p,t)
        Hexts  : energía extendida H + Eext
    """
    q, p, t, Eext = q0, p0, t0, E0

    traj, mom, times = [], [], []
    Hs, Hexts = [], []

    for _ in range(steps):
        # avanzar un paso
        q, p, t, Eext = strang_step(q, p, t, Eext, dt, E_func, E_dot_func)

        # registrar
        traj.append(q)
        mom.append(p)
        times.append(t)

        H = H_physical(q, p, t, E_func)
        Hs.append(H)
        Hexts.append(H + Eext)

    return (np.array(traj),
            np.array(mom),
            np.array(times),
            np.array(Hs),
            np.array(Hexts))


def simulate_rk4(q0, p0, t0, dt, steps, E_func):
    """
    Integra usando RK4 solo en el espacio físico (q, p, t).

    Devuelve:
        traj  : q(t)
        mom   : p(t)
        times : t
        Hs    : H(q,p,t)
    """
    q, p, t = q0, p0, t0
    traj, mom, times, Hs = [], [], [], []

    for _ in range(steps):
        traj.append(q)
        mom.append(p)
        times.append(t)
        Hs.append(H_physical(q, p, t, E_func))
        q, p, t = rk4_step(q, p, t, dt, E_func)

    return (np.array(traj),
            np.array(mom),
            np.array(times),
            np.array(Hs))


def jacobian_step_1d(q, p, t, Eext, dt, E_func, E_dot_func, eps=1e-6):
    """
    Jacobiano numérico del mapa de un paso de Strang en 1D
    para el vector (q, p, t, Eext).
    """
    x = np.array([q, p, t, Eext], dtype=float)
    n = 4
    J = np.zeros((n, n), dtype=float)

    q0, p0, t0, E0 = strang_step(q, p, t, Eext, dt, E_func, E_dot_func)
    f0 = np.array([q0, p0, t0, E0], dtype=float)

    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps

        q1, p1, t1, E1 = strang_step(
            x[0] + dx[0],
            x[1] + dx[1],
            x[2] + dx[2],
            x[3] + dx[3],
            dt,
            E_func,
            E_dot_func,
        )
        f = np.array([q1, p1, t1, E1], dtype=float)
        J[:, i] = (f - f0) / eps

    return J

