# stranglib/integrators.py
from typing import Tuple

import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.integrate import solve_ivp
from scipy import symbols, solve, Eq

from .system import m, e, H_physical

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

def symplectization1_step(p_n: np.ndarray, r_n: np.ndarray, q_n: np.ndarray, t_n: np.ndarray, E: callable, dE_ds: callable, f: callable, df_dr: callable, h: np.ndarray = 0.1, charge: float = 1, m: float = 1, C: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    p_np1: np.ndarray = p_n + h * charge * E(t_n + C)
    r_np1: np.ndarray = r_n + h * charge * q_n * dE_ds(t_n + C)
    q_np1: np.ndarray = q_n + h * f(r_np1) * p_np1
    t_np1: float = t_n + (h/m) * df_dr(r_np1) * (np.linalg.norm(p_np1)**2) #*standard 2-euclidean norm
    # print(h * charge * q_n * dE_ds(t_n + C))

    return p_np1, r_np1, q_np1, t_np1

def simulate_symplectization1(p0: np.ndarray, r0: np.ndarray, q0: np.ndarray, t0: float, t_target: float, E: callable, dE_ds: callable, f: callable, df_dr: callable, charge: float = 1, m: float = 1, C: float = 0, n_steps: int = 10) -> pd.DataFrame:   
    if t_target < t0:
        raise ValueError(f"Target time {t_target} must be greater or equal than the initial time {t0}")
    if t_target == t0:
        print(f"Warning, initial time is the same as target time ({t_target}), so the output will be the initial values")
        return p0, r0, q0, t0
    
    h: float = (t_target - t0)/n_steps
    results: pd.DataFrame = pd.DataFrame({'p': [p0], 'r': [r0], 'q': [q0], 't': [t0]}, dtype=object)
    p_n: np.ndarray = p0
    r_n: np.ndarray = r0
    q_n: np.ndarray = q0
    t_n: np.ndarray = t0
    for i in range(n_steps):
        p_n, r_n, q_n, t_n = symplectization1_step(p_n, r_n, q_n, t_n, E, dE_ds, f, df_dr, h, charge, m, C)

        #*Add nth step to results
        nth_step_results: pd.DataFrame = pd.DataFrame({'p': [p_n], 'r': [r_n], 'q': [q_n], 't': [t_n]}, dtype=object)
        results = pd.concat([results, nth_step_results], ignore_index=True)

    return results

def symplectization2_solve_rnp1() -> np.ndarray:
    r, r_n, B_n = symbols('r_np1 r_n B_n', real=True)
    f = r
    if f is None:
        raise NotImplemented('f must be a valid expression')
    g = r - B_n * f - r_n
    return solve(Eq(g, 0), r)[0]

def symplectization2_step(p_n: np.ndarray, r_n: np.ndarray, q_n: np.ndarray, t_n: np.ndarray, E: callable, dE_ds: callable, f: callable, df_dr: callable, r_np1_solver: callable= symplectization2_solve_rnp1, h: float = 0.1, charge: float = 1, m: float = 1, C: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    B_n: np.ndarray = h * charge * q_n * dE_ds(t_n + C)
    # print("B_n", B_n)
    r_np1: np.ndarray = r_np1_solver(r_n=r_n, B_n=B_n)
    # print("r_np1", r_np1)
    p_np1: np.ndarray = p_n + h * charge * E(t_n + C) * f(r_np1)
    q_np1: np.ndarray = q_n + (h / m) * p_np1
    t_np1: float = t_n - h * df_dr(r_np1) * charge * q_n * E(t_n + C)

    return p_np1, r_np1, q_np1, t_np1

def simulate_symplectization2(p0: np.ndarray, r0: np.ndarray, q0: np.ndarray, t0: float, t_target: float, E: callable, dE_ds: callable, f: callable, df_dr: callable, r_np1_solver: callable= symplectization2_solve_rnp1, charge: float = 1, m: float = 1, C: float = 0, n_steps: int = 10) -> pd.DataFrame:   
    if t_target < t0:
        raise ValueError(f"Target time {t_target} must be greater or equal than the initial time {t0}")
    if t_target == t0:
        print(f"Warning, initial time is the same as target time ({t_target}), so the output will be the initial values")
        return p0, r0, q0, t0
    
    h: float = (t_target - t0)/n_steps
    results: pd.DataFrame = pd.DataFrame({'p': [p0], 'r': [r0], 'q': [q0], 't': [t0]}, dtype=object)
    p_n: np.ndarray = p0
    r_n: np.ndarray = r0
    q_n: np.ndarray = q0
    t_n: np.ndarray = t0
    for i in range(n_steps):
        p_n, r_n, q_n, t_n = symplectization2_step(p_n, r_n, q_n, t_n, E, dE_ds, f, df_dr, r_np1_solver, h, charge, m, C)

        #*Add nth step to results
        nth_step_results: pd.DataFrame = pd.DataFrame({'p': [p_n], 'r': [r_n], 'q': [q_n], 't': [t_n]}, dtype=object)
        results = pd.concat([results, nth_step_results], ignore_index=True)

    return results