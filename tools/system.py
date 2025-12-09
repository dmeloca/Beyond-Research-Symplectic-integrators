import numpy as np

m = 1.0
e = 1.0

def make_field(omega=0.5):
    """
    Devuelve funciones E(t) y E_dot(t) = dE/dt para un campo sinusoidal.
    """
    def E(t):
        return np.sin(omega * t)

    def E_dot(t):
        return omega * np.cos(omega * t)

    return E, E_dot

def H_physical(q, p, t, E_func):
    """
    Hamiltoniano físico para 1D:
        H = p^2 / (2m) - e q E(t)
    """
    return 0.5 * (p**2) / m - e * q * E_func(t)
