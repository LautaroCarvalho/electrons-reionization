"""Cosmic time ↔ redshift and the Hubble rate, from astropy's Planck18 (assumption A10, decision D12).

Purpose
    One interpolator for z(t), t(z) and H(z) used by every figure (the notebook rebuilt it in almost every cell
    with 500–10000 points). astropy is too slow to call inside the ODE right-hand side, so ln t and ln H are
    tabulated on a grid in u = ln(1+z) and interpolated with cubic splines (monotone in t, smooth).

Library    astropy.cosmology.Planck18 (Planck 2018 VI, Table 2, TT,TE,EE+lowE+lensing+BAO; see A10).
From scratch the inversion t → z and the spline tables.

Accuracy   N_GRID is chosen by tests/test_cosmology.py: relative error < 1e-8 in t, H and (1+z) at points that are
           NOT grid nodes, against direct astropy evaluation (D12).
Range      0 ≤ z ≤ Z_MAX. Outside → ValueError (D08).
"""

import numpy as np
# [A10] flat ΛCDM, Planck 2018
from astropy.cosmology import Planck18
import astropy.units as u
from scipy.interpolate import CubicSpline

Z_MAX = 40.0        # covers z_init = 30 (adiabatic figure) and the Γ_ad map up to z = 15
N_GRID = 4000

_uz = np.linspace(0.0, np.log1p(Z_MAX), N_GRID)
_z = np.expm1(_uz)
_ln_t = np.log(Planck18.age(_z).to(u.s).value)
_ln_H = np.log(Planck18.H(_z).to(1 / u.s).value)
_t_of_u = CubicSpline(_uz, _ln_t)
_H_of_u = CubicSpline(_uz, _ln_H)
_u_of_lnt = CubicSpline(_ln_t[::-1], _uz[::-1])          # ln t increases as u decreases
T_MIN_S, T_MAX_S = float(np.exp(_ln_t[-1])), float(np.exp(_ln_t[0]))


def _check_z(z):
    z = np.asarray(z, dtype=float)
    if np.any(z < 0) or np.any(z > Z_MAX):
        raise ValueError(f"z outside [0, {Z_MAX}]")
    return z


def age(z):
    """Cosmic time t(z) [s]."""
    return np.exp(_t_of_u(np.log1p(_check_z(z))))


def hubble(z):
    """H(z) [s^-1]."""
    return np.exp(_H_of_u(np.log1p(_check_z(z))))


def redshift(t):
    """z(t) for cosmic time t [s]."""
    t = np.asarray(t, dtype=float)
    if np.any(t < T_MIN_S * (1 - 1e-12)) or np.any(t > T_MAX_S * (1 + 1e-12)):
        raise ValueError("t outside the tabulated range")
    return np.expm1(_u_of_lnt(np.log(t)))


def hubble_t(t):
    """H at cosmic time t [s^-1]."""
    return hubble(redshift(t))
