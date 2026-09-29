"""Kinematic up-scatter thresholds of CMB photons by inverse Compton (notebook cells 28–35, "Fotoionización secundaria").

Kinematics (derived, checked with sympy in tests/test_thresholds.py from 4-momentum conservation):
    head-on collision with back-scattering gives the maximum photon energy
        E1_max = g² ε / (1 + 2 g ε/mc²),   g = γ(1+β)
    whose γ ≫ 1 limit is B&G 1970 Eq. (2.50), E1 ≤ γmc² Γ/(1+Γ), Γ = 4εγ/mc².
    Inverting for a target E1: g = r + √(r² + E1/ε), r = E1/mc², and γ = (g² + 1)/(2g).
Photon energies: percentiles of the blackbody photon-number distribution x²/(eˣ−1) (x = ε/kT_CMB), total 2ζ(3):
    x_hi: fraction (1 − tol) of photons below it;  x_lo: fraction tol below it  (tol = upscatter_tolerance).
Curves (per z):
    K_cut(z):  electron energy for which an x_hi photon reaches upscatter_min_eV (10.2 eV, Lyα)
    K_hard(z): electron energy for which an x_lo photon reaches upscatter_target_eV (1 keV)
"""

import math

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from . import constants as K

P = K._P
_TOTAL = 2.0 * float(mp.zeta(3))            # ∫_0^∞ x²/(eˣ−1) dx = 2ζ(3)


def _cdf(x):
    val, _ = integrate.quad(lambda y: y * y / math.expm1(y) if y > 0 else 0.0, 0.0, x, epsabs=0.0, epsrel=1e-12)
    return val / _TOTAL


def photon_percentile(frac):
    """x such that a fraction `frac` of CMB photons has ε/kT below x."""
    return brentq(lambda x: _cdf(x) - frac, 1e-8, 60.0, xtol=1e-14)


def gamma_for_upscatter(E1_J, eps_J):
    r = E1_J / K.m_e_c2
    g = r + math.sqrt(r * r + E1_J / eps_J)
    return (g * g + 1.0) / (2.0 * g)


def threshold_curves(z):
    """(K_cut [eV], K_hard [eV]) arrays for the redshifts z."""
    tol = P["upscatter_tolerance"]
    x_hi, x_lo = photon_percentile(1.0 - tol), photon_percentile(tol)
    z = np.atleast_1d(np.asarray(z, dtype=float))
    kT = K.k_B * K.T_CMB0 * (1.0 + z)
    Kc = np.array([(gamma_for_upscatter(P["upscatter_min_eV"] * K.e, x_hi * t) - 1.0) * K.m_e_c2_eV for t in kT])
    Kh = np.array([(gamma_for_upscatter(P["upscatter_target_eV"] * K.e, x_lo * t) - 1.0) * K.m_e_c2_eV for t in kT])
    return Kc, Kh
