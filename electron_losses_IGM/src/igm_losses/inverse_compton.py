"""Inverse Compton energy loss on the CMB, general Klein–Nishina case (assumption A06, erratum E05, decision D15).

    −dE/dt = (4/3) σ_T c U_CMB γ²β² F_KN(b),     b = 4 γ k_B T_CMB / (m c²)

Physics
    F_KN(b) = (135/π⁴) ∫_0^∞ dx x³/(eˣ−1) ∫_0^1 dq q F(q, Γ)/(1+Γq)³,   Γ = x b,
    F(q, Γ) = 2q ln q + (1+2q)(1−q) + ½ (Γq)²(1−q)/(1+Γq)        [BlumenthalGould1970 Eqs. 2.48–2.49, p. 243]
    The factor q comes from weighting by the scattered photon energy ε1 = γmc² Γq/(1+Γq) in Eq. (2.56), p. 244
    (E05, confirmed by reading the PDF). 135/π⁴ normalizes F_KN → 1 in the Thomson limit (∫ q F(q,0) dq = 1/9,
    ∫ x³/(eˣ−1) dx = π⁴/15). The prefactor with β² is the Thomson loss for arbitrary β (γ²β² = p²c²/(mc²)²);
    B&G derive the KN kernel for γ ≫ 1, which is where F_KN differs from 1.
Limits (tests)
    small b: F ≈ 1 − (63/10)(⟨ε²⟩/⟨ε⟩)(γ/mc²) = 1 − 6.04 b  [B&G Eq. 2.28, p. 242; 24.15 γkT/mc² for a blackbody]
    large b: F → (45/(4π²)) (ln b − 5/6 − C_E − C_l) / b²   [B&G Eqs. 2.59–2.60, p. 244, normalized by Eq. 2.18]
Tabulation  scripts/compute_fkn_table.py (single writer) → provenance/data/F_KN_table.json, 200 points in
            b ∈ [1e-4, 1e4]; interpolation linear in log–log (D03, D15). Outside the table: F = 1 for b < 1e-4
            (error ≤ 6e-4, the first-order term) and the B&G asymptote for b > 1e4 (D15, explicit exception to D08).
"""

import json
import math
import pathlib

import mpmath as mp
import numpy as np
from scipy import integrate

from . import constants as K
from . import kinematics as KIN

ROOT = pathlib.Path(__file__).resolve().parents[3]
TABLE_FILE = ROOT / "provenance" / "data" / "F_KN_table.json"
B_MIN, B_MAX = 1.0e-4, 1.0e4
C_E = float(mp.euler)                                  # Euler's constant, B&G Eq. (2.60) quotes 0.5772
C_L = float(-6 / mp.pi ** 2 * mp.zeta(2, derivative=1))  # (6/π²) Σ ln k/k² = −(6/π²) ζ'(2); B&G quote 0.5700


def kernel(q, G):
    """B&G Eq. (2.48) bracket F(q, Γ)."""
    return 2 * q * math.log(q) + (1 + 2 * q) * (1 - q) + 0.5 * (G * q) ** 2 * (1 - q) / (1 + G * q) if q > 0 else 1.0


def compute_F_kn(b, epsrel=1e-10):
    """F_KN(b) by nested adaptive quadrature (inner q, outer x ∈ [0, 60]; e^-60 is negligible)."""
    def inner(x):
        G = x * b
        val, _ = integrate.quad(lambda q: q * kernel(q, G) / (1 + G * q) ** 3, 0.0, 1.0,
                                epsabs=0.0, epsrel=epsrel, limit=200,
                                points=[min(0.5, 1.0 / G)] if G > 2 else None)
        return val
    planck = lambda x: x ** 3 / math.expm1(x) if x > 0 else 0.0
    val, _ = integrate.quad(lambda x: planck(x) * inner(x), 0.0, 60.0, epsabs=0.0, epsrel=epsrel, limit=400)
    return 135.0 / math.pi ** 4 * val


def F_kn_asymptotic(b):
    """B&G Eq. (2.59) normalized to the Thomson loss (Eq. 2.18): (45/(4π²)) (ln b − 5/6 − C_E − C_l)/b²."""
    return 45.0 / (4.0 * math.pi ** 2) * (np.log(b) - 5.0 / 6.0 - C_E - C_L) / np.asarray(b) ** 2


_TABLE = None


def F_kn(b):
    """Interpolated F_KN(b) with the physical limits outside [1e-4, 1e4]."""
    global _TABLE
    if _TABLE is None:
        d = json.loads(TABLE_FILE.read_text(encoding="utf-8"))
        _TABLE = (np.log(d["b"]), np.log(d["F"]))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    out = np.exp(np.interp(np.log(np.clip(b, B_MIN, B_MAX)), *_TABLE))
    out[b < B_MIN] = 1.0
    hi = b > B_MAX
    out[hi] = F_kn_asymptotic(b[hi])
    return out if np.ndim(b) and b.size > 1 else float(out[0])


def b_parameter(K_J, T_CMB):
    """b = 4 γ k_B T / (m c²)."""
    return 4.0 * KIN.gamma(K_J) * K.k_B * T_CMB / K.m_e_c2


# [A06] target photons: CMB blackbody only
def loss_rate(K_J, medium):
    """−dE/dt [J s^-1] on the CMB at the medium's redshift."""
    return (4.0 / 3.0 * K.sigma_T * K.c * medium.U_CMB * KIN.gamma2beta2(K_J)
            * F_kn(b_parameter(K_J, medium.T_CMB)))
