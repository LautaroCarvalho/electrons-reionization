"""Collisional excitation of H(1s) → np (n = 2…10) by electron impact (assumption A13; D01, D02, E12, E19).

Cross section per level (T = kinetic energy of the incident electron):
    T ≤ 3 keV   BE-scaled plane-wave Born [StoneKimDesclaux2002 Eqs. 1–2]:
                σ = (4π a0² R/T) (R/E) ∫_{Qmin}^{Qmax} f(Q) dQ/Q · T/(T + B + E),
                Q = (K a0)², Qmin,max = (√T ∓ √(T−E))²/R (non-relativistic kinematics), exact hydrogen GOS
                f(Q) = (ΔE/R) Q⁻¹ · 3 |∫ R_10 R_n1 j_1(√Q r) r² dr|²  (atomic units; derived, checked against the
                closed form 2¹³3³/(4Q+9)⁶ for 1s→2p obtained with sympy, tests/test_excitation.py).
    3–10 keV    Eq. (5) of Stone et al.: σ = 4π a0² R/(T+B+E) [a ln(T/R) + b + c R/T]  ("should be used for T > 3 keV")
    T > 10 keV  relativistic Bethe form [Inokuti1971 Eqs. 4.26–4.27, p. 328] with a = M_n², b = M_n² ln 4c_n
                (Eq. 4.18): σ = 8π a0² (R/mc²)/β² {a [ln(β²/(1−β²)) − β²] + b − a ln(2R/mc²)}
                (Stone: "a relativistic form … should be used for T > 10 keV").
Data     E_n, f, a, b, c and B = 13.5984 eV from Stone Table 1 (provenance/data/stone2002_H_1s_np.yaml, E19).
         R = Ry∞ from constants (Stone print R = 13.61 eV, rounded). The GOS uses ΔE = Ry∞(1 − 1/n²).
Energy loss per unit path and density: S(T) = Σ_n σ_n(T) E_n  [m² J]; −dE/dt = n_HI v S.
Tabulation  only the Born+BE part (T ≤ 3 keV) is tabulated, by scripts/compute_excitation_table.py (single writer)
            → provenance/data/excitation_table.json, per level on a grid in ln(T − E_n); interpolation linear in
            ln σ vs ln(T − E_n) (D03); σ_n = 0 for T ≤ E_n. Above 3 keV the analytic forms are evaluated directly.
            T > 1e14 eV → ValueError (D04, D08).
"""

import json
import math
import pathlib

import numpy as np
import yaml
from scipy import integrate
from scipy.special import genlaguerre, spherical_jn

from . import constants as K
from . import kinematics as KIN

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = yaml.safe_load((ROOT / "provenance" / "data" / "stone2002_H_1s_np.yaml").read_text(encoding="utf-8"))
TABLE_FILE = ROOT / "provenance" / "data" / "excitation_table.json"
LEVELS = sorted(DATA["levels"])
B_EV = DATA["B_eV"]
R_EV = K.Ry_inf_eV
T_BORN_MAX_EV = 3.0e3        # Stone p. 328: Eq. (5) for T > 3 keV
T_REL_EV = 1.0e4             # Stone p. 328: relativistic form for T > 10 keV (E12 decision)
T_MAX_EV = 1.0e14            # D04


# ------------------------------------------------------------------ hydrogen radial functions (atomic units)
def _R_n1(n, r):
    norm = math.sqrt((2.0 / n) ** 3 * math.factorial(n - 2) / (2.0 * n * math.factorial(n + 1)))
    rho = 2.0 * r / n
    return norm * np.exp(-rho / 2.0) * rho * genlaguerre(n - 2, 3)(rho)


def gos(n, Q):
    """Generalized oscillator strength f(Q) of 1s → np, Q = (K a0)² (dimensionless)."""
    k = math.sqrt(Q)
    dE = 1.0 - 1.0 / n ** 2
    upper = 60.0 * n                    # e^{-r(1+1/n)} makes the tail beyond ~60 n a0 negligible
    val, _ = integrate.quad(lambda r: 2.0 * math.exp(-r) * _R_n1(n, r) * spherical_jn(1, k * r) * r * r,
                            0.0, upper, limit=400, epsabs=0.0, epsrel=1e-11)
    return dE / Q * 3.0 * val ** 2


# ------------------------------------------------------------------ cross sections (T, E in eV; σ in m²)
def sigma_born_be(n, T):
    E = DATA["levels"][n]["E_eV"]
    if T <= E:
        return 0.0
    qmin = (math.sqrt(T) - math.sqrt(T - E)) ** 2 / R_EV
    qmax = (math.sqrt(T) + math.sqrt(T - E)) ** 2 / R_EV
    F, _ = integrate.quad(lambda lnq: gos(n, math.exp(lnq)), math.log(qmin), math.log(qmax),
                          limit=200, epsabs=0.0, epsrel=1e-9)
    return 4.0 * math.pi * K.a0 ** 2 * R_EV / T * (R_EV / E) * F * T / (T + B_EV + E)


def sigma_asymptotic(n, T):
    L = DATA["levels"][n]
    E = L["E_eV"]
    return 4.0 * math.pi * K.a0 ** 2 * R_EV / (T + B_EV + E) * (L["a"] * math.log(T / R_EV) + L["b"] + L["c"] * R_EV / T)


def sigma_relativistic(n, T):
    L = DATA["levels"][n]
    b2 = float(KIN.beta2(T * K.e))
    g2b2 = float(KIN.gamma2beta2(T * K.e))      # = β²/(1−β²) exactly (no cancellation at T ≫ mc²)
    Rm = R_EV / K.m_e_c2_eV
    return 8.0 * math.pi * K.a0 ** 2 * Rm / b2 * (L["a"] * (math.log(g2b2) - b2) + L["b"] - L["a"] * math.log(2.0 * Rm))


def sigma_direct(n, T):
    """σ_n(T) [m²] evaluated directly (slow below 3 keV); used to build the table and in tests."""
    if T <= DATA["levels"][n]["E_eV"]:
        return 0.0
    if T <= T_BORN_MAX_EV:
        return sigma_born_be(n, T)
    if T <= T_REL_EV:
        return sigma_asymptotic(n, T)
    return sigma_relativistic(n, T)


# ------------------------------------------------------------------ tabulated form used by the ODE
_TAB = None


def _table():
    global _TAB
    if _TAB is None:
        d = json.loads(TABLE_FILE.read_text(encoding="utf-8"))
        _TAB = {int(n): (np.log(np.array(v["dT_eV"])), np.log(np.array(v["sigma_m2"]))) for n, v in d["levels"].items()}
    return _TAB


def sigma(n, T):
    """σ_n(T) [m²]: table (Born+BE) for T ≤ 3 keV, analytic Eq. (5) / Inokuti above. T in eV (scalar)."""
    E = DATA["levels"][n]["E_eV"]
    if T > T_MAX_EV * (1 + 1e-12):
        raise ValueError("T above the excitation range (1e14 eV, D04)")
    if T <= E:
        return 0.0
    if T > T_BORN_MAX_EV:
        return sigma_asymptotic(n, T) if T <= T_REL_EV else sigma_relativistic(n, T)
    ldT, ls = _table()[n]
    x = math.log(T - E)
    if x < ldT[0]:                      # between threshold and the first node: linear in (T − E) to zero
        return math.exp(ls[0]) * (T - E) / math.exp(ldT[0])
    if x > ldT[-1] + 1e-12:
        raise ValueError("inconsistent excitation table (does not reach 3 keV)")
    return math.exp(float(np.interp(x, ldT, ls)))


def stopping_cross_section(T):
    """S(T) = Σ_n σ_n E_n [m² J]."""
    return sum(sigma(n, T) * DATA["levels"][n]["E_eV"] * K.e for n in LEVELS)


# [A13] H(1s) → np only: Born + exact GOS + BE scaling, Stone et al. 2002
def loss_rate(K_J, medium):
    """−dE/dt [J s^-1] = n_HI v S(T)."""
    return medium.n_HI * float(KIN.speed(K_J)) * stopping_cross_section(float(K_J) / K.e)
