"""Bremsstrahlung energy loss of an electron in the IGM: neutral H (electron–atom) and ionized H (electron–proton).

Purpose
    −dE/dt_brems = v α r_e² (T + mc²) [ n_HI φ_n(T) + n_p φ_ep(T) ]            (Z = 1)
    with the radiative energy-loss function φ_rad(T) ≡ ∫ k (dσ/dk) dk / (α r_e² Z² (T + mc²)) (the scaled form
    used by Seltzer & Berger and by B&G 1970 Eqs. 3.53–3.55). Assumption A15, decision D18, erratum E07.

Neutral H, electron–atom (φ_n)  [Poskus2019; BREMS v1.5.8.9 CS_int_1.txt, 10 eV – 300 MeV]
    φ_n(T) = T ∫_0^0.95 CS(x) dx / (α r_e² (T + mc²)),  CS = (k/Z²) dσ/dk in mb, x = k/T.
    The tail x > 0.95 is ignored (user decision 2026-09-27; ~1–5 % of the integral).
    Missing grid points (zeros in the file, manual §9.6): rows with more than one missing k/T are dropped;
    a single missing k/T is filled by linear interpolation in x (k = 0 by linear extrapolation from x = 0.1, 0.2).
    Above 300 MeV: C¹ Hermite blend in (ln T, ln φ) over one decade to the complete-screening value of
    B&G 1970 Eq. (3.54), (4/3)φ1 − (1/3)φ2 with φ1, φ2 from Eq. (3.42), times the nuclear fraction ≈ 1/2
    (B&G p. 253: the total goes roughly as Z² + Z_el); constant above.

Ionized H, electron–proton (φ_ep)
    Low energy: exact non-relativistic K&L 1961 Gaunt factor, table from scripts/compute_phi_ep_table.py.
    High energy: B&G 1970 Eq. (3.53), nuclear term only: φ = 4 (ln 2E − 1/3), E = (T + mc²)/mc².
    The two cross at T*; C¹ Hermite blend in (ln T, ln φ) over [T*/f, f T*] (f = brems_ep_blend_factor).

Electron–electron bremsstrahlung: NOT included (user decision 2026-09-27). Justification valid in the
    non-relativistic regime: equal charge-to-mass ratio ⇒ no dipole moment ⇒ quadrupole emission,
    suppressed by v² [PradlerSemmelrock2021, pp. 2, 14–15; B&G 1970 footnote 18, p. 251]. At γ ≫ 1 e–e ≈ e–p
    (B&G p. 251), so the neglect underestimates bremsstrahlung by up to ~×2 there (flagged in A15).

Interpolation: linear in log–log (decision D03). Outside [10 eV, ∞) → ValueError (D08).
"""

import json
import math
import pathlib

import numpy as np
import yaml
from scipy.optimize import brentq

from . import constants as K

ROOT = pathlib.Path(__file__).resolve().parents[3]
CS_INT_FILE = ROOT / "provenance" / "data" / "CS_int_1.txt"
PHI_EP_FILE = ROOT / "provenance" / "data" / "phi_ep_KarzasLatter.json"
X_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
MB = 1.0e-31      # m² per millibarn (definition, 1 b = 1e-28 m²)

_P = {k: v["value"] for k, v in yaml.safe_load((ROOT / "provenance" / "parameters.yaml")
                                               .read_text(encoding="utf-8"))["parameters"].items()}


# --------------------------------------------------------------------------- helpers
def _hermite(u, u0, u1, y0, y1, s0, s1):
    """Cubic Hermite on [u0, u1] with values y0, y1 and slopes s0, s1 (dy/du)."""
    h = u1 - u0
    t = (u - u0) / h
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * y0 + h10 * h * s0 + h01 * y1 + h11 * h * s1


def _loglog_interp(T, Tg, yg):
    return np.exp(np.interp(np.log(T), np.log(Tg), np.log(yg)))


def _loglog_slope(f, T, rel=1e-4):
    """d ln f / d ln T by central differences."""
    return (math.log(f(T * (1 + rel))) - math.log(f(T / (1 + rel)))) / (math.log(1 + rel) * 2)


# --------------------------------------------------------------------------- neutral H
def load_neutral_table(path=CS_INT_FILE):
    """φ_n on the CS_int grid (T in eV), with the missing-point policy described in the module docstring."""
    lines = path.read_text(encoding="latin-1").splitlines()[1:]
    rows = np.array([[float(v) for v in ln.split()] for ln in lines if ln.strip()])
    T_eV = rows[:, 0] * 1.0e6
    cs = rows[:, 1::2]
    T_keep, phi = [], []
    for Ti, c in zip(T_eV, cs):
        missing = c == 0.0
        if missing.sum() > 1:
            continue
        c = c.copy()
        if missing.sum() == 1:
            j = int(np.where(missing)[0][0])
            c[j] = 2 * c[1] - c[2] if j == 0 else np.interp(X_GRID[j], X_GRID[~missing], c[~missing])
        integral_m2 = np.trapz(c, X_GRID) * MB * Ti          # ∫ k dσ/dk dk / eV-scale → (m² · eV)
        phi.append(integral_m2 / (K.alpha_re2 * (Ti + K.m_e_c2_eV)))
        T_keep.append(Ti)
    return np.array(T_keep), np.array(phi)


def bg_complete_screening_phi():
    """B&G 1970 Eq. (3.54) for H: (4/3)φ1 − (1/3)φ2 with the printed Eq. (3.42) values (atom = nucleus + electron)."""
    return 4.0 / 3.0 * _P["BG_phi1_H_complete"] - 1.0 / 3.0 * _P["BG_phi2_H_complete"]


class NeutralPhi:
    """φ_n(T) for neutral H: CS_int table, then C¹ blend to the B&G complete-screening nuclear value."""

    def __init__(self):
        self.Tg, self.phig = load_neutral_table()
        self.T1, self.T2 = _P["brems_neutral_T_join_lo"], _P["brems_neutral_T_join_hi"]
        if abs(self.Tg[-1] - self.T1) > 1e-6 * self.T1:
            raise ValueError("brems_neutral_T_join_lo must equal the last CS_int energy")
        self.phi_inf = _P["BG_nuclear_fraction_complete"] * bg_complete_screening_phi()
        self.s1 = math.log(self.phig[-1] / self.phig[-2]) / math.log(self.Tg[-1] / self.Tg[-2])

    def __call__(self, T_eV):
        T = np.atleast_1d(np.asarray(T_eV, dtype=float))
        if np.any(T < self.Tg[0]):
            raise ValueError(f"T below the neutral table ({self.Tg[0]} eV)")
        out = np.empty_like(T)
        a = T <= self.T1
        out[a] = _loglog_interp(T[a], self.Tg, self.phig)
        b = (T > self.T1) & (T < self.T2)
        out[b] = np.exp(_hermite(np.log(T[b]), math.log(self.T1), math.log(self.T2),
                                 math.log(self.phig[-1]), math.log(self.phi_inf), self.s1, 0.0))
        out[T >= self.T2] = self.phi_inf
        return out if np.ndim(T_eV) else float(out[0])


# --------------------------------------------------------------------------- ionized H (e–p)
def phi_bg_ep(T_eV):
    """B&G 1970 Eq. (3.53), nuclear term only (Z² = 1): φ = 4 (ln 2E − 1/3), E = (T + mc²)/mc² (γ ≫ 1)."""
    E = (np.asarray(T_eV, dtype=float) + K.m_e_c2_eV) / K.m_e_c2_eV
    return 4.0 * (np.log(2.0 * E) - 1.0 / 3.0)


class ElectronProtonPhi:
    """φ_ep(T): K&L table below the crossing, B&G (3.53) above, C¹ Hermite blend around the crossing T*."""

    def __init__(self):
        d = json.loads(PHI_EP_FILE.read_text(encoding="utf-8"))
        self.Tg, self.phig = np.array(d["T_eV"]), np.array(d["phi_rad"])
        kl = lambda T: float(_loglog_interp(T, self.Tg, self.phig))
        bg = lambda T: float(phi_bg_ep(T))
        self.T_star = math.exp(brentq(lambda u: math.log(kl(math.exp(u))) - math.log(bg(math.exp(u))),
                                      math.log(1.0e5), math.log(self.Tg[-1] / 2.01)))
        f = _P["brems_ep_blend_factor"]
        self.Ta, self.Tb = self.T_star / f, self.T_star * f
        if self.Tb > self.Tg[-1]:
            raise ValueError("blend window exceeds the K&L table")
        self.ya, self.yb = math.log(kl(self.Ta)), math.log(bg(self.Tb))
        self.sa, self.sb = _loglog_slope(kl, self.Ta), _loglog_slope(bg, self.Tb)

    def __call__(self, T_eV):
        T = np.atleast_1d(np.asarray(T_eV, dtype=float))
        if np.any(T < self.Tg[0]):
            raise ValueError(f"T below the K&L table ({self.Tg[0]} eV)")
        out = np.empty_like(T)
        a = T <= self.Ta
        out[a] = _loglog_interp(T[a], self.Tg, self.phig)
        b = (T > self.Ta) & (T < self.Tb)
        out[b] = np.exp(_hermite(np.log(T[b]), math.log(self.Ta), math.log(self.Tb),
                                 self.ya, self.yb, self.sa, self.sb))
        c = T >= self.Tb
        out[c] = phi_bg_ep(T[c])
        return out if np.ndim(T_eV) else float(out[0])


# --------------------------------------------------------------------------- loss rate
_NEUTRAL = None
_EP = None


def phi_neutral(T_eV):
    global _NEUTRAL
    if _NEUTRAL is None:
        _NEUTRAL = NeutralPhi()
    return _NEUTRAL(T_eV)


def phi_ep(T_eV):
    global _EP
    if _EP is None:
        _EP = ElectronProtonPhi()
    return _EP(T_eV)


# [A15] e–H: BREMS CS_int_1.txt + B&G (3.54); e–p: Karzas & Latter + B&G (3.53); e–e omitted
def loss_rate(K_J, n_HI, n_p):
    """−dE/dt [J s^-1] by bremsstrahlung for kinetic energy K [J] in a medium with n_HI, n_p [m^-3].

    −dE/dt = v α r_e² (K + mc²) [n_HI φ_n + n_p φ_ep], v = c β (relativistic kinematics).
    """
    K_J = np.asarray(K_J, dtype=float)
    gamma = 1.0 + K_J / K.m_e_c2
    v = K.c * np.sqrt(1.0 - 1.0 / gamma ** 2)
    T_eV = K_J / K.e
    return v * K.alpha_re2 * (K_J + K.m_e_c2) * (n_HI * phi_neutral(T_eV) + n_p * phi_ep(T_eV))
