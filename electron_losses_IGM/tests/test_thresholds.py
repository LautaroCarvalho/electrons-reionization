"""Tests of thresholds.py: Compton kinematics (sympy), percentile definitions, B&G Eq. 2.50 limit."""

import math
import sys
import pathlib

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import thresholds as TH, constants as K  # noqa: E402


def test_headon_kinematics_symbolic():
    """C25: 4-momentum conservation (1D head-on, back-scatter) gives E1 = g²ε/(1+2gε/m), g = γ(1+β)."""
    m, eps, E1, gam, b = sp.symbols("m epsilon E1 gamma beta", positive=True)
    E, p = gam * m, gam * b * m
    sols = sp.solve(sp.Eq((E + eps - E1) ** 2 - (p - eps - E1) ** 2, m ** 2), E1)
    sols = [sp.simplify(s.subs(gam, 1 / sp.sqrt(1 - b ** 2))) for s in sols]
    G = (1 + b) / sp.sqrt(1 - b ** 2)
    assert any(sp.simplify(s - G ** 2 * eps / (1 + 2 * G * eps / m)) == 0 for s in sols)


def test_inversion_roundtrip():
    eps, E1 = 1e-3 * K.e, 10.2 * K.e
    gam = TH.gamma_for_upscatter(E1, eps)
    g = gam * (1 + math.sqrt(1 - 1 / gam ** 2))
    assert abs(g * g * eps / (1 + 2 * g * eps / K.m_e_c2) / E1 - 1) < 1e-12


def test_bg_limit():
    """γ ≫ 1: E1_max → 4γ²ε/(1+4γε/mc²) (B&G Eq. 2.50)."""
    eps = 1e-3 * K.e
    gam = TH.gamma_for_upscatter(1e9 * K.e, eps)
    bg = 4 * gam ** 2 * eps / (1 + 4 * gam * eps / K.m_e_c2)
    assert abs(bg / (1e9 * K.e) - 1) < 1e-6


def test_percentiles():
    """The photon-number CDF of x²/(eˣ−1) against an independent mpmath quadrature normalized by 2ζ(3) (1e-10), and
    the percentiles solve CDF = tol, 1 − tol of that independent CDF. (The earlier version compared _cdf with itself.)"""
    import mpmath as mp
    cdf_ref = lambda x: mp.quad(lambda y: y * y / mp.expm1(y), [0, x]) / (2 * mp.zeta(3))
    for x in (0.1, 1.0, 2.82, 10.0):
        assert abs(TH._cdf(x) / float(cdf_ref(x)) - 1) < 1e-10
    tol = K._P["upscatter_tolerance"]
    for frac in (tol, 1 - tol):
        assert abs(float(cdf_ref(TH.photon_percentile(frac))) - frac) < 1e-10


def _gamma_direct(E1, eps):
    """γ from E1 = g²ε/(1 + 2gε/mc²), g = γ(1+β), solved numerically (not with the closed-form inversion)."""
    from scipy.optimize import brentq
    f = lambda lg: (lambda g_: g_ * g_ * eps / (1 + 2 * g_ * eps / K.m_e_c2) - E1)(
        math.exp(lg) * (1 + math.sqrt(1 - math.exp(-2 * lg))))
    return math.exp(brentq(f, 1e-12, 60.0, xtol=1e-15, rtol=1e-15))


def test_threshold_curves_vs_direct():
    """threshold_curves pairs x_hi (fraction 1 − tol below) with the 10.2 eV target and x_lo (fraction tol below)
    with the 1 keV target (module docstring). Rebuilt here with the independent CDF and the numerical γ; 1e-8."""
    import mpmath as mp
    P = K._P
    tol = P["upscatter_tolerance"]
    cdf = lambda x: mp.quad(lambda y: y * y / mp.expm1(y), [0, x]) / (2 * mp.zeta(3))
    x_hi = float(mp.findroot(lambda x: cdf(x) - (1 - tol), 8.0))
    x_lo = float(mp.findroot(lambda x: cdf(x) - tol, 0.3))
    z = [0.0, 10.0]
    Kc, Kh = TH.threshold_curves(z)
    for i, zi in enumerate(z):
        kT = K.k_B * K.T_CMB0 * (1 + zi)
        kc = (_gamma_direct(P["upscatter_min_eV"] * K.e, x_hi * kT) - 1) * K.m_e_c2_eV
        kh = (_gamma_direct(P["upscatter_target_eV"] * K.e, x_lo * kT) - 1) * K.m_e_c2_eV
        assert abs(Kc[i] / kc - 1) < 1e-8 and abs(Kh[i] / kh - 1) < 1e-8
