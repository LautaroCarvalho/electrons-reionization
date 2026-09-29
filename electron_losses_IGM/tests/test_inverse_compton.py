"""Tests of inverse_compton.py against the B&G 1970 limits (Eqs. 2.28, 2.59) and the constants of Eq. (2.60)."""

import math
import sys
import pathlib

import numpy as np
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import inverse_compton as IC, constants as K, medium as M  # noqa: E402


def test_thomson_normalization_symbolic():
    """C25: ∫_0^1 q F(q,0) dq = 1/9 and ∫ x³/(eˣ−1) = π⁴/15 ⇒ normalization 135/π⁴."""
    q = sp.symbols("q", positive=True)
    F0 = 2 * q * sp.log(q) + (1 + 2 * q) * (1 - q)
    assert sp.simplify(sp.integrate(q * F0, (q, 0, 1)) - sp.Rational(1, 9)) == 0
    assert abs(float(sp.Rational(135) / sp.pi ** 4 * sp.Rational(1, 9) * sp.pi ** 4 / 15) - 1) < 1e-15


def test_asymptote_normalization_symbolic():
    """(2.59)/(Thomson loss) with γ²(kT)² = b² (mc²)²/16 gives 45/(4π² b²) × [ln b − …]."""
    r0, m, c, kT, hbar, b, L = sp.symbols("r0 m c kT hbar b L", positive=True)
    U = sp.pi ** 2 / 15 * kT ** 4 / (hbar * c) ** 3
    gam = b * m * c ** 2 / (4 * kT)
    thomson = sp.Rational(4, 3) * sp.Rational(8, 3) * sp.pi * r0 ** 2 * c * gam ** 2 * U
    bg259 = sp.Rational(1, 6) * sp.pi * r0 ** 2 * (m * c * kT) ** 2 / hbar ** 3 * L
    assert sp.simplify(bg259 / thomson - 45 / (4 * sp.pi ** 2 * b ** 2) * L) == 0


def test_constants_of_eq_2_60():
    assert abs(IC.C_E - 0.5772) < 1e-4 and abs(IC.C_L - 0.5700) < 1e-4


def test_small_b_first_order():
    """F ≈ 1 − 6.04 b (B&G Eq. 2.28; 24.15/4 = 6.0375): relative residual O(b²)."""
    for b in (1e-4, 1e-3):
        assert abs((1 - IC.F_kn(b)) / (6.0375 * b) - 1) < 0.02


def test_large_b_asymptote():
    """At b = 1e3 and 1e4 the table agrees with Eq. (2.59) normalized, within 0.5 % (next order is O(1/ln b))."""
    for b in (1e3, 1e4 * 0.999):
        assert abs(IC.F_kn(b) / IC.F_kn_asymptotic(b) - 1) < 5e-3


def test_notebook_bug_is_absent():
    """E05: the notebook (no factor q) gives F(1) = 0.2988; the corrected kernel must give 0.147 ± 0.001."""
    assert abs(IC.F_kn(1.0) - 0.1473) < 1e-3


def test_table_nodes_vs_direct():
    """Interpolation error at off-node points below 1e-3 (200 log points, log–log)."""
    for b in (3.3e-3, 0.77, 42.0):
        assert abs(IC.F_kn(b) / IC.compute_F_kn(b) - 1) < 1e-3


def test_loss_rate_thomson_limit():
    """At 1 MeV (b ≈ 1e-7 at z = 0) the loss equals (4/3) σ_T c U γ²β² to 1e-6."""
    m = M.Medium(0.0, 0.0)
    K_J = 1e6 * K.e
    g = 1 + K_J / K.m_e_c2
    expected = 4 / 3 * K.sigma_T * K.c * m.U_CMB * (g * g - 1)
    assert abs(IC.loss_rate(K_J, m) / expected - 1) < 1e-6


def test_loss_rate_klein_nishina_regime_and_redshift():
    """The KN argument and its z dependence, which the Thomson-limit test cannot see (there F_KN = 1 for any b).
    b = 4 γ k_B T_CMB(z) / (m c²) [B&G 1970, Γ_e = 4εγ/mc² with ε → k_B T], T_CMB(z) = T_CMB0 (1+z), U = a T⁴,
    all written here from the constants, not from medium.py. At 10 TeV, z = 10, b ≈ 0.4, where F_KN is far from 1
    and steep, so a factor-2 error in b or a missing (1+z) changes the rate by far more than the criterion (1e-12).
    (A first version used 1 TeV assuming b ≈ 40; the guard below showed b = 0.04 there, so the energy was raised.)"""
    z, K_eV = 10.0, 1e13
    m = M.Medium(z, 1e-4)
    K_J = K_eV * K.e
    g = 1 + K_J / K.m_e_c2
    T = K.T_CMB0 * (1 + z)
    b = 4 * g * K.k_B * T / K.m_e_c2
    assert 0.1 < b < 1.0
    expected = 4 / 3 * K.sigma_T * K.c * K.a_rad * T ** 4 * (g * g - 1) * IC.F_kn(b)
    assert abs(IC.loss_rate(K_J, m) / expected - 1) < 1e-12
