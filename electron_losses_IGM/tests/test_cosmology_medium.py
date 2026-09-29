"""Tests of cosmology.py (D12: < 1e-8 vs astropy off-grid), medium.py and kinematics.py."""

import sys
import pathlib

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import cosmology as C, medium as M, kinematics as KIN, constants as K  # noqa: E402

RNG_SEED = 20260928
Z_TEST = np.sort(np.random.default_rng(RNG_SEED).uniform(0.01, 39.0, 200))   # off-grid points (np.random.default_rng)


def test_age_and_hubble_vs_astropy():
    t_ref = Planck18.age(Z_TEST).to(u.s).value
    H_ref = Planck18.H(Z_TEST).to(1 / u.s).value
    assert np.max(np.abs(C.age(Z_TEST) / t_ref - 1)) < 1e-8
    assert np.max(np.abs(C.hubble(Z_TEST) / H_ref - 1)) < 1e-8


def test_redshift_inverts_age():
    t_ref = Planck18.age(Z_TEST).to(u.s).value
    assert np.max(np.abs((1 + C.redshift(t_ref)) / (1 + Z_TEST) - 1)) < 1e-8


def test_hydrogen_density():
    """n_H(z) = (1−Y_p) Ω_b ρ_c (1+z)³/m_p; compare with the notebook value at z = 10 (258.7 m^-3, A02)."""
    n10 = M.Medium(10.0, 0.0).n_H
    rho_c = (Planck18.critical_density0).to(u.kg / u.m ** 3).value
    expected = (1 - 0.2454) * Planck18.Ob0 * rho_c * 11 ** 3 / K.m_p
    assert abs(n10 / expected - 1) < 1e-12
    assert 0.7 < n10 / 258.7 < 1.3        # same order as the notebook's hand-written normalization


def test_medium_fractions_and_fields():
    m = M.Medium(5.5, 1e-4)
    assert abs((m.n_HI + m.n_e) / m.n_H - 1) < 1e-15
    assert abs(m.B / (1e-13 * 6.5 ** 2) - 1) < 1e-15
    assert abs(m.U_CMB / (K.a_rad * (K.T_CMB0 * 6.5) ** 4) - 1) < 1e-15


def test_kinematics_limits():
    Kl = 1e-3 * K.e                     # 1 meV: β² → 2K/mc²
    assert abs(KIN.beta2(Kl) / (2 * Kl / K.m_e_c2) - 1) < 1e-8
    t = Kl / K.m_e_c2                     # γ²β² = 2t + t² exactly (t = K/mc²); γ² − 1 in floats would lose ~9 digits
    assert abs(KIN.gamma2beta2(Kl) / (2 * t + t * t) - 1) < 1e-14
    Kh = 1e14 * K.e
    assert abs(KIN.gamma2beta2(Kh) / (KIN.gamma(Kh) ** 2 - 1) - 1) < 1e-12
