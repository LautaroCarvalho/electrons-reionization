"""Tests of em_cascades.ic_spectrum (B&G 1970 Eq. 2.48 on the CMB). Criteria fixed before running."""

import math

import mpmath as mp
import pytest
from scipy import integrate

from em_cascades import ic_spectrum as ICS
from igm_losses import constants as K, inverse_compton as IC
from igm_losses.medium import Medium

P = K._P
T10 = K.T_CMB0 * (1 + P["z_init"])


def _moment(gam, kT, power):
    """∫ E1^power dN/(dt dE1) dE1 over (0, 1), in log E1."""
    f = lambda u: ICS.dN_dt_dE1(math.exp(u), gam, kT) * math.exp(u) ** (power + 1)
    E1max = 1 - 1e-12
    val, _ = integrate.quad(f, math.log(1e-16), math.log(E1max), epsabs=0.0, epsrel=1e-8, limit=400)
    return val


def test_thomson_limit_photon_count():
    """Γ ≪ 1 (γ = 1e3 at z_init): every scattering emits one photon, so ∫ dN/dt dE1 dE1 = σ_T c n_γ with
    n_γ = (2ζ(3)/π²)(kT/ħc)³ (blackbody photon density). Criterion 1e-4 (B&G drop q < 1/(4γ²), ~1e-7)."""
    kT = K.k_B * T10
    n_gamma = 2 * float(mp.zeta(3)) / math.pi ** 2 * (kT / (K.hbar * K.c)) ** 3
    assert abs(_moment(1e3, kT, 0) / (K.sigma_T * K.c * n_gamma) - 1) < 1e-4


@pytest.mark.parametrize("gam", [1e3, 1e6, 1e8])
def test_energy_moment_equals_ic_loss(gam):
    """B&G Eq. 2.56: γ m c² ∫ E1 dN/dE1 dE1 = −dE/dt, compared with the igm_losses route
    (4/3) σ_T c U γ²β² F_KN(b) with F_KN computed directly (not the table). Criterion 1e-5."""
    kT = K.k_B * T10
    K_J = (gam - 1) * K.m_e_c2
    ours = gam * K.m_e_c2 * _moment(gam, kT, 1)
    m = Medium(P["z_init"], 1e-4)
    b = float(IC.b_parameter(K_J, m.T_CMB))
    ref = 4 / 3 * K.sigma_T * K.c * m.U_CMB * (gam * gam - 1) * IC.compute_F_kn(b)
    assert abs(ours / ref - 1) < 1e-5


def test_no_photon_above_kinematic_maximum():
    """B&G Eq. 2.50: E1 ≤ Γ/(1+Γ) for the most energetic CMB photon; far above it (x > 700) the spectrum is 0."""
    kT = K.k_B * T10
    gam = 100.0
    Gmax = 4 * ICS.X_MAX * kT * gam / K.m_e_c2
    assert ICS.dN_dt_dE1(1.01 * Gmax / (1 + Gmax), gam, kT) == 0.0
