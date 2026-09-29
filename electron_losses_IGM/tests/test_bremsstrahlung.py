"""Tests of igm_losses.bremsstrahlung (A15, D18, D19). Criteria fixed before running."""

import math
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import bremsstrahlung as B, constants as K  # noqa: E402


def _dlog(f, T, rel=1e-5):
    return (math.log(f(T * (1 + rel))) - math.log(f(T / (1 + rel)))) / (2 * math.log(1 + rel))


def test_neutral_reproduces_table_nodes():
    """At the CS_int energies the interpolant must return the tabulated φ_n exactly (1e-12)."""
    Tg, phig = B.load_neutral_table()
    assert np.max(np.abs(B.phi_neutral(Tg) / phig - 1)) < 1e-12


def test_neutral_nonrelativistic_value():
    """Between 1 and 10 keV φ_n must be within 5 % of the non-relativistic Born value 16/3 (screening is weak)."""
    for T in (1e3, 3e3, 1e4):
        assert abs(B.phi_neutral(T) / (16 / 3) - 1) < 0.05


def test_neutral_high_energy_join_is_C1():
    """No jump and no slope jump at 300 MeV and 3 GeV (C¹), and φ_n = φ_inf above 3 GeV."""
    n = B.NeutralPhi()
    for Tj in (n.T1, n.T2):
        lo, hi = n(Tj * (1 - 1e-9)), n(Tj * (1 + 1e-9))
        assert abs(hi / lo - 1) < 1e-6
    assert abs(_dlog(n, n.T1 * (1 + 1e-3)) - n.s1) < 1e-2
    assert abs(_dlog(n, n.T2 * (1 - 1e-3))) < 1e-2
    assert abs(n(1e12) - 0.5 * (4 / 3 * 45.79 - 1 / 3 * 44.46)) < 1e-12


def test_bg_complete_screening_value():
    """B&G Eq. (3.54) with the printed Eq. (3.42) values: (4/3)45.79 − (1/3)44.46 = 46.2333…"""
    assert abs(B.bg_complete_screening_phi() - 46.23333333333) < 1e-9


def test_ep_equals_KL_below_and_BG_above():
    ep = B.ElectronProtonPhi()
    for T in (100.0, 1e3, 1e4, ep.Ta * 0.99):
        assert abs(ep(T) / float(B._loglog_interp(T, ep.Tg, ep.phig)) - 1) < 1e-12
    for T in (ep.Tb * 1.01, 1e7, 1e9):
        assert abs(ep(T) / float(B.phi_bg_ep(T)) - 1) < 1e-12


def test_ep_crossing_and_blend_are_C1():
    """T* is a genuine crossing (K&L = B&G there) and the blend is continuous with continuous slope."""
    ep = B.ElectronProtonPhi()
    kl = float(B._loglog_interp(ep.T_star, ep.Tg, ep.phig))
    assert abs(kl / float(B.phi_bg_ep(ep.T_star)) - 1) < 1e-8
    assert 1e5 < ep.T_star < 1e6
    for Tj, s in ((ep.Ta, ep.sa), (ep.Tb, ep.sb)):
        assert abs(ep(Tj * (1 + 1e-9)) / ep(Tj * (1 - 1e-9)) - 1) < 1e-6
        assert abs(_dlog(ep, Tj * (1 + 1e-3)) - s) < 2e-2 and abs(_dlog(ep, Tj * (1 - 1e-3)) - s) < 2e-2


def test_loss_rate_matches_BG_353_at_high_energy():
    """At 1 GeV with only protons: −dE/dt = 4 α r_e² c n (ln 2E − 1/3) E (B&G Eq. 3.53, Z² term), within 1e-6 (v→c)."""
    n_p, T_eV = 1.0, 1e9
    K_J = T_eV * K.e
    E = (K_J + K.m_e_c2)
    expected = 4 * K.alpha_re2 * K.c * n_p * (math.log(2 * E / K.m_e_c2) - 1 / 3) * E
    assert abs(B.loss_rate(K_J, 0.0, n_p) / expected - 1) < 1e-6


def test_phi_ep_formula_from_KL_eq22_symbolic():
    """C25: from K&L 1961 Eq. (22), dσ = (16π/(3√3)) (e²/ħc)³ Z² (ħ²/(2mE0)) (d(hν)/hν) g_ff (Gaussian units),
    the radiated energy ∫hν dσ = (16π/(3√3)) α³ Z² (ħ²/2m) ⟨g⟩ with ⟨g⟩ = ∫_0^1 g dx; with r_e = αħ/(mc) this is
    (8π/(3√3)) α r_e² Z² mc² ⟨g⟩, i.e. φ = ∫hν dσ / (α r_e² Z² (T + mc²)) = (8π/(3√3)) ⟨g⟩ mc²/(T + mc²)."""
    import sympy as sp
    a, hb, m, c, Z, T, G = sp.symbols("alpha hbar m c Z T G", positive=True)
    energy = sp.Rational(16) * sp.pi / (3 * sp.sqrt(3)) * a ** 3 * Z ** 2 * hb ** 2 / (2 * m) * G
    r_e = a * hb / (m * c)
    phi = energy / (a * r_e ** 2 * Z ** 2 * (T + m * c ** 2))
    assert sp.simplify(phi - 8 * sp.pi / (3 * sp.sqrt(3)) * G * m * c ** 2 / (T + m * c ** 2)) == 0


def test_phi_ep_table_node_vs_mean_gff():
    """Stale-table check: a node of provenance/data/phi_ep_KarzasLatter.json recomputed from gaunt_ff.mean_gff with
    the formula of the symbolic test above (1e-10; same algorithm and precision, so any difference means the table
    is out of date or was written with a different formula)."""
    from igm_losses.gaunt_ff import mean_gff
    ep = B.ElectronProtonPhi()
    i = int(np.argmin(np.abs(np.log(ep.Tg / 1e3))))
    T = float(ep.Tg[i])
    g = float(mean_gff(T / K.Ry_inf_eV))
    phi = 8 * math.pi / (3 * math.sqrt(3)) * g * K.m_e_c2_eV / (T + K.m_e_c2_eV)
    assert abs(float(ep.phig[i]) / phi - 1) < 1e-10


def test_neutral_loss_rate_matches_BG_354_at_high_energy():
    """At 1 TeV with only neutral H: −dE/dt = f_nuc × α r0² c n [(4/3)φ1 − (1/3)φ2] E, B&G 1970 Eq. (3.54) (p. 255)
    with φ1 = 45.79, φ2 = 44.46 for H [Eq. 3.42]; f_nuc = BG_nuclear_fraction_complete (0.5: e–e part omitted,
    A15/D19). v/c = 1 − 1e-13 at 1 TeV; criterion 1e-9."""
    n_HI, T_eV = 1.0, 1e12
    K_J = T_eV * K.e
    E = K_J + K.m_e_c2
    f_nuc = K._P["BG_nuclear_fraction_complete"]
    expected = f_nuc * K.alpha_re2 * K.c * n_HI * (4 / 3 * 45.79 - 1 / 3 * 44.46) * E
    assert abs(B.loss_rate(K_J, n_HI, 0.0) / expected - 1) < 1e-9


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        B.phi_neutral(5.0)
    with pytest.raises(ValueError):
        B.phi_ep(5.0)


def test_neutral_missing_points_handled():
    """Rows ≥ 50 MeV miss the k = 0 point (manual §9.6); if it were taken as 0, φ_n would oscillate.
    φ_n on the table nodes must be monotonically increasing from 1 MeV to 300 MeV, and the rows with
    more than one missing k/T (12–40 MeV) must be absent from the table."""
    Tg, phig = B.load_neutral_table()
    sel = Tg >= 1e6
    assert np.all(np.diff(phig[sel]) > 0)
    assert not np.any((Tg > 1.1e7) & (Tg < 4.5e7))
