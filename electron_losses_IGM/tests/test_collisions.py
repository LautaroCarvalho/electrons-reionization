"""Tests of excitation.py, ionization.py, coulomb.py and losses.py. Criteria fixed before running."""

import math
import sys
import pathlib

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import excitation as X, ionization as I, coulomb as C, losses as L, constants as K  # noqa: E402
from igm_losses.medium import Medium  # noqa: E402


# ------------------------------------------------------------------ excitation
@pytest.mark.parametrize("Q", [1e-4, 0.1, 1.0, 10.0, 100.0])
def test_gos_2p_closed_form(Q):
    """GOS(1s→2p) vs 2¹³3³/(4Q+9)⁶ (derived with sympy from the exact radial functions)."""
    assert abs(X.gos(2, Q) / (221184 / (4 * Q + 9) ** 6) - 1) < 1e-10


def test_gos_optical_limit_vs_stone_f():
    """GOS(Q→0) = optical f; Stone Table 1 prints f to 3–4 figures: agreement within 0.3 %."""
    for n in X.LEVELS:
        assert abs(X.gos(n, 1e-6) / X.DATA["levels"][n]["f"] - 1) < 3e-3


@pytest.mark.parametrize("n", [2, 3])
def test_born_be_vs_stone_table(n):
    """σ_BE at 11–20 eV vs Stone Table 1: within 0.5 % (Stone print R = 13.61 eV rounded)."""
    for T, ref in X.DATA["sigma_BE_check"][n].items():
        assert abs(X.sigma_born_be(n, float(T)) / (ref * 1e-20) - 1) < 5e-3


def test_table_matches_direct_below_3keV():
    for n in (2, 5, 10):
        for T in (15.0, 100.0, 1234.5, 2999.0):
            assert abs(X.sigma(n, T) / X.sigma_direct(n, T) - 1) < 2e-3


def test_switch_jumps_are_the_reported_ones():
    """E12: jump +0.1 % at 3 keV (< 0.3 %), +2.8 % at 10 keV (between 2 % and 4 %), all levels."""
    for n in X.LEVELS:
        j3 = X.sigma_asymptotic(n, 3000.0) / X.sigma_born_be(n, 3000.0) - 1
        j10 = X.sigma_relativistic(n, 1e4) / X.sigma_asymptotic(n, 1e4) - 1
        assert abs(j3) < 3e-3 and 0.02 < j10 < 0.04


def test_relativistic_form_reduces_to_eq5():
    """Inokuti (4.26) with the Stone identification → a ln(T/R) + b as β → 0 (T = 100 eV, no BE denominator)."""
    L2 = X.DATA["levels"][2]
    T = 100.0
    nonrel = 4 * math.pi * K.a0 ** 2 * X.R_EV / T * (L2["a"] * math.log(T / X.R_EV) + L2["b"])
    assert abs(X.sigma_relativistic(2, T) / nonrel - 1) < 1e-3


def test_excitation_above_range_raises():
    with pytest.raises(ValueError):
        X.sigma(2, 2e14)


# ------------------------------------------------------------------ ionization
def test_oscillator_moments_vs_kim_rudd_table():
    """N_i = Σ c_j/(j−1) = 0.4343 and M² = (R/B) D(∞) = 0.2834 (Kim & Rudd 1994 Table I)."""
    assert abs(I.N_I - 0.4343) < 1e-4
    assert abs(K.Ry_inf_eV / I.B_EV * I.D(1e30) - 0.2834) < 1e-4


def test_fano_slope_equals_M2():
    """Kim et al. 2000 Eq. 23: at T ≫ mc² the Fano plot slope → M² (independent structural check)."""
    def XY(T):
        tp = T / K.m_e_c2_eV
        b2 = 1 - 1 / (1 + tp) ** 2
        return math.log(tp * (tp + 2)) - b2, I.sigma(T) * b2 / (4 * math.pi * K.a0 ** 2 * K.alpha ** 2)
    (x1, y1), (x2, y2) = XY(1e9), XY(1e11)
    assert abs((y2 - y1) / (x2 - x1) / (K.Ry_inf_eV / I.B_EV * I.D(1e30)) - 1) < 1e-3


def test_mean_secondary_energy_closed_form_vs_mpmath():
    """D07/E15: closed form = mpmath quadrature (1e-10) and → ε̄ sin(π/p)/sin(2π/p) at T → ∞."""
    mp.mp.dps = 30
    for T in (50.0, 1e4, 1e9):
        X_ = (T - I.B_EV) / 2
        pts = [0] + [x for x in (8, 80, 800, 8e3, 8e4, 8e5, 8e6, 8e7) if x < X_] + [X_]
        num = mp.quad(lambda e: e / (1 + (e / 8) ** mp.mpf("2.1")), pts) / mp.quad(lambda e: 1 / (1 + (e / 8) ** mp.mpf("2.1")), pts)
        assert abs(I.mean_secondary_energy(T) / float(num) - 1) < 1e-10
    assert abs(I.mean_secondary_energy(1e40) / (8 * math.sin(math.pi / 2.1) / math.sin(2 * math.pi / 2.1)) - 1) < 1e-3


def test_notebook_quadrature_bug_absent():
    """E15: at 1e9 eV the notebook got −13.2 eV; the correct <ε> is 44.67 eV."""
    assert abs(I.mean_secondary_energy(1e9) - 44.67) < 0.01


def test_ionization_zero_below_threshold():
    assert I.sigma(13.6) == 0.0 and I.sigma(13.61) > 0.0


# ------------------------------------------------------------------ Coulomb
def test_coulomb_switch_at_beta_alpha():
    """E18: the classical and Born forms differ by ~1.1 % at β = α (reported jump; must stay below 2 %)."""
    m = Medium(10.0, 1e-4)
    Ka = K.m_e_c2 * (1 / math.sqrt(1 - K.alpha ** 2) - 1)
    jump = C.stopping_number(Ka * (1 + 1e-9), m.n_e) / C.stopping_number(Ka * (1 - 1e-9), m.n_e) - 1
    assert 0 < jump < 0.02


def test_coulomb_scaling_with_density():
    """−dE/dt ∝ n_e ln(1/ω_p) at fixed K: doubling x_e must scale the rate by 2 (B − ln√2)/B."""
    m1, m2 = Medium(10.0, 1e-4), Medium(10.0, 2e-4)
    Kj = 1e4 * K.e
    B1 = C.stopping_number(Kj, m1.n_e)
    assert abs(C.loss_rate(Kj, m2) / C.loss_rate(Kj, m1) - 2 * (B1 - 0.5 * math.log(2)) / B1) < 1e-12


# ------------------------------------------------------------------ registry
def test_registry_components_and_null_case():
    r = L.rates(1e6 * K.e, 10.0, 1e-4)
    assert set(r) == set(L.PROCESSES) and all(v > 0 for v in r.values())
    assert L.total_rate(1e6 * K.e, 10.0, 1e-4, include=()) == 0.0


def test_synchrotron_isotropic_is_two_thirds_of_perpendicular():
    m = Medium(10.0, 1e-4)
    Kj = 1e9 * K.e
    assert abs(L.synchrotron(Kj, m) / L.synchrotron(Kj, m, sin_theta=1.0) - 2 / 3) < 1e-14


# ------------------------------------------------------------------ absolute collisional losses (independent references)
def _hydrogen_I0_over_R():
    """ln(I0/R) = L(0) = Σ f ln(E/R) [Inokuti 1971 Eq. 4.62] from the exact H(1s) oscillator strengths
    f_1s→np = 2⁸ n⁵ (n−1)^(2n−4) / (3 (n+1)^(2n+4)) and df/d(E/R) = 2⁷ exp[−(4/k) arctan k] / (3 (E/R)⁴ (1 − e^(−2π/k))),
    k² = E/R − 1 (recalled, Bethe & Salpeter §69–71; validated here by the TRK sum rule Σf = 1 to 1e-12)."""
    with mp.workdps(30):
        f_n = lambda n: mp.mpf(2) ** 8 * n ** 5 * (n - 1) ** (2 * n - 4) / (3 * (n + 1) ** (2 * n + 4))
        k = lambda E: mp.sqrt(E - 1)
        dfdE = lambda E: mp.mpf(2) ** 7 * mp.exp(-4 / k(E) * mp.atan(k(E))) / (3 * E ** 4 * (1 - mp.exp(-2 * mp.pi / k(E))))
        pts = [1, 2, 10, mp.inf]
        trk = mp.nsum(f_n, [2, mp.inf]) + mp.quad(dfdE, pts)
        assert abs(trk - 1) < mp.mpf("1e-12")
        L0 = mp.nsum(lambda n: f_n(n) * mp.log(1 - 1 / mp.mpf(n) ** 2), [2, mp.inf]) + mp.quad(lambda E: dfdE(E) * mp.log(E), pts)
        return float(mp.exp(L0))


@pytest.mark.parametrize("T", [1e4, 1e5, 1e6])
def test_collisional_stopping_vs_bethe(T):
    """Excitation + ionization of neutral H vs the relativistic Bethe stopping for electrons with Møller exchange,
    Inokuti 1971 Eq. (4.65) (p. 333), Z = z = 1, I0 from _hydrogen_I0_over_R (14.99 eV):
        σ_st = 8π a0² (R/mv²) {ln[mv² T/(I0²(1−β²))] − [2√(1−β²) + β²] ln 2 + 1 − β² + ⅛[1 − √(1−β²)]²},
        −dE/dt = n_HI v R σ_st.
    Criterion fixed before running: 5 % (the code keeps np levels n ≤ 10 and the RBED continuum; missing n > 10
    oscillator strength ≈ 1 % of the logarithm; Bethe's O(R/T) terms negligible for T ≥ 10 keV)."""
    R_J = K.Ry_inf
    I0 = _hydrogen_I0_over_R() * R_J
    TJ = T * K.e
    b2 = 1 - 1 / (1 + TJ / K.m_e_c2) ** 2
    mv2 = K.m_e * K.c ** 2 * b2
    s = math.sqrt(1 - b2)
    bracket = (math.log(mv2 * TJ / (I0 ** 2 * (1 - b2))) - (2 * s + b2) * math.log(2) + 1 - b2 + (1 - s) ** 2 / 8)
    sigma_st = 8 * math.pi * K.a0 ** 2 * (R_J / mv2) * bracket
    m = Medium(10.0, 0.0)
    v = K.c * math.sqrt(b2)
    bethe = m.n_HI * v * R_J * sigma_st
    ours = X.loss_rate(TJ, m) + I.loss_rate(TJ, m)
    assert abs(ours / bethe - 1) < 0.05


def test_collisional_rates_scale_with_neutral_fraction():
    """A03: excitation and ionization act on n_HI = (1 − x_e) n_H only: rate(x_e = 0.5)/rate(x_e = 1e-4) = 0.5/0.9999."""
    Kj = 1e4 * K.e
    for f in (X.loss_rate, I.loss_rate):
        r = f(Kj, Medium(10.0, 0.5)) / f(Kj, Medium(10.0, 1e-4))
        assert abs(r / (0.5 / 0.9999) - 1) < 1e-12


@pytest.mark.parametrize("K_eV", [1e4, 11.0])
def test_coulomb_absolute_in_cgs(K_eV):
    """Gould 1972 in Gaussian units, an independent unit path: −dE/dx = (4π n_e e⁴/(m v²)) B [Eq. 1.1] with
    B = Eq. (5.5) for β > α (1e4 eV) and B = Eq. (2.20) ln[2(½m)v³δ^½/(Γ e² ω_p)], Γ = e^C (C = Euler), for β < α
    (11 eV); ω_p = (4π n_e e²/m)^½, ħ in erg s. Converted to J/s at the end; criterion 1e-10."""
    m = Medium(10.0, 1e-4)
    delta = K._P["delta_gould"]
    # Gaussian charge from e²/(4π ε0) [J m] × 1e9 [erg cm per J m]. "1 C = 10 c statC" is exact only for
    # μ0 = 4π×1e-7; with the SI 2019 / CODATA 2022 μ0 it is off by 1.3e-10 (first version of this test: 2.6e-10 in e⁴).
    e_g = math.sqrt(K.e ** 2 / (4 * math.pi * K.eps0) * 1e9)     # statC
    me_g, c_g, hbar_g = K.m_e * 1e3, K.c * 1e2, K.hbar * 1e7
    ne_g = m.n_e * 1e-6
    KJ = K_eV * K.e
    g = 1 + KJ / K.m_e_c2
    beta = math.sqrt(1 - 1 / g ** 2)
    v_g = beta * c_g
    wp = math.sqrt(4 * math.pi * ne_g * e_g ** 2 / me_g)
    if beta > K.alpha:
        B = (math.log(math.sqrt(g - 1) * beta * me_g * c_g ** 2 * math.sqrt(2 * delta) / (hbar_g * wp))
             + 0.5 * (1 + (2 * g - 1) / g ** 2) * math.log(1 - delta) + 0.5 * delta / (1 - delta)
             + 0.25 * ((g - 1) / g) ** 2 * delta ** 2)
    else:
        B = math.log(2 * 0.5 * me_g * v_g ** 3 * math.sqrt(delta) / (math.exp(float(mp.euler)) * e_g ** 2 * wp))
    dEdt_cgs = 4 * math.pi * ne_g * e_g ** 4 / (me_g * v_g ** 2) * B * v_g          # erg/s
    assert abs(C.loss_rate(KJ, m) / (dEdt_cgs * 1e-7) - 1) < 1e-10
