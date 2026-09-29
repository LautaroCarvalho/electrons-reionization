"""Tests of em_cascades.secondary and em_cascades.yields (objective 2). Criteria fixed before running."""

import math
import pathlib

import numpy as np
import pytest
from scipy import integrate

from em_cascades import fractions as FR, secondary as SE, yields as Y
from igm_losses import constants as K, ionization

P = K._P


def test_norm_closed_form_vs_quadrature():
    """Reference: quadrature in ln ε (a linear quad over [0, 1e8] misses the ε^-2.1 tail: 5 % off, first version)."""
    for X in (5.0, 100.0, 1e4, 1e8):
        head, _ = integrate.quad(lambda e: float(SE.weight(e)), 0, min(X, 1.0), epsrel=1e-12)
        tail, _ = integrate.quad(lambda u: float(SE.weight(math.exp(u))) * math.exp(u), 0.0, math.log(X),
                                 epsrel=1e-12, limit=400) if X > 1 else (0.0, 0)
        assert abs(SE.norm(X) / (head + tail) - 1) < 1e-8


def test_mean_energy_matches_igm_losses():
    """⟨ε⟩ through mean_over (g = ε on a dense grid, ε_min → 0) equals ionization.mean_secondary_energy (1e-3):
    the cascade uses exactly the secondary spectrum of the energy-loss rate (A14, A20)."""
    eps = np.logspace(-6, 7, 20000)
    for T in (50.0, 1e3, 1e5):
        assert abs(SE.mean_over(T, eps, eps, 1e-6) / ionization.mean_secondary_energy(T) - 1) < 1e-3


def test_fraction_above_floor_closed_form():
    """g ≡ 1 above ε_min: ⟨g⟩ = 1 − Z(ε_min)/Z(X) exactly (1e-4 on a dense grid)."""
    eps = np.logspace(-3, 7, 20000)
    for T in (100.0, 1e4):
        ref = 1 - SE.norm(P["K_floor"]) / SE.norm(SE.x_max(T))
        assert abs(SE.mean_over(T, eps, np.ones_like(eps), P["K_floor"]) - ref) < 1e-4


GRID = FR.energy_grid({"grid": {"K_max_eV": 1e6, "points_per_decade": 40}})


def test_ceiling_and_no_photons():
    """N ≤ E/R_H everywhere; without photons (or with an empty window) Ph ≡ 0."""
    C, Ph = Y.cascade(GRID, 1e-4, E_max_eV=0.0, photons=True)
    assert np.all(Ph == 0.0)
    assert np.all(C <= GRID / K.R_H_eV * (1 + 1e-12))


def test_grid_convergence_collisional():
    """40 → 80 points per decade changes C by < 1 % at 1e3, 1e4, 1e5 and 1e6 eV."""
    C1, _ = Y.cascade(GRID, 1e-4, 0.0, photons=False)
    G2 = FR.energy_grid({"grid": {"K_max_eV": 1e6, "points_per_decade": 80}})
    C2, _ = Y.cascade(G2, 1e-4, 0.0, photons=False)
    for E in (1e3, 1e4, 1e5, 1e6):
        a, b = np.interp(E, GRID, C1), np.interp(E, G2, C2)
        assert abs(a / b - 1) < 1e-2, E


def test_photon_branch_bounded_by_ic_energy():
    """With a window [R_H, 1 keV]: the IC energy radiated above R_H is at most the IC energy lost, so the photon
    branch Ph(E) ≤ (∫ L_ic/L_tot dE)/R_H (every photoionization and its tree cost ≥ R_H). Ph > 0 at 1e9 eV and
    negligible below 1 MeV (γ ≲ 3: only the Wien tail of the CMB reaches R_H; the first version demanded exactly 0,
    which the Wien tail never gives — measured 1e-156 … 6e-70)."""
    G = FR.energy_grid({"grid": {"K_max_eV": 1e9, "points_per_decade": 20}})
    C, Ph = Y.cascade(G, 1e-4, E_max_eV=1e3, photons=True)
    f = FR.integrated_fractions(G, 1e-4)
    assert Ph[-1] > 0 and np.all(Ph[G < 1e6] < 1e-30)
    assert np.all(Ph <= f["ic"] * G / K.R_H_eV * (1 + 1e-9))


SVS = __import__("yaml").safe_load((pathlib.Path(__file__).resolve().parents[2] / "provenance" / "data" /
                                    "shull1985_table1.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("x_e,tol", [(1e-4, 0.15), (1e-2, 0.15), (1e-1, 0.15), (0.5, 0.30)])
def test_collisional_yield_vs_shull_van_steenberg(x_e, tol):
    """External anchor: fraction of a 3 keV primary's energy that ends in H ionizations, φ = N_coll × I / E0 with
    I = 13.6 eV (SvS Fig. 2 caption), against SvS 1985 Table 1 (p. 269). Criterion fixed before running: 15 % for
    x ≤ 0.1, 30 % for x = 0.5 (small φ). Model differences: SvS include He (10 % by number, which takes part of the
    energy and of the ionizations) and use discrete Monte Carlo; here no He (A03), continuous slowing down (A20)."""
    E0, I = SVS["E0_eV"], SVS["I_eV"]
    G = FR.energy_grid({"grid": {"K_max_eV": E0, "points_per_decade": 40}})
    C, _ = Y.cascade(G, x_e, 0.0, photons=False)
    phi = C[-1] * I / E0
    ref = SVS["phi_HI"][x_e][0]
    assert abs(phi / ref - 1) < tol, (phi, ref)


def test_photon_grid_convergence():
    """Photon branch: 20 → 40 points per decade changes Ph(1e9 eV) by < 2 % (x_e = 1e-4, Verner window). Also
    covers the accuracy of the window quadrature (quad warns that epsrel = 1e-6 is not always reached)."""
    from em_cascades import photoionization as PI
    hi = PI.window(1e-4)[1]
    out = []
    for ppd in (20, 40):
        G = FR.energy_grid({"grid": {"K_max_eV": 1e9, "points_per_decade": ppd}})
        _, Ph = Y.cascade(G, 1e-4, hi, photons=True)
        out.append(Ph[-1])
    assert out[1] > 0 and abs(out[0] / out[1] - 1) < 2e-2, out


def test_photon_offspring_energy_bookkeeping():
    """A23: a photon of energy E_γ yields 1 + N(E_γ − R_H). A photoelectron at or below K_floor adds nothing (E_γ = R_H +
    10 eV → exactly 1), and one of 3 keV adds the cascade of a 3 keV electron. Catches a photoelectron given E_γ instead
    of E_γ − R_H (mutant not caught by the bound and convergence tests, EM_cascades/tests/mutants, 2026-09-28)."""
    G = FR.energy_grid({"grid": {"K_max_eV": 1e4, "points_per_decade": 40}})
    C, Ph = Y.cascade(G, 1e-4, 0.0, photons=False)
    N = C + Ph
    assert Y.photon_offspring(K.R_H + 10.0 * K.e, G, N) == 1.0
    ref = 1.0 + np.interp(np.log(3e3), np.log(G), N)
    assert abs(Y.photon_offspring(K.R_H + 3e3 * K.e, G, N) / ref - 1) < 1e-12
