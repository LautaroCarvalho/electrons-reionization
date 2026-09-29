"""Tests of integrate.py: analytic solution, null case, energy bookkeeping, floor event, time grid (D06)."""

import math
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import integrate as INT, constants as K, cosmology  # noqa: E402
from igm_losses.medium import Medium  # noqa: E402
import astropy.units as u  # noqa: E402
from astropy.cosmology import Planck18  # noqa: E402


def test_time_grid_is_union():
    """D06: the grid is the union of A (elapsed time, log from t_grid_A_start) and B (cosmic time, log), rebuilt here
    from the definitions in parameters.yaml; every A and B point must be in the grid (to 1e-12) and nothing else."""
    g = INT.time_grid(10.0, 5.5)
    assert np.all(np.diff(g) > 0)
    assert abs(g[0] / cosmology.age(10.0) - 1) < 1e-12 and abs(g[-1] / cosmology.age(5.5) - 1) < 1e-12
    P = K._P
    t0, t1 = float(cosmology.age(10.0)), float(cosmology.age(5.5))
    A = t0 + np.logspace(np.log10(P["t_grid_A_start"] * K.year), np.log10(t1 - t0), int(P["N_time_grid_A"]))
    B = np.logspace(np.log10(t0), np.log10(t1), int(P["N_time_grid_B"]))
    ref = np.unique(np.concatenate(([t0], A, B)))
    assert g.size == ref.size and np.max(np.abs(g / ref - 1)) < 1e-12


def test_null_case_no_processes():
    """C53: with no process K stays at K0 exactly."""
    out = INT.evolve(1e6, 10.0, 5.5, 1e-4, include=())
    assert np.all(out["K"] == 1e6)


def test_synchrotron_fixed_z_vs_analytic():
    """dK/dt = −C K(K+2mc²)/(mc²)² with C = (4/3)σ_T c U_B (fixed z) has the exact solution
    ln[K/(K+2mc²)] = ln[K0/(K0+2mc²)] − 2C t/mc². ODE vs closed form within 1e-6 relative."""
    K0 = 1e12
    out = INT.evolve(K0, 10.0, 5.5, 1e-4, include=("synchrotron",), fixed_z=True)
    C = 4 / 3 * K.sigma_T * K.c * Medium(10.0, 1e-4).U_B
    mc2 = K.m_e_c2
    t = out["t"] - out["t"][0]
    r0 = math.log(K0 * K.e / (K0 * K.e + 2 * mc2))
    r = r0 - 2 * C * t / mc2
    K_exact = 2 * mc2 * np.exp(r) / (1 - np.exp(r)) / K.e
    ok = K_exact > 1.0
    assert np.max(np.abs(out["K"][ok] / K_exact[ok] - 1)) < 1e-6


def test_augmented_energy_bookkeeping():
    """K0 − K(t) = Σ_i E_lost,i (within 1e-6 of K0) for all processes."""
    out = INT.evolve(1e7, 10.0, 9.0, 1e-4, augmented=True)
    lost = sum(out[k] for k in out if k.startswith("lost_"))
    assert np.max(np.abs((1e7 - out["K"]) - lost)) / 1e7 < 1e-6
    assert np.all(np.diff(out["D_proper"]) >= 0) and np.all(out["D_comoving"] >= out["D_proper"])


def test_floor_event_holds_energy():
    """A low-energy electron in a fully ionized medium reaches the 10.2 eV floor and stays there."""
    out = INT.evolve(100.0, 10.0, 5.5, 0.5)
    assert out["t_floor"] is not None
    assert abs(out["K"][-1] - K._P["K_floor"]) < 1e-12


def test_augmented_bookkeeping_after_floor():
    """After the floor event the accumulated losses are held at the event state, so K0 − K_floor = Σ_i E_lost,i
    on the whole grid (within 1e-6 of K0). Mutation: holding the last grid point before the event instead
    loses the energy of the last interval and breaks this."""
    K0 = 1e4
    out = INT.evolve(K0, 10.0, 5.5, 0.5, augmented=True)
    assert out["t_floor"] is not None
    lost = sum(out[k] for k in out if k.startswith("lost_"))
    assert np.max(np.abs((K0 - out["K"]) - lost)) / K0 < 1e-6


@pytest.mark.parametrize("K0", [1e3, 1e12])
def test_adiabatic_only_vs_exact_solution(K0):
    """A16 with the dynamical z(t) path: dp/dt = −H p has the exact solution p ∝ 1/a = (1+z), so
    p(z) c = p0 c (1+z)/(1+z_init) and K = √(p²c² + m²c⁴) − mc². Non-relativistic (1 keV) and ultra-relativistic
    (1 TeV) cases; criterion 1e-6 relative on the whole grid (ODE rtol = 1e-8)."""
    out = INT.evolve(K0, 10.0, 5.5, 1e-4, include=("adiabatic",))
    mc2 = K.m_e_c2_eV
    pc0 = math.sqrt(K0 * (K0 + 2 * mc2))
    pc = pc0 * (1 + out["z"]) / 11.0
    K_exact = pc * pc / (np.sqrt(pc * pc + mc2 * mc2) + mc2)       # = √(p²c²+m²c⁴) − mc² without cancellation
    assert out["t_floor"] is None
    assert np.max(np.abs(out["K"] / K_exact - 1)) < 1e-6


def test_distances_without_losses():
    """A09 with no losses (v = v0 constant): D_proper = v0 (t − t0) exactly, and D_comoving = β0 × (comoving distance
    of a photon between z_init and z) = β0 [χ(z_init) − χ(z)], with χ from astropy Planck18 (independent of the
    code's cosmology splines). Criteria: 1e-9 (proper), 1e-6 (comoving; astropy quadrature vs our splines)."""
    K0 = 1e6
    out = INT.evolve(K0, 10.0, 5.5, 1e-4, include=(), augmented=True)
    g = 1 + K0 / K.m_e_c2_eV
    beta0 = math.sqrt(1 - 1 / g ** 2)
    t = out["t"]
    Dp = beta0 * K.c * (t - t[0])
    ok = Dp > 0
    assert np.max(np.abs(out["D_proper"][ok] / Dp[ok] - 1)) < 1e-9
    zs = out["z"][ok][::200]
    chi0 = Planck18.comoving_distance(10.0).to(u.m).value
    chi = chi0 - Planck18.comoving_distance(zs).to(u.m).value
    # The reference is a difference of two ~3e26 m numbers: its own relative precision is ~1e-16 chi0/chi, which is
    # useless at the first grid points (0.5 at Δχ ≈ 6e10 m). Only points where that precision is better than 1e-9
    # are compared; the criterion 1e-6 is unchanged.
    use = 1e-16 * chi0 / chi < 1e-9
    Dc = out["D_comoving"][ok][::200]
    assert use.sum() >= 5
    assert np.max(np.abs(Dc[use] / (beta0 * chi[use]) - 1)) < 1e-6
