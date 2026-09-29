"""Tests of em_cascades.fractions (objective 1). Criteria fixed before running."""

import numpy as np
import pytest

from em_cascades import fractions as FR
from igm_losses import constants as K, integrate, losses

P = K._P
SPEC = {"grid": {"K_max_eV": 1.0e7, "points_per_decade": 40}}


def test_rate_fractions_sum_to_one():
    Kg = FR.energy_grid(SPEC)
    phi = FR.rate_fractions(Kg[::20], 1e-4)
    assert np.max(np.abs(sum(phi.values()) - 1)) < 1e-12


def test_integrated_budget_closes():
    """Σ_i f_i(K) + K_floor/K = 1 on the whole grid (trapezoid in K with Σ φ_i = 1): 1e-12."""
    Kg = FR.energy_grid(SPEC)
    f = FR.integrated_fractions(Kg, 0.5)
    assert np.max(np.abs(sum(f.values()) + P["K_floor"] / Kg - 1)) < 1e-12


@pytest.mark.parametrize("K0,x_e", [(1e4, 1e-4), (1e5, 1e-4), (1e7, 1e-4), (1e5, 0.5)])
def test_integrated_vs_time_integration(K0, x_e):
    """Independent route: the ODE in time at fixed z (integrate.evolve(fixed_z=True, augmented=True)) gives
    lost_i/K_ini at the floor; the energy integral must agree within 1e-3 (absolute, per process)."""
    spec = {"grid": {"K_max_eV": K0, "points_per_decade": 40}}
    Kg = FR.energy_grid(spec)
    f = FR.integrated_fractions(Kg, x_e)
    o = integrate.evolve(K0, P["z_init"], P["z_final"], x_e, fixed_z=True, augmented=True)
    assert o["t_floor"] is not None
    for p in losses.PROCESSES:
        assert abs(f[p][-1] - o["lost_" + p][-1] / K0) < 1e-3, p


def test_grid_convergence():
    """Doubling the points per decade changes every f_i(K_max) by less than 1e-3."""
    f1 = FR.integrated_fractions(FR.energy_grid(SPEC), 1e-4)
    f2 = FR.integrated_fractions(FR.energy_grid({"grid": {"K_max_eV": 1.0e7, "points_per_decade": 80}}), 1e-4)
    for p in losses.PROCESSES:
        assert abs(f1[p][-1] - f2[p][-1]) < 1e-3, p
