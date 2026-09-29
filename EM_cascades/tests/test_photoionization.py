"""Tests of em_cascades.photoionization (A22, A24). Criteria fixed before running."""

import math

import mpmath as mp
import numpy as np
import pytest

from em_cascades import photoionization as PI
from igm_losses import constants as K, cosmology
from igm_losses.medium import Medium

P = K._P


def sigma_exact(E_eV):
    """Exact hydrogenic σ from the continuum oscillator strength (recalled, Bethe & Salpeter §69–71; the same df/dE
    that satisfies the TRK sum rule to 1e-12 in electron_losses_IGM/tests/test_collisions.py):
    σ = (π e² ħ / (2 ε0 m c)) df/dE,  df/d(E/R) = 2⁷ exp[−(4/k) arctan k] / (3 (E/R)⁴ (1 − e^(−2π/k))), k² = E/R − 1."""
    R = K.R_H
    e = mp.mpf(E_eV * K.e / R)
    k = mp.sqrt(e - 1)
    dfde = mp.mpf(2) ** 7 * mp.exp(-4 / k * mp.atan(k)) / (3 * e ** 4 * (1 - mp.exp(-2 * mp.pi / k)))
    return math.pi * K.e ** 2 * K.hbar / (2 * K.eps0 * K.m_e * K.c) * float(dfde) / R


def test_verner_vs_exact_hydrogenic():
    """Verner's class-A fit (rms < 0.2 %, p. 495) vs the exact formula on 13.61 eV … 10 keV: rms < 0.5 %, max < 1 %."""
    E = np.logspace(np.log10(13.61), 4, 200)
    r = np.array([PI.sigma(x) / sigma_exact(x) - 1 for x in E])
    assert math.sqrt(np.mean(r ** 2)) < 5e-3 and np.max(np.abs(r)) < 1e-2


def test_threshold_value():
    """σ(E_th) ≈ 6.3e-18 cm² (Verner fit at 13.60 eV; exact hydrogenic value at threshold 6.30e-18 cm² by sigma_exact)."""
    assert abs(PI.sigma(13.60) / sigma_exact(13.60 + 1e-9) - 1) < 1e-2


@pytest.mark.parametrize("x_e", [1e-4, 1e-2, 1e-1, 0.5, 0.99])
def test_window_definition(x_e):
    """At E_max: n_HI σ c = H(z_init) (1e-9); the window starts at R_H."""
    lo, hi = PI.window(x_e)
    assert lo == K.R_H_eV and hi > lo
    m = Medium(P["z_init"], x_e)
    assert abs(m.n_HI * PI.sigma(hi) * K.c / float(cosmology.hubble(P["z_init"])) - 1) < 1e-9


def test_window_shrinks_with_ionization_and_can_be_empty():
    E = [PI.window(x)[1] for x in (1e-4, 1e-2, 1e-1, 0.5, 0.99)]
    assert all(a > b for a, b in zip(E, E[1:]))
    lo, hi = PI.window(1 - 1e-15)          # essentially no neutral H: no photon can ionize within a Hubble time
    assert hi == lo


def test_above_fit_range_raises():
    with pytest.raises(ValueError):
        PI.sigma(6e4)
