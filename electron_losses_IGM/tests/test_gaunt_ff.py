"""Tests of igm_losses.gaunt_ff (exact free-free Gaunt factor, Karzas & Latter 1961).

Pass criteria were fixed before running (see provenance/errata.yaml E07 and the F3 report of 2026-09-27).
Reference values (Born, Elwert) are recalled closed forms, independent of the K&L hypergeometric route.
"""

import sys
import pathlib

import mpmath as mp
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses.gaunt_ff import gff_eta, gff_emission, mean_gff, g_born, elwert_factor  # noqa: E402


@pytest.mark.parametrize("x", [0.1, 0.5, 0.9, 0.99])
def test_born_times_elwert_high_energy(x):
    """η ≪ 1 (E0 = 1e4 Z²Ry): exact g_ff must equal Born × Elwert to < 1e-4 relative."""
    E0 = mp.mpf(10) ** 4
    exact = gff_emission(E0, x * E0)
    approx = g_born(E0, x * E0) * elwert_factor(E0, x * E0)
    assert abs(exact / approx - 1) < 1e-4


def test_symmetry():
    """g_ff(η_i, η_f) = g_ff(η_f, η_i) (K&L Eq. 16 is symmetric)."""
    a, b = mp.mpf("0.7"), mp.mpf("2.3")
    assert abs(gff_eta(a, b) - gff_eta(b, a)) < mp.mpf("1e-25")


def test_precision_convergence():
    """30 and 60 digits agree to 1e-20 at the energy of the integration floor (10.2 eV ≈ 0.75 Ry)."""
    g30 = gff_emission("0.75", "0.375", dps=30)
    g60 = gff_emission("0.75", "0.375", dps=60)
    assert abs(g30 / g60 - 1) < mp.mpf("1e-20")


def test_born_integral_gives_16_over_3():
    """∫_0^1 g_Born dx = 2√3/π, so φ_rad = (8π/(3√3)) ⟨g⟩ = 16/3 (non-relativistic Bethe–Heitler)."""
    s = mp.quad(lambda x: g_born(1, x), [0, 1])  # g_born(E0, hν) depends only on x = hν/E0
    assert abs(s - 2 * mp.sqrt(3) / mp.pi) < mp.mpf("1e-12")
    assert abs(8 * mp.pi / (3 * mp.sqrt(3)) * s - mp.mpf(16) / 3) < mp.mpf("1e-12")


def test_mean_gff_tends_to_born_as_eta():
    """⟨g⟩ → 2√3/π as E0 → ∞, with a Coulomb correction of order η = E0^(-1/2).

    History (2026-09-27): the first version of this test demanded |⟨g⟩/⟨g⟩_Born − 1| < 1 % at E0 = 1e4 Z²Ry.
    That threshold was arbitrary and failed (1.22 %). Measured: 11.9 %, 3.88 %, 1.22 %, 0.385 % at
    E0 = 1e2, 1e3, 1e4, 1e5, i.e. proportional to η. The test now checks that scaling law instead of a
    tuned threshold: (ratio − 1)/η must be the same constant within 10 % at E0 = 1e3, 1e4, 1e5, and the
    ratio must decrease monotonically towards 1.
    """
    born = 2 * mp.sqrt(3) / mp.pi
    E = [mp.mpf(10) ** 3, mp.mpf(10) ** 4, mp.mpf(10) ** 5]
    dev = [mean_gff(e) / born - 1 for e in E]
    c = [d * mp.sqrt(e) for d, e in zip(dev, E)]          # (ratio − 1)/η with η = E0^(-1/2)
    assert all(d > 0 for d in dev) and dev[0] > dev[1] > dev[2]
    assert max(c) / min(c) - 1 < 0.10


def test_deliberate_disagreement_is_detected():
    """C18: at low energy (E0 = 1 Z²Ry) Born is badly wrong; the test must see a > 20 % difference."""
    assert abs(gff_emission(1, 0.5) / g_born(1, 0.5) - 1) > 0.2


def test_invalid_inputs():
    with pytest.raises(ValueError):
        gff_emission(1, 1.5)
    with pytest.raises(ValueError):
        gff_eta(1, 1)


def test_low_x_cut_is_negligible():
    """The x < X_MIN piece left out of ⟨g⟩ must be < 1e-12 of ⟨g⟩ at the floor and at high energy."""
    for E0 in ("0.75", "1e4"):
        val, bound = mean_gff(E0, return_cut_bound=True)
        assert bound / val < mp.mpf("1e-12")


def test_classical_limit_tends_to_kramers():
    """K&L define g_ff as the ratio to Kramers' classical cross-section (Eqs. 15–16). By the correspondence principle
    (recalled, not printed in K&L or R&L) g_ff → 1 as η_i, η_f → ∞ at fixed x = hν/E0. Criterion fixed before
    running: at x = 1/2, |g − 1| decreases monotonically for E0 = 1e-2, 1e-4, 1e-6 Z²Ry (η_i = 10, 100, 1000)
    and is below 1e-2 at the last one. This is the regime (η ≳ 1, near the 10.2 eV floor) where Born fails."""
    dev = [abs(gff_emission(mp.mpf(10) ** -k, mp.mpf(10) ** -k / 2) - 1) for k in (2, 4, 6)]
    assert dev[0] > dev[1] > dev[2]
    assert dev[2] < mp.mpf("1e-2")
