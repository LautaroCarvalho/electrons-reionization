"""Derived constants vs CODATA 2022 recommended values (independent check of constants.py).

Reference values are copied from references/Mohr (2024) CODATA 2022.pdf, Table XXXIII (PDF pp. 51–55).
Pass criterion (fixed before running): relative difference below 5 × the CODATA relative standard
uncertainty of the quantity, or 1e-12 for exact relations. Exception: the Stefan–Boltzmann constant is exact but
printed truncated to 10 digits, so its criterion is one unit of the last printed digit (1e-17 W m^-2 K^-4).
"""

import sys
import pathlib

import pytest
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from igm_losses import constants as K  # noqa: E402

CODATA2022 = {                       # value, relative standard uncertainty (Table XXXIII)
    "alpha": (7.2973525643e-3, 1.6e-10),
    "r_e": (2.8179403205e-15, 4.7e-10),
    "sigma_T": (6.6524587051e-29, 9.3e-10),
    "a0": (5.29177210544e-11, 1.6e-10),
    "hartree": (4.3597447222060e-18, 1.1e-12),
    "eps0": (8.8541878188e-12, 1.6e-10),         # vacuum electric permittivity, p. 51
    "m_e_c2": (8.1871057880e-14, 3.1e-10),       # electron mass energy equivalent [J], p. 52
}
SIGMA_SB_PRINTED = 5.670374419e-8                # Stefan–Boltzmann constant, exact, printed "5.670 374 419 ..." (p. 55)


@pytest.mark.parametrize("name", list(CODATA2022))
def test_against_codata(name):
    value, u = CODATA2022[name]
    ours = 2 * K.Ry_inf if name == "hartree" else getattr(K, name)
    assert abs(ours / value - 1) < 5 * u


def test_exact_relations_symbolic():
    """C25: r_e = α² a0, Ry∞ = α² mc²/2 = e²/(8π ε0 a0) and a_rad = 4σ_SB/c with σ_SB = (π²/60) k⁴/(ħ³c²)
    (CODATA's definition, p. 55) hold symbolically for the formulas used in constants.py. σ_T = (8π/3) r_e² is
    checked numerically in test_against_codata."""
    e, eps0, h, c, m = sp.symbols("e epsilon0 h c m", positive=True)
    hbar = h / (2 * sp.pi)
    alpha = e ** 2 / (2 * eps0 * h * c)
    r_e = e ** 2 / (4 * sp.pi * eps0 * m * c ** 2)
    a0 = hbar / (m * c * alpha)
    assert sp.simplify(r_e - alpha ** 2 * a0) == 0
    assert sp.simplify(alpha ** 2 * m * c ** 2 / 2 - e ** 2 / (8 * sp.pi * eps0 * a0)) == 0
    k = sp.symbols("k_B", positive=True)
    a_rad = 8 * sp.pi ** 5 * k ** 4 / (15 * h ** 3 * c ** 3)          # constants.py
    sigma_sb = sp.pi ** 2 / 60 * k ** 4 / (hbar ** 3 * c ** 2)        # CODATA 2022, Table XXXIII, p. 55
    assert sp.simplify(a_rad - 4 * sigma_sb / c) == 0


def test_radiation_constant_vs_stefan_boltzmann():
    """a_rad c/4 = σ_SB: ours vs the printed CODATA value within one unit of its last printed digit."""
    assert abs(K.a_rad * K.c / 4 - SIGMA_SB_PRINTED) < 1e-17


def test_reduced_mass_rydberg():
    """A12: R_H = Ry∞ m_p/(m_e+m_p) ≈ 13.5984 eV (the value Stone et al. 2002 quote, p. 328)."""
    assert abs(K.R_H_eV - 13.5984) < 5e-4


PRINTED_PRIMARIES = {        # value exactly as printed; source
    "c": (299792458.0, "SI 2019 Brochure §2.2 Table 1; CODATA 2022 Table XXXIII p. 51 (exact)"),
    "h": (6.62607015e-34, "SI 2019 §2.2 Table 1; CODATA 2022 p. 51 (exact)"),
    "e": (1.602176634e-19, "SI 2019 §2.2 Table 1; CODATA 2022 p. 51 (exact)"),
    "k_B": (1.380649e-23, "SI 2019 §2.2 Table 1; CODATA 2022 p. 55 (exact)"),
    "mu0": (1.25663706127e-6, "CODATA 2022 Table XXXIII p. 51: 1.256 637 061 27(20) × 10^-6 N A^-2"),
    "m_e": (9.1093837139e-31, "CODATA 2022 Table XXXIII p. 52: 9.109 383 7139(28) × 10^-31 kg"),
    "m_p": (1.67262192595e-27, "CODATA 2022 Table XXXIII p. 53: 1.672 621 925 95(52) × 10^-27 kg"),
}


@pytest.mark.parametrize("name", list(PRINTED_PRIMARIES))
def test_primary_transcription(name):
    """Transcription check of the primaries in parameters.yaml, digit for digit against the printed values.
    Needed because a wrong μ0 (e.g. 4π×1e-7) moves α and ε0 by less than their CODATA uncertainty, so the derived-
    constant tests above cannot see it (measured 2026-09-28)."""
    assert K._P[name] == PRINTED_PRIMARIES[name][0]          # the registry
    assert getattr(K, name) == PRINTED_PRIMARIES[name][0]    # the value the code actually uses


def test_T_CMB0_is_planck18():
    """T_CMB0 is the astropy Planck18 value (decision 2026-09-27, parameters.yaml)."""
    from astropy.cosmology import Planck18
    assert K.T_CMB0 == Planck18.Tcmb0.value
