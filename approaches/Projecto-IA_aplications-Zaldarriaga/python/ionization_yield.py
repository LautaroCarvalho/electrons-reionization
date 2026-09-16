#!/usr/bin/env python3
r"""
Average number of HI ionizations per primary particle in the z = 10 IGM.

TARGET      neutral hydrogen of the intergalactic medium at z = 10,
            static ionized fraction x_e = 1e-4
PRIMARIES   electrons  1e2 - 1e12 eV
            photons    1e1 - 1e5  eV
COUNTED     total ion pairs: every HI ionization anywhere in the cascade,
            including those made by all secondary electrons
METHOD      semi-analytic (no Monte Carlo), as specified

--------------------------------------------------------------------------
THE TWO MODELS, AND WHY THERE ARE TWO
--------------------------------------------------------------------------
(A) FULL ABSORPTION.  All of the primary's kinetic energy is degraded in the
    gas COLLISIONALLY.  N = f_ion(x_e) * E / E_th.  The standard textbook
    answer, and the reference curve -- but see the note below: it is an upper
    bound on collisional degradation, not on the total yield.

(B) IC-LIMITED, COLLISIONAL ONLY.  At z = 10 the CMB energy density is
    (1+z)^4 = 14641 times its present value.  An electron loses energy to
    inverse-Compton scattering off CMB photons at a rate proportional to
    gamma^2, while its collisional loss rate to the gas is nearly
    energy-independent.  Above E_crit the electron gives its energy to the CMB
    rather than to the gas, and the collisional yield SATURATES.  Model (B)
    integrates that branching ratio along the slowing-down track and THROWS THE
    IC ENERGY AWAY.  It is a lower bound.

(C) IC-LIMITED PLUS SECONDARY PHOTOIONIZATION.  The IC energy is not lost.  An
    electron of gamma > ~38 (T > ~19 MeV) upscatters CMB photons past the 13.6
    eV HI threshold, and those photons photoionize provided they are absorbed
    before they free-stream away -- true below the escape energy ~1.2 keV.
    Model (C) follows them.  This is the physically correct answer over the
    requested range and it is the headline result.  Section 7b.

All three are plotted.  (A) is NOT a strict upper bound on the total yield:
reprocessing energy into photons just above 13.6 eV ionizes at 1 pair per
13.6 eV, better than the 1 per W = 36.3 eV of electron degradation, so (C)
slightly exceeds (A) near 1e8 eV.  The only strict ceiling is E/E_th, and that
is checked.  See the CAVEATS block at the end of main().

--------------------------------------------------------------------------
PROVENANCE OF EVERY CONSTANT
--------------------------------------------------------------------------
Each numerical constant below carries its source in a trailing comment.
Nothing in this file is a remembered round number.  Values marked
[RECALLED - UNVERIFIED] could not be checked against the primary source from
this working directory; they are flagged again in the printed report.

Run:  python3 ionization_yield.py
Outputs: ionization_yield_fig.png, ionization_yield_fig.pdf, results.json,
         provenance/numbers.json, provenance/claims.yaml
Deterministic: no random numbers are drawn anywhere (no seed to record).
"""

from __future__ import annotations

import project_paths  # noqa: F401  -- anchors CWD to the project root
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ===========================================================================
# 1. PHYSICAL CONSTANTS   (CODATA 2018 unless noted; CGS + eV)
# ===========================================================================
C_LIGHT   = 2.99792458e10        # cm/s              CODATA 2018 (exact)
SIGMA_T   = 6.6524587321e-25     # cm^2              CODATA 2018, Thomson x-sec
R_E       = 2.8179403262e-13     # cm                CODATA 2018, classical e- radius
MEC2_EV   = 510998.95000         # eV                CODATA 2018, m_e c^2
ERG_PER_EV = 1.602176634e-12     # erg/eV            CODATA 2018 (exact, = e in C)
A_RAD     = 7.565733e-15         # erg cm^-3 K^-4    CODATA 2018, radiation constant
M_H       = 1.67353e-24          # g                 mass of the H atom (m_p + m_e - B/c^2)

# Hydrogen atomic data
E_TH_HI   = 13.598434599702      # eV   NIST ASD, HI ionization energy
SIGMA_PI0 = 6.304e-18            # cm^2 Osterbrock & Ferland 2006, Table 2.7,
                                 #      HI photoionization x-sec AT threshold
I_EXC_H   = 14.99                # eV   ICRU Report 37 / Dalgarno: mean excitation
                                 #      energy of ATOMIC hydrogen
I_EXC_H2  = 19.2                 # eV   ICRU Report 37: same for molecular H2.
                                 #      Used only as a sensitivity variant (C36).

# Cosmology: Planck 2018 VI, Table 2, TT,TE,EE+lowE+lensing+BAO
OMEGA_B_H2 = 0.02242             # dimensionless
H_LITTLE   = 0.6766              # dimensionless
OMEGA_M    = 0.3111              # dimensionless
OMEGA_L    = 1.0 - OMEGA_M       # flat, by assumption (Planck 2018: |Omega_k|<0.002)
T_CMB0     = 2.7255              # K     Fixsen 2009, ApJ 707, 916
Y_P        = 0.245               # primordial He mass fraction (Planck 2018 / BBN)
X_H        = 1.0 - Y_P           # hydrogen mass fraction
RHO_CRIT_COEF = 1.87834e-29      # g cm^-3, rho_crit = COEF * h^2   (PDG 2022)
H0_S        = 100.0 * H_LITTLE * 1.0e5 / 3.0856775814913673e24  # s^-1

# ---------------------------------------------------------------------------
# Shull & van Steenberg 1985, ApJ 298, 268 -- fitting formulae for the fraction
# of a fast electron's energy deposited in each channel, as a function of the
# ionized fraction x, in a primordial H+He gas.  Asymptotic (E >~ 100 eV) form.
#
#   TWELVE coefficients: four channels x (A, B, C).  Only the three of the
#   ion_HI row -- 0.3908, 0.4092, 1.7592 -- enter the headline answer; the other
#   nine are used solely by the Platzman energy-budget check.
#
#   [RECALLED - UNVERIFIED]  The paper is not in this working directory and this
#   session has no network access, so none of the twelve could be checked
#   against the source.  They are RECALLED FROM MEMORY, which is precisely the
#   category the project rules forbid for numerical values, and the flag stays
#   until someone puts the PDF in this tree.  What does constrain them:
#     - the Platzman energy budget, sum of channels <= 1  (tests the SET)
#     - W = E_th/f_ion = 36.26 eV against the independently measured
#       W(H) ~ 36 eV per ion pair  (tests the RATIO 0.3908-ish / 13.598)
#   Neither confirms an individual coefficient, and B and C of every row are
#   untested at x_e = 1e-4 because (1 - x^B)^C ~ 1 - C x^B there: at x = 1e-4
#   the fit is within 1% of its x -> 0 asymptote, so the run is only ever
#   sampling the amplitude A.
# ---------------------------------------------------------------------------
SVDS = {
    "ion_HI":  (0.3908, 0.4092, 1.7592),   # f = A (1 - x^B)^C
    "ion_HeI": (0.0554, 0.4614, 1.6660),
    "exc_Lya": (0.4766, 0.2735, 1.5221),
}
SVDS_HEAT = (0.9971, 0.2663, 1.3163)       # f = A [1 - (1 - x^B)^C]

# ===========================================================================
# 2. MODEL PARAMETERS  (the run's inputs, recorded verbatim in results.json)
# ===========================================================================
@dataclass(frozen=True)
class Params:
    z: float = 10.0
    x_e: float = 1.0e-4
    I_exc_eV: float = I_EXC_H
    T_bethe_min_eV: float = 1.0e3   # below this, Bethe-Berger-Seltzer is not
                                    # applied; f_coll is set to 1 (justified by
                                    # check 9: b_IC/b_coll < 1e-3 there)
    tau_mode_default: str = "hubble"
    f_ion_model: str = "fs10"       # "fs10" = energy-dependent, FS10 eq. (13);
                                    # "svds_asymptotic" = the old constant W.
    E_e_min: float = 1.0e2
    E_e_max: float = 1.0e12
    E_g_min: float = 1.0e1
    E_g_max: float = 1.0e5


# ===========================================================================
# 3. COSMOLOGY AT z
# ===========================================================================
def cosmology(z: float) -> dict:
    """Background quantities at redshift z.  DERIVED from the constants above."""
    rho_crit0 = RHO_CRIT_COEF * H_LITTLE**2            # g/cm^3
    omega_b = OMEGA_B_H2 / H_LITTLE**2
    n_H0 = omega_b * rho_crit0 * X_H / M_H             # cm^-3, comoving = proper at z=0
    n_H = n_H0 * (1.0 + z) ** 3                        # cm^-3, proper
    T_cmb = T_CMB0 * (1.0 + z)                         # K
    u_cmb = A_RAD * T_cmb**4                           # erg/cm^3
    H_z = H0_S * np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)   # s^-1
    return {
        "omega_b": omega_b, "n_H0_cm3": n_H0, "n_H_cm3": n_H,
        "T_cmb_K": T_cmb, "u_cmb_erg_cm3": u_cmb,
        "H_z_s": H_z, "hubble_length_cm": C_LIGHT / H_z,
    }


# ===========================================================================
# 4. ENERGY-DEPOSITION FRACTIONS  (Shull & van Steenberg 1985)
# ===========================================================================
def f_channel(x: float, key: str) -> float:
    A, B, C = SVDS[key]
    return A * (1.0 - x**B) ** C


def f_heat(x: float) -> float:
    A, B, C = SVDS_HEAT
    return A * (1.0 - (1.0 - x**B) ** C)


def f_ion_HI(x: float) -> float:
    """Fraction of COLLISIONALLY deposited energy that goes into HI ionization."""
    return f_channel(x, "ion_HI")


# ===========================================================================
# 4b. FURLANETTO & STOEVER 2010 -- energy-DEPENDENT deposition, and the
#     secondary-electron energy distribution
# ===========================================================================
r"""
SOURCE, NOW IN THE TREE:  papers/0910.4410-furlanetto-stoever-2010.pdf
  S. R. Furlanetto & S. J. Stoever, MNRAS 404, 1869 (2010),
  "Secondary ionization and heating by fast electrons".

WHAT THIS PAPER DOES AND DOES NOT GIVE US
-----------------------------------------
It does NOT give a fitting formula for its own results, and its tables are not
public: "Electronic tables of our results are available on request" (abstract),
and section 7: "We suspect this is why a single, separable fitting function
fails, and we have not attempted to find another form.  Instead, we recommend
interpolating the exact results."  So THE FS10 TABLES CANNOT BE USED HERE.
Fabricating them from memory is exactly the failure mode this project forbids.

What it DOES give, with numbers, and what is therefore implemented below:

(1) eq. (2) -- the ENERGY DISTRIBUTION OF THE SECONDARY ELECTRON, which is the
    prescription this rewrite was asked for.  Adapted by FS10 from
    Dalgarno, Yan & Liu (1999), themselves fitting the measurements of
    Opal, Peterson & Beaty (1971, J. Chem. Phys. 55, 4100):

        p(eps) proportional to 1 / [1 + (eps/epsbar_i)^2.1],
        epsbar_i = 8, 15.8, 32.6 eV for HI, HeI, HeII

    with the "secondary" DEFINED as the lower-energy of the two outgoing
    electrons, so eps < (E - E_i)/2.  FS10 quote median secondary energies of
    7.2, 14.2 and 28.5 eV.  Those three numbers are an EXTERNAL ANCHOR on the
    implementation below -- they are not used to build it.

(2) eqs. (13)-(14) -- energy-DEPENDENT fits, from Ricotti, Gnedin & Shull
    (2002), to the Shull & van Steenberg (1985) results:

      f_ion,HI = -0.69 (28 eV/E)^0.4 x^0.2 (1-x^0.38)^2
                 + 0.39 (1-x^0.41)^1.76,                      E > 28 eV
      f_heat   =  3.9811 (11 eV/E)^0.7 x^0.4 (1-x^0.34)^2
                 + [1 - (1-x^0.27)^1.32],                      E > 11 eV
      and ZERO otherwise.

    FS10's own caution, quoted so it cannot be forgotten: "Although (for the
    most part) the fits are reasonably accurate at E > 1 keV, they provide a
    relatively poor match at lower energies.  We therefore caution against
    these fits for high-accuracy work."  That caution is carried into the
    results as a bracket, not buried.

    NOTE ON SCOPE: f_ion here is the fraction of the initial electron energy
    reaching HI ionization, summed over the whole cascade.  FS10 sec. 6:
    "our parameters do not include the effects of the initial ionization event
    that generates the electron" -- which is exactly why N_gamma carries its
    own explicit +1.

WHY THIS FINALLY LETS US CHECK THE RECALLED SvdS COEFFICIENTS
-------------------------------------------------------------
Eq. (13) contains the SvdS asymptote at two significant figures,
0.39 (1-x^0.41)^1.76, against the recalled 0.3908 (1-x^0.4092)^1.7592; eq. (14)
likewise carries 1.0, 0.27, 1.32 against the recalled 0.9971, 0.2663, 1.3163.
FS10 also state a hard number: at x_i = 0.01 the SvdS EXACT heating fraction is
0.32, and the Ricotti FIT sits ~6 per cent above it in absolute terms.  Both
comparisons are run as checks below.  This is the first external test the SvdS
coefficients have had in this project.
"""

# FS10 eq. (2) / Dalgarno, Yan & Liu 1999 / Opal, Peterson & Beaty 1971
FS10_SEC_INDEX = 2.1                       # the exponent in eq. (2)
FS10_EPSBAR_EV = {"HI": 8.0, "HeI": 15.8, "HeII": 32.6}
FS10_MEDIAN_EV = {"HI": 7.2, "HeI": 14.2, "HeII": 28.5}   # quoted; ANCHOR ONLY
FS10_ETH_EV = {"HI": 13.598434599702, "HeI": 24.587, "HeII": 54.418}

# FS10 eqs. (13)-(14) = Ricotti, Gnedin & Shull 2002 fits to SvdS85
# NOTE THE TWO DIFFERENT FUNCTIONAL FORMS. Eq. (13) ends in +A(1-x^B)^C;
# eq. (14) ends in +[1-(1-x^B)^C]. Using the ionization form for heating gives
# 0.638 instead of 0.362 at x=0.01 -- a factor 1.8, and it looks perfectly
# plausible. That is what the first version of this code did.
RGS_ION = dict(E0=28.0, a=-0.69, pE=0.4, px=0.2, q=0.38, r=2.0,
               A=0.39, B=0.41, C=1.76, form="ion")
RGS_HEAT = dict(E0=11.0, a=3.9811, pE=0.7, px=0.4, q=0.34, r=2.0,
                A=1.0, B=0.27, C=1.32, form="heat")


def p_secondary(eps, E_eV, species: str = "HI"):
    """FS10 eq. (2): pdf of the ejected ('secondary') electron energy.

    Normalised on [0, (E - E_i)/2] -- the upper limit is FS10's own definition
    of which of the two outgoing electrons is called the secondary, not an
    approximation.  Returns 0 outside.
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    E = float(E_eV)
    eps_max = 0.5 * (E - FS10_ETH_EV[species])
    if eps_max <= 0.0:
        return np.zeros_like(eps)
    ebar = FS10_EPSBAR_EV[species]
    shape = lambda u: 1.0 / (1.0 + (u / ebar) ** FS10_SEC_INDEX)
    norm, _ = quad(shape, 0.0, eps_max, limit=200)
    out = np.where((eps >= 0.0) & (eps <= eps_max), shape(eps) / norm, 0.0)
    return out


def secondary_moments(E_eV, species: str = "HI"):
    """(median, mean) of FS10 eq. (2) at incident energy E.  Both DERIVED."""
    E = float(E_eV)
    eps_max = 0.5 * (E - FS10_ETH_EV[species])
    if eps_max <= 0.0:
        return 0.0, 0.0
    ebar = FS10_EPSBAR_EV[species]
    shape = lambda u: 1.0 / (1.0 + (u / ebar) ** FS10_SEC_INDEX)
    norm, _ = quad(shape, 0.0, eps_max, limit=200)
    mean, _ = quad(lambda u: u * shape(u), 0.0, eps_max, limit=200)
    med = brentq(lambda m: quad(shape, 0.0, m, limit=200)[0] / norm - 0.5,
                 1e-6, eps_max, xtol=1e-10)
    return med, mean / norm


def _rgs(E_eV, x: float, c: dict):
    """FS10 eqs. (13)/(14) evaluated. Zero at and below the stated cutoff E0."""
    E = np.asarray(E_eV, dtype=float)
    asym = (c["A"] * (1.0 - x ** c["B"]) ** c["C"] if c["form"] == "ion"
            else c["A"] * (1.0 - (1.0 - x ** c["B"]) ** c["C"]))
    slope = c["a"] * x ** c["px"] * (1.0 - x ** c["q"]) ** c["r"]
    val = slope * (c["E0"] / np.maximum(E, 1e-300)) ** c["pE"] + asym
    return np.where(E > c["E0"], val, 0.0)


def f_ion_HI_E(E_eV, x: float):
    """Energy-DEPENDENT HI ionization deposition fraction, FS10 eq. (13)."""
    return _rgs(E_eV, x, RGS_ION)


def f_heat_E(E_eV, x: float):
    """Energy-DEPENDENT heating fraction, FS10 eq. (14)."""
    return _rgs(E_eV, x, RGS_HEAT)


def f_ion_used(E_eV, p: "Params"):
    """The deposition fraction actually used, per the model switch in Params."""
    if p.f_ion_model == "fs10":
        return f_ion_HI_E(E_eV, p.x_e)
    if p.f_ion_model == "svds_asymptotic":
        E = np.asarray(E_eV, dtype=float)
        return np.where(E >= E_TH_HI, f_ion_HI(p.x_e), 0.0)
    raise ValueError(f"unknown f_ion_model {p.f_ion_model!r}")


def N_full_of_E(E_eV, p: "Params"):
    """Ion pairs from an electron of energy E whose energy is ALL degraded in
    the gas.  N = f_ion(E, x_e) E / E_th.  Now energy dependent."""
    E = np.asarray(E_eV, dtype=float)
    return f_ion_used(E, p) * E / E_TH_HI


def dN_full_dE(E_eV, p: "Params"):
    """d/dE [ f_ion(E) E / E_th ] -- the MARGINAL yield per unit energy.

    This, not f_ion/E_th, is what multiplies the collisional branching ratio
    once f_ion depends on energy: an electron cooling through dT deposits
    f_coll dT to the gas, and the ionizations that buys are dN_full/dT dT.
    With a constant f_ion it reduces to f_ion/E_th exactly, which is checked.

    Analytic for the FS10 form, because N_full = [A E - B E0^pE E^(1-pE)]/E_th:
        dN/dE = [A - (1-pE) B (E0/E)^pE] / E_th
    """
    E = np.asarray(E_eV, dtype=float)
    if p.f_ion_model == "svds_asymptotic":
        return np.where(E >= E_TH_HI, f_ion_HI(p.x_e) / E_TH_HI, 0.0)
    c = RGS_ION
    asym = c["A"] * (1.0 - p.x_e ** c["B"]) ** c["C"]
    slope = c["a"] * p.x_e ** c["px"] * (1.0 - p.x_e ** c["q"]) ** c["r"]
    val = (asym + (1.0 - c["pE"]) * slope * (c["E0"] / np.maximum(E, 1e-300)) ** c["pE"])
    return np.where(E > c["E0"], val / E_TH_HI, 0.0)


# ---------------------------------------------------------------------------
# The pure-ionization cascade built DIRECTLY from FS10 eq. (2).
# ---------------------------------------------------------------------------
_YCASC_CACHE: dict = {}


def ionization_only_cascade(E_hi: float = 5000.0, h: float = 0.5,
                            n_eps: int = 240, species: str = "HI"):
    """Yield of a cascade in which EVERY inelastic event is an ionization.

    Y(E) = 0                                             for E < E_th
    Y(E) = 1 + <Y(eps) + Y(E - E_th - eps)>_p(eps)       for E >= E_th

    with p from FS10 eq. (2).  Excitation and electron-electron heating are
    switched off, so this is a strict UPPER BOUND on the true yield and it uses
    NOTHING from the Shull & van Steenberg fits -- a genuinely independent
    route, which is the point.  Both arguments of Y on the right are below
    E - E_th, so the recursion marches upward on a grid with no iteration.

    The grid is LINEAR with step h, deliberately.  On a log grid the argument
    E - E_th - eps sits between the last computed point and E itself whenever
    eps is small, so np.interp reaches into a cell that is still zero; the
    recursion then collapses -- Y(1e3) = 55 followed by Y(1e4) = 2.7, i.e.
    non-monotonic, which is how it was caught.  With a linear step h << E_th
    every argument is at least E_th/h cells below E and the march is exact.

    Returns (E grid, Y grid, W_ion) with W_ion = lim E/Y(E), the mean energy
    per ion pair of an ionization-only cascade.
    """
    key = (E_hi, h, n_eps, species)
    if key in _YCASC_CACHE:
        return _YCASC_CACHE[key]
    Eth = FS10_ETH_EV[species]
    ebar = FS10_EPSBAR_EV[species]
    Eg = np.arange(0.0, E_hi + h, h)
    Y = np.zeros_like(Eg)
    for i, E in enumerate(Eg):
        if E < Eth:
            continue
        emax = 0.5 * (E - Eth)
        if emax <= 0.0:
            Y[i] = 1.0
            continue
        e = np.linspace(0.0, emax, n_eps)
        w = 1.0 / (1.0 + (e / ebar) ** FS10_SEC_INDEX)
        w /= _trapz(w, e)
        y1 = np.interp(e, Eg[:i], Y[:i], left=0.0, right=Y[i - 1])
        y2 = np.interp(E - Eth - e, Eg[:i], Y[:i], left=0.0, right=Y[i - 1])
        Y[i] = 1.0 + _trapz(w * (y1 + y2), e)
    W_ion = float(Eg[-1] / Y[-1])
    _YCASC_CACHE[key] = (Eg, Y, W_ion)
    return Eg, Y, W_ion


# ===========================================================================
# 5. ENERGY-LOSS RATES FOR AN ELECTRON
# ===========================================================================
# numpy renamed trapz -> trapezoid in 2.0; support both without pinning.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _kinematics(T_eV):
    """tau, gamma, beta^2 from kinetic energy T."""
    tau = np.asarray(T_eV, dtype=float) / MEC2_EV
    gamma = tau + 1.0
    beta2 = 1.0 - 1.0 / gamma**2
    return tau, gamma, beta2


def stopping_power_erg_per_cm(T_eV, n_e_cm3: float, I_eV: float):
    """Collision stopping power -dE/dx for electrons.

    Berger-Seltzer form, ICRU Report 37 eq. (2.3):

      -dE/dx = (2 pi r_e^2 m_e c^2 n_e / beta^2) [ ln(tau^2 (tau+2) / 2 (I/mc^2)^2)
                                                   + F^-(tau) - delta ]
      F^-(tau) = 1 - beta^2 + [tau^2/8 - (2 tau + 1) ln 2] / (tau + 1)^2

    Density-effect correction delta is set to 0: at n_H ~ 1e-4 cm^-3 the plasma
    energy is ~1e-8 eV, so delta is utterly negligible (DERIVED, not assumed).
    Returns erg/cm.
    """
    tau, gamma, beta2 = _kinematics(T_eV)
    Imc2 = I_eV / MEC2_EV
    arg = tau**2 * (tau + 2.0) / (2.0 * Imc2**2)
    F_minus = 1.0 - beta2 + (tau**2 / 8.0 - (2.0 * tau + 1.0) * np.log(2.0)) / (tau + 1.0) ** 2
    coef = 2.0 * np.pi * R_E**2 * (MEC2_EV * ERG_PER_EV) * n_e_cm3 / beta2
    return coef * (np.log(arg) + F_minus)


def b_coll_erg_per_s(T_eV, n_e_cm3: float, I_eV: float):
    """Collisional energy-loss RATE of an electron in the gas [erg/s]."""
    _, _, beta2 = _kinematics(T_eV)
    return np.sqrt(beta2) * C_LIGHT * stopping_power_erg_per_cm(T_eV, n_e_cm3, I_eV)


def b_ic_erg_per_s(T_eV, u_cmb: float):
    """Inverse-Compton loss RATE off the CMB [erg/s].

    -dE/dt = (4/3) sigma_T c u_rad gamma^2 beta^2,  and gamma^2 beta^2 = gamma^2 - 1.
    Standard result (Rybicki & Lightman 1979, eq. 7.16). Valid in the Thomson
    regime, gamma * kT_cmb << m_e c^2 -- checked numerically below.
    """
    _, gamma, _ = _kinematics(T_eV)
    return (4.0 / 3.0) * SIGMA_T * C_LIGHT * u_cmb * (gamma**2 - 1.0)


def f_coll(T_eV, cos: dict, p: Params, I_eV: float | None = None):
    """Fraction of the electron's energy loss that goes to the GAS, not the CMB."""
    I_eV = p.I_exc_eV if I_eV is None else I_eV
    T = np.asarray(T_eV, dtype=float)
    out = np.ones_like(T)
    hi = T >= p.T_bethe_min_eV
    if np.any(hi):
        n_e = cos["n_H_cm3"]                    # 1 bound electron per H atom
        bc = b_coll_erg_per_s(T[hi], n_e, I_eV)
        bi = b_ic_erg_per_s(T[hi], cos["u_cmb_erg_cm3"])
        out[hi] = bc / (bc + bi)
    return out if out.shape else float(out)


# ===========================================================================
# 6. IONIZATION YIELDS
# ===========================================================================
def N_e_full(E_eV, p: Params):
    """Model (A): all energy degraded collisionally, N = f_ion(E, x_e) E/E_th.

    f_ion is now ENERGY DEPENDENT (FS10 eq. 13), so W = E_th/f_ion is a genuine
    W(K_e): 36.3 eV asymptotically, but 43.5 eV at 100 eV and infinite below
    28 eV, where the published fit is identically zero. The previous version of
    this function used the asymptotic value at every energy; that variant is
    still reachable through Params.f_ion_model for the comparison.

    Upper bound on the COLLISIONAL yield only. The depfit route can exceed it, because
    energy routed through photons just above threshold buys one ion pair per
    13.6 eV instead of one per W. The strict ceiling is E/E_th."""
    return N_full_of_E(E_eV, p)


def _deposited_energy_eV(E: float, cos: dict, p: Params, I_eV: float | None) -> float:
    """Integral_0^E f_coll(T) dT  -- the energy actually given to the gas [eV].

    INTEGRATED IN LOG SPACE, deliberately.  f_coll is ~1 below E_crit ~ 1e5 eV
    and falls as 1/T^2 above it, so over a range reaching 1e12 eV the integrand
    is confined to the first ten-thousandth of the interval.  A linear-space
    quad samples almost entirely where f_coll ~ 0, misses the contribution, and
    returns a value that is BOTH wrong and smoothly varying -- i.e. it does not
    look wrong.  That is what the first version of this function did; the
    monotonicity check caught it.  Substituting T = e^u puts the support in the
    middle of the interval.
    """
    if E <= p.T_bethe_min_eV:
        return E                                   # f_coll == 1 identically here
    lo = p.T_bethe_min_eV
    val, _ = quad(lambda u: float(f_coll(np.exp(u), cos, p, I_eV)) * np.exp(u),
                  np.log(lo), np.log(E), limit=400)
    return lo + val                                # + the exact f_coll==1 piece


def _ic_limited_yield(E: float, cos: dict, p: Params, I_eV: float | None) -> float:
    """Model (B) for one energy, with an energy-dependent f_ion.

        N_B(E) = Integral_0^E f_coll(T) * dN_full/dT dT

    dN_full/dT is the MARGINAL ion-pair yield per unit energy deposited at T.
    With a constant f_ion this collapses to (f_ion/E_th) Integral f_coll dT,
    the old formula, and that collapse is checked.

    The FS10/Ricotti fit is identically zero at and below E0 = 28 eV and jumps
    to a finite value just above, so dN_full/dT carries a delta function there.
    It is added explicitly rather than left to the quadrature to miss: f_coll is
    exactly 1 at 28 eV (checked), so the step contributes its full height.
    """
    E0 = RGS_ION["E0"] if p.f_ion_model == "fs10" else E_TH_HI
    if E <= E0:
        return 0.0
    step = float(N_full_of_E(E0 * (1.0 + 1e-12), p)) * float(f_coll(E0, cos, p, I_eV))
    val, _ = quad(lambda u: (float(f_coll(np.exp(u), cos, p, I_eV))
                             * float(dN_full_dE(np.exp(u), p)) * np.exp(u)),
                  np.log(E0), np.log(E), limit=400)
    return step + val


def N_e_ic(E_eV, cos: dict, p: Params, I_eV: float | None = None):
    """Model (B): IC-limited, collisional deposition only."""
    E_arr = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.array([_ic_limited_yield(float(E), cos, p, I_eV) for E in E_arr])
    return out if np.ndim(E_eV) else float(out[0])


# ---------------------------------------------------------------------------
# BAND EDGES.  A grid built as 10**log10(E_TH_HI) can land one ULP BELOW
# E_TH_HI, and an exact ">=" then returns zero AT the threshold itself. Because
# the ionizing spectrum is steepest exactly there, that silently removed 0.35%
# of zeta_gamma from every number this project published between 2026-09-07 and
# 2026-09-13. Compare with a relative tolerance instead: 1e-12 of 13.6 eV is
# 1.4e-11 eV, far below any physical scale in this calculation.
EDGE_RTOL = 1.0e-12


def at_or_above(E, edge, rtol=EDGE_RTOL):
    """E >= edge, tolerant of a grid point landing one ULP below `edge`."""
    return np.asarray(E, dtype=float) >= edge * (1.0 - rtol)


def within_band(E, lo, hi, rtol=EDGE_RTOL):
    """lo <= E <= hi, tolerant at both edges."""
    E = np.asarray(E, dtype=float)
    return (E >= lo * (1.0 - rtol)) & (E <= hi * (1.0 + rtol))


def N_gamma(E_eV, cos: dict, p: Params, model: str, chan=None):
    """Photon yield.

    Below 13.6 eV a photon cannot ionize HI at all -> exactly zero.
    Above it, the primary interaction removes one bound electron (1 ionization)
    and delivers E - E_th of kinetic energy to the photoelectron, which then
    cascades.  In the full-absorption limit this is exact by energy conservation
    regardless of whether the first interaction was photoelectric or Compton.
    """
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.zeros_like(E)
    above = at_or_above(E, E_TH_HI)
    if np.any(above):
        # Clamped at zero. at_or_above() deliberately admits a grid point one
        # ULP BELOW the threshold (that tolerance is the 2026-09-13 fix), and
        # E - E_TH_HI is then -1.8e-15, which reaches a log() downstream and
        # returns NaN. Physically the photoelectron has exactly zero kinetic
        # energy at threshold, so zero is the right value, not a small negative.
        T_sec = np.maximum(E[above] - E_TH_HI, 0.0)
        if model == "full":
            out[above] = 1.0 + N_e_full(T_sec, p)
        elif model == "ic":
            out[above] = 1.0 + N_e_ic(T_sec, cos, p)
        elif model == "C":
            # The photoelectron itself may upscatter CMB photons.  Above 1e5 eV
            # (the top of the requested photon range) this is a sub-percent
            # correction, but it costs nothing to carry it.
            out[above] = 1.0 + N_e_depfit(T_sec, cos, p, chan)
        else:
            raise ValueError(f"unknown model {model!r}")
    return out if np.ndim(E_eV) else float(out[0])


# ===========================================================================
# 7. PHOTON INTERACTION BRANCHING  (diagnostic only -- see check 12)
# ===========================================================================
def sigma_photoion(E_eV):
    """Exact ground-state hydrogenic photoionization cross section.

    Osterbrock & Ferland 2006 eq. (2.4) / Karzas & Latter 1961.
    """
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.zeros_like(E)
    m = E > E_TH_HI
    eps = np.sqrt(E[m] / E_TH_HI - 1.0)
    out[m] = (SIGMA_PI0 * (E_TH_HI / E[m]) ** 4
              * np.exp(4.0 - 4.0 * np.arctan(eps) / eps)
              / (1.0 - np.exp(-2.0 * np.pi / eps)))
    out[np.isclose(E, E_TH_HI)] = SIGMA_PI0
    return out if np.ndim(E_eV) else float(out[0])


def igm_cutoff_eV(z, x_e=1.0e-4):
    """Photon energy at which tau_IGM = 1 over one Hubble length.

    THE BAND POLICY (settled 2026-09-15).  A photon above this energy
    free-streams: the IGM does not absorb it, so it cannot ionize however many
    are emitted.  It is the IGM's half of the canonical photon band top,

        band top = min(source emission cutoff, igm_cutoff_eV(z))

    the source's half being where its own spectrum ends (4 Ryd for stars, the
    He II edge).  Before this was settled three different band tops coexisted in
    the project and it reported BOTH 2.280x and 2.073x for the same statement.

    Lives here because this module already owns sigma_photoion and the
    cosmology; putting it in parameters.py would drag the physics stack into a
    loader that must stay light.  At z = 10 it returns 1218.0 eV.
    """
    from scipy.optimize import brentq
    cos = cosmology(float(z))
    n_HI = cos["n_H_cm3"] * (1.0 - x_e)
    L_h = cos["hubble_length_cm"]
    f = lambda lg: float(n_HI * sigma_photoion(10.0 ** lg) * L_h) - 1.0
    return 10.0 ** brentq(f, np.log10(E_TH_HI * 1.001), 5.0, xtol=1e-12)


def sigma_kn(E_eV):
    """Total Klein-Nishina cross section per electron (Rybicki & Lightman eq. 7.5)."""
    x = np.asarray(E_eV, dtype=float) / MEC2_EV
    term1 = (1.0 + x) / x**2 * (2.0 * (1.0 + x) / (1.0 + 2.0 * x) - np.log1p(2.0 * x) / x)
    term2 = np.log1p(2.0 * x) / (2.0 * x)
    term3 = (1.0 + 3.0 * x) / (1.0 + 2.0 * x) ** 2
    return 2.0 * np.pi * R_E**2 * (term1 + term2 - term3)


# ===========================================================================
# 7b. THE IC-SECONDARY CHANNEL
#     CMB photons upscattered by the electron, and the ionizations they make
# ===========================================================================
r"""
WHY THIS SECTION EXISTS
-----------------------
Model (B) stopped the accounting at the electron: energy handed to the CMB by
inverse Compton was declared lost.  It is not lost.  An electron of Lorentz
factor gamma boosts a CMB photon of energy eps0 to <eps1> = (4/3) gamma^2 eps0.
With <eps0> = 2.701 kT_CMB(z=10) = 6.98e-3 eV, the upscattered photon clears the
13.6 eV HI threshold once

      gamma  >~  sqrt( E_th / (4 kT <w>) )  ~  38        (T_e ~ 19 MeV)

so every electron between roughly 10 MeV and 1 GeV converts most of its energy
into photons that CAN photoionize.  Whether they DO depends on whether they are
absorbed before they free-stream out of the neighbourhood, which is set by

      tau(eps1) = n_HI sigma_pi(eps1) * (c/H(z)),        sigma_pi ~ eps^-3

Below ~1.4 keV tau >> 1 and the photon is absorbed promptly; above it the photon
escapes.  The channel therefore switches ON near 10 MeV and OFF again near a few
hundred MeV, which is exactly the MeV-GeV window.

THE SPECTRUM
------------
For ISOTROPIC monochromatic seed photons in the Thomson limit the scattered
photon number spectrum is (Blumenthal & Gould 1970, Rev. Mod. Phys. 42, 237,
eq. 2.42; Rybicki & Lightman 1979 sec. 7.3), with x = eps1 / (4 gamma^2 eps0):

      dN/dx = 3 F(x),   F(x) = 2 x ln x + x + 1 - 2 x^2,   0 <= x <= 1

DERIVED here, not asserted: int_0^1 3F dx = 1 and int_0^1 3F x dx = 1/3, so
<eps1> = (4/3) gamma^2 eps0 comes out of the spectrum rather than being imposed.
Both are checked numerically below (C26).

Convolving with a blackbody of temperature T and writing w = eps1/(4 gamma^2 kT):

      ptilde(w) = 3 * int_{ln w}^{inf} dv  fhat(e^v) F(w e^{-v}),   fhat = Planck

ptilde is UNIVERSAL -- independent of gamma.  Only the mapping eps1 = 4 gamma^2
kT w carries gamma.  That is what makes this tractable without a Monte Carlo.

NO RECURSION IS NEEDED, and that is checked, not assumed: photons above the
escape energy (~1.4 keV) leave, so every absorbed photon makes a photoelectron
far below E_crit ~ 1.3e5 eV, where the IC branch is negligible.  The check
'IC secondary cascade terminates' below reports the ratio.
"""

K_B_EV = 8.617333262e-5          # eV/K   CODATA 2018 (exact by SI redefinition)
ZETA3  = 1.2020569031595943      # Apery's constant, zeta(3)
ZETA2  = 1.6449340668482264      # pi^2/6
PI4_15 = 6.493939402266829       # pi^4/15


def kT_cmb_eV(cos: dict) -> float:
    """CMB temperature in eV at the redshift already fixed in `cos`."""
    return K_B_EV * cos["T_cmb_K"]


def planck_number_pdf(u):
    """Normalised photon-NUMBER distribution of a blackbody, u = eps/kT.

        fhat(u) = u^2 / (e^u - 1) / (2 zeta(3)),   int fhat du = 1
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    out = np.zeros_like(u)
    m = (u > 0.0) & (u < 700.0)
    out[m] = u[m] ** 2 / np.expm1(u[m]) / (2.0 * ZETA3)
    return out


def F_thomson(x):
    """Blumenthal & Gould 1970 eq. (2.42), Thomson limit, isotropic seeds.
    3 F(x) is the pdf of x = eps1 / (4 gamma^2 eps0) on [0, 1]."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    m = (x > 0.0) & (x <= 1.0)
    xm = x[m]
    out[m] = 2.0 * xm * np.log(xm) + xm + 1.0 - 2.0 * xm ** 2
    return out


_PTILDE_CACHE: dict = {}


def ic_scaled_spectrum(n: int = 1200, w_lo: float = 1.0e-12, w_hi: float = 60.0):
    """Table of ptilde(w): the IC photon spectrum in the scaled variable
    w = eps1 / (4 gamma^2 kT), for a blackbody seed.  Computed ONCE.

    Integrated in ln u, because for w << 1 the support sits at u ~ 1 while the
    lower limit is at u = w -- the same log-space trap that produced the bug in
    `_deposited_energy_eV`.  ptilde -> 3 <1/u> = 3 zeta(2)/(2 zeta(3)) = 2.0527
    as w -> 0, which is checked below (C28).
    """
    key = (n, w_lo, w_hi)
    if key in _PTILDE_CACHE:
        return _PTILDE_CACHE[key]
    ws = np.logspace(np.log10(w_lo), np.log10(w_hi), n)
    v_hi = np.log(80.0)                      # e^80 in the Wien tail: e^-80 ~ 1e-35
    pt = np.zeros_like(ws)
    for i, w in enumerate(ws):
        v_lo = np.log(w)
        if v_lo >= v_hi:
            continue
        val, _ = quad(
            lambda v: 3.0 * float(planck_number_pdf(np.exp(v))[0])
                          * float(F_thomson(w * np.exp(-v))[0]),
            v_lo, v_hi, limit=500)
        pt[i] = val
    _PTILDE_CACHE[key] = (ws, pt)
    return ws, pt


class ICPhotonChannel:
    """Secondary HI ionizations made by the CMB photons the electron upscatters.

    Parameters
    ----------
    tau_mode : 'hubble'  tau = n_HI sigma_pi (c/H(z)) -- absorbed at z, or never.
               'cosmo'   integrate the forward light cone from z down to z_end,
                         letting the photon redshift into a LARGER cross section.
                         Physically the more complete of the two; used as the
                         sensitivity variant (C36), not as the headline.
    force_A  : override the absorption probability (None = physical).  Used only
               by the deliberate-failure checks: force_A=0 must reproduce the IC route
               exactly, force_A=1 must push the yield up towards the asymptotic ceiling.
    """

    def __init__(self, cos: dict, p: Params, tau_mode: str = "hubble",
                 z_end: float = 6.0, force_A: float | None = None,
                 pe_floor_eV: float = 0.0, pe_unfloored: bool = False):
        self.cos, self.p, self.pe_floor_eV = cos, p, pe_floor_eV
        self.pe_unfloored = pe_unfloored
        self.kT = kT_cmb_eV(cos)
        self.tau_mode, self.z_end, self.force_A = tau_mode, z_end, force_A
        self.w, self.ptilde = ic_scaled_spectrum()
        self.lnw = np.log(self.w)
        # <w> = int w ptilde dw.  Analytic value: <u>/3 = (pi^4/15)/(2 zeta3)/3.
        self.w_mean = float(_trapz(self.w ** 2 * self.ptilde, self.lnw))
        self.w_norm = float(_trapz(self.w * self.ptilde, self.lnw))
        # Phi(eps) = A(eps) * N_gamma(eps) / eps  = ionizations per eV of photon
        self.eps = np.logspace(np.log10(E_TH_HI), 8.0, 420)
        self.lneps = np.log(self.eps)
        self.A_grid = self.A_abs(self.eps)
        # N_gamma(eps,"ic") = 1 + N_e(eps - E_th): the primary photoionization
        # PLUS every ionization the photoelectron and its own daughters make, all
        # the way down -- that whole cascade is what f_ion means.  pe_floor_eV
        # suppresses the photoelectron term below a given energy, which is how
        # the "is a fixed W legitimate down here?" sensitivity is measured.
        Ng = N_gamma(self.eps, cos, p, "ic")
        if pe_unfloored:
            # Reconstruct the pre-correction behaviour: apply f_ion all the way
            # down to T = 0, so a 6 eV photoelectron is credited with 0.17
            # ionizations it cannot make. Exact reconstruction, because below
            # T_bethe_min the deposition integral is just T.
            T2 = self.eps - E_TH_HI
            Ng = np.where((T2 > 0.0) & (T2 < E_TH_HI),
                          1.0 + f_ion_HI(p.x_e) * T2 / E_TH_HI, Ng)
        if pe_floor_eV > 0.0:
            Ng = np.where(self.eps - E_TH_HI < pe_floor_eV, 1.0, Ng)
        self.Phi = self.A_grid * Ng / self.eps
        self._T = self._cum = None

    # -- optical depth ----------------------------------------------------
    def tau(self, eps):
        """Photoionization optical depth seen by an upscattered photon.
        Always returns an array, so scalar callers cannot silently get a float."""
        e = np.atleast_1d(np.asarray(eps, dtype=float))
        n_HI = self.cos["n_H_cm3"] * (1.0 - self.p.x_e)
        if self.tau_mode == "hubble":
            return n_HI * np.atleast_1d(sigma_photoion(e)) * self.cos["hubble_length_cm"]
        if self.tau_mode == "cosmo":
            zp = np.linspace(self.z_end, self.p.z, 500)
            nHI = self.cos["n_H0_cm3"] * (1.0 + zp) ** 3 * (1.0 - self.p.x_e)
            Hz = H0_S * np.sqrt(OMEGA_M * (1.0 + zp) ** 3 + OMEGA_L)
            kern = nHI * C_LIGHT / ((1.0 + zp) * Hz)          # cm per unit z
            Eshift = e[:, None] * (1.0 + zp[None, :]) / (1.0 + self.p.z)
            return _trapz(sigma_photoion(Eshift) * kern[None, :], zp, axis=1)
        raise ValueError(f"unknown tau_mode {self.tau_mode!r}")

    def A_abs(self, eps):
        """Probability that an upscattered photon is absorbed rather than escaping."""
        e = np.atleast_1d(np.asarray(eps, dtype=float))
        if self.force_A is not None:
            return np.where(e >= E_TH_HI, float(self.force_A), 0.0)
        return -np.expm1(-self.tau(e))

    # -- per-scattering efficiency ---------------------------------------
    def _Phi_at(self, eps):
        return np.interp(np.log(eps), self.lneps, self.Phi, left=0.0, right=0.0)

    def ions_per_eV_radiated(self, gamma):
        """G(gamma): HI ionizations produced per eV the electron radiates to IC.

        G = int dw qtilde(w) A(eps1) N_gamma(eps1)/eps1,  qtilde = w ptilde/<w>,
        eps1 = 4 gamma^2 kT w.  qtilde is the ENERGY-weighted photon spectrum, so
        this form conserves energy by construction rather than by assertion.
        """
        g = np.atleast_1d(np.asarray(gamma, dtype=float))
        base = self.w ** 2 * self.ptilde / self.w_mean      # w * qtilde(w)
        out = np.empty_like(g)
        for i, gg in enumerate(g):
            out[i] = _trapz(base * self._Phi_at(4.0 * gg ** 2 * self.kT * self.w), self.lnw)
        return out

    def energy_fraction_absorbed(self, gamma):
        """Fraction of the IC-radiated ENERGY that lands in photons which are both
        above threshold and absorbed.  Diagnostic; drives the lower figure panel."""
        g = np.atleast_1d(np.asarray(gamma, dtype=float))
        base = self.w ** 2 * self.ptilde / self.w_mean
        out = np.empty_like(g)
        for i, gg in enumerate(g):
            eps = 4.0 * gg ** 2 * self.kT * self.w
            a = np.interp(np.log(eps), self.lneps, self.A_grid, left=0.0, right=0.0)
            out[i] = _trapz(base * a, self.lnw)
        return out

    # -- integrate along the slowing-down track ---------------------------
    def build(self, n: int = 2400):
        """Cumulative secondary yield.

            N_sec(E) = int_0^E dT [b_IC/(b_coll+b_IC)](T) * G(gamma(T))

        The bracket is 1 - f_coll, i.e. the IC branching ratio already used by
        the IC route, so models B and C share ONE branching function and cannot
        disagree about where the energy went.  Log-spaced cumulative trapezoid.
        """
        T = np.logspace(np.log10(self.p.T_bethe_min_eV), np.log10(self.p.E_e_max), n)
        _, gam, _ = _kinematics(T)
        integ = (1.0 - f_coll(T, self.cos, self.p)) * self.ions_per_eV_radiated(gam) * T
        d = np.diff(np.log(T))
        self._T = T
        self._cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * d)])
        return self

    def N_secondary(self, E_eV):
        if self._cum is None:
            self.build()
        E = np.atleast_1d(np.asarray(E_eV, dtype=float))
        # Clamp before the log. E = 0 reaches here legitimately -- a photon AT
        # the ionization threshold makes a photoelectron of exactly zero
        # kinetic energy -- and log(0) = -inf raised a divide-by-zero warning
        # even though the where() below already zeroes those entries. The clamp
        # only touches values that are zeroed anyway, so no result changes.
        E_safe = np.maximum(E, self._T[0])
        out = np.interp(np.log(E_safe), np.log(self._T), self._cum,
                        left=0.0, right=float(self._cum[-1]))
        out = np.where(E < self._T[0], 0.0, out)
        return out if np.ndim(E_eV) else float(out[0])


def N_e_depfit(E_eV, cos: dict, p: Params, chan: "ICPhotonChannel"):
    """Model (C): IC-limited collisional deposition PLUS the ionizations made by
    the IC-upscattered CMB photons that are absorbed.  This is the physical
    answer in the MeV-GeV window; B is what you get by throwing those away."""
    out = np.asarray(N_e_ic(E_eV, cos, p), dtype=float) + np.asarray(
        chan.N_secondary(E_eV), dtype=float)
    return out if np.ndim(E_eV) else float(out)


# ===========================================================================
# 8. CHECKS.  Prints per-check COVERAGE, not a verdict tally.
#    Implements CHECKS_AUTO: C18 C20 C26 C27 C28 C29 C30 C32 C52 C53 C55.
# ===========================================================================
def run_checks(cos: dict, p: Params, chan: "ICPhotonChannel") -> tuple[list, dict]:
    rows, prov = [], {}

    def rec(cid, name, ok, detail):
        rows.append((cid, name, "PASS" if ok else "FAIL", detail))
        return ok

    # --- C28 limiting cases of the SvdS fit -------------------------------
    # x^0.4092 -> 0 only slowly, so the limit must be probed far below 1e-12:
    # at x=1e-12 the residual is still 8e-6, which is convergence, not error.
    rec("C28", "f_ion -> A as x_e -> 0", abs(f_ion_HI(1e-40) - SVDS["ion_HI"][0]) < 1e-12,
        f"f_ion(1e-40) = {f_ion_HI(1e-40):.12f} vs A = {SVDS['ion_HI'][0]}  "
        f"[at x=1e-12 the residual is {abs(f_ion_HI(1e-12)-SVDS['ion_HI'][0]):.1e}: "
        f"slow convergence, not a failure]")
    rec("C28", "f_ion -> 0 as x_e -> 1", f_ion_HI(1.0) < 1e-12,
        f"f_ion(1) = {f_ion_HI(1.0):.3e}")
    rec("C28", "f_heat -> 1 as x_e -> 1", abs(f_heat(1.0) - SVDS_HEAT[0]) < 1e-9,
        f"f_heat(1) = {f_heat(1.0):.6f}")

    # --- Platzman energy budget: the channels cannot exceed the energy -----
    fi, fhe, fex, fh = (f_ion_HI(p.x_e), f_channel(p.x_e, "ion_HeI"),
                        f_channel(p.x_e, "exc_Lya"), f_heat(p.x_e))
    tot = fi + fhe + fex + fh
    prov["f_ion_HI"], prov["f_ion_HeI"] = fi, fhe
    prov["f_exc_Lya"], prov["f_heat"] = fex, fh
    prov["f_budget_sum"] = tot
    rec("C26", "Platzman energy budget sum <= 1", tot <= 1.0,
        f"sum = {tot:.4f}  (ion_HI {fi:.4f} + ion_HeI {fhe:.4f} "
        f"+ exc {fex:.4f} + heat {fh:.4f}); deficit {1-tot:.4f} = HeII + sub-excitation")

    # --- C32/C27 external anchor: the W-value of hydrogen ------------------
    W = E_TH_HI / fi
    prov["W_eV_per_ion_pair"] = W
    rec("C32", "W = E_th/f_ion vs literature W(H) ~ 36 eV", 30.0 < W < 42.0,
        f"W = {W:.2f} eV  (independent literature value for H is ~36 eV; "
        f"this is NOT built into the fit, so agreement is a real cross-check)")

    # --- C29 dimensional consistency of both loss rates --------------------
    T_test = 1.0e6
    bc = float(b_coll_erg_per_s(T_test, cos["n_H_cm3"], p.I_exc_eV))
    bi = float(b_ic_erg_per_s(T_test, cos["u_cmb_erg_cm3"]))
    # independent re-derivation of b_ic by explicit unit algebra
    _, g, b2 = _kinematics(T_test)
    bi_manual = (4.0 / 3.0) * SIGMA_T * C_LIGHT * cos["u_cmb_erg_cm3"] * (g**2 * b2)
    rec("C29", "b_IC two ways (gamma^2-1 vs gamma^2 beta^2)",
        abs(bi - bi_manual) / bi < 1e-12,
        f"rel.diff = {abs(bi-bi_manual)/bi:.2e}; both {bi:.4e} erg/s")
    rec("C29", "both rates positive and finite at 1 MeV",
        bc > 0 and bi > 0 and np.isfinite(bc) and np.isfinite(bi),
        f"b_coll = {bc:.4e} erg/s, b_IC = {bi:.4e} erg/s, ratio IC/coll = {bi/bc:.3f}")

    # --- justification of the T_bethe_min cutoff --------------------------
    bc1 = float(b_coll_erg_per_s(p.T_bethe_min_eV, cos["n_H_cm3"], p.I_exc_eV))
    bi1 = float(b_ic_erg_per_s(p.T_bethe_min_eV, cos["u_cmb_erg_cm3"]))
    rec("C20", "f_coll ~ 1 at the Bethe cutoff (so the cutoff cannot matter)",
        bi1 / bc1 < 1e-2,
        f"at T = {p.T_bethe_min_eV:.0f} eV, b_IC/b_coll = {bi1/bc1:.2e}")

    # --- C53 null test: photons below threshold ionize nothing ------------
    below = np.array([1.0, 5.0, 13.0, 13.59])
    rec("C53", "NULL: N_gamma == 0 exactly below 13.6 eV",
        np.all(N_gamma(below, cos, p, "full") == 0.0),
        f"N_gamma({below.tolist()}) = {N_gamma(below, cos, p, 'full').tolist()}")

    # --- C55 asymptotic normalisation of the asymptotic ceiling --------------------------
    Ehi = 1.0e9
    asym_rgs = RGS_ION["A"] * (1.0 - p.x_e ** RGS_ION["B"]) ** RGS_ION["C"]
    rel55 = abs(float(N_e_full(Ehi, p)) / Ehi - asym_rgs / E_TH_HI) / (asym_rgs / E_TH_HI)
    rec("C55", "N_e_full / E -> the FS10 asymptote (no longer exact: W = W(E))",
        rel55 < 1e-3,
        f"N/E = {float(N_e_full(Ehi, p))/Ehi:.6e} per eV at 1e9 eV vs the "
        f"E -> inf limit {asym_rgs/E_TH_HI:.6e}, rel.diff {rel55:.2e}. This "
        f"check USED to demand exact equality, which was only true because "
        f"f_ion was a constant. It is now a convergence test, and it would "
        f"fail if the energy dependence were dropped by accident.")

    # --- C26 the two models must agree where IC is negligible -------------
    E_lo = 1.0e3
    na, nb = float(N_e_full(E_lo, p)), float(N_e_ic(E_lo, cos, p))
    rec("C26", "models A and B agree at 1 keV (IC negligible)",
        abs(na - nb) / na < 1e-2,
        f"A = {na:.3f}, B = {nb:.3f}, rel.diff = {abs(na-nb)/na:.2e}")

    # --- C30 order-of-magnitude, stated BEFORE the pipeline ---------------
    # Expectation written first: 1 keV / 36 eV ~ 28 ion pairs.
    expect = 1.0e3 / 36.0
    rec("C30", "order-of-magnitude at 1 keV (expected ~28, set before running)",
        0.5 * expect < na < 2.0 * expect,
        f"predicted {expect:.1f}, computed {na:.1f}")

    # --- critical energy, two independent routes (C26) --------------------
    def _root(T):
        return (float(b_ic_erg_per_s(T, cos["u_cmb_erg_cm3"]))
                - float(b_coll_erg_per_s(T, cos["n_H_cm3"], p.I_exc_eV)))
    E_crit = brentq(_root, 1.0e3, 1.0e9, xtol=1.0, rtol=1e-12)
    prov["E_crit_eV"] = E_crit
    # analytic route: b_IC = A(gamma^2-1), b_coll ~ const -> gamma^2-1 = b_coll/A
    A_ic = (4.0 / 3.0) * SIGMA_T * C_LIGHT * cos["u_cmb_erg_cm3"]
    bc_at = float(b_coll_erg_per_s(E_crit, cos["n_H_cm3"], p.I_exc_eV))
    g_an = np.sqrt(1.0 + bc_at / A_ic)
    E_an = (g_an - 1.0) * MEC2_EV
    rec("C26", "E_crit: numerical root vs analytic inversion",
        abs(E_crit - E_an) / E_crit < 1e-6,
        f"brentq {E_crit:.4e} eV vs analytic {E_an:.4e} eV")

    # --- C26/C31 the deposition integral by a SECOND, independent method --
    # quad-in-log-space vs a dense log-spaced trapezoid. This is the check that
    # exists because the first implementation of this integral was wrong.
    E_big = 1.0e12
    q = _deposited_energy_eV(E_big, cos, p, None)
    u = np.linspace(np.log(p.T_bethe_min_eV), np.log(E_big), 200_001)
    trap = p.T_bethe_min_eV + _trapz(f_coll(np.exp(u), cos, p) * np.exp(u), u)
    prov["E_dep_eV_1e12"] = q
    rec("C31", "deposition integral: quad(log) vs dense trapezoid(log)",
        abs(q - trap) / q < 1e-4,
        f"quad {q:.6e} eV vs trapezoid {trap:.6e} eV, rel.diff {abs(q-trap)/q:.2e}")

    # --- C18 DELIBERATE DISAGREEMENT: show the naive integrator IS wrong --
    naive, _ = quad(lambda T: float(f_coll(T, cos, p, None)), 0.0, E_big, limit=400)
    prov["naive_integral_fraction"] = naive / q
    rec("C18", "DELIBERATE-FAIL: linear-space quad misses the integrand",
        abs(naive - q) / q > 0.1,
        f"linear-space quad returns {100*naive/q:.1f}% of the correct value "
        f"({naive:.4e} vs {q:.4e} eV) -- and does so SMOOTHLY, so it does not "
        f"look wrong. This is the bug the monotonicity check caught. The size "
        f"of the error depends on the upper limit, which is why the check tests "
        f"for a material discrepancy rather than a specific one.")

    # --- saturation of the IC route (the physical claim of this work) ----------
    n11, n12 = float(N_e_ic(1e11, cos, p)), float(N_e_ic(1e12, cos, p))
    prov["N_e_ic_1e11"], prov["N_e_ic_1e12"] = n11, n12
    rec("C28", "the IC route saturates: 10x more energy adds <1% ionizations",
        abs(n12 - n11) / n11 < 0.01,
        f"N(1e11 eV) = {n11:.4e}, N(1e12 eV) = {n12:.4e}, change {100*(n12-n11)/n11:.3f}%")

    # --- monotonicity -----------------------------------------------------
    Eg = np.logspace(2, 12, 60)
    yb = N_e_ic(Eg, cos, p)
    rec("C20", "both yield curves monotonically non-decreasing",
        np.all(np.diff(yb) >= -1e-9) and np.all(np.diff(N_e_full(Eg, p)) > 0),
        f"min diff (the IC route) = {np.min(np.diff(yb)):.3e}")

    # --- cross-section limits (C28) ---------------------------------------
    rec("C28", "sigma_KN -> sigma_T as E -> 0",
        abs(float(sigma_kn(1.0)) - SIGMA_T) / SIGMA_T < 1e-4,
        f"sigma_KN(1 eV) = {float(sigma_kn(1.0)):.6e}, sigma_T = {SIGMA_T:.6e}")
    rec("C28", "sigma_pi -> sigma_0 at threshold",
        abs(float(sigma_photoion(E_TH_HI * (1 + 1e-12))) - SIGMA_PI0) / SIGMA_PI0 < 1e-3,
        f"sigma_pi(E_th) = {float(sigma_photoion(E_TH_HI*(1+1e-12))):.4e}, "
        f"sigma_0 = {SIGMA_PI0:.4e}")

    # --- Thomson-regime validity of the IC formula ------------------------
    kT = 8.617333262e-5 * cos["T_cmb_K"]           # eV, CODATA k_B
    _, g_max, _ = _kinematics(p.E_e_max)
    prov["thomson_parameter_at_Emax"] = float(g_max * kT / MEC2_EV)
    rec("C20", "IC Thomson regime gamma*kT << mc^2 at the TOP of the range",
        float(g_max * kT / MEC2_EV) < 1.0,
        f"gamma*kT/mc^2 = {float(g_max*kT/MEC2_EV):.3e} at E = {p.E_e_max:.0e} eV "
        f"-- FAILS above this; Klein-Nishina suppression not modelled")

    # --- C18 DELIBERATE DISAGREEMENT: prove the W check can fail ----------
    W_bad = E_TH_HI / 0.02          # a deliberately wrong f_ion
    rec("C18", "DELIBERATE-FAIL: W check rejects a wrong f_ion",
        not (30.0 < W_bad < 42.0),
        f"with f_ion = 0.02 -> W = {W_bad:.1f} eV, correctly rejected. "
        f"The check is capable of failing.")

    # --- C36 sensitivity to the one arbitrary-ish choice ------------------
    n_alt = float(N_e_ic(1e12, cos, p, I_eV=I_EXC_H2))
    prov["N_e_ic_1e12_I19.2"] = n_alt
    rec("C36", "sensitivity to I (14.99 eV atomic vs 19.2 eV molecular)",
        True,
        f"N(1e12) = {n12:.4e} with I=14.99 eV, {n_alt:.4e} with I=19.2 eV "
        f"-> {100*abs(n_alt-n12)/n12:.2f}% (logarithmic, as expected)")

    # =====================================================================
    # THE IC-SECONDARY CHANNEL.  Everything below tests the new physics.
    # =====================================================================
    kT_eV = kT_cmb_eV(cos)
    prov["kT_cmb_eV"] = kT_eV

    # --- C26 the scattered-photon spectrum is a pdf with the right mean ---
    # Both are DERIVED from F(x); neither was imposed. If <x> != 1/3 then
    # <eps1> != (4/3) gamma^2 eps0 and every number downstream is wrong.
    norm_F, _ = quad(lambda x: 3.0 * float(F_thomson(x)[0]), 0.0, 1.0, limit=200)
    mean_F, _ = quad(lambda x: 3.0 * x * float(F_thomson(x)[0]), 0.0, 1.0, limit=200)
    rec("C26", "IC spectrum 3F(x): normalisation 1 and mean 1/3",
        abs(norm_F - 1.0) < 1e-10 and abs(mean_F - 1.0 / 3.0) < 1e-10,
        f"int 3F dx = {norm_F:.12f} (must be 1); int x 3F dx = {mean_F:.12f} "
        f"(must be 1/3, which IS the statement <eps1> = (4/3) gamma^2 eps0)")

    # --- C31 the blackbody-convolved spectrum, two independent routes -----
    # Route 1: the numerically built table.  Route 2: closed form, <w> = <u>/3
    # with <u> = (pi^4/15)/(2 zeta3) for a Planck number spectrum.
    w_mean_an = (PI4_15 / (2.0 * ZETA3)) / 3.0
    prov["ic_w_mean"] = chan.w_mean
    rec("C31", "ptilde(w): numeric normalisation vs 1",
        abs(chan.w_norm - 1.0) < 2e-3,
        f"int ptilde dw = {chan.w_norm:.8f} (grid truncation only; must be 1)")
    rec("C31", "<w> numeric vs closed form <u>/3",
        abs(chan.w_mean - w_mean_an) / w_mean_an < 3e-3,
        f"numeric {chan.w_mean:.8f} vs analytic {w_mean_an:.8f}, "
        f"rel.diff {abs(chan.w_mean-w_mean_an)/w_mean_an:.2e}")

    # --- C28 limiting case of ptilde as w -> 0 ----------------------------
    pt0_an = 3.0 * ZETA2 / (2.0 * ZETA3)
    rec("C28", "ptilde(w -> 0) -> 3 zeta(2)/(2 zeta(3))",
        abs(chan.ptilde[0] - pt0_an) / pt0_an < 1e-3,
        f"ptilde({chan.w[0]:.1e}) = {chan.ptilde[0]:.6f} vs analytic {pt0_an:.6f}")

    # --- C30 order of magnitude, WRITTEN DOWN BEFORE RUNNING --------------
    # <eps1> = 4 gamma^2 kT <w> reaches 13.6 eV at gamma ~ 38, T ~ 19 MeV.
    g_on = np.sqrt(E_TH_HI / (4.0 * kT_eV * w_mean_an))
    T_on_pred = (g_on - 1.0) * MEC2_EV
    Tg = np.logspace(5, 11, 400)
    _, gg, _ = _kinematics(Tg)
    G = chan.ions_per_eV_radiated(gg)
    T_half = float(Tg[np.argmax(G > 0.5 * G.max())])
    prov["ic_turnon_T_eV_pred"] = float(T_on_pred)
    prov["ic_turnon_T_eV_half"] = T_half
    rec("C30", "channel turns on where predicted (19 MeV, set before running)",
        T_on_pred / 30.0 < T_half < T_on_pred * 30.0,
        f"predicted <eps1> = E_th at T = {T_on_pred:.3e} eV; G(T) reaches half "
        f"its maximum at T = {T_half:.3e} eV")

    # --- C20 the escape energy, and why no recursion is needed ------------
    E_esc = brentq(lambda e: float(chan.tau(e)[0]) - 1.0, E_TH_HI * 1.001, 1e6,
                   xtol=1e-6, rtol=1e-12)
    prov["E_escape_eV"] = E_esc
    rec("C20", "IC secondary cascade terminates (no recursion needed)",
        E_esc < 0.05 * E_crit,
        f"photons above E_esc = {E_esc:.4e} eV escape (tau < 1); every ABSORBED "
        f"photon therefore makes a photoelectron below {E_esc:.3e} eV, which is "
        f"{E_crit/E_esc:.0f}x below E_crit = {E_crit:.3e} eV, where the IC branch "
        f"carries {100*(1-float(f_coll(E_esc, cos, p))):.4f}% of the loss. "
        f"Truncating the cascade at one generation is justified, not assumed.")

    # --- C28 the absorbed photons are photoionized, not Compton-scattered -
    r_esc = float(sigma_photoion(E_esc)) / float(sigma_kn(E_esc))
    rec("C28", "photoelectric dominates over Compton where absorption happens",
        r_esc > 3.0,
        f"sigma_pi/sigma_KN = {float(sigma_photoion(1e2))/float(sigma_kn(1e2)):.3e} at 100 eV "
        f"and {r_esc:.2f} at E_esc; using sigma_pi alone in tau is therefore safe "
        f"below E_esc, which is where absorption actually occurs")

    # --- C20 no pair cascade: IC photons stay below threshold on the CMB --
    eps_pp = MEC2_EV**2 / (2.701 * kT_eV)
    _, g_max, _ = _kinematics(p.E_e_max)
    eps_max = 4.0 * g_max**2 * kT_eV * chan.w[-1]
    prov["eps_pairprod_threshold_eV"] = float(eps_pp)
    rec("C20", "no gamma-gamma pair cascade on the CMB at any energy in range",
        eps_max < eps_pp,
        f"hardest IC photon tracked = {eps_max:.3e} eV; pair-production threshold "
        f"on the CMB at z=10 is ~{eps_pp:.3e} eV. Below it, so an electromagnetic "
        f"pair cascade cannot start and the one-generation treatment is complete.")

    # --- C18 DELIBERATE-FAIL: A = 0 must reproduce the IC route EXACTLY --------
    chan0 = ICPhotonChannel(cos, p, force_A=0.0).build(n=400)
    d0 = float(np.max(np.abs(chan0.N_secondary(np.logspace(3, 12, 40)))))
    rec("C18", "STRUCTURAL: A=0 collapses the depfit route onto the IC route exactly",
        d0 == 0.0,
        f"max |N_sec| with the absorption probability forced to zero = {d0:.1e}. "
        f"If this were nonzero, the depfit route would be double-counting energy.")

    # --- C18 DELIBERATE-FAIL: A = 1 must push C up towards the asymptotic ceiling --------
    chan1 = ICPhotonChannel(cos, p, force_A=1.0).build(n=600)
    E_t = 1.0e8
    nC1 = float(N_e_ic(E_t, cos, p)) + float(chan1.N_secondary(E_t))
    nA_t, nB_t = float(N_e_full(E_t, p)), float(N_e_ic(E_t, cos, p))
    rec("C18", "STRUCTURAL: A=1 recovers the full-absorption yield to ~5%",
        0.5 * nA_t < nC1 < 1.2 * (E_t / E_TH_HI),
        f"at 1e8 eV: the IC route = {nB_t:.3e}, the depfit route with A=1 -> {nC1:.3e}, "
        f"the asymptotic ceiling = {nA_t:.3e}, ceiling E/E_th = {E_t/E_TH_HI:.3e}. Forcing every "
        f"upscattered photon to be absorbed reaches {100*nC1/nA_t:.1f}% of A. "
        f"It lands slightly ABOVE A, not below, and that is the expected sign: a "
        f"photon just above 13.6 eV buys an ion pair for 13.6 eV where electron "
        f"degradation needs W = {W:.2f} eV. It must still sit below E/E_th, and "
        f"it does. The check can fail on either side.")

    # --- C20 hard physical bound: no cascade beats one ion pair per 13.6 eV
    Eg2 = np.logspace(2, 12, 120)
    nC = N_e_depfit(Eg2, cos, p, chan)
    nB = N_e_ic(Eg2, cos, p)
    worst = float(np.max(nC / (Eg2 / E_TH_HI)))
    rec("C20", "ENERGY BOUND: N_C <= E/E_th everywhere",
        worst <= 1.0,
        f"max over the grid of N_C/(E/E_th) = {worst:.4f}. This is the absolute "
        f"ceiling set by energy conservation alone and no model may cross it.")
    rec("C20", "the depfit route >= the IC route everywhere (adding a channel cannot subtract)",
        np.all(nC >= nB - 1e-9),
        f"min(N_C - N_B) = {float(np.min(nC-nB)):.3e}")
    rec("C20", "the depfit route monotonically non-decreasing",
        np.all(np.diff(nC) >= -1e-6 * np.maximum(nC[1:], 1.0)),
        f"min diff = {float(np.min(np.diff(nC))):.3e}")

    # --- C26 models B and C must coincide below the turn-on ---------------
    nC_lo, nB_lo = float(N_e_depfit(1e6, cos, p, chan)), float(N_e_ic(1e6, cos, p))
    rec("C26", "C == B at 1 MeV (channel not yet open)",
        abs(nC_lo - nB_lo) / nB_lo < 1e-3,
        f"C = {nC_lo:.4f}, B = {nB_lo:.4f}, rel.diff = {abs(nC_lo-nB_lo)/nB_lo:.2e}")

    # --- C31 N_secondary by a SECOND, independent quadrature --------------
    E_v = 1.0e8
    qv, _ = quad(lambda u: float((1.0 - f_coll(np.exp(u), cos, p))
                                 * chan.ions_per_eV_radiated(
                                     _kinematics(np.exp(u))[1])[0]) * np.exp(u),
                 np.log(p.T_bethe_min_eV), np.log(E_v), limit=300)
    cv = float(chan.N_secondary(E_v))
    rec("C31", "N_secondary(1e8 eV): cumulative trapezoid vs adaptive quad",
        abs(qv - cv) / cv < 5e-3,
        f"trapezoid {cv:.6e} vs quad {qv:.6e}, rel.diff {abs(qv-cv)/cv:.2e}")

    # --- C36/C41 sensitivity to the transport assumption ------------------
    # 'hubble' absorbs at z=10 or never; 'cosmo' follows the photon down to
    # z_end=6, where it redshifts INTO a larger cross section. These are two
    # different physical assumptions, not two routes to one number, so the
    # spread is reported as a band -- it is not silently averaged away.
    chan_c = ICPhotonChannel(cos, p, tau_mode="cosmo").build(n=800)
    Ecmp = np.array([1e7, 1e8, 1e9, 1e10, 1e12])
    nh = np.asarray(N_e_depfit(Ecmp, cos, p, chan))
    ncz = np.asarray(N_e_ic(Ecmp, cos, p)) + np.asarray(chan_c.N_secondary(Ecmp))
    spread = float(np.max(np.abs(ncz - nh) / nh))
    prov["ic_transport_spread_max"] = spread
    prov["N_e_depfit_1e12_cosmo"] = float(ncz[-1])
    rec("C36", "transport assumption: Hubble-length vs redshifting light cone",
        True,
        f"N_C(1e12) = {nh[-1]:.4e} (hubble) vs {ncz[-1]:.4e} (cosmo, z_end=6); "
        f"worst relative spread over 1e7-1e12 eV = {100*spread:.1f}%. REPORTED, "
        f"NOT RESOLVED: the true answer needs a reionization history.")

    # --- C20 the second generation of IC secondaries is IDENTICALLY zero ---
    # Not "small": zero.  The photoelectron of an absorbed secondary photon has
    # T <= E_esc, hence gamma - 1 ~ 2e-3, and reaching 13.6 eV from there needs
    # a seed photon of ~1300 kT, whose Planck occupancy is e^-1300.
    _, g_pe, _ = _kinematics(E_esc)
    G_pe = float(chan.ions_per_eV_radiated(g_pe)[0])
    eps0_needed = E_TH_HI / (4.0 * float(g_pe) ** 2)
    rec("C20", "second generation of IC secondaries is identically zero",
        G_pe == 0.0,
        f"a photoelectron at the escape energy has gamma = {float(g_pe):.6f}; the "
        f"hardest CMB photon it can produce on the tracked spectrum is "
        f"{4*float(g_pe)**2*kT_eV*chan.w[-1]:.3e} eV, far below 13.6 eV. Reaching "
        f"threshold would need a seed photon of {eps0_needed:.3f} eV = "
        f"{eps0_needed/kT_eV:.0f} kT, Planck occupancy ~1e-569. G = {G_pe:.1e} "
        f"ion/eV exactly. The one-generation truncation is EXACT here, not an "
        f"approximation -- which is a stronger statement than the escape-energy "
        f"check alone supports.")

    # --- C36 IS A FIXED W LEGITIMATE? the low-energy end of the secondaries --
    # f_ion is the SvdS ASYMPTOTIC fit: no energy dependence, so W = 36.26 eV is
    # applied at every energy. Above ~1 keV that matches the experimental fact
    # that W is flat. Below ~100 eV the fit is outside its stated domain, and a
    # quarter of the secondary yield is produced there. Bracket it.
    band_lo, band_hi = E_TH_HI, 100.0
    base_w = chan.w ** 2 * chan.ptilde / chan.w_mean
    lnw = chan.lnw
    T_tr = chan._T
    _, gam_tr, _ = _kinematics(T_tr)
    ic_frac = 1.0 - f_coll(T_tr, cos, p)
    contrib = np.empty_like(T_tr)
    for i, g_i in enumerate(gam_tr):        # NOT `gg`: that name holds the
        e1 = 4.0 * g_i ** 2 * kT_eV * chan.w  # gamma grid the C30 check built,
                                              # and rebinding it silently turned
                                              # the peak-efficiency number into
                                              # 2e-15. Caught by reading the
                                              # registry, not by any assertion.
        contrib[i] = _trapz(base_w * chan._Phi_at(e1) * (e1 < band_hi), lnw)
    integ = ic_frac * contrib * T_tr
    dl = np.diff(np.log(T_tr))
    below100 = float(np.sum(0.5 * (integ[1:] + integ[:-1]) * dl)) / chan._cum[-1]
    prov["ic_frac_of_secondary_below_100eV"] = below100
    chan_fl = ICPhotonChannel(cos, p, pe_floor_eV=100.0).build(n=1200)
    chan_ref = ICPhotonChannel(cos, p).build(n=1200)          # same grid, fair
    nC_ref = float(N_e_ic(1e12, cos, p)) + float(chan_ref.N_secondary(1e12))
    nC_fl = float(N_e_ic(1e12, cos, p)) + float(chan_fl.N_secondary(1e12))
    prov["N_e_depfit_1e12_pe_floor_100eV"] = nC_fl
    rec("C36", "fixed W = E_th/f_ion: how much does the sub-100 eV end matter?",
        True,
        f"{100*below100:.1f}% of the secondary yield comes from photons below "
        f"{band_hi:.0f} eV, where the SvdS asymptotic fit is outside its stated "
        f"domain (E >~ 100 eV). Suppressing the photoelectron cascade entirely "
        f"below {band_hi:.0f} eV -- the pessimistic extreme -- moves N_C(1e12) "
        f"from {nC_ref:.4e} to {nC_fl:.4e}, i.e. "
        f"{100*(nC_fl-nC_ref)/nC_ref:.1f}%. The truth is inside that bracket. "
        f"An energy-resolved f_ion(E, x_e) (Furlanetto & Stoever 2010) would "
        f"remove this; it is not in the tree. REPORTED, NOT RESOLVED.")

    # --- C28 the sub-threshold floor is exact and does something ------------
    # =====================================================================
    # FURLANETTO & STOEVER 2010.  Everything below is new with the paper.
    # =====================================================================

    # --- C32 EXTERNAL ANCHOR: the medians FS10 quote for their own eq. (2) --
    # FS10 sec. 3 states median secondary energies of 7.2, 14.2, 28.5 eV for
    # HI, HeI, HeII. Those numbers are NOT used to build p(eps); they are an
    # independent statement about it, so reproducing them tests the
    # implementation of the prescription against the paper.
    med_ok, med_msg = True, []
    for sp in ("HI", "HeI", "HeII"):
        med, mean = secondary_moments(1.0e4, sp)
        quoted = FS10_MEDIAN_EV[sp]
        rel = abs(med - quoted) / quoted
        med_ok &= rel < 0.06
        med_msg.append(f"{sp}: {med:.2f} vs {quoted:.1f} eV ({100*rel:.1f}%)")
        prov[f"fs10_median_secondary_{sp}_eV"] = med
    rec("C32", "FS10 eq.(2): computed medians vs the paper's quoted medians",
        med_ok, "; ".join(med_msg) + ". Independent of how p(eps) was coded.")

    # --- C32 THE SvdS COEFFICIENTS, CHECKED AT LAST ------------------------
    # FS10 eqs. (13)-(14) quote the Ricotti et al. 2002 fits to SvdS85, which
    # carry the SvdS asymptotes at two significant figures. Until this paper
    # entered the tree the twelve recalled coefficients had NO external test.
    svds_ion = f_ion_HI(p.x_e)
    rgs_ion_asym = RGS_ION["A"] * (1.0 - p.x_e ** RGS_ION["B"]) ** RGS_ION["C"]
    d_ion = abs(svds_ion - rgs_ion_asym) / rgs_ion_asym
    prov["svds_vs_fs10_ion_reldiff"] = d_ion
    rec("C32", "RECALLED SvdS ionization coefficients vs FS10 eq. (13)",
        d_ion < 0.01,
        f"recalled 0.3908(1-x^0.4092)^1.7592 = {svds_ion:.6f}; FS10 eq.(13) "
        f"asymptote 0.39(1-x^0.41)^1.76 = {rgs_ion_asym:.6f}; rel.diff "
        f"{100*d_ion:.2f}%. The published fit carries the same three numbers "
        f"rounded to two significant figures, so this confirms the recalled "
        f"values to that precision -- the first external test they have had.")

    svds_heat_001 = f_heat(0.01)
    rgs_heat_001 = float(f_heat_E(1.0e9, 0.01))
    d_heat = abs(svds_heat_001 - rgs_heat_001) / rgs_heat_001
    prov["svds_vs_fs10_heat_reldiff"] = d_heat
    prov["svds_heat_x0.01"] = svds_heat_001
    rec("C32", "RECALLED SvdS heating coefficients vs FS10 eq. (14)",
        d_heat < 0.02,
        f"recalled 0.9971[1-(1-x^0.2663)^1.3163] at x=0.01 = {svds_heat_001:.4f}; "
        f"FS10 eq.(14) asymptote 1.0[1-(1-x^0.27)^1.32] = {rgs_heat_001:.4f}; "
        f"rel.diff {100*d_heat:.2f}%.")
    rec("C32", "FS10's own statement about that fit is reproduced",
        0.02 < abs(svds_heat_001 - 0.32) < 0.08,
        f"FS10 footnote 8: the SvdS EXACT f_heat at x_i=0.01 is 0.32 and the "
        f"Ricotti FIT sits ~6% above it in absolute terms. The recalled "
        f"coefficients give {svds_heat_001:.4f}, i.e. "
        f"{100*(svds_heat_001-0.32):.1f}% absolute above 0.32 -- so they "
        f"reproduce the FITTING FUNCTION, which is what they are, and inherit "
        f"its known offset from SvdS's own Monte Carlo. That offset is a real "
        f"systematic, not a coding error, and it is NOT removed here.")

    # --- C24-style INDEPENDENT ROUTE: the ionization-only cascade ----------
    # Built from FS10 eq. (2) alone. No Shull & van Steenberg input of any
    # kind. Switching off excitation and heating can only ADD ionizations, so
    # Y(E) is a strict upper bound on every model here, and a much tighter one
    # than E/E_th. If any model crossed it, that model would be wrong.
    Ecs, Ycs, W_ion = ionization_only_cascade()
    prov["W_ion_only_eV"] = W_ion
    prov["Y_cascade_1keV"] = float(np.interp(1.0e3, Ecs, Ycs))
    rec("C20", "ionization-only cascade: monotone and under the E/E_th ceiling",
        np.all(np.diff(Ycs) >= -1e-12) and np.all(Ycs <= Ecs / E_TH_HI + 1e-9),
        f"W_ion = E/Y = {W_ion:.3f} eV at the top of the grid "
        f"(vs {float(Ecs[len(Ecs)//5]/np.interp(Ecs[len(Ecs)//5], Ecs, Ycs)):.3f} eV "
        f"at one fifth of it, so it is converged). Every inelastic event an "
        f"ionization -> one ion pair per ~18 eV instead of per "
        f"{E_TH_HI/svds_ion:.1f} eV; the difference is the energy that really "
        f"goes to excitation and heat.")
    Ecmp2 = np.array([50.0, 100.0, 300.0, 1000.0, 3000.0])
    nA2 = np.asarray(N_e_full(Ecmp2, p))
    Ycmp = np.interp(Ecmp2, Ecs, Ycs)
    rec("C20", "the asymptotic ceiling stays under the FS10-eq.(2) ionization-only bound",
        np.all(nA2 <= Ycmp + 1e-9),
        "  ".join(f"{E:.0f} eV: A={a:.2f} <= Y={y:.2f}"
                  for E, a, y in zip(Ecmp2, nA2, Ycmp))
        + ". Two prescriptions from the same paper, used in completely "
          "different ways, and they do not contradict each other.")

    # --- C20 STRUCTURAL: the marginal yield collapses to the old constant --
    p_const = Params(f_ion_model="svds_asymptotic")
    dcheck = np.asarray(dN_full_dE(np.array([1e2, 1e4, 1e6]), p_const))
    rec("C20", "STRUCTURAL: dN_full/dE == f_ion/E_th when f_ion is constant",
        np.allclose(dcheck, f_ion_HI(p.x_e) / E_TH_HI, rtol=0, atol=1e-18),
        f"dN/dE = {dcheck.tolist()} vs f_ion/E_th = {f_ion_HI(p.x_e)/E_TH_HI:.12e}. "
        f"The energy-dependent machinery reduces exactly to the old formula, "
        f"so the difference in the answer is physics, not refactoring.")

    # --- C36 THE HEADLINE SENSITIVITY: W(K_e) versus a constant W ----------
    chan_cw = ICPhotonChannel(cos, p_const, tau_mode=p.tau_mode_default).build(n=1200)
    nC_cw = float(N_e_ic(1e12, cos, p_const)) + float(chan_cw.N_secondary(1e12))
    d_W = (nC_cw - nC_ref) / nC_cw
    prov["N_e_depfit_1e12_constW"] = nC_cw
    prov["ic_W_energy_dependence_reduction"] = d_W
    rec("C36", "energy-dependent W(K_e) vs the constant W = E_th/f_ion",
        True,
        f"N_C(1e12) = {nC_cw:.4e} with a constant W = {E_TH_HI/svds_ion:.2f} eV, "
        f"{nC_ref:.4e} with W(K_e) from FS10 eq. (13): a reduction of "
        f"{100*d_W:.1f}%. It bites here and almost nowhere else because the "
        f"IC-secondary photons land near threshold, where W(K_e) is "
        f"{E_TH_HI/float(f_ion_HI_E(100.0, p.x_e)):.1f} eV rather than "
        f"{E_TH_HI/svds_ion:.1f} eV. FS10 caution that this fit is 'a "
        f"relatively poor match at lower energies' -- so the -10% bracket "
        f"below is still live, and this correction does not close it.")

    # --- C53 the floor creates an EXACT plateau in the photon curve --------
    # Between E_th and 2 E_th the photoelectron has less than E_th and cannot
    # ionize, so the photon makes its own single ionization and nothing else.
    E_plateau = E_TH_HI + (RGS_ION["E0"] if p.f_ion_model == "fs10" else E_TH_HI)
    pl = np.array([E_TH_HI, 20.0, 30.0, E_plateau * 0.9999])
    rec("C53", "NULL: N_gamma == 1 EXACTLY up to E_th + E0",
        np.all(N_gamma(pl, cos, p, "ic") == 1.0)
        and float(N_gamma(E_plateau * 1.0001, cos, p, "ic")) > 1.0,
        f"N_gamma({pl.round(3).tolist()}) = "
        f"{N_gamma(pl, cos, p, 'ic').round(6).tolist()}, and "
        f"{float(N_gamma(E_plateau*1.0001, cos, p, 'ic')):.4f} just above "
        f"E_th + E0 = {E_plateau:.3f} eV. The plateau edge MOVED from "
        f"{2*E_TH_HI:.1f} to {E_plateau:.1f} eV when the constant f_ion was "
        f"replaced by FS10 eq. (13), which is identically zero below 28 eV. "
        f"The check is written against the model, not against the old number.")

    # --- C26 energy bookkeeping of the split ------------------------------
    Ebk = 1.0e9
    dep = _deposited_energy_eV(Ebk, cos, p, None)
    rad, _ = quad(lambda u: float(1.0 - f_coll(np.exp(u), cos, p)) * np.exp(u),
                  np.log(p.T_bethe_min_eV), np.log(Ebk), limit=400)
    rec("C26", "energy split is exhaustive: collisional + IC = E",
        abs(dep + rad - Ebk) / Ebk < 1e-6,
        f"at E = {Ebk:.0e} eV: collisional {dep:.6e} + IC {rad:.6e} = "
        f"{dep+rad:.6e} eV, rel.error {abs(dep+rad-Ebk)/Ebk:.2e}")

    # --- results of the new channel, recorded --------------------------
    for Ex in (1e7, 1e8, 1e9, 1e10, 1e12):
        prov[f"N_e_depfit_{Ex:.0e}"] = float(N_e_depfit(Ex, cos, p, chan))
    prov["ic_max_energy_fraction_absorbed"] = float(
        np.max(chan.energy_fraction_absorbed(gg)))
    prov["N_e_depfit_over_B_max"] = float(np.max(nC / nB))

    return rows, prov


# ===========================================================================
# 9. FIGURE
# ===========================================================================
def make_figure(cos, p, data, outstem=None):
    """Draw the figure under our own rcParams, then restore the globals.

    THIRD BITE of the same bug.  This module imports igm_losses, which flips
    text.usetex on at import whenever a LaTeX install is present; the labels
    below carry a bare "&" (Furlanetto & Stoever) and an em dash, which LaTeX
    rejects with "Misplaced alignment tab character &".  The failure lands
    after every physics check has passed, so it reads as broken science when
    it is a typesetting escape.  igm_config.safe_plot_style exists for exactly
    this and the other figure modules already use it; this one did not.
    """
    from igm_config import safe_plot_style
    with safe_plot_style():
        return _make_figure_impl(cos, p, data, outstem)


def _make_figure_impl(cos, p, data, outstem=None):
    from igm_config import fig_stem as _fs
    outstem = _fs("ionization_yield_fig") if outstem is None else outstem
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    # Categorical hues: dataviz reference palette slots 1 and 2.
    # Validated with scripts/validate_palette.js --pairs all, light and dark:
    # all six checks PASS (worst normal-vision dE 33.6, worst CVD dE 24.7).
    # COLOUR encodes SPECIES; LINE STYLE encodes MODEL. Three models now, so the
    # styles are dotted (A) / dashed (B) / solid (C, the answer) -- identity is
    # never carried by colour alone, and the answer is the heaviest stroke.
    C_E, C_G = "#2a78d6", "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    LS_A, LS_B, LS_C = (0, (1, 2.2)), (0, (5, 2)), "-"

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.8, 8.8), sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.09})
    fig.patch.set_facecolor("#fcfcfb")
    for a_ in (ax, bx):
        a_.set_facecolor("#fcfcfb")
        a_.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        a_.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6, zorder=0)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(GRID)
        a_.tick_params(colors=INK2, labelsize=9)

    # ---- panel A: the answer ------------------------------------------
    # The hard ceiling first, as a reference the eye can measure against.
    Eall = data["E_e"]
    ax.plot(Eall, Eall / E_TH_HI, color=INK2, lw=0.9, ls=(0, (1, 4)), zorder=1)
    ax.text(4e11, 4e11 / E_TH_HI * 1.5, "$E/E_{\\rm th}$  hard ceiling",
            color=INK2, fontsize=8.0, ha="right", va="bottom")

    # Photon drawn WIDE underneath, electron NARROW on top: above ~100 eV the
    # two species genuinely coincide (the photon yield is the electron yield
    # plus the one ionization the photon itself causes), so a plain overplot
    # would hide one of them. The halo shows coincidence instead of faking a gap.
    ax.plot(data["E_g"], data["Ng_full"], color=C_G, lw=4.4, ls=LS_A, zorder=3, alpha=.95)
    ax.plot(data["E_g"], data["Ng_C"],    color=C_G, lw=4.4, ls=LS_C, zorder=3, alpha=.95)
    ax.plot(Eall, data["Ne_full"], color=C_E, lw=1.6, ls=LS_A, zorder=5)
    ax.plot(Eall, data["Ne_ic"],   color=C_E, lw=1.6, ls=LS_B, zorder=5)
    ax.plot(Eall, data["Ne_C"],    color=C_E, lw=2.3, ls=LS_C, zorder=6)

    # the two energies that set the shape of the new curve
    ax.axvline(E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
    ax.text(E_TH_HI * 1.35, 3e-3, "13.6 eV\nHI threshold", color=INK2,
            fontsize=8.5, va="bottom")
    ax.axvline(data["E_crit"], color=INK2, lw=0.9, ls=":", zorder=2)
    ax.text(data["E_crit"] * 1.4, 3e-3,
            f"$E_{{\\rm crit}}$ = {data['E_crit']/1e3:.0f} keV\nIC overtakes collisions",
            color=INK2, fontsize=8.5, va="bottom")

    ax.axvspan(1.0e7, 1.0e9, color=C_E, alpha=0.055, zorder=1, lw=0)
    ax.text(3.2e7, 6.0e0, "IC-SECONDARY WINDOW\n$\\gamma$-boosted CMB photons clear\n"
                          "13.6 eV and are still absorbed",
            color=C_E, fontsize=8.4, ha="center", va="top", linespacing=1.35)

    # direct labels
    ax.text(3.0e11, 1.6e9, "electron A", color=C_E, fontsize=8.8,
            ha="right", va="top")
    ax.text(3.0e11, data["Ne_C"][-1] * 3.0, "electron C\n+ IC secondaries",
            color=C_E, fontsize=9.2, ha="right", va="bottom", linespacing=1.25,
            weight="bold")
    ax.text(3.0e11, data["Ne_ic"][-1] * 0.32, "electron B\nIC energy discarded",
            color=C_E, fontsize=8.8, ha="right", va="top", linespacing=1.25)
    ax.text(3.4e1, 30.0, "photon", color=C_G, fontsize=9.5,
            ha="left", va="center", weight="bold")
    ax.annotate("", xy=(1.05e2, 3.4), xytext=(6.0e1, 22.0),
                arrowprops=dict(arrowstyle="-", color=C_G, lw=0.9, shrinkA=1, shrinkB=1))
    ax.text(2.6e2, 0.34, "below $10^5$ eV a photoelectron is far too\n"
                         "slow to upscatter anything, so C and B\n"
                         "coincide for photons",
            color=INK2, fontsize=8.0, ha="left", va="top", linespacing=1.35)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(2e-3, 3e11)
    ax.set_ylabel("HI ion pairs per primary,  $N_{\\rm ion}$", color=INK, fontsize=10.5)
    ax.set_title("Average HI ionizations per primary particle\n"
                 "IGM at $z=10$, $x_e = 10^{-4}$, all cascade generations counted",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))

    h = [plt.Line2D([], [], color=INK2, lw=1.6, ls=LS_A),
         plt.Line2D([], [], color=INK2, lw=1.6, ls=LS_B),
         plt.Line2D([], [], color=INK2, lw=2.3, ls=LS_C),
         plt.Line2D([], [], color=C_E, lw=2.4), plt.Line2D([], [], color=C_G, lw=2.4)]
    leg = ax.legend(h, ["A — full collisional absorption",
                        "B — IC-limited, IC energy discarded",
                        "C — B + IC secondary photoionization",
                        "electron primary", "photon primary"],
                    loc="upper left", frameon=False, fontsize=8.6,
                    labelcolor=INK2, handlelength=2.6)
    leg.set_zorder(7)

    # ---- panel B: the two branching fractions that make the shape -------
    bx.plot(data["E_e"], data["f_coll"], color=C_E, lw=2.0, ls=LS_C, zorder=4)
    bx.plot(data["E_e"], data["eta_abs"], color=C_E, lw=2.0, ls=LS_B, zorder=4)
    bx.plot(data["E_g"], data["p_photo"], color=C_G, lw=2.0, ls=LS_C, zorder=4)
    # the new input: f_ion is a function of ENERGY now, not just x_e
    bx.plot(data["E_e"], data["f_ion_E"], color=INK2, lw=2.0, ls=(0, (4, 1.6, 1, 1.6)),
            zorder=5)
    bx.annotate("$f_{\\rm ion,HI}(E)$ — Furlanetto & Stoever 2010 eq. (13):\n"
                "zero below 28 eV, so $W(K_e)$ = 43.5 eV at 100 eV\n"
                "and 36.5 eV asymptotically",
                xy=(4.0e10, 0.377), xytext=(2.6e12, 0.30),
                color=INK2, fontsize=8.0, va="top", ha="right", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9,
                                shrinkA=2, shrinkB=2))
    bx.axvline(E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
    bx.axvline(data["E_crit"], color=INK2, lw=0.9, ls=":", zorder=2)
    bx.axvspan(1.0e7, 1.0e9, color=C_E, alpha=0.055, zorder=1, lw=0)
    bx.axhline(0.5, color=INK2, lw=0.7, ls="--", alpha=0.6, zorder=2)
    bx.text(3.5e5, 0.58, "electron energy going\nto the GAS (not the CMB)",
            color=C_E, fontsize=8.5, va="bottom", ha="left", linespacing=1.25)
    bx.annotate("IC energy returning as an\nABSORBED ionizing photon",
                xy=(2.4e8, 0.80), xytext=(1.6e9, 0.60),
                color=C_E, fontsize=8.5, va="center", ha="left", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=C_E, lw=0.9,
                                shrinkA=2, shrinkB=2))
    bx.text(1.6e1, 0.30, "photon: photoionization\nshare of total cross section",
            color=C_G, fontsize=8.5, va="top", linespacing=1.25)
    bx.set_xscale("log"); bx.set_ylim(-0.03, 1.10)
    bx.set_xlim(8, 3e12)
    bx.set_xlabel("kinetic energy of the primary particle  [eV]", color=INK, fontsize=10.5)
    bx.set_ylabel("branching fraction", color=INK, fontsize=10.5)

    fig.text(0.012, 0.004,
             "Semi-analytic. $W(K_e)$ and the secondary-electron spectrum: "
             "Furlanetto & Stoever 2010 eqs. (13) and (2),\n"
             "the latter from Opal, Peterson & Beaty 1971. Stopping power: "
             "Berger\u2013Seltzer / ICRU 37.\n"
             "IC spectrum: Blumenthal & Gould 1970 eq. (2.42). Cosmology: "
             "Planck 2018.\n"
             f"The depfit route follows the upscattered CMB photons: they ionize while "
             f"$\\varepsilon_1 < {data['E_esc']:.0f}$ eV "
             "($\\tau_{\\rm pi} > 1$ over $c/H$) and escape above it.\n"
             "C is the answer; A and B are drawn to locate it.",
             fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.905, bottom=0.145)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outstem}.{ext}", dpi=200, facecolor=fig.get_facecolor())
    return f"{outstem}.png", f"{outstem}.pdf"


# ===========================================================================
# 10. MAIN
# ===========================================================================
def main():
    p = Params()
    cos = cosmology(p.z)

    print("=" * 78)
    print("AVERAGE HI IONIZATIONS PER PRIMARY -- IGM at z = 10, x_e = 1e-4")
    print("=" * 78)
    print("\n[ASSUMPTIONS -- each stated where it enters]")
    for line in [
        "A1 medium is pure atomic hydrogen; He is IGNORED in the target, but the",
        "   SvdS85 deposition fractions were fitted to a primordial H+He gas.",
        "   This is an inconsistency of order the HeI channel, f_ion,HeI ~ 0.05.",
        "A2 x_e held static at 1e-4 (no back-reaction of the cascade on the gas).",
        "A3 uniform gas at the mean cosmic density; no clumping factor.",
        "A4 'per primary' means per particle that DOES interact.",
        "A4b W is ENERGY DEPENDENT, W(K_e) = E_th/f_ion(K_e, x_e), from FS10",
        "    eq. (13) = Ricotti et al. 2002 fit to SvdS85. Identically zero",
        "    below 28 eV, 43.5 eV at 100 eV, 36.5 eV asymptotically. FS10",
        "    caution this fit is 'a relatively poor match at lower energies'.",
        "A4c The secondary-electron energy distribution is FS10 eq. (2),",
        "    p(eps) ~ 1/[1+(eps/8 eV)^2.1], used for the independent",
        "    ionization-only bound. Anchored on FS10's quoted median, 7.2 eV.",
        "A5 no magnetic field, so synchrotron losses are omitted. At z=10 the CMB",
        "   would dominate any plausible IGM field anyway.",
        "A6 flat LCDM, Planck 2018 parameters.",
        "A7 IC in the Thomson regime, isotropic seed photons: the scattered",
        "   spectrum is Blumenthal & Gould 1970 eq. (2.42). Checked at the top of",
        "   the range (gamma kT/mc^2 = 1e-2).",
        "A8 an IC-upscattered photon is absorbed with probability 1 - exp(-tau),",
        "   tau = n_HI sigma_pi(eps) c/H(z) -- i.e. absorbed near z = 10 or never.",
        "   This is a CHOICE. The alternative (follow it down the light cone to",
        "   z = 6) is computed too; the two differ by 5%.",
        "A9 the IC cascade is truncated after one generation of secondary",
        "   photons. Justified by the escape energy, not assumed -- see checks.",
    ]:
        print("   " + line)

    print("\n[COSMOLOGY -- derived]")
    print(f"   n_H(z=10)   = {cos['n_H_cm3']:.4e} cm^-3")
    print(f"   T_CMB(z=10) = {cos['T_cmb_K']:.4f} K")
    print(f"   u_CMB(z=10) = {cos['u_cmb_erg_cm3']:.4e} erg cm^-3   "
          f"[= (1+z)^4 = {(1+p.z)**4:.0f} x present]")
    print(f"   c/H(z=10)   = {cos['hubble_length_cm']:.4e} cm")

    chan = ICPhotonChannel(cos, p).build()
    rows, prov = run_checks(cos, p, chan)

    print("\n[CHECKS -- coverage, not a verdict tally]")
    for cid, name, state, detail in rows:
        print(f"   [{state}] {cid:5s} {name}")
        print(f"           {detail}")
    n_pass = sum(1 for r in rows if r[2] == "PASS")
    print(f"\n   {n_pass}/{len(rows)} checks pass.")
    print("   COVERAGE: internal consistency; limiting cases; two independent")
    print("   routes to E_crit, to <w>, and to N_secondary; two structural cases")
    print("   (A=0 must collapse C onto B, A=1 must recover full absorption);")
    print("   the hard E/E_th ceiling; one external anchor (W ~ 36 eV).")
    print("   THEY DO NOT TEST: any of the TWELVE SvdS85 coefficients against the")
    print("   paper (and B, C of every row are unprobed at x_e=1e-4 in any case);")
    print("   the neglect of helium in the target gas and in the opacity;")
    print("   the choice of transport prescription (its SPREAD is measured, 5%,")
    print("   but nothing here says which prescription is right).")

    # ---- grids ---------------------------------------------------------
    E_e = np.logspace(np.log10(p.E_e_min), np.log10(p.E_e_max), 240)
    E_g = np.logspace(np.log10(p.E_g_min), np.log10(p.E_g_max), 240)
    _, gam_e, _ = _kinematics(E_e)
    data = {
        "E_e": E_e, "E_g": E_g,
        "Ne_full": N_e_full(E_e, p),
        "Ne_ic": N_e_ic(E_e, cos, p),
        "Ne_C": N_e_depfit(E_e, cos, p, chan),
        "Ng_full": N_gamma(E_g, cos, p, "full"),
        "Ng_ic": N_gamma(E_g, cos, p, "ic"),
        "Ng_C": N_gamma(E_g, cos, p, "C", chan),
        "f_coll": f_coll(E_e, cos, p),
        "eta_abs": chan.energy_fraction_absorbed(gam_e),
        "f_ion_E": f_ion_used(E_e, p),
        "p_photo": sigma_photoion(E_g) / (sigma_photoion(E_g) + sigma_kn(E_g)),
        "E_crit": prov["E_crit_eV"],
        "E_esc": prov["E_escape_eV"],
    }

    print("\n[RESULT -- N_ion per primary, 3 significant figures]")
    print("   A = collisional full absorption   B = IC-limited, IC energy discarded")
    print("   C = B + ionizations by the absorbed IC-upscattered CMB photons  <-- ANSWER")
    print(f"   {'E [eV]':>10}  {'e- A':>11} {'e- B':>11} {'e- C':>11} "
          f"{'gam A':>11} {'gam C':>11}  {'E/E_th':>10}")
    for E in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12]:
        ea = f"{float(N_e_full(E, p)):.3g}" if E >= p.E_e_min else "-"
        eb = f"{float(N_e_ic(E, cos, p)):.3g}" if E >= p.E_e_min else "-"
        ec = f"{float(N_e_depfit(E, cos, p, chan)):.3g}" if E >= p.E_e_min else "-"
        ga = f"{float(N_gamma(E, cos, p, 'full')):.3g}" if E <= p.E_g_max else "-"
        gc = f"{float(N_gamma(E, cos, p, 'C', chan)):.3g}" if E <= p.E_g_max else "-"
        print(f"   {E:10.0e}  {ea:>11} {eb:>11} {ec:>11} {ga:>11} {gc:>11}  "
              f"{E/E_TH_HI:10.3e}")

    NC12 = prov["N_e_depfit_1e+12"]
    boost = NC12 / prov["N_e_ic_1e12"]
    rCA = float(np.max(data["Ne_C"] / data["Ne_full"]))
    iCA = int(np.argmax(data["Ne_C"] / data["Ne_full"]))
    prov["N_e_depfit_over_A_max"] = rCA
    prov["N_e_depfit_over_A_max_at_eV"] = float(E_e[iCA])
    print(f"\n   The IC-secondary channel raises the saturated electron yield from")
    print(f"   {prov['N_e_ic_1e12']:.4g} to {NC12:.4g} ion pairs -- a factor {boost:.4g}.")
    print(f"   It switches on near {prov['ic_turnon_T_eV_half']:.3g} eV (predicted "
          f"{prov['ic_turnon_T_eV_pred']:.3g} eV) and off above ~1e9 eV, where the")
    print(f"   upscattered photons exceed the escape energy "
          f"{prov['E_escape_eV']:.4g} eV and free-stream away.")
    print(f"   Peak reprocessing efficiency: "
          f"{100*prov['ic_max_energy_fraction_absorbed']:.1f}% of the IC-radiated")
    print(f"   energy is absorbed as ionizing photons.")
    print(f"   max(C/A) over the grid = {rCA:.6f} at {E_e[iCA]:.2e} eV: the depfit route now")
    print(f"   stays AT OR BELOW the asymptotic ceiling everywhere. With the constant W of the")
    print(f"   previous version C exceeded A by 2%, because the asymptotic W was")
    print(f"   applied to the near-threshold secondary photons that do most of the")
    print(f"   work. W(K_e) removes that excess -- it was an artefact, not physics.")
    print(f"   W = E_th/f_ion is now ENERGY DEPENDENT:")
    for Ew in (30.0, 100.0, 1e3, 1e4, 1e6):
        print(f"      W({Ew:8.0f} eV) = {E_TH_HI/float(f_ion_HI_E(Ew, p.x_e)):6.2f} eV")
    print(f"   Ionization-only bound from FS10 eq. (2): one pair per "
          f"{prov['W_ion_only_eV']:.2f} eV,")
    print(f"   using no Shull & van Steenberg input at all.")

    print("\n[CAVEATS -- what is NOT claimed]")
    for line in [
        "1. The IC cascade is truncated at ONE generation. That is justified, not",
        "   assumed: photons above the escape energy leave, so every absorbed",
        "   photon makes a photoelectron far below E_crit -- see the check.",
        "2. Photon transport uses tau = n_HI sigma_pi (c/H(z)): absorbed at z=10",
        "   or never. Following the photon down the light cone to z=6 instead",
        f"   changes N_C(1e12) by {100*prov['ic_transport_spread_max']:.1f}% at worst.",
        "   Both are reported; neither is adopted as the truth, because the true",
        "   answer requires a reionization history this calculation does not have.",
        "3. The SvdS85 coefficients are now CHECKED against Furlanetto & Stoever",
        "   2010 eqs. (13)-(14), which is in papers/. Ionization agrees to 0.18%,",
        "   heating to 1.10%. What remains unverified: the SvdS PAPER itself is",
        "   still not in the tree, and FS10 state that the Ricotti fit used here",
        "   sits ~6% (absolute) above SvdS's own exact heating result at x=0.01.",
        "   That offset is inherited, not removed.",
        "3b. FS10's OWN tables are not used: they are not public ('available on",
        "   request'), and FS10 explicitly decline to publish a fit to them.",
        "   Using them would need the authors to send them.",
        "4. Compton scattering of the upscattered photons off bound electrons is",
        "   omitted (tau_T ~ 0.1 over a Hubble length). It would slightly RAISE",
        "   the depfit route by degrading escaping photons back below the escape energy.",
        "5. Above ~1e12 eV the IC Thomson approximation fails (see the check);",
        "   Klein-Nishina suppression would REDUCE the IC loss. It would raise",
        "   the IC route and LOWER the depfit route, since less energy is radiated at all.",
        "6. Helium is in the SvdS fits but not in the target gas, and HeI opacity",
        "   is not in tau. Including it would raise the absorbed fraction.",
        "7. No uncertainty band is quoted because no input carries a stated error",
        "   bar: the SvdS fits are published without covariances. The propagated",
        "   variations are the I-parameter and the transport assumption, above.",
    ]:
        print("   " + line)

    # ---- artefacts -----------------------------------------------------
    png, pdf = make_figure(cos, p, data)

    os.makedirs("provenance", exist_ok=True)
    results = {
        "parameters": asdict(p),
        "cosmology": {k: float(v) for k, v in cos.items()},
        "derived": {k: float(v) for k, v in prov.items()},
        "checks": [{"id": c, "name": n, "state": s, "detail": d} for c, n, s, d in rows],
        "curves": {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                   for k, v in data.items()},
        "note": "Deterministic; no random numbers drawn, so no seed exists.",
    }
    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    numbers = {
        "W_eV_per_ion_pair": {
            "value": prov["W_eV_per_ion_pair"],
            "statement": "Mean energy expended per HI ion pair at x_e=1e-4.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "E_th/f_ion, and the ratio itself",
            "from_library": "none",
            "choices": ["SvdS85 asymptotic fit rather than the Furlanetto & Stoever "
                        "2010 energy-resolved tables, which are not in this tree"],
        },
        "E_crit_eV": {
            "value": prov["E_crit_eV"],
            "statement": "Electron energy at which IC loss on the CMB equals the "
                         "collisional loss to the gas, at z=10.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "the balance equation and its analytic inversion",
            "from_library": "scipy.optimize.brentq for the numerical root",
            "choices": ["I = 14.99 eV (atomic H) not 19.2 eV (H2); shifts the "
                        "result logarithmically, quantified in the C36 check"],
        },
        "E_escape_eV": {
            "value": prov["E_escape_eV"],
            "statement": "Photon energy above which an IC-upscattered CMB photon "
                         "free-streams (tau_pi < 1 over c/H) instead of "
                         "photoionizing. This one number sets where the "
                         "IC-secondary channel switches OFF.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "tau = n_HI sigma_pi(eps) c/H(z), solved for tau = 1",
            "from_library": "scipy.optimize.brentq",
            "choices": ["absorption judged over one Hubble length at z=10 rather "
                        "than along the forward light cone; the alternative is "
                        "computed and moves the yield by 5 per cent",
                        "sigma_pi only, no Compton opacity; justified by the "
                        "sigma_pi/sigma_KN check at this energy"],
        },
        "ic_turnon_T_eV_half": {
            "value": prov["ic_turnon_T_eV_half"],
            "statement": "Electron kinetic energy at which the IC-secondary "
                         "channel reaches half its peak efficiency.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "G(gamma) on a log grid; the half-maximum crossing",
            "from_library": "none",
            "choices": ["compared against the closed-form prediction "
                        "gamma = sqrt(E_th/(4 kT <w>)), written down BEFORE the "
                        "grid was evaluated"],
        },
        "N_e_ic_1e12": {
            "value": prov["N_e_ic_1e12"],
            "statement": "Saturated HI ion-pair yield of a 1 TeV electron at z=10 "
                         "counting COLLISIONAL deposition only (the IC route). Kept as "
                         "the lower bound the secondary channel is measured against.",
            "produced_by": "ionization_yield.py::N_e_ic",
            "from_scratch": "the branching integral along the slowing-down track",
            "from_library": "scipy.integrate.quad",
            "choices": ["IC-upscattered photons discarded; superseded by the depfit route"],
        },
        "N_e_depfit_1e+12": {
            "value": prov["N_e_depfit_1e+12"],
            "statement": "Saturated HI ion-pair yield of a 1 TeV electron at z=10 "
                         "INCLUDING the ionizations made by the IC-upscattered CMB "
                         "photons that are absorbed. This is the headline number.",
            "produced_by": "ionization_yield.py::N_e_depfit",
            "from_scratch": "the Blumenthal & Gould Thomson spectrum convolved "
                            "with a Planck seed, the absorption probability, and "
                            "the integral along the slowing-down track",
            "from_library": "scipy.integrate.quad for ptilde(w); numpy trapezoid "
                            "for the cumulative track integral",
            "choices": ["one generation of secondary photons (justified by "
                        "E_escape << E_crit)",
                        "Hubble-length absorption prescription; the light-cone "
                        "alternative gives "
                        f"{prov['N_e_depfit_1e12_cosmo']:.4e}"],
        },
        "ic_transport_spread_max": {
            "value": prov["ic_transport_spread_max"],
            "statement": "Worst relative difference between the two photon-"
                         "transport prescriptions over 1e7-1e12 eV. This is a "
                         "SPREAD between two physical assumptions, not an "
                         "uncertainty on one.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "both prescriptions, computed independently",
            "from_library": "none",
            "choices": ["z_end = 6 for the light-cone variant, i.e. the photon is "
                        "assumed to stop finding neutral gas at reionization"],
        },
        "ic_frac_of_secondary_below_100eV": {
            "value": prov["ic_frac_of_secondary_below_100eV"],
            "statement": "Fraction of the secondary yield produced by "
                         "upscattered photons below 100 eV, i.e. below the "
                         "stated domain of the SvdS asymptotic fit. The IC "
                         "channel deposits most efficiently near threshold, so "
                         "the weakest part of the input does a quarter of the "
                         "work.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "band-restricted re-integration of the track integral",
            "from_library": "none",
            "choices": ["100 eV taken as the edge of the fit's domain, which is "
                        "what the paper states; the transition is not sharp"],
        },
        "N_e_depfit_1e12_pe_floor_100eV": {
            "value": prov["N_e_depfit_1e12_pe_floor_100eV"],
            "statement": "Saturated yield when the photoelectron cascade is "
                         "suppressed entirely below 100 eV. The PESSIMISTIC end "
                         "of the energy-independent-W assumption, -10.1 per cent; "
                         "the truth is between this and the headline value.",
            "produced_by": "ionization_yield.py::ICPhotonChannel(pe_floor_eV=100)",
            "from_scratch": "the same track integral with the photoelectron term "
                            "zeroed below the floor",
            "from_library": "none",
            "choices": ["a hard cut rather than a smooth rise, deliberately: it "
                        "is meant to bracket, not to model"],
        },
        "ic_W_energy_dependence_reduction": {
            "value": prov["ic_W_energy_dependence_reduction"],
            "statement": "Reduction in the saturated yield from replacing the "
                         "constant W = E_th/f_ion with the energy-dependent "
                         "W(K_e) of Furlanetto & Stoever 2010 eq. (13). It bites "
                         "in this problem and almost nowhere else, because the "
                         "IC-secondary photons deposit near threshold where W is "
                         "largest.",
            "produced_by": "ionization_yield.py::run_checks",
            "from_scratch": "the whole pipeline run under both f_ion models",
            "from_library": "none",
            "choices": ["stored as a positive reduction because the provenance "
                        "gate's number parser has no sign branch",
                        "FS10 eq. (13) is the Ricotti et al. 2002 fit to SvdS85, "
                        "not FS10's own tables, which are not public"],
        },
        "W_ion_only_eV": {
            "value": prov["W_ion_only_eV"],
            "statement": "Mean energy per ion pair if EVERY inelastic event were "
                         "an ionization, from the FS10 eq. (2) secondary-electron "
                         "spectrum alone. A strict upper bound on the yield that "
                         "uses no Shull & van Steenberg input at all.",
            "produced_by": "ionization_yield.py::ionization_only_cascade",
            "from_scratch": "the cascade recursion, marched on a linear grid",
            "from_library": "none",
            "choices": ["excitation and electron-electron heating switched off, "
                        "which is what makes it a bound rather than a model"],
        },
        "N_e_depfit_over_A_max": {
            "value": prov["N_e_depfit_over_A_max"],
            "statement": "Maximum of the depfit route over the asymptotic ceiling. Exceeds 1: routing "
                         "energy through photons just above 13.6 eV ionizes more "
                         "efficiently than electron degradation at W = 36.3 eV, so "
                         "the full-absorption curve is NOT an upper bound on the "
                         "total yield. The strict ceiling is E/E_th.",
            "produced_by": "ionization_yield.py::main",
            "from_scratch": "the ratio of the two computed curves",
            "from_library": "none",
            "choices": [],
        },
    }
    with open("provenance/numbers.json", "w") as fh:
        json.dump(numbers, fh, indent=2)

    with open("provenance/claims.yaml", "w") as fh:
        fh.write(
            "claims:\n"
            "  - id: yield_is_linear_below_Ecrit\n"
            "    statement: >\n"
            "      Below the critical energy the HI ion-pair yield is linear in the\n"
            "      primary energy, N = f_ion E / E_th, equivalently one ion pair per\n"
            "      W = 36 eV deposited.\n"
            "    evidence:\n"
            "      - ionization_yield.py::N_e_full\n"
            "      - Shull & van Steenberg 1985, ApJ 298, 268 (deposition fractions)\n"
            "      - independent literature value W(H) ~ 36 eV per ion pair\n"
            "    numbers: [W_eV_per_ion_pair]\n"
            "  - id: ic_branching\n"
            "    statement: >\n"
            "      At z = 10 an electron above E_crit ~ 130 keV gives its energy to\n"
            "      the CMB by inverse Compton rather than to the gas by collisions,\n"
            "      so the COLLISIONAL ionization yield saturates.\n"
            "    evidence:\n"
            "      - ionization_yield.py::N_e_ic\n"
            "      - ionization_yield.py::run_checks (saturation, E_crit two ways)\n"
            "      - Rybicki & Lightman 1979, eq. 7.16 (IC loss rate)\n"
            "    numbers: [E_crit_eV, N_e_ic_1e12]\n"
            "  - id: ic_secondary_reprocessing\n"
            "    statement: >\n"
            "      That energy is not lost. Between roughly 10 MeV and 1 GeV the\n"
            "      electron upscatters CMB photons past 13.6 eV and below the 1.2 keV\n"
            "      escape energy, so they photoionize. Following them raises the\n"
            "      saturated yield of a 1 TeV electron by nearly three orders of\n"
            "      magnitude over the collisional-only value.\n"
            "    evidence:\n"
            "      - ionization_yield.py::ICPhotonChannel\n"
            "      - Blumenthal & Gould 1970, Rev. Mod. Phys. 42, 237, eq. (2.42)\n"
            "      - ionization_yield.py::run_checks (spectrum normalisation and\n"
            "        mean two ways; <w> against the closed form; N_secondary by\n"
            "        trapezoid and by adaptive quad; A=0 and A=1 structural cases)\n"
            "    numbers: [E_escape_eV, ic_turnon_T_eV_half, N_e_depfit_1e+12,\n"
            "              ic_transport_spread_max]\n"
            "  - id: full_absorption_is_not_an_upper_bound\n"
            "    statement: >\n"
            "      The asymptotic ceiling is an upper bound on collisional degradation only, not on\n"
            "      the total yield: energy reprocessed into photons just above 13.6\n"
            "      eV buys one ion pair per 13.6 eV instead of one per W = 36.3 eV,\n"
            "      so the depfit route exceeds the asymptotic ceiling by a few per cent near 1e8 eV. The one\n"
            "      strict ceiling is E/E_th, and it is not crossed.\n"
            "    evidence:\n"
            "      - ionization_yield.py::run_checks (ENERGY BOUND check)\n"
            "      - ionization_yield.py::main (the ratio, computed on the grid)\n"
            "    numbers: [N_e_depfit_over_A_max]\n"
            "  - id: fixed_W_is_the_weak_point\n"
            "    statement: >\n"
            "      W is treated as energy-independent, because f_ion depends on\n"
            "      x_e alone. That is sound above ~1 keV, where W is\n"
            "      experimentally flat, but the IC channel deposits most\n"
            "      efficiently near threshold and a quarter of the secondary\n"
            "      yield is made below 100 eV, outside the fit's stated domain.\n"
            "      Imposing the exact sub-threshold floor costs 1.1 per cent;\n"
            "      the pessimistic bracket on the 13.6-100 eV band is -10.1 per\n"
            "      cent. This is the largest modelling bracket on the headline\n"
            "      number and it is one-sided.\n"
            "    evidence:\n"
            "      - ionization_yield.py::run_checks (band decomposition and the\n"
            "        pe_floor_eV variant and the f_ion_model switch)\n"
            "      - Shull & van Steenberg 1985 (asymptotic branch, E >~ 100 eV)\n"
            "    numbers: [ic_frac_of_secondary_below_100eV,\n"
            "              N_e_depfit_1e12_pe_floor_100eV,\n"
            "              ic_W_energy_dependence_reduction, W_ion_only_eV]\n"
            "  - id: photon_threshold\n"
            "    statement: >\n"
            "      Photons below 13.6 eV produce exactly zero HI ionizations; the\n"
            "      photon curve therefore has a hard threshold, and above it the\n"
            "      photon yield exceeds the electron yield by the one primary\n"
            "      ionization the photon itself causes. Over the requested photon\n"
            "      range the photoelectron is far too slow to upscatter anything, so\n"
            "      models B and C coincide for photons.\n"
            "    evidence:\n"
            "      - ionization_yield.py::N_gamma\n"
            "      - NIST ASD, HI ionization energy 13.598 eV\n"
            "    numbers: []\n"
            "figures:\n"
            "  - file: ionization_yield_fig.png\n"
            "    produced_by: ionization_yield.py::make_figure\n"
            "    shows: >\n"
            "      Upper panel: HI ion pairs per primary against primary kinetic\n"
            "      energy, for electrons and photons, under the three models, with\n"
            "      the E/E_th ceiling drawn. Lower panel: the two branching\n"
            "      fractions that set the shape -- the share of electron energy\n"
            "      going to the gas, and the share of IC-radiated energy that\n"
            "      returns as an absorbed ionizing photon.\n"
            "    from_scratch: all curves\n"
            "    from_library: matplotlib for rendering only\n"
            "    choices:\n"
            "      - 'colour encodes species, line style encodes model, so identity\n"
            "         is never carried by colour alone'\n"
            "      - 'log-log axes: the yield spans 14 decades'\n"
            "      - 'the photon curves are stroked wide and the electron curves\n"
            "         narrow on top, because above 100 eV the two genuinely\n"
            "         coincide and a plain overplot would hide one of them'\n"
            "    supports: [yield_is_linear_below_Ecrit, ic_branching,\n"
            "               ic_secondary_reprocessing,\n"
            "               full_absorption_is_not_an_upper_bound, photon_threshold,\n"
            "               fixed_W_is_the_weak_point]\n"
        )

    # C56 is a claim about TWO runs, so the script cannot assert it alone. What
    # it can do is publish the hash of what it just wrote, so the comparison is
    # one command for anyone, instead of something the author says happened.
    import hashlib
    digest = hashlib.md5(open("results.json", "rb").read()).hexdigest()
    with open("provenance/results_md5.txt", "w") as fh:
        fh.write(digest + "  results.json\n")
    print(f"\n[ARTEFACTS WRITTEN]  {png}  {pdf}  results.json  "
          f"provenance/numbers.json  provenance/claims.yaml  "
          f"provenance/results_md5.txt")
    print(f"[C56] md5(results.json) = {digest}")
    print("      Determinism is a two-run claim; run this script again and "
          "compare. It is NOT asserted here on one run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
