#!/usr/bin/env python3
r"""
W-values: the energy price of one ionization, for either channel.

    W  =  <E> / <N_ion>        [eV per ion pair]

the W-value of radiation physics. Because zeta is a LINEAR functional of the
emissivity against a kernel N(E) that belongs to the IGM and not to the source,
the whole spectrum reaches zeta through this one scalar:

    zeta = C L / (n_H W)

verified against the full integral to 8e-5 (check S2 / X2).

WHY THIS MODULE EXISTS.  source_map.py and make_xcomp_definitions.py each grew
their own implementation of these integrals. They agreed to 3.7e-8, which is
reassuring but is not a reason to keep two -- the next edit to one of them would
not have been mirrored in the other. This is now the single implementation; both
import it. Check X2 still compares the two call paths, so the convergence is
gated rather than assumed.

The expensive parts -- N_gamma and the electron yield -- are evaluated ONCE on a
master log grid at import and interpolated thereafter. Check S4 gates that
shortcut against direct evaluation: a speed trick that moves a number is a bug.
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import parameters as PR          # the single source of truth

import numpy as np

import ionization_yield as IY
import photon_vs_electron as P
import yield_comparison as YC

Z_SNAP = PR.Z                     # parameters.yaml
YC.set_redshift(Z_SNAP)
PAR = IY.Params(z=Z_SNAP)
COS = IY.cosmology(Z_SNAP)
CHAN = IY.ICPhotonChannel(COS, PAR).build()

CONV = P.comoving_Mpc3_to_proper_cm3(Z_SNAP)
N_H = COS["n_H_cm3"]
X_E = PR.X_E                      # parameters.yaml
N_HI = N_H * (1.0 - X_E)
L_HUB = COS["hubble_length_cm"]

NGRID = PR.MASTER_GRID_N          # parameters.yaml
E_TOP_MASTER = PR.E_TOP_MASTER_EV # parameters.yaml


# ===========================================================================
# 1. MASTER GRIDS.  N_gamma and the electron yield are the expensive calls, so
#    each is evaluated ONCE and interpolated.  Check S4 gates that shortcut
#    against direct evaluation -- a speed trick that changes a number is a bug.
# ===========================================================================
_GP = np.linspace(np.log10(IY.E_TH_HI), np.log10(E_TOP_MASTER), NGRID)
_EP = 10.0 ** _GP
_NGAM = IY.N_gamma(_EP, COS, PAR, "C", CHAN)
_TAU = N_HI * np.array([float(IY.sigma_photoion(e)) for e in _EP]) * L_HUB
_FABS = 1.0 - np.exp(-_TAU)

_GE = np.linspace(2.0, 12.0, NGRID)
_EE = 10.0 ** _GE
_NE = P._electron_yield(_EE, COS, PAR, CHAN)


def tau_igm(E_eV):
    """HI optical depth over one Hubble length at Z_SNAP."""
    E = np.asarray(E_eV, dtype=float)
    return N_HI * np.asarray(IY.sigma_photoion(E), dtype=float) * L_HUB


def _ngam(g, absorb=True, exact=False):
    if exact:
        N = np.asarray(IY.N_gamma(10.0 ** g, COS, PAR, "C", CHAN), dtype=float)
        if absorb:
            N = N * (1.0 - np.exp(-tau_igm(10.0 ** g)))
        return N
    N = np.interp(g, _GP, _NGAM)
    return N * np.interp(g, _GP, _FABS) if absorb else N


def _ne(g, exact=False):
    if exact:
        return np.asarray(P._electron_yield(10.0 ** g, COS, PAR, CHAN), dtype=float)
    return np.interp(g, _GE, _NE)


# ===========================================================================
# 2. W:  eV per ionization ACTUALLY ACHIEVED.
# ===========================================================================
def W_gamma(alpha, E_top, E_th=None, absorb=True, n=NGRID, exact=False):
    """Single power law, f_nu ~ nu^-alpha, i.e. dN/dE ~ E^-(alpha+1).

    NOTE the exact identity  Gamma = alpha + 1  with the X-ray photon index,
    since X-ray work defines N(E) ~ E^-Gamma.  The two conventions meet here.
    """
    E_th = IY.E_TH_HI if E_th is None else E_th
    g = np.linspace(np.log10(E_th), np.log10(E_top), n)
    E = 10.0 ** g
    w = E ** (-(alpha + 1.0)) * np.log(10.0) * E          # dN/dlog10E
    N = _ngam(g, absorb=absorb, exact=exact)
    return float(IY._trapz(w * E, g) / IY._trapz(w * N, g))


def W_gamma_broken(G1, E_break, dG, E_top, E_th=None, absorb=True, n=NGRID):
    """Broken power law in PHOTON-INDEX convention: dN/dE ~ E^-G1 below the
    break, E^-(G1+dG) above, continuous at the break.  This is the form
    Gladstone, Roberts & Done (2009) fit to ULX spectra (their Table 6)."""
    E_th = IY.E_TH_HI if E_th is None else E_th
    g = np.linspace(np.log10(E_th), np.log10(E_top), n)
    E = 10.0 ** g
    G2 = G1 + dG
    dN = np.where(E < E_break, E ** (-G1), E_break ** (G2 - G1) * E ** (-G2))
    w = dN * np.log(10.0) * E
    N = _ngam(g, absorb=absorb)
    return float(IY._trapz(w * E, g) / IY._trapz(w * N, g))


def W_e(p, E_min, E_max, n=NGRID, exact=False):
    """CR electron injection dN/dE ~ E^-p over [E_min, E_max]."""
    g = np.linspace(np.log10(E_min), np.log10(E_max), n)
    E = 10.0 ** g
    w = E ** (-p) * np.log(10.0) * E
    N = _ne(g, exact=exact)
    return float(IY._trapz(w * E, g) / IY._trapz(w * N, g))


def W_bounds_gamma(E_lo, E_hi, absorb=True, n=NGRID):
    """W is the p(E)N(E)-weighted mean of the monoenergetic W(E) = E/N(E), so
    ANY spectrum on this support has W between these two numbers.  A rigorous
    uncertainty that needs no assumption about the spectral shape."""
    g = np.linspace(np.log10(E_lo), np.log10(E_hi), n)
    WE = (10.0 ** g) / _ngam(g, absorb=absorb)
    return float(WE.min()), float(WE.max())


def W_bounds_e(E_lo, E_hi, n=NGRID):
    g = np.linspace(np.log10(E_lo), np.log10(E_hi), n)
    WE = (10.0 ** g) / _ne(g)
    return float(WE.min()), float(WE.max())


def zeta_from_W(L_erg_s, W_eV):
    """zeta [s^-1 per H atom] from a luminosity per comoving Mpc^3."""
    return CONV * L_erg_s * P.EV_PER_ERG / (N_H * W_eV)

