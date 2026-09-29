"""Number of H ionizations produced by an electron and its whole cascade until thermalization (objective 2).

Definitions (A19 fixed z, A20 continuous slowing down, A21–A23, A17 passive medium). For an electron of kinetic
energy E, every descendant is followed down to K_floor (A07):
    C(E)   collisional ionizations of the tree that are NOT below an IC photon
    Ph(E)  photoionizations by IC photons in the window [R_H, E_max] (A22) plus EVERYTHING the photoelectrons
           (energy E_γ − R_H, A23) produce afterwards
    N(E) = C(E) + Ph(E);   ceiling N ≤ E/R_H (each ionization costs at least R_H).
Energy integrals at fixed z (derived; dt = dE'/L_tot):
    C(E)  = ∫_{K_floor}^{E} dE' w(E') [1 + ⟨C⟩(E')]
    Ph(E) = ∫_{K_floor}^{E} dE' { w(E') ⟨Ph⟩(E') + S(E')/L_tot(E') }
    w = n_HI σ_ion v / L_tot   (ionizations per unit energy lost, [J^-1]),   ⟨g⟩ = mean over p(ε|E') (secondary.py),
    S(E') = ∫_{R_H}^{E_max} dE_γ dN/(dt dE_γ) [1 + C(E_γ − R_H) + Ph(E_γ − R_H)]   (ic_spectrum.py).
Solved bottom-up on the grid of parameters.yaml → figures.ionization_yield.grid (D22): the secondaries of E' have
ε ≤ (E' − B)/2 and the photoelectrons ≤ E_max − R_H, both below the previous grid point (grid ratio 10^(1/40) < 2,
and window photons need γ ≳ 20, i.e. E' ≳ 10 MeV ≫ E_max), so their C and Ph are already known. Trapezoid in E'.
Units: J inside, eV at the interface. E_max comes from photoionization.window() (A22, A24).
"""

import math

import numpy as np
from scipy import integrate

from . import _paths  # noqa: F401
from . import ic_spectrum, secondary
from igm_losses import constants as K
from igm_losses import ionization, losses
from igm_losses.medium import Medium

P = K._P


def _w_and_L(E_eV, z, x_e):
    m = Medium(z, x_e)
    EJ = E_eV * K.e
    r = losses.rates(EJ, z, x_e)
    L = sum(r.values())
    v = K.c * math.sqrt(1.0 - 1.0 / (1.0 + EJ / K.m_e_c2) ** 2)
    return m.n_HI * ionization.sigma(E_eV) * v / L, L


# [A23] a photoionization counts 1, plus the whole tree of its photoelectron of energy E_γ − R_H
def photon_offspring(eg_J, E_known_eV, N_known):
    """1 + N(E_γ − R_H): ionizations due to one photon of energy eg_J [J] that photoionizes H; N_known = C + Ph
    tabulated on E_known_eV (ascending, starting at K_floor); photoelectrons at or below K_floor add nothing (A20)."""
    pe = (eg_J - K.R_H) / K.e                                          # photoelectron energy [eV]
    if pe <= P["K_floor"]:
        return 1.0
    return 1.0 + float(np.interp(math.log(pe), np.log(E_known_eV), N_known))


# [A19] [A20] [A21] [A22] [A23] [A17]
def cascade(E_grid_eV, x_e, E_max_eV, z=None, photons=True):
    """Arrays C, Ph (same length as E_grid_eV, which must start at K_floor) for window [R_H, E_max_eV]."""
    z = P["z_init"] if z is None else z
    E = np.asarray(E_grid_eV, float)
    if abs(E[0] / P["K_floor"] - 1) > 1e-12:
        raise ValueError("the grid must start at K_floor")
    T_cmb = K.T_CMB0 * (1.0 + z)
    lo, hi = K.R_H, E_max_eV * K.e
    n = len(E)
    C, Ph = np.zeros(n), np.zeros(n)
    fC, fPh = np.zeros(n), np.zeros(n)            # integrands d C/dE', d Ph/dE' at the grid points [J^-1]
    floor = P["K_floor"]
    for k in range(n):
        known = slice(0, max(k, 1))
        w, L = _w_and_L(E[k], z, x_e)
        mC = secondary.mean_over(E[k], E[known], C[known], floor)
        mPh = secondary.mean_over(E[k], E[known], Ph[known], floor)
        S = 0.0
        if photons and hi > lo:
            Nk = (C + Ph)[known]

            def g(u):
                eg = math.exp(u)
                return ic_spectrum.dN_dt_deps1(E[k] * K.e, eg, T_cmb) * eg * photon_offspring(eg, E[known], Nk)
            S, _ = integrate.quad(g, math.log(lo), math.log(hi), epsabs=0.0, epsrel=1e-6, limit=200)
        fC[k] = w * (1.0 + mC)
        fPh[k] = w * mPh + S / L
        if k:
            dE = (E[k] - E[k - 1]) * K.e
            C[k] = C[k - 1] + 0.5 * (fC[k] + fC[k - 1]) * dE
            Ph[k] = Ph[k - 1] + 0.5 * (fPh[k] + fPh[k - 1]) * dE
    return C, Ph
