"""Spectrum of the photons that an electron up-scatters from the CMB by inverse Compton (A06, A21).

Physics  [BlumenthalGould1970 Eq. 2.48, p. 243], exact for any Γ_ε, only assumption γ ≫ 1:
    dN/(dt dE1) = (2π r0² m c³ / γ) ∫ dε n(ε)/ε · F(q, Γ_ε),
    F(q, Γ) = 2q ln q + (1+2q)(1−q) + ½ (Γq)²(1−q)/(1+Γq)       (= igm_losses.inverse_compton.kernel)
    E1 = ε1/(γ m c²),  Γ_ε = 4εγ/(m c²),  q = E1/[Γ_ε (1 − E1)],  1/(4γ²) ≤ q ≤ 1   [Eqs. 2.47, 2.49, 2.51]
    CMB photons  n(ε) dε = [π² (ħc)³]^-1 ε² dε/(e^{ε/kT} − 1)                        [Eq. 2.58]
    With x = ε/kT:  n(ε) dε / ε = (kT)² x dx / [π² (ħc)³ (e^x − 1)].
    Per unit scattered-photon energy ε1:  dN/(dt dε1) = dN/(dt dE1) / (γ m c²).
Checks (tests)  Thomson limit: total scattering rate = σ_T c n_γ, n_γ = (2ζ(3)/π²)(kT/ħc)³ [each scattering emits
    one photon]; energy: γ m c² ∫ E1 dN/dE1 dE1 = −dE/dt [Eq. 2.56] = igm_losses IC loss (F_KN route, independent).
Regime  γ ≫ 1 (A21). Photons reaching the photoionization window come from γ ≳ 20 (checked in tests with the
    numbers of the problem).
Units   SI (J, s, m); energies in J inside, eV at the plotting interface only.
"""

import math

import numpy as np
from scipy import integrate

from . import _paths  # noqa: F401
from igm_losses import constants as K
from igm_losses import kinematics as KIN
from igm_losses.inverse_compton import kernel

R0 = K.r_e
PREF_CMB = 1.0 / (math.pi ** 2 * (K.hbar * K.c) ** 3)     # [π²(ħc)³]^-1  [J^-3 m^-3]
X_MAX = 700.0                                             # e^-700 underflows: no CMB photons beyond


def _x_integrand(x, E1, gam, kT):
    """Integrand in x = ε/kT of dN/(dt dE1), without the constant 2π r0² m c³/γ · (kT)² PREF_CMB."""
    G = 4.0 * x * kT * gam / K.m_e_c2
    q = E1 / (G * (1.0 - E1))
    if q > 1.0 or q < 1.0 / (4.0 * gam * gam):
        return 0.0
    return x / math.expm1(x) * kernel(q, G)


def dN_dt_dE1(E1, gam, kT):
    """dN/(dt dE1) [s^-1] for scattered energy E1 = ε1/(γ m c²) ∈ (0, 1), electron Lorentz factor gam, CMB kT [J]."""
    if not 0.0 < E1 < 1.0:
        return 0.0
    x_lo = E1 * K.m_e_c2 / (4.0 * kT * gam * (1.0 - E1))      # q = 1  (maximum-energy kinematics, Eq. 2.50)
    x_hi = gam * E1 * K.m_e_c2 / (kT * (1.0 - E1))              # q = 1/(4γ²)
    lo, hi = x_lo, min(x_hi, X_MAX)
    if lo >= hi:
        return 0.0
    pts = [p for p in (1.0, 3.0, 10.0) if lo < p < hi]
    val, _ = integrate.quad(_x_integrand, lo, hi, args=(E1, gam, kT), points=pts or None, epsabs=0.0,
                            epsrel=1e-9, limit=200)
    return 2.0 * math.pi * R0 ** 2 * K.m_e * K.c ** 3 / gam * kT ** 2 * PREF_CMB * val


def dN_dt_deps1(K_J, eps1_J, T_CMB):
    """dN/(dt dε1) [s^-1 J^-1]: photons per unit scattered energy ε1 [J] from an electron of kinetic energy K_J [J]."""
    gam = float(KIN.gamma(K_J))
    kT = K.k_B * T_CMB
    return dN_dt_dE1(eps1_J / (gam * K.m_e_c2), gam, kT) / (gam * K.m_e_c2)


def photon_rate(K_J, T_CMB, eps_lo_J, eps_hi_J):
    """Number of scattered photons per second with ε1 ∈ [eps_lo_J, eps_hi_J] [s^-1]."""
    if eps_hi_J <= eps_lo_J:
        return 0.0
    f = lambda u: dN_dt_deps1(K_J, math.exp(u), T_CMB) * math.exp(u)
    val, _ = integrate.quad(f, math.log(eps_lo_J), math.log(eps_hi_J), epsabs=0.0, epsrel=1e-7, limit=200)
    return val
