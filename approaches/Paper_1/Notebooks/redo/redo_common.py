"""
redo_common.py -- INDEPENDENT re-implementation of the pieces of physics that
enter Fig. 4 of manuscript_13page (ionization yield / W-value) and the
heat-per-ionization number quoted in Sect. 9.4 of manuscript.tex.

Nothing here is imported from photon_cascade.py / cascade_yield.py /
cascade_traj.py.  Only ``igm_losses`` is reused, and only for the primitives
that are not under test: physical constants, the Planck18 cosmology tables, the
seven loss rates, and the atomic cross-sections (RBEB ionization, PWB
excitation).  The IC differential spectrum, its quadrature, the photon opacity,
the cascade recursion and every accumulator are written again from scratch,
with different quadratures and a different bookkeeping, so that agreement with
the published tables is a real check and not a tautology.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import igm_losses as L                                            # noqa: E402

EV = L.EV_MKS
E0_EV = L.E0 / EV                       # electron rest energy in eV
B_H = L.THRESHOLD_EV_ION                # 13.6057 eV
KT0 = L.K_B * L.T_CMB_0                 # CMB temperature today, in J


# =====================================================================
# 1.  Inverse-Compton differential photon spectrum (Blumenthal & Gould 1970)
# =====================================================================
# Independent quadrature: composite Simpson on a fine logarithmic grid in
# x = epsilon/(kT), instead of the 128-node Gauss-Legendre rule of
# photon_cascade.py.  Nodes and weights therefore have nothing in common.
_XN = 801
_LX = np.linspace(np.log(1.0e-5), np.log(80.0), _XN)
_XG = np.exp(_LX)
_SIMP = np.ones(_XN)
_SIMP[1:-1:2] = 4.0
_SIMP[2:-1:2] = 2.0
_SIMP *= (_LX[1] - _LX[0]) / 3.0
# weight for  int n(eps)/eps deps  ->  (8 pi (kT)^2 / (h^3 c^3)) int x/(e^x-1) dx
_W_NUMBER = _SIMP * _XG * _XG / np.expm1(_XG)      # extra x from dx = x dlnx
# weight for  int n(eps) eps deps  (energy density, used for normalisation)
_W_ENERGY = _SIMP * _XG**4 / np.expm1(_XG)


def _planck_prefactor(kT):
    return 8.0 * np.pi * kT**2 / (L.PLANCK_CONSTANT_MKS**3 * L.C_LIGHT**3)


def u_cmb(z):
    """CMB energy density [J m^-3] from the same quadrature, for closure."""
    kT = KT0 * (1.0 + z)
    return _planck_prefactor(kT) * kT**2 * float(np.sum(_W_ENERGY))


def ic_dnde(K_eV, z, Eg_eV):
    """dN_gamma / (dt dE_gamma)  [s^-1 eV^-1] for one electron of energy K_eV.

    Isotropic Klein--Nishina kernel of Blumenthal & Gould (1970), Eq. (2.48),
    integrated over the Planck spectrum of the CMB at redshift z.
    """
    gamma = 1.0 + float(K_eV) / E0_EV
    kT = KT0 * (1.0 + float(z))
    eps = _XG * kT                                   # seed photon energies [J]
    Gam = 4.0 * gamma * eps / L.E0                   # Gamma = 4 gamma eps/mc^2

    Eg = np.atleast_1d(np.asarray(Eg_eV, float))
    EgJ = Eg[:, None] * EV
    den = Gam[None, :] * (gamma * L.E0 - EgJ)
    q = np.divide(EgJ, den, out=np.full_like(den, np.inf), where=den > 0.0)
    ok = (q >= 1.0 / (4.0 * gamma**2)) & (q <= 1.0)
    qs = np.clip(q, 1.0e-300, 1.0)
    Gq = Gam[None, :] * qs
    G = (2.0 * qs * np.log(qs) + (1.0 + 2.0 * qs) * (1.0 - qs)
         + 0.5 * Gq**2 * (1.0 - qs) / (1.0 + Gq))
    G = np.where(ok, np.maximum(G, 0.0), 0.0)

    pref = 3.0 * L.THOMSON_CROSS_SECTION_MKS * L.C_LIGHT / (4.0 * gamma**2)
    return pref * _planck_prefactor(kT) * (G @ _W_NUMBER) * EV   # per eV


def ic_emitted_power(K_eV, z, n=900):
    """int E_gamma dN/dE_gamma dE_gamma  [J s^-1]: total up-scattered power."""
    gamma = 1.0 + float(K_eV) / E0_EV
    kT = KT0 * (1.0 + float(z))
    lo = 1.0e-4 * kT / EV                       # far below any seed photon
    hi = 0.999999 * (gamma - 1.0 / (4.0 * gamma)) * E0_EV + kT / EV
    Eg = np.geomspace(lo, hi, n)
    dnde = ic_dnde(K_eV, z, Eg)
    return float(np.trapz(dnde * Eg, Eg)) * EV


# =====================================================================
# 2.  Photon opacity of the IGM and the free-streaming ceiling
# =====================================================================
SIGMA_PI_THRESHOLD = 6.30e-18 * 1.0e-4          # m^2, H(1s) at 13.6 eV


def sigma_pi(E_eV):
    """H(1s) photoionization cross-section [m^2] (Karzas & Latter 1961)."""
    x = np.atleast_1d(np.asarray(E_eV, float)) / B_H
    out = np.zeros_like(x)
    m = x > 1.0
    e = np.sqrt(x[m] - 1.0)
    out[m] = (SIGMA_PI_THRESHOLD * x[m] ** -4
              * np.exp(4.0 - 4.0 * np.arctan(e) / e)
              / (1.0 - np.exp(-2.0 * np.pi / e)))
    return out


def sigma_kn(E_eV):
    """Total Klein--Nishina cross-section per electron [m^2]."""
    a = np.atleast_1d(np.asarray(E_eV, float)) * EV / L.E0
    s = L.THOMSON_CROSS_SECTION_MKS
    return s * 0.75 * (((1.0 + a) / a**3)
                       * (2.0 * a * (1.0 + a) / (1.0 + 2.0 * a) - np.log1p(2.0 * a))
                       + np.log1p(2.0 * a) / (2.0 * a)
                       - (1.0 + 3.0 * a) / (1.0 + 2.0 * a) ** 2)


def photon_mfp(E_eV, z):
    """Proper mean free path [m]."""
    nH = L.n_HI(z)
    return 1.0 / (nH * sigma_pi(E_eV) + (1.0 + L.ION_FRACTION) * nH * sigma_kn(E_eV))


def hubble_length(z):
    return L.C_LIGHT / float(L.Planck18.H(z).to(L.u.s**-1).value)


def e_free(z):
    """Photon energy [eV] at which the proper mfp equals c/H(z)."""
    f = lambda lg: float(photon_mfp(10.0 ** lg, z)[0]) / hubble_length(z) - 1.0
    return 10.0 ** brentq(f, np.log10(20.0), 5.0, xtol=1e-10)


# =====================================================================
# 3.  Atomic input reused from igm_losses (not under test)
# =====================================================================
def sigma_ion(K_eV):
    """RBEB collisional ionization cross-section [m^2]."""
    K_eV = np.atleast_1d(np.asarray(K_eV, float))
    out = np.array([L.coll_ionisation_cross_section(k * EV) if k > B_H else 0.0
                    for k in K_eV])
    return out if out.size > 1 else float(out[0])


def secondary_pdf(K_eV):
    """Normalised BEB secondary-electron spectrum p(eps|K); (eps [eV], pdf)."""
    return L.secondary_pdf(K_eV)


# =====================================================================
# 4.  Cosmological transport of the IC photons
# =====================================================================
# The band model of stage2 replaces radiative transfer by a step function:
# probability one between B_H and E_max, zero outside, evaluated at the
# emission redshift.  Sect. 6.5(ii) of
# manuscript/code_anatomy_ionization_yield.pdf shows that this is the largest
# single approximation in the whole calculation, because a photon that is
# transparent when it is emitted becomes opaque as it redshifts:
# sigma_pi ~ E^-3 grows as (1+z)^-3 while the gas thins only as (1+z)^3, so
# n_HI sigma_pi is nearly constant in proper units while the horizon c/H grows.
#
# The functions below replace the step function by the exact first-interaction
# probability along the light path.  For a photon emitted with energy Eg at
# z_emit, define the proper-time opacities
#
#     d tau_pi  / dt = c n_HI(z) sigma_pi (E(z))
#     d tau_tot / dt = c n_HI(z) [sigma_pi + (1+x_e) sigma_KN](E(z))
#
# with E(z) = Eg (1+z)/(1+z_emit).  The probability that the FIRST interaction
# is a photoionization and that it happens in dt around t is
#
#     dP = (d tau_pi / dt) exp(-tau_tot(t)) dt ,
#
# so any per-absorption quantity Q (the ionization count 1 + Y_e, the
# photoelectron heat H_q, ...) is folded as  int dP Q(E(t) - B_H, z(t)).
#
# Compton scattering is treated as terminating the photon.  That is
# conservative: in reality a scattered photon continues with slightly less
# energy and a correspondingly larger sigma_pi, so the true absorbed fraction
# is a little higher than what this returns.


def absorption_kernel(z_emit, Eg_eV, z_min, n_path=192):
    """First-interaction photoionization kernel along the light path.

    Returns ``(w, Ke, zp)`` with

      w  : (nE, n_path) quadrature weights, such that ``(w * Q).sum(axis=1)``
           evaluates  int dP Q  for any Q sampled on the same nodes.
           ``w.sum(axis=1)`` is the probability that the photon is eventually
           photoionized before ``z_min``.
      Ke : (nE, n_path) photoelectron kinetic energy E(z) - B_H at each node,
           clipped at zero.
      zp : (n_path,) the redshift nodes of the path.

    The path is logarithmic in (1+z).  The quadrature is done in optical-depth
    space rather than in time, integrating exp(-tau) analytically over each
    segment and treating the rest of the integrand as linear in tau.  That is
    what keeps the scheme exact when the photon is absorbed almost
    immediately: a 500 eV photon at z = 20 has d tau/dt so large that a
    time-space trapezoid rule overshoots unity by a per cent, while this rule
    is bounded by construction (the weights are non-negative and their sum is
    1 - exp(-tau_max) times the photoionization branching ratio).
    """
    z_emit = float(z_emit)
    Eg = np.atleast_1d(np.asarray(Eg_eV, float))
    if z_min >= z_emit:
        return (np.zeros((Eg.size, 1)), np.zeros((Eg.size, 1)),
                np.array([z_emit]))

    zp = np.expm1(np.linspace(np.log1p(z_emit), np.log1p(z_min), n_path))
    tp = L.age_s(zp)                                   # increasing with time

    # redshifted photon energy on the path, (nE, n_path)
    E = Eg[:, None] * (1.0 + zp)[None, :] / (1.0 + z_emit)
    nH = L.n_HI(zp)[None, :]

    flat = E.ravel()
    k_pi = (nH * sigma_pi(flat).reshape(E.shape)) * L.C_LIGHT
    k_kn = (nH * (1.0 + L.ION_FRACTION)
            * sigma_kn(flat).reshape(E.shape)) * L.C_LIGHT
    k_tot = k_pi + k_kn
    frac = np.divide(k_pi, k_tot, out=np.zeros_like(k_pi), where=k_tot > 0.0)

    # tau_tot(t): cumulative trapezoid of k_tot dt from the emission
    dt = np.diff(tp)[None, :]
    seg = 0.5 * (k_tot[:, 1:] + k_tot[:, :-1]) * dt
    tau = np.concatenate([np.zeros((E.shape[0], 1)),
                          np.cumsum(seg, axis=1)], axis=1)

    # exact segment integrals of exp(-tau) against a linear ramp
    t1, t2 = tau[:, :-1], tau[:, 1:]
    dtau = np.maximum(t2 - t1, 0.0)
    u1, u2 = np.exp(-t1), np.exp(-t2)
    small = dtau < 1.0e-8
    ratio = np.divide(u1 - u2, dtau, out=np.zeros_like(dtau), where=~small)
    a1 = np.where(small, 0.5 * u1 * dtau, u1 - ratio)      # weight of node i
    a2 = np.where(small, 0.5 * u1 * dtau, ratio - u2)      # weight of node i+1

    w = np.zeros_like(tau)
    w[:, :-1] += a1
    w[:, 1:] += a2
    w *= frac                       # only photoionizations are counted

    Ke = np.maximum(E - B_H, 0.0)
    return w, Ke, zp
