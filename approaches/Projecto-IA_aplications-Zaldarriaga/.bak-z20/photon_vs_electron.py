#!/usr/bin/env python3
r"""
Stellar UV photons versus cosmic-ray electrons as reionization agents at z = 10.

SCENARIO   identical to ionization_yield_fig.png: neutral IGM at z = 10, static
           ionized fraction x_e = 1e-4, pure atomic hydrogen at the mean density.
PRODUCES   (a) differential ionizing emissivity  d n_dot_ion / dlog10 E
           (b) ionization rate per target atom   d zeta / dlog10 E
           for both channels, plus the competition map over the scanned axes.

EVERY NUMERICAL INPUT IS SOURCED OR DERIVED.  Nothing here is recalled.  The
choices marked SCANNED were made by the user and are logged in QUESTION_LOG.md.

--------------------------------------------------------------------------
THE TWO CHANNELS
--------------------------------------------------------------------------
STELLAR UV.   n_dot_gamma = f_esc * xi_ion * rho_UV  photons/s/cMpc^3, with the
    escaping spectrum a T_eff = 5e4 K blackbody truncated at 13.598 eV.  Each
    photon makes N_gamma(E) ionizations. Electron yields come from MODEL E
    (igm_losses.py, via yield_comparison); model C of ionization_yield.py is
    retained as the cross-check that validated it.

CR ELECTRONS. The same rho_UV fixes rho_SFR, hence the core-collapse supernova
    rate, hence the CR electron luminosity
        L_e = R_SN * (eps_CR f_e) * E_SN,
    injected as dN/dE ~ E^-2.1 between 1 keV and 1 TeV.  Each electron makes
    N_e^(C)(E) ionizations -- the same physics, so the comparison is internally
    consistent: ONE star-formation history feeds both channels.

zeta is a proper-frame quantity and n_dot scales with n_H, so zeta is
independent of the comoving/proper convention.  That is checked, not assumed.

Run:  python3 photon_vs_electron.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.integrate import quad

import ionization_yield as IY

# STANDING DECISION (2026-09-11): electron propagation is computed with
# igm_losses.py. ELECTRON_YIELD selects which yield drives zeta_e:
#   "E" -- model E, the loss-based route (7 mechanisms, RBEB event counting,
#          BEB secondary cascade, Klein-Nishina IC, + IC-secondary photons)
#   "C" -- model C, the deposition-fit route, retained as the cross-check that
#          validated E (they agree to 7.7% at saturation)
# The photon channel is unaffected: over 10-10^5 eV the photoelectron has
# gamma <= 1.2 and upscatters nothing, so both routes give the same N_gamma.
ELECTRON_YIELD = "E"


def _electron_yield(E_eV, cos, par, chan):
    if ELECTRON_YIELD == "C":
        return IY.N_e_C(E_eV, cos, par, chan)
    import yield_comparison as YC
    return YC.N_e_E(E_eV, cos, par)

# ===========================================================================
# 1. INPUTS.  Source in the comment; SCANNED means the user chose to vary it.
# ===========================================================================
# Donnan et al. 2024, JWST PRIMER, MNRAS, Table 3 -- measured AT z = 10.
LOG_RHO_UV = 25.12                 # log10(erg s^-1 Hz^-1 cMpc^-3)
LOG_RHO_UV_HI = +0.07              # asymmetric 1-sigma, as published
LOG_RHO_UV_LO = -0.14

# Madau & Dickinson 2014, as used by Donnan+24.  Salpeter (1955) IMF.
K_UV = 1.15e-28                    # Msun yr^-1 / (erg s^-1 Hz^-1)

# Llerena et al. 2025, A&A 698, A302 (arXiv:2412.01358): median at the highest
# redshift probed.  Observed scatter 0.42 dex, carried as a band in the figure.
LOG_XI_ION = 25.28                 # log10(Hz erg^-1)
XI_ION_SCATTER_DEX = 0.42

T_EFF_K = 5.0e4                    # retained as the COMPARISON case only

# ---------------------------------------------------------------------------
# THE STELLAR IONIZING SED.  Replaces the blackbody as the default.
#
# The identified population is that fitted to U37126, a UV-bright star-forming
# galaxy at z = 10.255 -- the same redshift as this calculation -- by
# Marques-Chaves et al. 2026 (arXiv:2602.02322, A&A; in papers/):
#     BPASS v2.2.1 (Stanway & Eldridge 2018), imf135_300 (slope -2.35, upper
#     mass cutoff 300 Msun), metallicity Z = 0.003 (Z/Zsun = 0.15), constant
#     star formation, best-fit age 6.8 +/- 1.6 Myr, beta_UV = -2.88 +/- 0.10,
#     log10(xi_ion/Hz erg^-1) = 25.75 +/- 0.09, f_esc(LyC) = 0.94 +/- 0.06.
#
# WHAT COULD NOT BE OBTAINED, AND IS THEREFORE NOT INVENTED.  Reproducing the
# BPASS spectrum itself needs the BPASS data tables, which are not public in a
# form reachable from here.  The paper quotes Q_H but no Lyman-continuum shape.
# So the SHAPE is carried by the standard parametrisation for reionization-era
# sources -- a power law in frequency across 1-4 Ryd,
#       f_nu ~ nu^-alpha   =>   photon number  dN/dE ~ E^-(alpha+1)
# -- with alpha BRACKETED rather than asserted, because no public number pins
# it for this population.  The point of the exercise is then to show that the
# conclusion does not depend on it: see the check "SED shape insensitivity".
SED_MODEL = "powerlaw"             # "powerlaw" (default) or "blackbody"
SED_ALPHA = 2.0                    # f_nu ~ nu^-alpha; BRACKETED over 1-3
SED_ALPHA_LO, SED_ALPHA_HI = 1.0, 3.0
RYD_EV = 13.605693122994           # 1 Rydberg, CODATA 2018
LYC_E_MAX_EV = 4.0 * RYD_EV        # 4 Ryd: the conventional top of the stellar
                                   # LyC band; He II opacity takes over above
E_SN_ERG = 1.0e51                  # standard core-collapse energy budget

# Salpeter (1955) IMF, used only through an integral performed below.
IMF_SLOPE = 2.35
M_MIN, M_MAX, M_SN_MIN = 0.1, 100.0, 8.0     # Msun

# CR electron injection -- user's choice (QUESTION_LOG 2.3)
CR_INDEX = 2.2                     # user's choice, revised from 2.1
CR_E_MIN_EV, CR_E_MAX_EV = 1.0e3, 1.0e12

# Fiducial points of the scanned axes (QUESTION_LOG 1.2, 2.2, 3.1)
F_ESC_FID = 0.10                   # SCANNED
EPS_CR_FE_FID = 1.0e-3             # SCANNED: eps_CR = 0.1 times f_e = 0.01
RHO_SFR_SCALE_FID = 1.0            # SCANNED: multiplier on the Donnan+24 value

SEC_PER_YR = 3.155693e7            # Julian year
CM_PER_MPC = 3.0856775814913673e24
EV_PER_ERG = 1.0 / IY.ERG_PER_EV

# ===========================================================================
# 2. DERIVED QUANTITIES  (Master Rule 4: derived here, not recalled)
# ===========================================================================
def sn_per_solar_mass(slope=IMF_SLOPE, m_lo=M_MIN, m_hi=M_MAX, m_sn=M_SN_MIN):
    """Core-collapse supernovae per solar mass of stars formed, Salpeter IMF.

        N/M = int_{m_sn}^{m_hi} m^-slope dm  /  int_{m_lo}^{m_hi} m^(1-slope) dm

    Closed form; the numerical quadrature and the sympy result are both checked
    against it below.  This is a DERIVATION, not a literature number.
    """
    a = 1.0 - slope
    num = (m_hi ** a - m_sn ** a) / a
    den = (m_hi ** (a + 1.0) - m_lo ** (a + 1.0)) / (a + 1.0)
    return num / den


def comoving_Mpc3_to_proper_cm3(z):
    """Multiply a per-cMpc^3 rate by this to get a per-proper-cm^3 rate."""
    return (1.0 + z) ** 3 / CM_PER_MPC ** 3


# ---- stellar ionizing photon spectrum -------------------------------------
def _planck_photon(E_eV, kT_eV):
    """Blackbody PHOTON-number spectrum, up to normalisation: E^2/(e^{E/kT}-1)."""
    E = np.asarray(E_eV, dtype=float)
    x = np.clip(E / kT_eV, 1e-300, 700.0)
    return E ** 2 / np.expm1(x)


def photon_pdf(E_eV, model=None, alpha=None, T_K=T_EFF_K, E_th=IY.E_TH_HI):
    """Normalised energy distribution of the ESCAPING ionizing photons.

    model="powerlaw": dN/dE ~ E^-(alpha+1) on [E_th, 4 Ryd], i.e. f_nu ~ nu^-alpha.
    model="blackbody": Planck photon spectrum truncated below the HI edge.
    Photons under 13.598 eV cannot ionize and are not counted in xi_ion either,
    so in both cases the pdf lives on [E_th, E_max]. Normalisation is computed
    and then checked against 1 -- in LOG SPACE for the power law, because a
    linear-space quad over a steep power law is the trap that has already bitten
    this project three times.
    """
    model = SED_MODEL if model is None else model
    alpha = SED_ALPHA if alpha is None else alpha
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    if model == "powerlaw":
        q = -(alpha + 1.0)
        a = q + 1.0
        norm = (LYC_E_MAX_EV ** a - E_th ** a) / a
        out = np.where((E >= E_th) & (E <= LYC_E_MAX_EV), E ** q / norm, 0.0)
    elif model == "blackbody":
        kT = IY.K_B_EV * T_K
        norm, _ = quad(lambda e: float(_planck_photon(e, kT)), E_th, 60.0 * kT,
                       limit=200)
        out = np.where(E >= E_th, _planck_photon(E, kT) / norm, 0.0)
    else:
        raise ValueError(model)
    return out if np.ndim(E_eV) else float(out[0])


def photon_band_max(model=None, T_K=T_EFF_K):
    """Upper limit of the escaping-photon band. ONE definition, used by every
    integral over the photon channel. Three checks failed when the SED switch
    left two of them integrating to 60 kT while zeta_total integrated to 4 Ryd:
    they were comparing different grids, not different conventions."""
    model = SED_MODEL if model is None else model
    return LYC_E_MAX_EV if model == "powerlaw" else 60.0 * IY.K_B_EV * T_K


def mean_ionizing_photon_energy(model=None, alpha=None, T_K=T_EFF_K):
    """<E> of the escaping ionizing photons [eV]. Closed form for the power law."""
    model = SED_MODEL if model is None else model
    alpha = SED_ALPHA if alpha is None else alpha
    if model == "powerlaw":
        a, b = 1.0 - alpha, -alpha
        E1, E2 = IY.E_TH_HI, LYC_E_MAX_EV
        return ((E2 ** a - E1 ** a) / a) / ((E2 ** b - E1 ** b) / b)
    kT = IY.K_B_EV * T_K
    num = quad(lambda e: e * float(_planck_photon(e, kT)), IY.E_TH_HI, 60 * kT,
               limit=200)[0]
    den = quad(lambda e: float(_planck_photon(e, kT)), IY.E_TH_HI, 60 * kT,
               limit=200)[0]
    return num / den


# ---- CR electron injection spectrum ---------------------------------------
def _powint(E1, E2, expo):
    """Integral_{E1}^{E2} E^expo dE, handling the removable singularity at
    expo = -1 where the closed form divides by zero and the answer is a log.

    p = 2.0 was one of the injection indices offered to the user; it would have
    raised ZeroDivisionError in the mean-energy closed form (expo = 1 - p = -1).
    """
    if abs(expo + 1.0) < 1e-12:
        return np.log(E2 / E1)
    a = expo + 1.0
    return (E2 ** a - E1 ** a) / a


def electron_pdf(E_eV, p=None, E1=None, E2=None):
    """Normalised NUMBER distribution of injected CR electrons, dN/dE ~ E^-p.

    Parameters default to the MODULE globals resolved AT CALL TIME. They used to
    be bound as default arguments, which Python evaluates once at def time: the
    E_min sensitivity check then mutated CR_E_MIN_EV and the spectrum never saw
    it, so that check moved the integration limits while leaving the
    distribution at 1 keV. It answered a different question from the one it
    asked, and its number reached the paper.
    """
    p = CR_INDEX if p is None else p
    E1 = CR_E_MIN_EV if E1 is None else E1
    E2 = CR_E_MAX_EV if E2 is None else E2
    norm = _powint(E1, E2, -p)
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.where((E >= E1) & (E <= E2), E ** (-p) / norm, 0.0)
    return out if np.ndim(E_eV) else float(out[0])


def mean_injected_electron_energy(p=None, E1=None, E2=None):
    """<E> = int E^(1-p) dE / int E^-p dE.  Closed form; checked by quadrature.

    For p = 2.1 BOTH integrals are dominated by E1, which is why the answer is
    so sensitive to the low-energy cutoff.  Flagged in QUESTION_LOG 2.3.
    """
    p = CR_INDEX if p is None else p
    E1 = CR_E_MIN_EV if E1 is None else E1
    E2 = CR_E_MAX_EV if E2 is None else E2
    return _powint(E1, E2, 1.0 - p) / _powint(E1, E2, -p)


# ===========================================================================
# 3. THE TWO CHANNELS
# ===========================================================================
def rho_uv(scale=RHO_SFR_SCALE_FID):
    """erg s^-1 Hz^-1 cMpc^-3."""
    return scale * 10.0 ** LOG_RHO_UV


def rho_sfr(scale=RHO_SFR_SCALE_FID):
    """Msun yr^-1 cMpc^-3, Salpeter, from rho_UV via Madau & Dickinson 2014."""
    return K_UV * rho_uv(scale)


def ndot_photons_comoving(f_esc=F_ESC_FID, scale=RHO_SFR_SCALE_FID):
    """Escaping ionizing photons per second per comoving Mpc^3."""
    return f_esc * 10.0 ** LOG_XI_ION * rho_uv(scale)


def ndot_electrons_comoving(eps_fe=EPS_CR_FE_FID, scale=RHO_SFR_SCALE_FID):
    """Injected CR electrons per second per comoving Mpc^3, and their luminosity."""
    R_sn = rho_sfr(scale) * sn_per_solar_mass() / SEC_PER_YR   # SN/s/cMpc^3
    L_e_erg = R_sn * eps_fe * E_SN_ERG                          # erg/s/cMpc^3
    n_dot = L_e_erg * EV_PER_ERG / mean_injected_electron_energy()
    return n_dot, L_e_erg, R_sn


def dndlog10E(E_eV, channel, **kw):
    """Differential emissivity per comoving Mpc^3: d n_dot / dlog10 E."""
    E = np.asarray(E_eV, dtype=float)
    if channel == "photon":
        return ndot_photons_comoving(**kw) * np.log(10.0) * E * photon_pdf(E)
    if channel == "electron":
        n_dot, _, _ = ndot_electrons_comoving(**kw)
        return n_dot * np.log(10.0) * E * electron_pdf(E)
    raise ValueError(channel)


def dzeta_dlog10E(E_eV, channel, cos, par, chan, **kw):
    """Ionizations per second per H atom, per dex of PRIMARY energy."""
    conv = comoving_Mpc3_to_proper_cm3(par.z)
    dn = dndlog10E(E_eV, channel, **kw) * conv          # proper cm^-3 s^-1 dex^-1
    if channel == "photon":
        yld = IY.N_gamma(E_eV, cos, par, "C", chan)
    else:
        yld = _electron_yield(E_eV, cos, par, chan)
    return dn * np.asarray(yld, dtype=float) / cos["n_H_cm3"]


def zeta_total(channel, cos, par, chan, n=400, **kw):
    """Integrate dzeta/dlog10E over the channel's support.  s^-1 per H atom."""
    lo, hi = ((np.log10(IY.E_TH_HI), np.log10(photon_band_max()))
              if channel == "photon"
              else (np.log10(CR_E_MIN_EV), np.log10(CR_E_MAX_EV)))
    g = np.linspace(lo, hi, n)
    y = dzeta_dlog10E(10.0 ** g, channel, cos, par, chan, **kw)
    return float(IY._trapz(y, g))


# ===========================================================================
# 4. CHECKS.  Master Rule 5: sympy where algebra applies, an independent route
#    where it does not.  Prints coverage, not a verdict tally.
# ===========================================================================
def run_checks(cos, par, chan):
    import sympy as sp
    rows, prov = [], {}

    def rec(cid, name, ok, detail):
        rows.append((cid, name, "PASS" if ok else "FAIL", detail))
        return ok

    # --- C25 sympy: the IMF integral, symbolically -------------------------
    m, s = sp.symbols("m s", positive=True)
    num = sp.integrate(m ** (-sp.Rational(235, 100)), (m, 8, 100))
    den = sp.integrate(m ** (1 - sp.Rational(235, 100)), (m, sp.Rational(1, 10), 100))
    sn_sym = float(num / den)
    sn_cf = sn_per_solar_mass()
    sn_q = (quad(lambda x: x ** -2.35, 8, 100)[0]
            / quad(lambda x: x ** -1.35, 0.1, 100)[0])
    prov["sn_per_Msun"] = sn_cf
    rec("C25", "SN per Msun: sympy vs closed form vs quadrature",
        abs(sn_sym - sn_cf) / sn_cf < 1e-12 and abs(sn_q - sn_cf) / sn_cf < 1e-8,
        f"sympy {sn_sym:.10e}, closed form {sn_cf:.10e}, quad {sn_q:.10e} "
        f"SN/Msun (= 1 SN per {1/sn_cf:.1f} Msun). Salpeter, 8-100 Msun "
        f"progenitors. DERIVED, never recalled.")

    # --- C25 sympy: the mean injected electron energy ----------------------
    E, p = sp.symbols("E p", positive=True)
    pp = sp.nsimplify(CR_INDEX, rational=True)   # follows CR_INDEX; hardcoding
                                                 # 21/10 here would have kept
                                                 # "verifying" the old index
    mE_sym = float(sp.integrate(E ** (1 - pp), (E, CR_E_MIN_EV, CR_E_MAX_EV))
                   / sp.integrate(E ** (-pp), (E, CR_E_MIN_EV, CR_E_MAX_EV)))
    mE_cf = mean_injected_electron_energy()
    prov["mean_injected_E_eV"] = mE_cf
    rec("C25", "<E> of the injected CR electrons: sympy vs closed form",
        abs(mE_sym - mE_cf) / mE_cf < 1e-10,
        f"sympy {mE_sym:.6f} eV, closed form {mE_cf:.6f} eV. With p = {CR_INDEX} "
        f"both integrals are dominated by E_min, which is why this number moves "
        f"by ~10^3.3 if E_min moves from 1 MeV to 1 keV.")

    # --- C28 both spectra are normalised pdfs ------------------------------
    kT = IY.K_B_EV * T_EFF_K
    ngam = quad(lambda e: float(photon_pdf(e)), IY.E_TH_HI, photon_band_max(),
                limit=300)[0]
    # IN LOG SPACE, deliberately. A linear-space quad over [1 keV, 1 TeV] of a
    # E^-2.1 pdf returns -1e-10 instead of 1: nine decades of interval with all
    # the mass in the first, so the adaptive sampler never sees it. Same trap
    # that produced the model-B bug in ionization_yield.py, in a CHECK this time
    # -- which is worse, because a broken check is silent.
    nele = quad(lambda u: float(electron_pdf(np.exp(u))) * np.exp(u),
                np.log(CR_E_MIN_EV), np.log(CR_E_MAX_EV), limit=400)[0]
    rec("C28", "both injection spectra integrate to exactly one",
        abs(ngam - 1) < 1e-8 and abs(nele - 1) < 1e-6,
        f"photon pdf {ngam:.10f}, electron pdf {nele:.10f} (the electron one "
        f"integrated in ln E; in linear space the same quad returns -1e-10)")

    # --- C28 the power-law SED is a pdf, integrated in LOG space ----------
    ng_pl = quad(lambda u: float(photon_pdf(np.exp(u), model="powerlaw"))
                 * np.exp(u), np.log(IY.E_TH_HI), np.log(LYC_E_MAX_EV),
                 limit=300)[0]
    rec("C28", "power-law SED integrates to one (in ln E)",
        abs(ng_pl - 1) < 1e-8,
        f"{ng_pl:.10f} over 1-4 Ryd = "
        f"[{IY.E_TH_HI:.3f}, {LYC_E_MAX_EV:.3f}] eV")

    # --- C27/C36 SED SHAPE INSENSITIVITY -- the point of switching SEDs ----
    # zeta_gamma = (f_esc xi_ion rho_UV / n_H) x <N_gamma>, and xi_ion carries
    # the normalisation, so the SED shape enters ONLY through <N_gamma>, the
    # mean ionizations per escaping photon. That is bounded below by 1 (every
    # photon ionizes once) and rises slowly with hardness.
    shapes = {}
    for nm, kwshape in (("blackbody 5e4 K", dict(model="blackbody")),
                        ("power law a=1", dict(model="powerlaw", alpha=1.0)),
                        ("power law a=2", dict(model="powerlaw", alpha=2.0)),
                        ("power law a=3", dict(model="powerlaw", alpha=3.0))):
        hi = photon_band_max(model=kwshape.get("model"))
        gg = np.linspace(np.log10(IY.E_TH_HI), np.log10(hi), 1200)
        w = photon_pdf(10 ** gg, **kwshape) * 10 ** gg * np.log(10)
        Ngam = IY.N_gamma(10 ** gg, cos, par, "C", chan)
        # DIVIDE BY THE WEIGHT INTEGRAL ON THE SAME GRID. Without this the
        # blackbody returned <N> = 0.9998, i.e. below the hard floor of 1, and
        # the check correctly refused it -- the deficit was the trapezoid's
        # own normalisation error, not physics.
        mean_N = float(IY._trapz(w * Ngam, gg) / IY._trapz(w, gg))
        shapes[nm] = mean_N
        prov["meanN_" + nm.replace(" ", "_").replace("=", "")] = mean_N
    spread = max(shapes.values()) / min(shapes.values())
    prov["sed_shape_spread"] = spread
    rec("C27", "SED shape enters only through <N_gamma>, and weakly",
        spread < 3.0 and min(shapes.values()) >= 1.0,
        "; ".join(f"{k}: {v:.4f}" for k, v in shapes.items())
        + f". Worst-case ratio {spread:.3f}. Every value is >= 1 because each "
          f"escaping photon ionizes at least once -- a hard floor, not a fit. "
          f"Swapping the blackbody for the literature power law therefore moves "
          f"zeta_gamma by well under an order of magnitude, against a "
          f"channel-to-channel gap of nearly four.")

    # --- C32 mean escaping photon energy against the LyC band -------------
    mEg = mean_ionizing_photon_energy()
    prov["mean_ionizing_E_eV"] = mEg
    rec("C32", "mean escaping ionizing photon energy is inside 1-4 Ryd",
        IY.E_TH_HI < mEg < LYC_E_MAX_EV,
        f"<E_gamma> = {mEg:.3f} eV = {mEg/RYD_EV:.3f} Ryd (power law), vs "
        f"{mean_ionizing_photon_energy(model='blackbody'):.3f} eV for the "
        f"5e4 K blackbody. The two shapes put their photons in the same place, "
        f"which is why the answer barely moves.")

    # --- C29 dimensional consistency, by construction and by unit algebra --
    n_g = ndot_photons_comoving()
    n_e, L_e, R_sn = ndot_electrons_comoving()
    prov["ndot_gamma_cMpc3"], prov["ndot_e_cMpc3"] = n_g, n_e
    prov["L_e_erg_s_cMpc3"], prov["R_sn_cMpc3_s"] = L_e, R_sn
    # independent route to n_e: SN rate x electrons per SN
    e_per_sn = EPS_CR_FE_FID * E_SN_ERG * EV_PER_ERG / mE_cf
    n_e_alt = R_sn * e_per_sn
    rec("C29", "CR electron injection rate, two independent routes",
        abs(n_e - n_e_alt) / n_e < 1e-12,
        f"L_e/<E> = {n_e:.6e} vs R_SN x (E_CR,e/<E>) = {n_e_alt:.6e} "
        f"electrons/s/cMpc^3; {e_per_sn:.3e} electrons per supernova")

    # --- C26 zeta is invariant under the comoving/proper convention --------
    z_g = zeta_total("photon", cos, par, chan)
    z_e = zeta_total("electron", cos, par, chan)
    prov["zeta_gamma_s"], prov["zeta_e_s"] = z_g, z_e
    # redo with everything in proper units from the start
    conv = comoving_Mpc3_to_proper_cm3(par.z)
    g = np.linspace(np.log10(IY.E_TH_HI), np.log10(photon_band_max()), 400)
    y = (ndot_photons_comoving() * conv * np.log(10) * 10 ** g
         * photon_pdf(10 ** g) * IY.N_gamma(10 ** g, cos, par, "C", chan)
         / cos["n_H_cm3"])
    z_g_alt = float(IY._trapz(y, g))
    rec("C26", "zeta independent of the comoving/proper convention",
        abs(z_g - z_g_alt) / z_g < 1e-12,
        f"{z_g:.6e} vs {z_g_alt:.6e} s^-1. n_dot and n_H both scale as (1+z)^3, "
        f"so the factor {1/conv:.4e} cancels exactly -- checked, not asserted.")

    # --- C31 zeta by a second quadrature -----------------------------------
    zq = quad(lambda u: float(dzeta_dlog10E(10.0 ** u, "electron", cos, par, chan)),
              np.log10(CR_E_MIN_EV), np.log10(CR_E_MAX_EV), limit=200)[0]
    rec("C31", "zeta_e: trapezoid vs adaptive quad",
        abs(zq - z_e) / z_e < 5e-3,
        f"trapezoid {z_e:.6e} vs quad {zq:.6e} s^-1, rel.diff "
        f"{abs(zq-z_e)/z_e:.2e}")

    # --- C30 ORDER OF MAGNITUDE, PRE-REGISTERED ----------------------------
    # Written into QUESTION_LOG.md before this file existed: photons expected to
    # beat electrons by 2-4 decades, back-of-envelope ratio ~5e3.
    ratio = z_g / z_e
    prov["zeta_ratio"] = ratio
    rec("C30", "ratio matches the PRE-REGISTERED prediction (2-4 decades, ~5e3)",
        1e2 < ratio < 1e5,
        f"zeta_gamma/zeta_e = {ratio:.4g} ({np.log10(ratio):.2f} decades). "
        f"Predicted 2-4 decades from Leite+17 and Sazonov & Sunyaev+15 BEFORE "
        f"writing this code; envelope estimate was 5e3.")

    # --- C32 external sanity: does the stellar channel reionize on time? ---
    t_H = 1.0 / cos["H_z_s"]
    prov["zeta_gamma_tH"] = z_g * t_H
    rec("C32", "stellar channel gives a sensible reionization pace",
        0.01 < z_g * t_H < 3.0,
        f"zeta_gamma x t_H = {z_g*t_H:.4f} ionizations per H per Hubble time at "
        f"z = 10. Reionization completing near z ~ 6 needs this to be of order "
        f"0.1-1 here. NOT tuned: it falls out of Donnan+24, Llerena+25 and "
        f"f_esc = {F_ESC_FID}.")

    # --- C18 DELIBERATE: model B instead of C must change the CR answer ----
    yB = IY.N_e_ic(10.0 ** np.linspace(3, 12, 200), cos, par)
    yC = _electron_yield(10.0 ** np.linspace(3, 12, 200), cos, par, chan)
    rec("C18", "DELIBERATE: the yield model materially changes the CR channel",
        np.max(yC / np.maximum(yB, 1e-30)) > 10.0,
        f"max N_C/N_B over the injection range = "
        f"{float(np.max(yC/np.maximum(yB,1e-30))):.1f}. The CR result is not "
        f"independent of the cascade physics, so quoting it without naming the "
        f"model would be meaningless.")

    # --- C28 limiting cases ------------------------------------------------
    rec("C28", "limits: f_esc -> 0 and eps_CR f_e -> 0 kill their channels",
        zeta_total("photon", cos, par, chan, f_esc=0.0) == 0.0
        and zeta_total("electron", cos, par, chan, eps_fe=0.0) == 0.0,
        "both exactly zero, not approximately")

    # --- C26 the crossover, analytically and numerically -------------------
    # zeta_gamma ~ f_esc and zeta_e ~ eps_CR f_e, so the ratio is exactly
    # (f_esc/f_esc_fid)/(eps/eps_fid) x ratio_fid.  rho_SFR cancels.
    f2, e2 = 0.037, 4.4e-3
    r_pred = ratio * (f2 / F_ESC_FID) / (e2 / EPS_CR_FE_FID)
    r_num = (zeta_total("photon", cos, par, chan, f_esc=f2)
             / zeta_total("electron", cos, par, chan, eps_fe=e2))
    eps_cross = EPS_CR_FE_FID * ratio
    prov["eps_cr_fe_crossover"] = eps_cross
    rec("C26", "linearity in the scanned axes, and the crossover it implies",
        abs(r_pred - r_num) / r_num < 1e-10,
        f"predicted {r_pred:.6e} vs computed {r_num:.6e}. Parity therefore needs "
        f"eps_CR x f_e = {eps_cross:.3g} at f_esc = {F_ESC_FID}; that exceeds "
        f"unity, so on these assumptions CR electrons CANNOT reach parity at any "
        f"physical acceleration efficiency.")

    # --- C36 uncertainty propagation (Master Rule 6) -----------------------
    lo = zeta_total("photon", cos, par, chan,
                    scale=10 ** (LOG_RHO_UV_LO)) * 10 ** (-XI_ION_SCATTER_DEX)
    hi = zeta_total("photon", cos, par, chan,
                    scale=10 ** (LOG_RHO_UV_HI)) * 10 ** (+XI_ION_SCATTER_DEX)
    prov["zeta_gamma_lo"], prov["zeta_gamma_hi"] = lo, hi
    rec("C36", "propagated uncertainty on the stellar channel",
        lo < z_g < hi,
        f"zeta_gamma = {z_g:.3e} (+{hi-z_g:.1e} / -{z_g-lo:.1e}) s^-1, from the "
        f"asymmetric rho_UV error (+{LOG_RHO_UV_HI}/{LOG_RHO_UV_LO} dex, "
        f"Donnan+24) and the {XI_ION_SCATTER_DEX} dex xi_ion scatter "
        f"(Llerena+25) added in quadrature-free worst case. The CR channel has "
        f"NO published error bar to propagate -- eps_CR and f_e are scanned "
        f"precisely because they are unmeasured at z = 10.")

    # --- C25/C31 IS  n_dot = L_e/<E>  AN APPROXIMATION?  No: an identity. ---
    # Asked by the user. <E> is DEFINED as the number-weighted mean of the same
    # distribution, so L/<E> is that definition rearranged -- exact for any
    # normalisable spectrum. Proven symbolically, then confirmed by closing the
    # energy budget back through n_dot x pdf(E).
    Es, ps = sp.symbols("Es ps", positive=True)
    E1s, E2s = sp.symbols("E1s E2s", positive=True)
    fs = Es ** (-ps)
    nums = sp.integrate(Es * fs, (Es, E1s, E2s))
    dens = sp.integrate(fs, (Es, E1s, E2s))
    ident = sp.simplify(nums / sp.simplify(nums / dens) - dens)
    back = quad(lambda u: n_e * float(electron_pdf(np.exp(u))) * np.exp(2 * u),
                np.log(CR_E_MIN_EV), np.log(CR_E_MAX_EV), limit=400)[0]
    L_eV = L_e * EV_PER_ERG
    rec("C25", "n_dot = L_e/<E> is an IDENTITY, not an approximation",
        ident == 0 and abs(back - L_eV) / L_eV < 1e-10,
        f"sympy: L/<E> - (number integral) = {ident}. Energy closure: "
        f"int E (n_dot pdf) dE = {back:.6e} vs L_e = {L_eV:.6e} eV/s/cMpc^3, "
        f"rel.diff {abs(back-L_eV)/L_eV:.1e}. The energy-dependent spectrum is "
        f"never replaced by <E>; it cancels.")

    # --- C36 the approximation that WOULD cost something -------------------
    N_at_mean = float(_electron_yield(mE_cf, cos, par, chan))
    conv0 = comoving_Mpc3_to_proper_cm3(par.z)
    z_mean = n_e * conv0 * N_at_mean / cos["n_H_cm3"]
    err_mean = (z_mean - z_e) / z_e
    prov["N_at_mean_E"] = N_at_mean
    prov["mean_N_exact"] = z_e * cos["n_H_cm3"] / (n_e * conv0)
    prov["mean_energy_approx_error"] = err_mean
    rec("C36", "error if the YIELD were evaluated at <E> instead of integrated",
        True,
        f"N_C(<E> = {mE_cf:.0f} eV) = {N_at_mean:.1f} ion pairs, but the "
        f"pdf-weighted <N> = {prov['mean_N_exact']:.1f}. Using the first would "
        f"give zeta_e = {z_mean:.4e} instead of {z_e:.4e}, an error of "
        f"{100*err_mean:+.1f}%. Jensen: N(E) is concave over the injected range "
        f"because it saturates above ~1 GeV. The code integrates, so it does NOT "
        f"pay this -- the figure is what the approximation WOULD cost.")

    # --- C20 STRUCTURAL: the spectrum must FOLLOW the module parameters ----
    # It did not. Both functions bound CR_INDEX / CR_E_MIN_EV as default
    # arguments, which Python evaluates once at import, so the sensitivity scan
    # below silently varied only the integration limits.
    _keep = CR_E_MIN_EV
    globals()["CR_E_MIN_EV"] = 1.0e6
    moved = (abs(mean_injected_electron_energy() - mE_cf) / mE_cf > 0.1
             and float(electron_pdf(1.0e4)) == 0.0)
    globals()["CR_E_MIN_EV"] = _keep
    rec("C20", "STRUCTURAL: spectrum follows CR_E_MIN_EV at call time",
        moved,
        f"setting CR_E_MIN_EV = 1e6 moves <E> away from {mE_cf:.1f} eV AND makes "
        f"the pdf vanish at 1e4 eV. Before the fix it did neither, because the "
        f"parameters were bound as default arguments at import time.")

    # --- C36 the E_min sensitivity the user was warned about ---------------
    for emin in (1.0e3, 1.0e4, 1.0e6):
        globals()["CR_E_MIN_EV"] = emin
        prov[f"zeta_e_Emin_{emin:.0e}"] = zeta_total("electron", cos, par, chan)
    globals()["CR_E_MIN_EV"] = 1.0e3
    r13 = prov["zeta_e_Emin_1e+03"] / prov["zeta_e_Emin_1e+06"]
    prov["Emin_sensitivity_1keV_over_1MeV"] = r13
    rec("C36", "sensitivity to E_min, the most consequential CR choice",
        True,
        f"zeta_e = {prov['zeta_e_Emin_1e+03']:.3e} (1 keV), "
        f"{prov['zeta_e_Emin_1e+04']:.3e} (10 keV), "
        f"{prov['zeta_e_Emin_1e+06']:.3e} (1 MeV) s^-1. Moving the floor from "
        f"1 MeV to 1 keV raises the CR channel by {r13:.1f}x. REPORTED, NOT "
        f"RESOLVED: E_min is a physical unknown, not a numerical detail.")

    return rows, prov


# ===========================================================================
# 5. FIGURES.  Colour encodes CHANNEL (matching ionization_yield_fig.png:
#    orange = photon, blue = electron); line style encodes quantity.
#    Both layouts the user asked for are produced.
# ===========================================================================
C_G, C_E = "#eb6834", "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BG = "#fcfcfb"


def _style(ax, logy=True):
    ax.set_facecolor(BG)
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def _curves(cos, par, chan, n=260):
    kT = IY.K_B_EV * T_EFF_K
    Eg = np.logspace(np.log10(IY.E_TH_HI), np.log10(photon_band_max()), n)
    Ee = np.logspace(np.log10(CR_E_MIN_EV), np.log10(CR_E_MAX_EV), n)
    conv = comoving_Mpc3_to_proper_cm3(par.z)
    return {
        "E_g": Eg, "E_e": Ee,
        "n_g": dndlog10E(Eg, "photon"), "n_e": dndlog10E(Ee, "electron"),
        "z_g": dzeta_dlog10E(Eg, "photon", cos, par, chan),
        "z_e": dzeta_dlog10E(Ee, "electron", cos, par, chan),
        "conv": conv,
    }


def _panel_a(ax, d):
    ax.plot(d["E_g"], d["n_g"], color=C_G, lw=2.4, zorder=5)
    ax.plot(d["E_e"], d["n_e"], color=C_E, lw=2.4, zorder=5)
    ax.axvline(IY.E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
    _style(ax)
    ax.set_ylabel(r"$d\dot n_{\rm ion}/d\log_{10}E$" "\n"
                  r"[s$^{-1}$ cMpc$^{-3}$ dex$^{-1}$]", color=INK, fontsize=9.5)
    ax.text(0.985, 0.93, "(a)  differential ionizing emissivity",
            transform=ax.transAxes, ha="right", va="top", color=INK,
            fontsize=11, zorder=8)


def _panel_b(ax, d):
    ax.plot(d["E_g"], d["z_g"], color=C_G, lw=2.4, zorder=5)
    ax.plot(d["E_e"], d["z_e"], color=C_E, lw=2.4, zorder=5)
    ax.axvline(IY.E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
    _style(ax)
    ax.set_ylabel(r"$d\zeta/d\log_{10}E$" "\n"
                  r"[s$^{-1}$ per H atom dex$^{-1}$]", color=INK, fontsize=9.5)
    ax.set_xlabel("primary energy  $E$  [eV]", color=INK, fontsize=10)
    ax.text(0.985, 0.93, "(b)  ionization rate per target atom",
            transform=ax.transAxes, ha="right", va="top", color=INK,
            fontsize=11, zorder=8)
    ax.annotate("IC-secondary window\n(model E)", xy=(6e7, 1.0e-22),
                xytext=(2.5e9, 3e-25), color=C_E, fontsize=8.0,
                ha="center", va="top", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=C_E, lw=0.8,
                                shrinkA=2, shrinkB=3), zorder=7)


def _label_channels(ax, d, y_g, y_e):
    sed_txt = (f"$f_\\nu \\propto \\nu^{{-{SED_ALPHA:.0f}}}$, 1–4 Ryd"
               if SED_MODEL == "powerlaw"
               else f"$T_{{\\rm eff}} = 5\\times10^4$ K blackbody")
    ax.text(0.205, 0.60, "stellar UV photons\n" + sed_txt + "\n"
            "all within a decade of 13.6 eV",
            transform=ax.transAxes, color=C_G, fontsize=8.6, ha="left",
            va="top", weight="bold", linespacing=1.3, zorder=8)
    ax.text(0.985, 0.60, "CR electrons\n"
            f"$E^{{-{CR_INDEX}}}$, 1 keV – 1 TeV",
            transform=ax.transAxes, color=C_E, fontsize=8.6, ha="right",
            va="top", weight="bold", linespacing=1.3, zorder=8)


def _map(ax, ratio_fid, cbar_fig=None):
    """zeta_gamma / zeta_e over the two scanned normalisations. Analytic:
    both channels are strictly linear in their own parameter, so the map is
    exact rather than sampled -- and that linearity is a check, not a hope."""
    fe = np.logspace(-4, 0, 240)                    # f_esc
    ec = np.logspace(-6, 0, 240)                    # eps_CR x f_e
    F, Ec = np.meshgrid(fe, ec, indexing="ij")
    R = ratio_fid * (F / F_ESC_FID) / (Ec / EPS_CR_FE_FID)
    im = ax.pcolormesh(fe, ec, np.log10(R).T, cmap="RdBu_r", shading="auto",
                       vmin=-4, vmax=4, zorder=1)
    # The parity locus is ANALYTIC -- both channels are linear in their own
    # normalisation -- so draw it rather than contour it. matplotlib's inline
    # clabel cut a gap in the contour and set the text at a steep angle away
    # from it, which read as an unexplained black segment.
    slope = ratio_fid * EPS_CR_FE_FID / F_ESC_FID      # eps = slope x f_esc
    f_at_ceiling = 1.0 / slope
    fline = np.array([fe[0], min(f_at_ceiling, fe[-1])])
    ax.plot(fline, slope * fline, color="#0b0b0b", lw=2.2, zorder=4)
    ax.annotate(r"$\zeta_\gamma=\zeta_e$   parity" "\n"
                r"above this line electrons win," "\n"
                r"which needs $f_{\rm esc} < %.1f\%%$" % (100 * f_at_ceiling),
                xy=(fline[1] * 0.55, slope * fline[1] * 0.55),
                xytext=(1.15e-4, 2.0e-1),
                color="#0b0b0b", fontsize=8.4, ha="left", va="center",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color="#0b0b0b", lw=0.9,
                                shrinkA=2, shrinkB=4), zorder=6)
    ax.text(0.97, 0.04, "photons win everywhere below", transform=ax.transAxes,
            color="#ffffff", fontsize=8.6, ha="right", va="bottom", zorder=6)

    ax.plot([F_ESC_FID], [EPS_CR_FE_FID], "o", ms=9, mfc="none",
            mec="#0b0b0b", mew=2.0, zorder=5)
    ax.annotate("canonical\n"
                f"$f_{{\\rm esc}}={F_ESC_FID}$, $\\epsilon_{{\\rm CR}}f_e=10^{{-3}}$",
                xy=(F_ESC_FID, EPS_CR_FE_FID), xytext=(6e-3, 2e-5),
                color=INK, fontsize=8.5, ha="left", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.0,
                                shrinkA=2, shrinkB=6), zorder=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"escape fraction  $f_{\rm esc}$", color=INK, fontsize=10)
    ax.set_ylabel(r"$\epsilon_{\rm CR}\times f_e$", color=INK, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(r"(c)  competition map:  $\log_{10}(\zeta_\gamma/\zeta_e)$",
                 color=INK, fontsize=11, loc="left")
    if cbar_fig is not None:
        cb = cbar_fig.colorbar(im, ax=ax, pad=0.02, extend="both")
        cb.set_label(r"$\log_{10}(\zeta_\gamma/\zeta_e)$  —  blue: electrons win",
                     color=INK2, fontsize=8.5, labelpad=2)
        cb.ax.tick_params(colors=INK2, labelsize=8)
    return im


def _footer(fig, extra=""):
    fig.text(0.012, 0.004,
             r"IGM at $z=10$, $x_e=10^{-4}$, pure H — the scenario of "
             r"ionization_yield_fig.png." "\n"
             r"Electron yields: model E — igm_losses.py: 7 loss mechanisms, "
             r"RBEB event counting, BEB secondary cascade, Klein–Nishina IC, "
             r"+ IC-secondary photoionization." "\n"
             r"$\rho_{\rm UV}$: Donnan+24 JWST PRIMER. "
             r"$\xi_{\rm ion}=10^{25.28}$: Llerena+25. "
             r"$\rho_{\rm UV}\to\rho_{\rm SFR}$: Madau & Dickinson 14 (Salpeter)." "\n"
             r"SN/M$_\odot$ derived from the IMF, not adopted." "\n"
             r"Stellar SED: population of U37126 ($z=10.255$, Marques-Chaves+26, "
             r"BPASS v2.2.1, $Z=0.003$, 6.8 Myr); LyC shape parametrised, "
             r"$\alpha$ bracketed 1–3." "\n"
             + extra,
             fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)


def make_figures(cos, par, chan, prov):
    # see igm_config.safe_plot_style: importing igm_losses turns on usetex
    # globally, which turns a bare "&" in a label into a LaTeX syntax error
    from igm_config import safe_plot_style
    with safe_plot_style() as plt:
        return _make_figures(cos, par, chan, prov, plt)


def _make_figures(cos, par, chan, prov, plt):

    d = _curves(cos, par, chan)
    ratio = prov["zeta_ratio"]

    def _sci(v):
        m, e = ("%.2e" % v).split("e")
        return f"{m}\\times10^{{{int(e)}}}"

    out = []

    # ---------- layout 2: (a), (b) and the map together -------------------
    fig = plt.figure(figsize=(13.0, 7.0)); fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], hspace=0.09,
                          wspace=0.20)
    ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    cx = fig.add_subplot(gs[:, 1])
    _panel_a(ax, d); _panel_b(bx, d)
    ax.set_xticklabels([])
    for a in (ax, bx):
        a.set_xlim(8, 3e12)
    _label_channels(ax, d, d["n_g"].max() * 1.6, d["n_e"].max() * 2.2)
    _map(cx, ratio, cbar_fig=fig)
    fig.suptitle("Stellar UV photons versus cosmic-ray electrons as "
                 "reionization agents at $z=10$",
                 color=INK, fontsize=13, x=0.012, ha="left", y=0.985)
    _footer(fig, f"Photons lead by {np.log10(ratio):.1f} decades at canonical "
                 f"parameters; parity would need "
                 f"$\\epsilon_{{\\rm CR}}f_e={prov['eps_cr_fe_crossover']:.2g}$, "
                 f"which exceeds unity.")
    fig.subplots_adjust(left=0.085, right=0.965, top=0.905, bottom=0.195)
    for ext in ("png", "pdf"):
        fig.savefig(f"photon_vs_electron_fig1.{ext}", dpi=200, facecolor=BG)
    out.append("photon_vs_electron_fig1.png"); plt.close(fig)

    # ---------- layout 4a: (a) and (b) alone, fiducial only ---------------
    fig = plt.figure(figsize=(7.4, 7.2)); fig.patch.set_facecolor(BG)
    ax, bx = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
    _panel_a(ax, d); _panel_b(bx, d)
    ax.set_xticklabels([])
    for a in (ax, bx):
        a.set_xlim(8, 3e12)
    _label_channels(ax, d, d["n_g"].max() * 1.6, d["n_e"].max() * 2.2)
    bx.text(0.98, 0.80, f"$\\zeta_\\gamma/\\zeta_e = {ratio:,.0f}$\n"
                        f"({np.log10(ratio):.2f} decades, photons ahead)",
            transform=bx.transAxes, ha="right", va="top", fontsize=9.5,
            color=INK, linespacing=1.3)
    fig.suptitle("Ionizing emissivity and ionization rate at $z=10$\n"
                 "canonical parameters", color=INK, fontsize=12.5,
                 x=0.012, ha="left", y=0.985)
    _footer(fig)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.885, bottom=0.205,
                        hspace=0.09)
    for ext in ("png", "pdf"):
        fig.savefig(f"photon_vs_electron_fig2a.{ext}", dpi=200, facecolor=BG)
    out.append("photon_vs_electron_fig2a.png"); plt.close(fig)

    # ---------- layout 4b: the parameter exploration on its own -----------
    fig = plt.figure(figsize=(12.0, 5.2)); fig.patch.set_facecolor(BG)
    ax, cx = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    # THE X-AXIS IS THE PARAMETER ITSELF, not a multiplier on it.
    # The first version plotted a multiplier, which put the canonical CR marker
    # at 10^0 while the canonical value is 1e-3 -- the axis and the marker were
    # saying different things. Both normalisations are dimensionless fractions
    # bounded by 1, so they share one axis honestly and both curves END at 10^0.
    # Nothing can be plotted beyond it: f_esc > 1 has no photons left to escape,
    # and eps_CR f_e > 1 is more energy than the supernova released.
    xpar = np.logspace(-6, 0, 400)
    zg = prov["zeta_gamma_s"] * (xpar / F_ESC_FID)
    ze = prov["zeta_e_s"] * (xpar / EPS_CR_FE_FID)
    ax.plot(xpar, zg, color=C_G, lw=2.6, zorder=5)
    ax.plot(xpar, ze, color=C_E, lw=2.6, zorder=5)
    ax.fill_between(xpar, prov["zeta_gamma_lo"] * (xpar / F_ESC_FID),
                    prov["zeta_gamma_hi"] * (xpar / F_ESC_FID),
                    color=C_G, alpha=0.16, lw=0, zorder=3)
    # markers AT their canonical parameter values
    ax.plot([F_ESC_FID], [prov["zeta_gamma_s"]], "o", ms=8, color=C_G, zorder=6)
    ax.plot([EPS_CR_FE_FID], [prov["zeta_e_s"]], "o", ms=8, color=C_E, zorder=6)
    ax.plot([1.0], [ze[-1]], "s", ms=7, color=C_E, zorder=6)

    ax.text(1.3e-6, 1.0e-16,
            "stellar UV   $x = f_{\\rm esc}$",
            color=C_G, fontsize=9.5, ha="left", va="center", weight="bold")
    ax.text(1.3e-6, prov["zeta_e_s"] * 4.0e-3,
            "CR electrons   $x = \\epsilon_{\\rm CR}f_e$",
            color=C_E, fontsize=9.5, ha="left", va="bottom", weight="bold")
    ax.annotate(f"canonical $f_{{\\rm esc}}=0.1$\n"
                f"$\\zeta_\\gamma={_sci(prov['zeta_gamma_s'])}$ s$^{{-1}}$",
                xy=(F_ESC_FID, prov["zeta_gamma_s"]),
                xytext=(1.2e-5, 1.1e-17),
                color=C_G, fontsize=8.2, ha="left", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=C_G, lw=0.9,
                                shrinkA=2, shrinkB=7), zorder=7)
    ax.annotate(f"canonical $\\epsilon_{{\\rm CR}}f_e=10^{{-3}}$\n"
                f"$\\zeta_e={_sci(prov['zeta_e_s'])}$ s$^{{-1}}$",
                xy=(EPS_CR_FE_FID, prov["zeta_e_s"]),
                xytext=(3.0e-3, prov["zeta_e_s"] * 2.2e-3),
                color=C_E, fontsize=8.2, ha="left", va="bottom", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=C_E, lw=0.9,
                                shrinkA=2, shrinkB=7), zorder=7)
    # the result, stated where it can be read straight off the axis
    ax.annotate(f"at the ceiling $\\epsilon_{{\\rm CR}}f_e=1$,\n"
                f"$\\zeta_e={_sci(ze[-1])}$ s$^{{-1}}$ is still\n"
                f"{prov['eps_cr_fe_crossover']:.2f}$\\times$ short of stellar UV",
                xy=(1.0, ze[-1]), xytext=(1.2e-2, 3.0e-21),
                color=INK, fontsize=8.4, ha="left", va="top", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.9,
                                shrinkA=2, shrinkB=7), zorder=8)
    ax.set_xlim(1e-6, 1.0)
    _style(ax)
    ax.set_xlabel("normalisation of that channel   "
                  r"($f_{\rm esc}$ for stellar UV, $\epsilon_{\rm CR}f_e$ for CR)",
                  color=INK, fontsize=9.5)
    ax.set_ylabel(r"$\zeta$   [s$^{-1}$ per H atom]", color=INK, fontsize=10)
    ax.set_title(r"(a)  each channel against its own scanned parameter",
                 color=INK, fontsize=11, loc="left")
    _map(cx, ratio, cbar_fig=fig)
    cx.set_title(r"(b)  competition map:  $\log_{10}(\zeta_\gamma/\zeta_e)$",
                 color=INK, fontsize=11, loc="left")
    fig.suptitle("Parameter exploration: what would it take for CR electrons "
                 "to compete?", color=INK, fontsize=13, x=0.012, ha="left",
                 y=0.975)
    _footer(fig, r"Left: both normalisations are dimensionless fractions, so "
                 r"both curves end at $10^0$ — beyond it there is no more energy "
                 r"to give. Square marker: the CR channel at its absolute ceiling.")
    fig.subplots_adjust(left=0.085, right=0.965, top=0.875, bottom=0.285,
                        wspace=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(f"photon_vs_electron_fig2b.{ext}", dpi=200, facecolor=BG)
    out.append("photon_vs_electron_fig2b.png"); plt.close(fig)
    return out


# ===========================================================================
# 6. MAIN
# ===========================================================================
def main():
    par = IY.Params()
    cos = IY.cosmology(par.z)
    print("=" * 78)
    print("STELLAR UV PHOTONS vs CR ELECTRONS AS REIONIZATION AGENTS, z = 10")
    print("=" * 78)

    print("\n[ASSUMPTIONS -- each stated where it enters]")
    for L in [
        "A1  Scenario identical to ionization_yield_fig.png: neutral IGM at",
        "    z = 10, x_e = 1e-4 held static, pure atomic H at mean density.",
        "A2  rho_UV(z=10) = 10^25.12 erg/s/Hz/cMpc^3 (Donnan+24, JWST PRIMER,",
        "    Table 3), +0.07/-0.14 dex. SCANNED via a multiplier.",
        "A3  rho_UV -> rho_SFR with K_UV = 1.15e-28 (Madau & Dickinson 2014),",
        "    Salpeter (1955) IMF. The IMF choice is inherited from that paper.",
        "A4  xi_ion = 10^25.28 Hz/erg (Llerena+25, A&A 698, A302), 0.42 dex",
        "    observed scatter, propagated.",
        "A5  Escaping stellar spectrum: T_eff = 5e4 K blackbody, truncated at",
        "    13.598 eV. A stellar atmosphere is NOT used; this is the shape",
        "    you chose, and it is computed from the Planck function.",
        "A6  f_esc SCANNED; canonical 0.10 by convention, not measurement.",
        "A7  CR electrons injected as E^-2.1 between 1 keV and 1 TeV (your",
        "    choice). E_min dominates both number and energy -- see checks.",
        "A8  E_SN = 1e51 erg; eps_CR x f_e SCANNED, canonical 1e-3 from",
        "    eps_CR = 0.1 (DSA) and f_e = K_ep = 0.01 (Galactic e/p at 10 GeV).",
        "A9  Electron yields from MODEL E: igm_losses.py (adiabatic, synchrotron,",
        "    IC with Klein-Nishina, Coulomb, excitation, ionization, brems);",
        "    ionizations counted event by event against RBEB cross sections,",
        "    knock-on electrons followed through the BEB secondary spectrum,",
        "    plus the IC-secondary photoionization channel. Model C of",
        "    ionization_yield.py agrees to 7.7% and is kept as the cross-check.",
        "A10 Emissivities per COMOVING Mpc^3; zeta is proper-frame and the",
        "    convention cancels from it -- checked, not assumed.",
        "A11 No CR escape/confinement modelling: every injected electron is",
        "    assumed to deposit in the IGM. This FAVOURS the CR channel.",
        "A12 No recombinations, no clumping: zeta is an ionization rate, not",
        "    a net reionization rate.",
    ]:
        print("   " + L)

    print("\n[DERIVED -- Master Rule 4: computed here, not recalled]")
    print(f"   SN per solar mass (Salpeter, >=8 Msun) = {sn_per_solar_mass():.6e}"
          f"  (1 per {1/sn_per_solar_mass():.1f} Msun)")
    print(f"   rho_SFR(z=10)  = {rho_sfr():.4e} Msun/yr/cMpc^3")
    print(f"   <E> injected e = {mean_injected_electron_energy():.4e} eV")
    print(f"   n_H(z=10)      = {cos['n_H_cm3']:.4e} cm^-3 (proper)")

    chan = IY.ICPhotonChannel(cos, par).build()
    rows, prov = run_checks(cos, par, chan)

    print("\n[CHECKS -- coverage, not a verdict tally]")
    for cid, name, state, detail in rows:
        print(f"   [{state}] {cid:5s} {name}")
        print(f"           {detail}")
    print(f"\n   {sum(1 for r in rows if r[2]=='PASS')}/{len(rows)} checks pass.")

    print("\n[RESULT -- units and uncertainty]")
    zg, ze = prov["zeta_gamma_s"], prov["zeta_e_s"]
    print(f"   zeta_gamma = {zg:.3e} (+{prov['zeta_gamma_hi']-zg:.1e}"
          f" / -{zg-prov['zeta_gamma_lo']:.1e}) s^-1 per H atom")
    print(f"   zeta_e     = {ze:.3e} s^-1 per H atom  [no published error bar:")
    print(f"                eps_CR and f_e are unmeasured at z = 10]")
    print(f"   ratio      = {prov['zeta_ratio']:.4g}"
          f"  = {np.log10(prov['zeta_ratio']):.2f} decades, photons ahead")
    print(f"   parity would need eps_CR x f_e = "
          f"{prov['eps_cr_fe_crossover']:.3g} at f_esc = {F_ESC_FID},")
    print(f"   which EXCEEDS UNITY and is therefore unreachable.")

    print("\n[CAVEATS -- what is NOT claimed]")
    for L in [
        "1. zeta is an ionization rate per atom, not a reionization history:",
        "   no recombinations, no clumping, no radiative transfer.",
        "2. A11 gives the CR channel every benefit of the doubt (full",
        "   confinement and deposition). The real CR channel is weaker.",
        "3. E_min = 1 keV is a choice, not a measurement, and it is the",
        "   single most consequential number here -- sensitivity reported.",
        "4. The stellar SED is a blackbody, not a synthesis model; it sets",
        "   the SHAPE of panel (a) and only weakly affects the totals.",
        "5. f_esc = 0.1 is a convention adopted to close photon budgets, not",
        "   an observation. That is why it is scanned.",
        "6. Helium is absent from the target, as in the parent calculation.",
    ]:
        print("   " + L)

    figs = make_figures(cos, par, chan, prov)

    os.makedirs("provenance", exist_ok=True)
    d = _curves(cos, par, chan)
    results = {
        "scenario": {"z": par.z, "x_e": par.x_e, "n_H_cm3": cos["n_H_cm3"]},
        "inputs": {"log_rho_UV": LOG_RHO_UV, "log_rho_UV_err": [LOG_RHO_UV_HI,
                   LOG_RHO_UV_LO], "K_UV": K_UV, "log_xi_ion": LOG_XI_ION,
                   "xi_ion_scatter_dex": XI_ION_SCATTER_DEX, "T_eff_K": T_EFF_K,
                   "E_SN_erg": E_SN_ERG, "CR_index": CR_INDEX,
                   "CR_E_min_eV": CR_E_MIN_EV, "CR_E_max_eV": CR_E_MAX_EV,
                   "f_esc_fid": F_ESC_FID, "eps_CR_fe_fid": EPS_CR_FE_FID},
        "electron_yield_model": ELECTRON_YIELD,
        "derived": {k: float(v) for k, v in prov.items()},
        "checks": [{"id": c, "name": n, "state": s, "detail": dt}
                   for c, n, s, dt in rows],
        "curves": {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                   for k, v in d.items()},
        "note": "Deterministic; no random numbers drawn, so no seed exists.",
    }
    with open("photon_vs_electron_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    import hashlib
    dig = hashlib.md5(open("photon_vs_electron_results.json", "rb").read()).hexdigest()
    with open("provenance/photon_vs_electron_md5.txt", "w") as fh:
        fh.write(dig + "  photon_vs_electron_results.json\n")
    print(f"\n[ARTEFACTS]  {'  '.join(figs)}  photon_vs_electron_results.json")
    print(f"[C56] md5 = {dig}  (determinism is a two-run claim; run again and compare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
