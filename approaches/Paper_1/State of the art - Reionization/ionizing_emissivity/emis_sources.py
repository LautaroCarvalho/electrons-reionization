"""
emis_sources.py -- volumetric injection spectra for the two populations.

Every normalisation is a MEASURED or PUBLISHED number from the corpus; the
only quantity computed here rather than quoted is the IMF integral giving the
stellar mass per core-collapse supernova, which is done from scratch.

PHOTONS
-------
P1  Lyman-alpha forest measurement.  Gaikwad et al. (2023), arXiv:2304.02038,
    Table 3, z = 6.00:
        ndot_ion = 0.701 (+0.357 / -0.191) x 10^51  s^-1 cMpc^-3
        Gamma_HI = 0.145 (+0.157 / -0.087) x 10^-12 s^-1
        alpha_s  = 2.0 +/- 0.6      (eps_nu ~ nu^-alpha_s)
P2  JWST/NIRSpec inference.  Giovinazzo et al. (2026), arXiv:2607.22834,
    Table B.1: log10(ndot_ion / s^-1 Mpc^-3) at integer redshift 5..15.
    Giovinazzo et al. do not quote a spectral index, so the Gaikwad alpha_s
    is used for the SHAPE in both cases; this is flagged.

    The photon NUMBER spectrum follows from eps_nu ~ nu^-alpha_s:
        dndot/dE ~ E^-(alpha_s + 1),   E >= E_th(H I),
    normalised so that its integral above 13.598 eV equals ndot_ion.  This is
    the same relation verified symbolically in ../photon_energy_fractions.py.

COSMIC-RAY ELECTRONS
--------------------
Tied to the SAME star formation that makes the photons, by instruction.

    (1) invert Robertson et al. (2015), arXiv:1502.02024, Eq. (1),
            ndot_ion = f_esc xi_ion rho_SFR
        with their fiducial f_esc = 0.2 and log10 xi_ion = 53.14
        [s^-1 per (Msun/yr)], to get rho_SFR(z) from the chosen ndot_ion;
    (2) convert to a supernova energy rate with E_SN = 10^51 erg
        (Woosley & Weaver 1995, ApJS 101, 181) and the stellar mass per
        core-collapse supernova obtained HERE by integrating the
        Chabrier (2003), PASP 115, 763, disc IMF over 8-100 Msun
        progenitors (0.1-100 Msun normalisation);
    (3) apply the cosmic-ray efficiencies collected by Gessey-Jones et al.
        (2023), MNRAS 526, 4262: "cosmic rays carry away between 10 and 50
        per cent of the initial kinetic energy of the shock" (Drury et al.
        1989; Berezinskii et al. 1990; Caprioli & Spitkovsky 2014), and "the
        majority of this energy is in cosmic ray protons ... with per cent
        level portions of the energy in the form of alpha particles and
        electrons".  Hence eta_cr in [0.1, 0.5] and f_e in [0.01, 0.02].
    (4) distribute that power over an injection spectrum.  Two prescriptions,
        which together define the band:
          Tueros et al. (2014), arXiv:1409.6225, Sect. 2:
              dN/dE ~ E^-2.2 (Drury 1983 DSA), 1 MeV <= E <= 100 TeV
          Gessey-Jones et al. (2023), Eq. (23) and Sect. 2.1:
              dN/dE ~ E^-2.0, 1 keV <= E <= 10^9 MeV = 1 PeV

BENCHMARK
---------
Tueros et al. (2014) quote I_1MeV^e ~ 1.5 x 10^4 ionizations per injected
1 MeV electron for their spectrum; emis_verify.py block D reproduces that
integral with the engines here.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from emis_common import E_TH, N_H_COM, MPC, YR_S

# ---------------------------------------------------------------------------
# Photons
# ---------------------------------------------------------------------------
ALPHA_S, ALPHA_S_ERR = 2.0, 0.6                 # Gaikwad+2023 Table 3
GAIKWAD_Z = 6.00
GAIKWAD_NDOT = (0.701e51, +0.357e51, -0.191e51)  # s^-1 cMpc^-3
GAIKWAD_GAMMA_HI = (0.145e-12, +0.157e-12, -0.087e-12)   # s^-1

# Giovinazzo+2026 Table B.1: z -> (log10 ndot, +err, -err)
GIOVINAZZO = {
    15: (49.30, 0.35, 0.29), 14: (49.41, 0.34, 0.28), 13: (49.70, 0.33, 0.27),
    12: (49.88, 0.32, 0.27), 11: (50.10, 0.20, 0.18), 10: (50.27, 0.17, 0.17),
    9: (50.47, 0.09, 0.10), 8: (50.64, 0.09, 0.09), 7: (50.83, 0.09, 0.09),
    6: (51.05, 0.09, 0.09), 5: (51.12, 0.09, 0.09),
}


def ndot_ion(z, which="giovinazzo"):
    """Ionizing photon emissivity [s^-1 cMpc^-3] and its 1-sigma range."""
    if which == "gaikwad":
        v, up, dn = GAIKWAD_NDOT
        return v, v + up, v + dn
    zs = np.array(sorted(GIOVINAZZO))
    lg = np.array([GIOVINAZZO[int(k)][0] for k in zs])
    ep = np.array([GIOVINAZZO[int(k)][1] for k in zs])
    em = np.array([GIOVINAZZO[int(k)][2] for k in zs])
    zc = np.clip(z, zs[0], zs[-1])
    c = np.interp(zc, zs, lg)
    return (10.0 ** c, 10.0 ** (c + np.interp(zc, zs, ep)),
            10.0 ** (c - np.interp(zc, zs, em)))


def photon_spectrum(E_eV, z, which="giovinazzo", alpha_s=ALPHA_S):
    """dndot/dE  [photons s^-1 cMpc^-3 eV^-1].

    eps_nu ~ nu^-alpha_s  =>  dndot/dE ~ E^-(alpha_s+1) above E_th(H I),
    normalised to ndot_ion.  For alpha_s > 0 the integral converges and the
    normalisation is analytic:  N0 = ndot_ion * alpha_s / E_th^(-alpha_s).
    """
    Eth = E_TH["HI"]
    n_tot = ndot_ion(z, which)[0]
    E = np.asarray(E_eV, float)
    return np.where(E >= Eth,
                    n_tot * alpha_s * Eth ** alpha_s * E ** (-alpha_s - 1.0),
                    0.0)


# ---------------------------------------------------------------------------
# Cosmic-ray electrons
# ---------------------------------------------------------------------------
F_ESC_R15 = 0.2                       # Robertson+2015, fiducial
LOG_XI_ION_SFR = 53.14                # Robertson+2015, s^-1 per (Msun/yr)
E_SN_ERG = 1.0e51                     # Woosley & Weaver 1995
ETA_CR = (0.10, 0.50)                 # Gessey-Jones+2023 Sect. 2.1
F_ELECTRON = (0.01, 0.02)             # "per cent level" (ibid.)
ERG_PER_EV = 1.602176634e-12

# Chabrier (2003) individual-star disc IMF, integrated here.
_LOG_MC, _SIG_C, _ALPHA_HI = np.log10(0.079), 0.69, 1.3


def _chabrier_dn_dlogm(m):
    m = np.asarray(m, float)
    lo = np.exp(-(np.log10(m) - _LOG_MC) ** 2 / (2.0 * _SIG_C ** 2))
    norm = np.exp(-(0.0 - _LOG_MC) ** 2 / (2.0 * _SIG_C ** 2))
    return np.where(m <= 1.0, lo, norm * m ** (-_ALPHA_HI))


def mass_per_ccsn(m_lo=0.1, m_hi=100.0, m_sn=8.0):
    """Stellar mass formed per core-collapse supernova [Msun], Chabrier 2003."""
    n, _ = quad(lambda lm: _chabrier_dn_dlogm(10.0 ** lm),
                np.log10(m_sn), np.log10(m_hi), limit=200)
    mt, _ = quad(lambda lm: 10.0 ** lm * _chabrier_dn_dlogm(10.0 ** lm),
                 np.log10(m_lo), np.log10(m_hi), limit=200)
    return mt / n


M_PER_CCSN = mass_per_ccsn()


def rho_sfr(z, which="giovinazzo"):
    """Cosmic SFR density [Msun yr^-1 cMpc^-3], inverted from ndot_ion."""
    return ndot_ion(z, which)[0] / (F_ESC_R15 * 10.0 ** LOG_XI_ION_SFR)


def cr_electron_power(z, which="giovinazzo", eta_cr=None, f_e=None):
    """Energy injected in cosmic-ray ELECTRONS [eV s^-1 cMpc^-3], and range."""
    r = rho_sfr(z, which)                       # Msun/yr/cMpc^3
    e_sn = r * (E_SN_ERG / M_PER_CCSN) / YR_S   # erg s^-1 cMpc^-3
    lo = e_sn * ETA_CR[0] * F_ELECTRON[0] / ERG_PER_EV
    hi = e_sn * ETA_CR[1] * F_ELECTRON[1] / ERG_PER_EV
    if eta_cr is None:
        return np.sqrt(lo * hi), lo, hi
    return e_sn * eta_cr * f_e / ERG_PER_EV, lo, hi


CR_SPECTRA = {
    # label: (index beta such that dN/dE ~ E^-beta, E_min[eV], E_max[eV], ref)
    "Tueros+2014":      (2.2, 1.0e6, 1.0e14, "arXiv:1409.6225 Sect. 2"),
    "GesseyJones+2023": (2.0, 1.0e3, 1.0e15, "arXiv:2304.07201 Eq. (23)"),
}


def cr_spectrum(E_eV, z, model="Tueros+2014", which="giovinazzo", level="mid"):
    """dN/dE  [electrons s^-1 cMpc^-3 eV^-1] for the named prescription."""
    beta, Emin, Emax, _ = CR_SPECTRA[model]
    P = cr_electron_power(z, which)
    P = {"mid": P[0], "lo": P[1], "hi": P[2]}[level]
    # normalise: P = N0 Int E^(1-beta) dE
    if abs(beta - 2.0) < 1e-12:
        I = np.log(Emax / Emin)
    else:
        I = (Emax ** (2.0 - beta) - Emin ** (2.0 - beta)) / (2.0 - beta)
    N0 = P / I
    E = np.asarray(E_eV, float)
    return np.where((E >= Emin) & (E <= Emax), N0 * E ** (-beta), 0.0)


if __name__ == "__main__":
    print("Chabrier (2003) IMF integral computed here:")
    print("   stellar mass per core-collapse SN = %.2f Msun"
          " (=> %.3f CCSN per 100 Msun)" % (M_PER_CCSN, 100.0 / M_PER_CCSN))
    print("\nPhoton emissivity [s^-1 cMpc^-3]")
    print("   %-6s %-14s %-14s %-14s" % ("z", "Gaikwad+2023", "Giovinazzo+2026",
                                         "ratio J/G"))
    g = ndot_ion(6.0, "gaikwad")[0]
    for z in (5, 6, 7, 8, 10, 12, 15):
        j = ndot_ion(z, "giovinazzo")[0]
        print("   %-6d %-14.4g %-14.4g %-14.3f"
              % (z, g if z == 6 else float("nan"), j, j / g))
    print("\nDerived star formation and cosmic-ray power (Giovinazzo norm.)")
    print("   %-6s %-16s %-18s %s" % ("z", "rho_SFR", "E_CRe [eV/s/cMpc^3]",
                                      "range"))
    for z in (6, 8, 10, 12):
        r = rho_sfr(z)
        m, lo, hi = cr_electron_power(z)
        print("   %-6d %-16.4g %-18.4g %.3g - %.3g" % (z, r, m, lo, hi))
    print("\nInjection spectra at z = 6 [particles s^-1 cMpc^-3 eV^-1]")
    print("   %-12s %-14s %-14s %-14s" % ("E [eV]", "photons", "CR Tueros",
                                          "CR GesseyJones"))
    for E in (14.0, 1e2, 1e3, 1e6, 1e9, 1e12):
        print("   %-12.4g %-14.4g %-14.4g %-14.4g"
              % (E, photon_spectrum(E, 6.0), cr_spectrum(E, 6.0, "Tueros+2014"),
                 cr_spectrum(E, 6.0, "GesseyJones+2023")))
    n_int = quad(lambda lE: photon_spectrum(np.exp(lE), 6.0) * np.exp(lE),
                 np.log(E_TH["HI"]), np.log(1e6), limit=300)[0]
    print("\n   normalisation check: Int dndot/dE dE above 13.598 eV = %.4e"
          "  vs ndot_ion = %.4e  (ratio %.6f)"
          % (n_int, ndot_ion(6.0)[0], n_int / ndot_ion(6.0)[0]))
