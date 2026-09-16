"""
stage9_verify_budget.py -- provenance and verification of the cosmic-ray
                           electron energy budget used in stage9_budget.py
==========================================================================

The number 1e46 erg per Msun of star formation is NOT quoted by any single
paper.  It is a product of four inputs.  This script (a) shows the chain
explicitly, (b) recomputes the IMF factor from scratch, and (c) replaces the
whole chain with a direct observational anchor from the Milky Way, which is
an independent determination of the same quantity.

A.  The assumed chain
       E_CRe/M_* = K_ep * eps_CR * E_SN / M_per_SN
    E_SN        1e51 erg per core-collapse SN (canonical "one foe";
                Woosley & Weaver 1995, ApJS 101, 181; Janka 2012,
                Ann. Rev. Nucl. Part. Sci. 62, 407)
    M_per_SN    recomputed here from the Chabrier (2003, PASP 115, 763) and
                Salpeter (1955, ApJ 121, 161) IMFs -- NOT taken on faith
    eps_CR      0.1 (range 0.1-0.5): Drury, Markiewicz & Voelk (1989) A&A
                225, 179; Berezinskii et al. (1990); Caprioli & Spitkovsky
                (2014) ApJ 783, 91.  As collected by Gessey-Jones et al.
                (2023) MNRAS 526, 4262: "cosmic rays carry away between 10
                and 50 per cent of the initial kinetic energy of the shock".
    K_ep        0.01.  Gessey-Jones et al. (2023): "The majority of this
                energy is in cosmic ray protons ... with per cent level
                portions of the energy in the form of alpha particles and
                electrons."  Measured value below.

B.  The observational anchor  (this is the reference the number should carry)
    Strong et al. (2010) ApJL 722, L58, Table 2, give the INJECTED CR
    luminosity of the Milky Way split by species, and Licquia & Newman
    (2015) ApJ 806, 96 give the Milky Way SFR.  Their ratio is the cosmic-ray
    electron energy injected per solar mass of star formation, measured
    rather than assumed.

Run:  python stage9_verify_budget.py
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

YR_S = 3.155760e7

# --------------------------------------------------------------------------
# A2.  IMF factor: how many core-collapse supernovae per solar mass formed?
# --------------------------------------------------------------------------
LOG_MC, SIGMA_C, ALPHA_HI = np.log10(0.079), 0.69, 1.3      # Chabrier (2003)
M_LO, M_HI, M_SN = 0.1, 100.0, 8.0


def chabrier_dn_dlogm(m):
    """dn/dlog m for the Chabrier (2003) individual-star disc IMF."""
    m = np.asarray(m, float)
    lo = np.exp(-(np.log10(m) - LOG_MC) ** 2 / (2.0 * SIGMA_C ** 2))
    norm = np.exp(-(0.0 - LOG_MC) ** 2 / (2.0 * SIGMA_C ** 2))   # value at m=1
    hi = norm * m ** (-ALPHA_HI)
    return np.where(m <= 1.0, lo, hi)


def salpeter_dn_dlogm(m):
    """dn/dm ~ m^-2.35  =>  dn/dlog m ~ m^-1.35 (Salpeter 1955)."""
    return np.asarray(m, float) ** (-1.35)


def m_per_sn(dn_dlogm):
    """Stellar mass formed per core-collapse supernova [Msun]."""
    n_sn, _ = quad(lambda lm: dn_dlogm(10.0 ** lm),
                   np.log10(M_SN), np.log10(M_HI), limit=200)
    m_tot, _ = quad(lambda lm: 10.0 ** lm * dn_dlogm(10.0 ** lm),
                    np.log10(M_LO), np.log10(M_HI), limit=200)
    return m_tot / n_sn, n_sn, m_tot


def block_A():
    print("=" * 78)
    print("A.  THE ASSUMED CHAIN   E_CRe/M_* = K_ep * eps_CR * E_SN / M_per_SN")
    print("=" * 78)
    print("\n   A1  IMF factor, recomputed here from the IMF definitions")
    print("       (progenitor mass range %.0f-%.0f Msun, IMF integrated %.1f-%.0f)"
          % (M_SN, M_HI, M_LO, M_HI))
    print("       %-14s %-16s %s" % ("IMF", "Msun per CCSN", "CCSN per 100 Msun"))
    out = {}
    for lbl, f in (("Chabrier 2003", chabrier_dn_dlogm),
                   ("Salpeter 1955", salpeter_dn_dlogm)):
        mps, _, _ = m_per_sn(f)
        out[lbl] = mps
        print("       %-14s %-16.4g %.3f" % (lbl, mps, 100.0 / mps))
    print("       -> the value 100 Msun/CCSN assumed in stage9_budget.py is the")
    print("          Chabrier result to %.0f%%; Salpeter would give %.0f Msun."
          % (100.0 * abs(out["Chabrier 2003"] - 100.0) / 100.0,
             out["Salpeter 1955"]))

    E_SN, EPS_CR, K_EP = 1.0e51, 0.10, 0.01
    for lbl, mps in out.items():
        e_cr = EPS_CR * E_SN / mps
        print("\n   A2  %s:" % lbl)
        print("       E_SN/M_*  = %.3g erg/Msun" % (E_SN / mps))
        print("       E_CR/M_*  = %.3g erg/Msun   (eps_CR = %.2f)" % (e_cr, EPS_CR))
        print("       E_CRe/M_* = %.3g erg/Msun   (K_ep   = %.2f)"
              % (K_EP * e_cr, K_EP))
    return out


# --------------------------------------------------------------------------
# B.  Observational anchor: the Milky Way
# --------------------------------------------------------------------------
# Strong et al. (2010) ApJL 722, L58, Table 2, "Luminosity of the Galaxy for
# various processes, 1e38 erg/s", cosmic rays 0.1-100 GeV.
# Columns: DR model 1,2,3 then PD model 1,2,3.
STRONG_TOT = np.array([805.0, 790.0, 698.0, 780.0, 723.0, 660.0]) * 1.0e38
STRONG_P = np.array([737.0, 724.0, 633.0, 718.0, 662.0, 601.0]) * 1.0e38
STRONG_EPRIM = np.array([8.8, 11.1, 13.4, 8.65, 10.5, 12.7]) * 1.0e38
SFR_MW, SFR_MW_ERR = 1.65, 0.19       # Licquia & Newman (2015) ApJ 806, 96


def block_B():
    print("\n" + "=" * 78)
    print("B.  OBSERVATIONAL ANCHOR -- the Milky Way (measured, not assumed)")
    print("=" * 78)
    print("\n   Strong et al. (2010) ApJL 722, L58, Table 2 -- INJECTED CR")
    print("   luminosity, 0.1-100 GeV, six propagation models [erg/s]:")
    print("     total CR      %.3g - %.3g   (headline value 7.9e40)"
          % (STRONG_TOT.min(), STRONG_TOT.max()))
    print("     protons       %.3g - %.3g" % (STRONG_P.min(), STRONG_P.max()))
    print("     primary e-    %.3g - %.3g" % (STRONG_EPRIM.min(),
                                              STRONG_EPRIM.max()))
    kep = STRONG_EPRIM / STRONG_P
    print("\n   => MEASURED injection-power ratio K_ep = L(prim e-)/L(p)")
    print("      = %.4f - %.4f   (assumed in stage9_budget.py: 0.01)"
          % (kep.min(), kep.max()))

    print("\n   Licquia & Newman (2015) ApJ 806, 96:  SFR(MW) = %.2f +/- %.2f"
          " Msun/yr" % (SFR_MW, SFR_MW_ERR))
    e_cr = STRONG_TOT * YR_S / SFR_MW
    e_cre = STRONG_EPRIM * YR_S / SFR_MW
    # propagate the SFR uncertainty (the model spread already brackets the rest)
    rel = SFR_MW_ERR / SFR_MW
    print("\n   => ENERGY INJECTED PER SOLAR MASS OF STAR FORMATION [erg/Msun]")
    print("      all cosmic rays      %.3g - %.3g  (+/- %.0f%% from SFR)"
          % (e_cr.min(), e_cr.max(), 100 * rel))
    print("      cosmic-ray ELECTRONS %.3g - %.3g  (+/- %.0f%% from SFR)"
          % (e_cre.min(), e_cre.max(), 100 * rel))
    lo = e_cre.min() * (1.0 - rel)
    hi = e_cre.max() * (1.0 + rel)
    print("      -> full range including the SFR error: %.2g - %.2g erg/Msun"
          % (lo, hi))
    print("      -> best single value: %.2g erg/Msun"
          % (np.median(e_cre)))
    return np.median(e_cr), np.median(e_cre), lo, hi


def block_C(imf, e_cr_obs, e_cre_obs, lo, hi):
    print("\n" + "=" * 78)
    print("C.  COMPARISON AND VERDICT")
    print("=" * 78)
    assumed_cr = 0.10 * 1.0e51 / 100.0
    assumed_cre = 0.01 * assumed_cr
    print("\n   %-34s %-16s %-16s %s"
          % ("quantity [erg/Msun]", "assumed", "measured (MW)", "ratio"))
    print("   %-34s %-16.3g %-16.3g %.2f"
          % ("all cosmic rays", assumed_cr, e_cr_obs, e_cr_obs / assumed_cr))
    print("   %-34s %-16.3g %-16.3g %.2f"
          % ("cosmic-ray electrons", assumed_cre, e_cre_obs,
             e_cre_obs / assumed_cre))
    print("\n   The assumed value is LOW by a factor %.1f, i.e. conservative"
          % (e_cre_obs / assumed_cre))
    print("   in the direction that matters (it understates the CR electron")
    print("   contribution).  Redoing the stage9 comparison with the measured")
    print("   value:")
    import stage9_budget as G
    n_uv, n_x, n_x_hi, n_cre, n_crp = G.delivered()
    scale = e_cre_obs / assumed_cre
    print("     CR electrons / UV, assumed  = %.2e" % (n_cre / n_uv))
    print("     CR electrons / UV, measured = %.2e  (range %.1e - %.1e)"
          % (n_cre * scale / n_uv,
             n_cre * (lo / assumed_cre) / n_uv,
             n_cre * (hi / assumed_cre) / n_uv))
    print("   The conclusion is unchanged: cosmic-ray electrons deliver ~1e-4")
    print("   of the ionizations that escaping stellar UV does.")
    print("\n   CAVEAT, flagged: the Milky Way anchor is a present-day,")
    print("   high-metallicity, strongly magnetised disc.  At z ~ 10 the SN")
    print("   energy per Msun is plausibly HIGHER (top-heavy IMF, more massive")
    print("   progenitors) and the CR escape fraction into the IGM larger")
    print("   (Leite et al. 2017 MNRAS 469, 416).  Both push the same way and")
    print("   neither is a factor 1e4.  A factor 10 enhancement would still")
    print("   leave CR electrons at ~1e-3 of the UV ionizations.")


if __name__ == "__main__":
    imf = block_A()
    e_cr_obs, e_cre_obs, lo, hi = block_B()
    block_C(imf, e_cr_obs, e_cre_obs, lo, hi)
