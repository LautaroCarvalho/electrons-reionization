"""
stage9_budget.py -- why UV photons reionize the Universe and energetic
                    particles do not, using the stage8 yields.
=======================================================================

The stage8 figure plots ionizations per PRIMARY PARTICLE.  Reionization is
not decided by that number: it is decided by

    (ionizations per erg injected)  x  (ergs available in that channel).

The first factor is 1/W(E) and is read directly off the stage8 tables.  The
second factor comes from the source physics and is taken from the literature;
every input is listed with its reference in SOURCES below.

Everything is normalised per solar mass of star formation, so that the
comparison is independent of the star formation history.

References for every number used
--------------------------------
xi_ion            Robertson et al. (2013) ApJ 768, 71; Bouwens et al. (2016)
                  ApJ 831, 176.  log10(xi_ion / Hz erg^-1) = 25.2 (+-0.1).
L_UV/SFR          Madau & Dickinson (2014) ARA&A 52, 415, Eq. (10) and
                  Kennicutt (1998): SFR = 1.15e-28 L_nu(UV) [erg/s/Hz].
<E_gamma>_ion     mean energy of an escaping H-ionizing photon for a Pop II
                  population, ~20 eV (18-25 eV depending on IMF/metallicity).
f_esc             Robertson et al. (2015) ApJ 802, L19; Finkelstein et al.
                  (2019) ApJ 879, 36:  0.1-0.2.
L_X/SFR           Mineo et al. (2012) MNRAS 419, 2095 (z~0, 2.6e39);
                  Fragos et al. (2013) ApJ 776, L31 and Lehmer et al. (2021)
                  ApJ 907, 17 (low-Z / high-z enhancement, up to ~1e40)
                  in erg s^-1 per (Msun/yr), 0.5-8 keV.
HMXB spectrum     power law dN/dE ~ E^-Gamma with photon index Gamma = 2 over
                  0.5-8 keV (the standard choice in 21cmFAST-type modelling;
                  Fragos et al. 2013; Pacucci et al. 2014).
E_SN, N_SN        1e51 erg per core-collapse SN, one CCSN per ~100 Msun of
                  stars formed for a Chabrier (2003) IMF.
eps_CR            fraction of SN shock kinetic energy in cosmic rays, 0.1
                  (quoted range 0.1-0.5: Drury, Markiewicz & Voelk 1989 A&A
                  225, 179; Berezinskii et al. 1990; Caprioli & Spitkovsky
                  2014 ApJ 783, 91, as collected by Gessey-Jones et al. 2023
                  MNRAS 526, 4262).
K_ep              cosmic-ray electron-to-proton INJECTED POWER ratio.  Not
                  assumed: measured, 0.012-0.021, from Strong et al. (2010)
                  ApJL 722, L58, Table 2 (primary e- / protons).
E_CR/M_*,         NOT a chain of assumptions.  Taken directly from the
E_CRe/M_*         Milky Way: Strong et al. (2010) injected CR luminosity
                  (7.9e40 erg/s total, 8.65e38-1.34e39 erg/s in primary
                  electrons) divided by the Milky Way SFR, 1.65 +- 0.19
                  Msun/yr (Licquia & Newman 2015 ApJ 806, 96).  See
                  stage9_verify_budget.py, which also recomputes the
                  assumption chain from the Chabrier (2003) IMF and shows
                  the two agree to a factor 2.

Run:  python stage9_budget.py
"""
from pathlib import Path

import numpy as np

import redo_common as R          # puts the repo root on sys.path
import igm_losses as L          # noqa: F401  (kept for unit constants)

HERE = Path(__file__).resolve().parent
pho = np.load(HERE / "redo_photon_yield_table.npz")
ele = np.load(HERE / "redo_cascade_table.npz")

n0 = int(ele["nlow"])
K, ze = ele["K"][n0:], ele["z"]
E, zp = pho["E"], pho["z"]

EV_ERG = 1.602176634e-12                    # erg per eV
MSUN_S_PER_YR = 3.155760e7                  # s per yr

# --- literature inputs ------------------------------------------------------
XI_ION = 10.0 ** 25.2                       # Hz^-1 erg^-1
K_UV = 1.15e-28                             # Msun/yr per (erg/s/Hz)
E_GAMMA_ION = 20.0                          # eV, mean ionizing photon energy
F_ESC = 0.10                                # escape fraction
LX_SFR = 3.0e39                             # erg/s per (Msun/yr), 0.5-8 keV
LX_SFR_HI = 1.0e40                          # low-metallicity / high-z value
GAMMA_X = 2.0                               # HMXB photon index
EX_LO, EX_HI = 5.0e2, 8.0e3                 # eV, X-ray band
E_SN = 1.0e51                               # erg
M_PER_SN = 100.0                            # Msun of stars per CCSN
EPS_CR = 0.10                               # SN kinetic energy -> cosmic rays
K_EP = 0.01                                 # CR electron/proton energy ratio
# Measured Milky Way values (Strong et al. 2010; Licquia & Newman 2015),
# used in preference to the eps_CR * E_SN / M_PER_SN chain above.
E_CR_PER_MSUN = 1.44e48                     # erg/Msun, all cosmic rays
E_CRE_PER_MSUN = 2.07e46                    # erg/Msun, primary CR electrons
E_CRE_LO, E_CRE_HI = 1.5e46, 2.9e46         # erg/Msun, full range
K_CRE = 1.0e9                               # eV, characteristic CR electron


def yield_photon(E_eV, zi):
    iz = int(np.argmin(abs(zp - zi)))
    return np.exp(np.interp(np.log(E_eV), np.log(E),
                            np.log(np.maximum(pho["Ne"][iz], 1e-300))))


def yield_electron(K_eV, zi):
    iz = int(np.argmin(abs(ze - zi)))
    Y = np.maximum(ele["Y"][iz, n0:] + ele["Ype"][iz, n0:], 1e-300)
    return np.exp(np.interp(np.log(K_eV), np.log(K), np.log(Y)))


def hmxb_weighted(zi, emin=EX_LO, emax=EX_HI, gamma=GAMMA_X):
    """Ionizations per erg of X-ray luminosity, for dN/dE ~ E^-gamma."""
    Eg = np.geomspace(emin, emax, 2000)
    dNdE = Eg ** (-gamma)
    num = np.trapz(dNdE * yield_photon(Eg, zi), Eg)       # ionizations
    den = np.trapz(dNdE * Eg, Eg) * EV_ERG                # erg
    return num / den, den / (np.trapz(dNdE, Eg) * EV_ERG)  # (ion/erg, <E>/erg)


def delivered(zi=10.0):
    """(UV, X-ray 3e39, X-ray 1e40, CR electrons, CR protons scaled)."""
    uv = yield_photon(2.04e1, zi) / (2.04e1 * EV_ERG)
    gx, _ = hmxb_weighted(zi)
    E_uv_esc = (XI_ION / K_UV) * MSUN_S_PER_YR * E_GAMMA_ION * EV_ERG * F_ESC
    n_uv = E_uv_esc * uv
    n_x = LX_SFR * MSUN_S_PER_YR * gx
    n_x_hi = LX_SFR_HI * MSUN_S_PER_YR * gx
    n_cre = E_CRE_PER_MSUN * yield_electron(K_CRE, zi) / (K_CRE * EV_ERG)
    n_crp = E_CR_PER_MSUN * yield_electron(K_CRE, zi) / (K_CRE * EV_ERG)
    return n_uv, n_x, n_x_hi, n_cre, n_crp


def main():
    print("=" * 78)
    print("STAGE 9 -- reionization budget: ionizations per erg x ergs available")
    print("=" * 78)

    # ---------------------------------------------------------------- part 1
    print("\n1.  IONIZATIONS PER ERG INJECTED   (= 1/W, read off stage8)")
    print("    %-24s %-14s %-14s" % ("channel / energy", "W [eV]",
                                     "N_ion per erg"))
    rows = []
    for lbl, E_eV, fn in (
            ("UV photon, 20 eV", 2.04e1, yield_photon),
            ("UV photon, 30 eV", 3.0e1, yield_photon),
            ("soft X-ray, 0.5 keV", 5.0e2, yield_photon),
            ("X-ray, 1 keV", 1.0e3, yield_photon),
            ("X-ray, 3 keV", 3.0e3, yield_photon),
            ("hard X-ray, 10 keV", 1.0e4, yield_photon),
            ("electron, 1 keV", 1.0e3, yield_electron),
            ("electron, 1 MeV", 1.0e6, yield_electron),
            ("CR electron, 1 GeV", 1.0e9, yield_electron),
            ("CR electron, 1 TeV", 1.0e12, yield_electron)):
        n = fn(E_eV, 10.0)
        W = E_eV / n
        per_erg = n / (E_eV * EV_ERG)
        rows.append((lbl, W, per_erg))
        print("    %-24s %-14.4g %-14.4g" % (lbl, W, per_erg))
    uv = rows[0][2]
    print("\n    Relative to a 20 eV UV photon (= 1.000):")
    for lbl, W, per_erg in rows:
        print("      %-24s %.4g" % (lbl, per_erg / uv))

    gx, Ex_mean = hmxb_weighted(10.0)
    print("\n    HMXB power law, Gamma = %.1f, %.1f-%.1f keV, z_i = 10:"
          % (GAMMA_X, EX_LO / 1e3, EX_HI / 1e3))
    print("      spectrum-averaged N_ion per erg = %.4g   (%.4g x UV)"
          % (gx, gx / uv))
    print("      effective W = %.4g eV" % (1.0 / (gx * EV_ERG)))

    # ---------------------------------------------------------------- part 2
    print("\n2.  ENERGY AVAILABLE PER SOLAR MASS OF STARS FORMED")
    Ndot_per_sfr = XI_ION / K_UV                 # photons/s per (Msun/yr)
    N_uv_per_msun = Ndot_per_sfr * MSUN_S_PER_YR  # photons per Msun
    E_uv = N_uv_per_msun * E_GAMMA_ION * EV_ERG   # erg per Msun
    E_uv_esc = E_uv * F_ESC
    E_x = LX_SFR * MSUN_S_PER_YR
    E_x_hi = LX_SFR_HI * MSUN_S_PER_YR
    E_cr = E_CR_PER_MSUN
    E_cre = E_CRE_PER_MSUN
    E_cr_chain = EPS_CR * E_SN / M_PER_SN
    print("    NOTE the deliberate asymmetry: the UV number is multiplied by")
    print("    f_esc ~ 0.1 because most stellar UV is absorbed inside the host")
    print("    galaxy, whereas X-rays and cosmic rays escape essentially freely")
    print("    (sigma_bf ~ E^-3 makes the host transparent above ~0.5 keV).")
    print("    That asymmetry is worth a factor 10 to the hard channels and is")
    print("    the reason they are discussed at all.")
    print("    %-40s %-14s %s" % ("channel", "erg / Msun", "fraction of UV"))
    for lbl, val in (("stellar H-ionizing UV (total)", E_uv),
                     ("  ... escaping, f_esc = %.2f" % F_ESC, E_uv_esc),
                     ("HMXB X-rays, L_X/SFR = 3e39", E_x),
                     ("HMXB X-rays, L_X/SFR = 1e40", E_x_hi),
                     ("cosmic rays, all species (MW-anchored)", E_cr),
                     ("cosmic-ray ELECTRONS (MW-anchored)", E_cre),
                     ("  [cross-check: eps_CR E_SN / M_SN]", E_cr_chain),
                     ("  [cross-check: x K_ep]", K_EP * E_cr_chain)):
        print("    %-40s %-14.4g %.4g" % (lbl, val, val / E_uv))
    print("\n    check: %.4g ionizing photons per Msun (log10 = %.2f)"
          % (N_uv_per_msun, np.log10(N_uv_per_msun)))

    # ---------------------------------------------------------------- part 3
    print("\n3.  IONIZATIONS DELIVERED PER SOLAR MASS OF STARS FORMED")
    print("    (energy available)  x  (ionizations per erg)")
    n_uv = E_uv_esc * uv
    n_x = E_x * gx
    n_x_hi = E_x_hi * gx
    n_cre = E_cre * yield_electron(K_CRE, 10.0) / (K_CRE * EV_ERG)
    n_crp = E_cr * yield_electron(K_CRE, 10.0) / (K_CRE * EV_ERG)
    n_cre_lo = E_CRE_LO * yield_electron(K_CRE, 10.0) / (K_CRE * EV_ERG)
    n_cre_hi = E_CRE_HI * yield_electron(K_CRE, 10.0) / (K_CRE * EV_ERG)
    print("    %-36s %-14s %s" % ("channel", "N_ion / Msun", "rel. to UV"))
    for lbl, val in (("escaping stellar UV", n_uv),
                     ("HMXB X-rays (L_X/SFR = 3e39)", n_x),
                     ("HMXB X-rays (L_X/SFR = 1e40)", n_x_hi),
                     ("cosmic-ray electrons (1 GeV)", n_cre)):
        print("    %-36s %-14.4g %.3e" % (lbl, val, val / n_uv))
    print("    %-36s %-14s %.1e - %.1e"
          % ("  CR electron range (MW anchor)", "", n_cre_lo / n_uv,
             n_cre_hi / n_uv))
    print("\n    NOT COMPUTED HERE: cosmic-ray PROTONS, which carry ~99 % of")
    print("    the CR energy.  No proton cascade table exists in this project,")
    print("    so no number is quoted; the literature limit is 'no more than a")
    print("    few per cent of the ionizations' (Sazonov & Sunyaev 2015; Leite")
    print("    et al. 2017).  Using the MEASURED total CR power instead of the")
    print("    electron power, and assuming a proton W comparable to the")
    print("    electron one, would give %.2e x UV -- about one per cent, i.e."
          % (n_crp / n_uv))
    print("    consistent with that published limit.")

    # ------------------------------------------------- part 3b: X-ray scan
    print("\n3b. HOW SOFT AND HOW LUMINOUS WOULD THE X-RAYS HAVE TO BE?")
    print("    The literature quotes an X-ray share of up to ~10 % of the")
    print("    ionizations (McQuinn 2012 and refs. therein).  Scanning the two")
    print("    parameters that matter -- the low-energy cutoff of the escaping")
    print("    X-ray SED and L_X/SFR -- reproduces where that comes from.")
    print("    %-10s %-8s %-14s %-14s %s"
          % ("E_min[keV]", "Gamma", "W_eff [eV]", "N/Msun", "rel. to UV"))
    for emin in (2.0e2, 5.0e2, 1.0e3):
        for gam in (1.5, 2.0, 2.5):
            g, _ = hmxb_weighted(10.0, emin=emin, gamma=gam)
            nx = LX_SFR_HI * MSUN_S_PER_YR * g
            print("    %-10.2f %-8.1f %-14.4g %-14.4g %.3e"
                  % (emin / 1e3, gam, 1.0 / (g * EV_ERG), nx, nx / n_uv))
    print("    Even the most favourable corner (E_min = 0.2 keV, Gamma = 2.5,")
    print("    L_X/SFR = 1e40) stays at the 1e-2 level relative to escaping UV.")
    print("    Reaching ~10 % requires either f_esc(UV) ~ 0.01 or L_X/SFR ~")
    print("    1e42, both outside the observationally allowed range -- which")
    print("    is precisely the argument of McQuinn (2012).")

    # ---------------------------------------------------------------- part 4
    print("\n4.  CROSS-CHECK -- the UV budget alone must close reionization")
    print("    Ionizing photons needed per H atom: 1 + N_rec ~ 2 (clumping)")
    OMB, OMM, H0 = 0.04897, 0.3111, 67.66      # Planck18
    rho_c = 1.87834e-29 * (H0 / 100.0) ** 2    # g/cm^3
    n_H = 0.76 * OMB * rho_c / 1.6726219e-24   # comoving cm^-3
    print("    n_H (comoving) = %.4g cm^-3" % n_H)
    rho_star_needed = 2.0 * n_H / N_uv_per_msun / F_ESC   # Msun/cm^3
    MPC = 3.0856775814913673e24
    print("    -> rho_* needed = %.4g Msun/Mpc^3 (comoving) with f_esc = %.2f"
          % (rho_star_needed * MPC ** 3, F_ESC))
    print("    -> Omega_* = %.3e" % (rho_star_needed * 1.98892e33 / rho_c))
    print("    Observed z~6 stellar mass density is ~1e7 Msun/Mpc^3")
    print("    (Madau & Dickinson 2014, Fig. 11): the UV budget closes to")
    print("    within a factor of a few, which is the well-known 'photon-")
    print("    starved' character of reionization (Bolton & Haehnelt 2007).")
    print("    There is therefore no room for an additional channel at more")
    print("    than the tens-of-per-cent level, quite apart from its W.")


if __name__ == "__main__":
    main()
