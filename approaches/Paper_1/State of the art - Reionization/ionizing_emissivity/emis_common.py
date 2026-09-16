"""
emis_common.py -- constants, cosmology, target densities and photoionization
                  cross-sections shared by the ionizing-emissivity branches.

EVERY number in this module is traceable to a specific paper and equation or
table.  Nothing is invented; nothing is carried over from other parts of this
project without being re-derived here from the cited source.

SOURCES
-------
Cosmology      Planck Collaboration 2018 VI (arXiv:1807.06209), Table 1 (TT,TE,
               EE+lowE+lensing):  Omega_b h^2 = 0.02237, h = 0.6736,
               Y_p = 0.2454 (BBN-consistent helium mass fraction).
Constants      CODATA 2018 (exact SI definitions where applicable).
Thresholds     NIST ASD: E(H I) = 13.598 eV, E(He I) = 24.587 eV,
               E(He II) = 54.418 eV.  Verner et al. (1996) Table 1 quote
               13.60, 24.59, 54.42 eV; the NIST values are used for the
               thresholds and the Verner values inside their own fit, exactly
               as the fit was constructed.
sigma_pi       Verner, Ferland, Korista & Yakovlev (1996), ApJ 465, 487
               (arXiv:astro-ph/9601009), Eq. (1) and Table 1.
               Selected by the user over the Osterbrock & Ferland fits.
               VALIDITY: the fits are constructed over E_th < E < E_max with
               E_max = 5.000e4 eV for all three ions.  Above 50 keV the same
               expression is used, which reproduces the correct
               NON-RELATIVISTIC asymptote sigma ~ E^-3.5 by construction
               (Verner et al. 1996, text below their Eq. 1).  This is flagged
               wherever the >50 keV region is plotted; above ~511 keV Compton
               scattering dominates the total opacity in any case.

HELIUM IONIZATION STATE
-----------------------
By explicit instruction, helium is taken to be FULLY NEUTRAL,
n_HeI = n_He = (Y_p / 4 X_p) n_H, independent of the hydrogen ionized
fraction x_e.  This is deliberately an UPPER BOUND on the helium target
density and is internally inconsistent with x_e > 0; it is flagged as such in
every figure and table produced from this module.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 0.  Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
EV_ERG = 1.602176634e-12            # erg per eV                     (exact)
H_PLANCK = 6.62607015e-27           # erg s                          (exact)
C_LIGHT = 2.99792458e10             # cm/s                           (exact)
K_B = 1.380649e-16                  # erg/K                          (exact)
M_E_C2 = 510998.95                  # eV, electron rest energy
MB = 1.0e-18                        # cm^2 per megabarn (Verner+1996 units)
MPC = 3.0856775814913673e24         # cm
YR_S = 3.155760e7                   # s
GYR_S = 3.1556952e16                # s

# ---------------------------------------------------------------------------
# 1.  Cosmology -- Planck Collaboration 2018 VI, Table 1
# ---------------------------------------------------------------------------
OMEGA_B_H2 = 0.02237
H_HUBBLE = 0.6736
Y_P = 0.2454                        # helium MASS fraction
X_P = 1.0 - Y_P
RHO_CRIT_H2 = 1.87834e-29           # g cm^-3, critical density / h^2
M_H_ATOM = 1.67353e-24              # g, hydrogen atom mass

N_H_COM = X_P * OMEGA_B_H2 * RHO_CRIT_H2 / M_H_ATOM     # cm^-3 comoving
N_HE_OVER_N_H = Y_P / (4.0 * X_P)                       # 0.08130
N_HE_COM = N_HE_OVER_N_H * N_H_COM

# ---------------------------------------------------------------------------
# 2.  Ionization thresholds (NIST Atomic Spectra Database)
# ---------------------------------------------------------------------------
E_TH = {"HI": 13.598, "HeI": 24.587, "HeII": 54.418}

# ---------------------------------------------------------------------------
# 3.  Verner et al. (1996) photoionization cross-sections
#     Table 1 rows for H I, He I, He II, transcribed verbatim.
#     Columns: E_th[eV], E_max[eV], E_0[eV], sigma_0[Mb], y_a, P, y_w, y_0, y_1
# ---------------------------------------------------------------------------
VERNER = {
    #        E_th      E_max    E_0        sig_0     y_a       P       y_w     y_0        y_1
    "HI":   (1.360e1, 5.000e4, 4.298e-1, 5.475e4, 3.288e1, 2.963, 0.000,  0.000,    0.000),
    "HeI":  (2.459e1, 5.000e4, 1.361e1,  9.492e2, 1.469e0, 3.188, 2.039,  4.434e-1, 2.136),
    "HeII": (5.442e1, 5.000e4, 1.720e0,  1.369e4, 3.288e1, 2.963, 0.000,  0.000,    0.000),
}
VERNER_EMAX = 5.000e4               # eV -- fitted range ceiling, all three ions


def sigma_pi(E_eV, ion="HI"):
    """Photoionization cross-section [cm^2].  Verner et al. (1996) Eq. (1).

        y      = sqrt( (E/E_0 - y_0)^2 + y_1^2 )
        F(y)   = [ (y-1)^2 + y_w^2 ] y^(0.5 P - 5.5) (1 + sqrt(y/y_a))^(-P)
        sigma  = sigma_0 F(y)  [Mb]

    Zero below threshold.  Above E_max = 50 keV the same expression is used
    (correct non-relativistic E^-3.5 asymptote); see the module docstring.
    """
    Eth, _Emax, E0, s0, ya, P, yw, y0, y1 = VERNER[ion]
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.zeros_like(E)
    m = E >= Eth
    if not m.any():
        return out if np.ndim(E_eV) else out[0]
    x = E[m] / E0 - y0
    y = np.sqrt(x * x + y1 * y1)
    F = ((x - 1.0) ** 2 + yw * yw) * y ** (0.5 * P - 5.5) \
        * (1.0 + np.sqrt(y / ya)) ** (-P)
    out[m] = s0 * F * MB
    return out if np.ndim(E_eV) else out[0]


def sigma_pi_total_per_H(E_eV, x_HeI=1.0):
    """Bound-free opacity per hydrogen NUCLEUS [cm^2], H I + He I.

    Helium fully neutral by instruction: x_HeI = 1.  He II is not populated,
    so its cross-section is provided by sigma_pi but not summed here.
    """
    return sigma_pi(E_eV, "HI") + x_HeI * N_HE_OVER_N_H * sigma_pi(E_eV, "HeI")


# ---------------------------------------------------------------------------
# 4.  Target number densities
# ---------------------------------------------------------------------------
def n_target_comoving(species):
    """Comoving number density of the ionization target [cm^-3].

    Helium fully neutral (instruction), hydrogen fully neutral for the target
    count -- the emissivity per target atom is quoted per AVAILABLE target, so
    the neutral fraction multiplies out of the comparison and is stated
    separately rather than folded in.
    """
    if species == "HI":
        return N_H_COM
    if species in ("HeI", "HeII"):
        return N_HE_COM
    raise ValueError(species)


def n_proper(species, z):
    return n_target_comoving(species) * (1.0 + np.asarray(z, float)) ** 3


if __name__ == "__main__":
    print("Planck 2018 VI:  Omega_b h^2 = %.5f  h = %.4f  Y_p = %.4f"
          % (OMEGA_B_H2, H_HUBBLE, Y_P))
    print("n_H  comoving = %.4e cm^-3   (= %.4e cMpc^-3)"
          % (N_H_COM, N_H_COM * MPC ** 3))
    print("n_He/n_H      = %.5f" % N_HE_OVER_N_H)
    print("n_He comoving = %.4e cm^-3" % N_HE_COM)
    print("\nVerner et al. (1996) cross-sections [cm^2]")
    print("  %-10s %-13s %-13s %-13s" % ("E [eV]", "H I", "He I", "He II"))
    for E in (13.7, 24.7, 54.5, 100.0, 300.0, 1.0e3, 1.0e4, 5.0e4):
        print("  %-10.4g %-13.4e %-13.4e %-13.4e"
              % (E, sigma_pi(E, "HI"), sigma_pi(E, "HeI"), sigma_pi(E, "HeII")))
