#!/usr/bin/env python3
"""
Photoionization rates per target atom, Gamma_i, for H I, He I and He II from the
escaping ionizing background at the end of reionization -- in the same units as
a cosmic-ray ionization rate zeta (s^-1 per target particle), so the two can be
compared directly.

ANCHOR
------
Gamma_HI is *measured*, so nothing about hydrogen needs to be derived.  The
helium rates are obtained as ratios to it, which cancels the (poorly known)
normalisation of the background and leaves only the SPECTRAL SHAPE and the
photoionization cross-sections.

    Gamma_i = Integral_{nu_i}^{inf}  (4 pi J_nu / h nu) sigma_i(nu) dnu

    Gamma_i / Gamma_HI = [ Int_{nu_i} f(nu) sigma_i(nu) dnu/nu ]
                         / [ Int_{nu_H} f(nu) sigma_H(nu) dnu/nu ]

with J_nu = J_0 f(nu).  Three choices of f(nu) are compared (see SHAPES).

SOURCES  (every input below is from the 57-paper corpus unless marked)
---------------------------------------------------------------------
Gaikwad et al. 2023 (arXiv:2304.02038), Table 3:
    Gamma_HI(z=6.00) = 0.145 (+0.157/-0.087) x 10^-12 s^-1
    Gamma_HI(z=4.90) = 0.501 (+0.275/-0.232) x 10^-12 s^-1
    eps_912(z=6.00)  = 0.929 (+0.052/-0.050) x 10^25 erg s^-1 cMpc^-3 Hz^-1
    lambda_mfp(z=6)  = 8.318 (+7.531/-4.052) h^-1 cMpc
    ndot(z=6.00)     = 0.701 (+0.357/-0.191) x 10^51 s^-1 cMpc^-3
    alpha_s          = 2.0 +/- 0.6      (spectral index of the ionizing sources)
Planck Collaboration 2018 VI (arXiv:1807.06209): Omega_b h^2, h, Y_p
Eide et al. 2020 (arXiv:2009.06631), Fig. 1: BPASS stellar SED shape (read off)
Murphy et al. 2021 (arXiv:2105.06900), Table 4: Pop III N_H, N_HeI, N_HeII
Verner et al. 1996 / Osterbrock & Ferland 2006: photoionization cross-sections
    (standard textbook fits -- NOT from the corpus, flagged in the write-up)
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad

# ----------------------------------------------------------------------
# 0.  Constants (CGS) and cosmology
# ----------------------------------------------------------------------
h_pl   = 6.62607015e-27      # erg s      (CODATA 2018, exact)
eV     = 1.602176634e-12     # erg        (exact)
Mpc    = 3.0856775814913673e24   # cm
m_H    = 1.67353e-24         # g
rho_c_h2 = 1.87834e-29       # g cm^-3

Ob_h2, hub, Yp = 0.02237, 0.6736, 0.2454      # Planck 2018 VI
Xp = 1.0 - Yp
nH_com = Xp * Ob_h2 * rho_c_h2 / m_H          # comoving cm^-3
nHe_over_nH = Yp / (4.0 * Xp)

E_HI, E_HeI, E_HeII = 13.598, 24.587, 54.418          # eV (NIST)
nu_HI, nu_HeI, nu_HeII = [E * eV / h_pl for E in (E_HI, E_HeI, E_HeII)]

# ----------------------------------------------------------------------
# 1.  Photoionization cross-sections  (Osterbrock & Ferland 2006, App.
#     analytic fits; Verner et al. 1996)
# ----------------------------------------------------------------------
def sig_HI(nu):
    x = nu / nu_HI
    return 6.30e-18 * (1.34 * x**-2.99 - 0.34 * x**-3.99)

def sig_HeI(nu):
    x = nu / nu_HeI
    return 7.42e-18 * (1.66 * x**-2.05 - 0.66 * x**-3.05)

def sig_HeII(nu):                      # hydrogenic, Z = 2
    x = nu / nu_HeII
    return 1.58e-18 * (1.34 * x**-2.99 - 0.34 * x**-3.99)

# ----------------------------------------------------------------------
# 2.  Symbolic check of the closed form used as a sanity limit
#     For J ~ nu^-a and sigma ~ nu^-3 :  Gamma = 4 pi sigma_0 J(nu_i) / (h(a+3))
# ----------------------------------------------------------------------
nu_s, a_s, nu0, J0, s0, hh = sp.symbols('nu alpha nu_0 J_0 sigma_0 h', positive=True)
integrand = (4*sp.pi*J0*(nu_s/nu0)**(-a_s) / (hh*nu_s)) * s0*(nu_s/nu0)**(-3)
Gamma_sym = sp.integrate(integrand, (nu_s, nu0, sp.oo))
closed    = 4*sp.pi*s0*J0/(hh*(a_s+3))
assert sp.simplify(Gamma_sym - closed) == 0
print("[sympy] Gamma = 4*pi*sigma_0*J(nu_i)/(h*(alpha+3))  ...  VERIFIED")

# dimensional check.  [J_nu] = erg cm^-2 s^-1 Hz^-1 = energy/length^2,
# [h nu] = energy, [sigma] = length^2, [dnu] = 1/time.
from sympy.physics.units import time as T, length as L, energy as EN
from sympy.physics.units.systems.si import dimsys_SI
dim_J    = EN / L**2                      # specific intensity
dim_G    = (dim_J / EN) * L**2 * (1/T)    # (J/h nu) * sigma * dnu
print("[sympy] [Gamma] == 1/time :",
      dimsys_SI.equivalent_dims(sp.simplify(dim_G), 1/T))
assert dimsys_SI.equivalent_dims(sp.simplify(dim_G), 1/T)
print()

# ----------------------------------------------------------------------
# 3.  Spectral shapes  f(nu), normalised to f(nu_HI) = 1
# ----------------------------------------------------------------------
def f_powerlaw(alpha):
    return lambda nu: (nu / nu_HI) ** (-alpha)

def f_bpass(a1=0.97, a2=2.18, a3=23.8):
    """Piecewise power law read off Eide et al. 2020 Fig. 1 (stellar/BPASS
    curve, z~6): log10 S drops 26.3 -> 26.05 -> 25.30 across the H I, He I and
    He II edges, then falls off a cliff.  Read-off uncertainty ~0.2 dex."""
    def f(nu):
        if nu < nu_HeI:
            return (nu / nu_HI) ** (-a1)
        A = (nu_HeI / nu_HI) ** (-a1)
        if nu < nu_HeII:
            return A * (nu / nu_HeI) ** (-a2)
        B = A * (nu_HeII / nu_HeI) ** (-a2)
        return B * (nu / nu_HeII) ** (-a3)
    return f

# ----------------------------------------------------------------------
# 4.  Gamma ratios
# ----------------------------------------------------------------------
def gamma_integral(f, sig, nu_min, nu_max_factor=1e4):
    g = lambda lnu: f(np.exp(lnu)) * sig(np.exp(lnu))      # dnu/nu = dln(nu)
    val, _ = quad(g, np.log(nu_min), np.log(nu_min * nu_max_factor), limit=300)
    return val

def ratios(f):
    gH   = gamma_integral(f, sig_HI,   nu_HI)
    gHe1 = gamma_integral(f, sig_HeI,  nu_HeI)
    gHe2 = gamma_integral(f, sig_HeII, nu_HeII)
    return gHe1 / gH, gHe2 / gH

SHAPES = {
    "power law  alpha_s = 1.4 (-1sigma)": f_powerlaw(1.4),
    "power law  alpha_s = 2.0 (fiducial)": f_powerlaw(2.0),
    "power law  alpha_s = 2.6 (+1sigma)": f_powerlaw(2.6),
    "BPASS stellar SED (Eide+2020 Fig.1)": f_bpass(),
}

print("Gamma ratios (independent of the background normalisation)")
print("  spectral shape                        Gamma_HeI/Gamma_HI   Gamma_HeII/Gamma_HI")
R = {}
for name, f in SHAPES.items():
    r1, r2 = ratios(f)
    R[name] = (r1, r2)
    print("  %-36s %12.3f %20.3e" % (name, r1, r2))

# ----------------------------------------------------------------------
# 5.  Anchor on the measured Gamma_HI  ->  absolute rates
# ----------------------------------------------------------------------
G_HI = {"z=6.0": (0.145e-12, +0.157e-12, -0.087e-12),
        "z=4.9": (0.501e-12, +0.275e-12, -0.232e-12)}

print("\nAbsolute photoionization rates, s^-1 per target particle")
print("  (Gamma_HI measured: Gaikwad et al. 2023, Table 3)")
for zlab, (g, up, dn) in G_HI.items():
    print("\n  %s   Gamma_HI = %.3e  (+%.3e / %.3e) s^-1" % (zlab, g, up, dn))
    for name, (r1, r2) in R.items():
        print("      %-36s Gamma_HeI = %.3e   Gamma_HeII = %.3e"
              % (name, g*r1, g*r2))

# ----------------------------------------------------------------------
# 6.  Photon NUMBER rates (the literal "photons per unit time")
# ----------------------------------------------------------------------
ndot6 = 0.701e51                        # s^-1 cMpc^-3, Gaikwad+2023 z=6.00
nH_per_cMpc3 = nH_com * Mpc**3
print("\nEscaping ionizing photon output at z = 6")
print("  ndot_HI            = %.3e s^-1 cMpc^-3" % ndot6)
print("  <n_H> comoving     = %.4e cm^-3  = %.4e cMpc^-3" % (nH_com, nH_per_cMpc3))
print("  photons per H atom = %.3e s^-1   (= %.2f per H per Gyr)"
      % (ndot6/nH_per_cMpc3, ndot6/nH_per_cMpc3*3.1557e16))

# photon-number fractions above each edge, for the same shapes
def photon_frac(f, nu_edge):
    num = gamma_integral(lambda n: f(n), lambda n: 1.0, nu_edge)
    den = gamma_integral(lambda n: f(n), lambda n: 1.0, nu_HI)
    return num / den
print("\n  fraction of escaping ionizing photons above each edge")
for name, f in SHAPES.items():
    print("      %-36s >24.6 eV: %6.3f    >54.4 eV: %10.3e"
          % (name, photon_frac(f, nu_HeI), photon_frac(f, nu_HeII)))

# Pop III photon-number ratios, Murphy et al. 2021 Table 4 (non-rotating,
# alpha_ov = 0.1); SB13 IMF alpha=0.17 and Salpeter alpha=2.35
print("\n  Pop III photon-number ratios (Murphy+2021, Table 4)")
for lab, NH, NHe1, NHe2 in [("SB13 IMF  a=0.17", 1.1264e62, 4.3221e61, 2.0709e60),
                            ("Salpeter  a=2.35", 8.3354e61, 2.5874e61, 8.0269e59)]:
    aeff1 = -np.log(NHe1/NH)/np.log(E_HeI/E_HI)
    aeff2 = -np.log(NHe2/NH)/np.log(E_HeII/E_HI)
    print("      %-18s N_HeI/N_H = %.3f  N_HeII/N_H = %.4f   "
          "(effective alpha: %.2f below He II, %.2f overall)"
          % (lab, NHe1/NH, NHe2/NH, aeff1, aeff2))

# ----------------------------------------------------------------------
# 7.  Independent cross-check: rebuild Gamma_HI from eps_912 and lambda_mfp
#     Local-source approximation, 4 pi J_nu = eps_nu^proper * lambda^proper
# ----------------------------------------------------------------------
z = 6.0
eps912_com = 0.929e25 / Mpc**3                      # erg s^-1 Hz^-1 ccm^-3
eps912_pro = eps912_com * (1+z)**3                  # proper
lam_pro    = (8.318/hub) / (1+z) * Mpc              # h^-1 cMpc -> proper cm
alpha = 2.0
Gamma_check = eps912_pro * 6.30e-18 * lam_pro / (h_pl * (alpha + 3))
print("\nCross-check (local-source approximation, independent of Gamma_HI):")
print("  eps_912 proper = %.3e erg s^-1 Hz^-1 cm^-3" % eps912_pro)
print("  lambda proper  = %.3e cm  (= %.2f pMpc)" % (lam_pro, lam_pro/Mpc))
print("  Gamma_HI(rebuilt) = %.3e s^-1   vs measured %.3e s^-1  -> ratio %.2f"
      % (Gamma_check, G_HI["z=6.0"][0], Gamma_check/G_HI["z=6.0"][0]))
