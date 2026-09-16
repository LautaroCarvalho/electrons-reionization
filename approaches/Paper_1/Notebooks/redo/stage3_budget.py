"""
Stage 3a: the two downstream numbers of Sect. 9.4 of manuscript.tex, redone.

(i) the Thomson-depth increment implied by a given N_ion/n_H;
(ii) an INDEPENDENT source budget for the cosmic-ray electron energy density,
     which the manuscript does not do: it infers the leptonic contribution by
     borrowing the hadronic Delta T = 10-200 K of Leite et al. (2017).
"""
import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18

SIGMA_T = 6.652458732e-29           # m^2
C = 2.99792458e8
KB_EV = 8.617333262e-5              # eV/K
ERG = 1.0e-7                        # J
MPC3_CM3 = (3.0856775814913673e24) ** 3

rho_b = (Planck18.Ob0 * Planck18.critical_density0).to(u.kg / u.m**3).value
X_H = 0.7535
nH0_cm3 = rho_b * X_H / 1.6735575e-27 * 1e-6      # comoving, cm^-3
MU = 1.0 + (rho_b * (1 - X_H) / (4 * 1.6735575e-27)) / (rho_b * X_H / 1.6735575e-27)


def tau_full(z1, z2):
    z = np.linspace(z1, z2, 20001)
    Hz = Planck18.H(z).to(u.s**-1).value
    ne = nH0_cm3 * 1e6 * (1 + z) ** 3
    return float(np.trapz(SIGMA_T * ne * C / ((1 + z) * Hz), z))


def theta(Q_eV, mu=MU):
    """Delta T per unit N_ion/n_H  [K]."""
    return (2.0 / 3.0) * Q_eV / (mu * KB_EV)


print(f"comoving n_H  = {nH0_cm3:.3e} cm^-3   (manuscript: 1.94e-7)")
print(f"mu (H+He, neutral) = {MU:.4f}          (manuscript: 1.081)")

print("\n--- the lock, Eq. (16) ---")
for Q in (6.0, 6.5, 9.6):
    print(f"  Q = {Q:4.1f} eV/ionization  ->  Theta = {theta(Q):.3e} K "
          f"(manuscript quotes 4.3e4 - 6.9e4)")

print("\n--- (i) implied ionizations and Thomson depth ---")
tau_eor = tau_full(5.5, 20.0)
print(f"  tau of a fully ionized IGM over 5.5 < z < 20 : {tau_eor:.4f}")
print(f"  {'DeltaT [K]':>10} {'N_ion/n_H':>11} {'dtau':>10} {'dtau/sigma(tau)':>16}")
for dT in (10.0, 200.0):
    X = dT / theta(6.0)
    print(f"  {dT:10.0f} {X:11.3e} {X*tau_eor:10.3e} {X*tau_eor/0.007:16.3f}")

print("\n--- (ii) independent cosmic-ray electron budget from star formation ---")
print("  u_e(comoving) = eps_CR * (K_e/p) * (E_SN/M_*) * rho_*(z_i->z_f)")
E_SN, M_PER_SN = 1.0e51, 100.0                    # erg, M_sun per SN
cases = [("conservative", 0.05, 0.01, 2.0e-3),
         ("fiducial",     0.10, 0.02, 5.0e-3),
         ("optimistic",   0.15, 0.05, 1.5e-2)]
t10 = Planck18.age(10.0).to(u.yr).value
t6 = Planck18.age(6.0).to(u.yr).value
dt = t6 - t10
print(f"  integration window z=10 -> 6 : {dt/1e6:.0f} Myr")
print(f"  {'case':>13} {'eps_CR':>7} {'K_e/p':>7} {'SFRD':>9} {'u_e [erg/cm3]':>14} "
      f"{'eps_H [eV]':>11} {'N_ion/n_H':>11} {'DeltaT [K]':>11}")
for name, eps_cr, kep, sfrd in cases:
    rho_star = sfrd * dt                                   # M_sun / Mpc^3
    u_e = rho_star * (E_SN / M_PER_SN) * eps_cr * kep / MPC3_CM3   # erg/cm^3
    eps_H = u_e / nH0_cm3 / (1.602176634e-12)              # eV per H atom
    for Wbar, tag in ((37.4, "soft p=2.5"), (107.0, "DSA p=2.0")):
        X = eps_H / Wbar
        dT = X * theta(6.0)
        print(f"  {name+' '+tag:>24} {eps_cr:5.2f} {kep:7.3f} {sfrd:9.1e} "
              f"{u_e:14.2e} {eps_H:11.2e} {X:11.2e} {dT:11.2e}")

print("\n  For comparison, the energy density needed for ONE ionization per H:")
for Wbar in (37.4, 107.0, 290.0):
    print(f"    Wbar = {Wbar:6.1f} eV  ->  u_e = {nH0_cm3*Wbar*1.602176634e-12:.2e}"
          f" erg/cm^3 comoving")
