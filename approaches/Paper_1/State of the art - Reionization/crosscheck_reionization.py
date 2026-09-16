"""Independent cross-checks for the reionization summary.
All INPUT numbers are cited in the .tex; nothing here is invented."""
import sympy as sp

# --- Physical constants (CODATA 2018) and cosmology (Planck 2018 TT,TE,EE+lowE+lensing) ---
rho_crit_h2 = 1.87834e-29   # g cm^-3 ; critical density / h^2  (Planck 2018 VI, Table 1 conventions)
Ob_h2       = 0.02237       # Planck Collaboration 2018 (arXiv:1807.06209), Table 1
h           = 0.6736
Yp          = 0.2454        # Planck 2018 VI BBN-consistent helium mass fraction
Xp          = 1 - Yp
m_H         = 1.67353e-24   # g  (hydrogen atom mass)
Mpc_cm      = 3.0856775814913673e24
Gyr_s       = 3.1556952e16
Myr_s       = 3.1556952e13

# 1) Comoving mean hydrogen number density
n_H = Xp * Ob_h2 * rho_crit_h2 / m_H          # cm^-3 (comoving)
print(f"[1] <n_H> comoving  = {n_H:.4e} cm^-3  = {n_H*Mpc_cm**3:.4e} cMpc^-3")

# 2) Ionizing photons per H atom per Gyr for the measured emissivity of
#    Gaikwad et al. 2023 (arXiv:2304.02038), Table 3: ndot ~ 0.6e51 s^-1 cMpc^-3 at 4.9<z<6.0
ndot = 0.6e51                                  # s^-1 cMpc^-3
rate = ndot / (n_H * Mpc_cm**3)                # s^-1 per H atom
print(f"[2] photons per H per Gyr = {rate*Gyr_s:.2f}")

# 3) Recombination time (Robertson et al. 2015, arXiv:1502.02024, eq. 5)
#    t_rec = [C_HII alpha_B(T) (1+Yp/4Xp) <n_H> (1+z)^3]^-1
alpha_B = 1.43e-13    # cm^3 s^-1 at T = 2e4 K, case B (Osterbrock & Ferland 2006,
                      # Table 2.1).  Robertson et al. 2015 evaluate alpha_B at
                      # T = 20,000 K.  NOTE: 2.59e-13 is the 1e4 K value, not 2e4 K.
for C in (1.0, 3.0):
    for z in (6.0, 7.0, 8.0):
        t_rec = 1.0/(C*alpha_B*(1+Yp/(4*Xp))*n_H*(1+z)**3)
        print(f"[3] C_HII={C:.0f} z={z:.0f}: t_rec = {t_rec/Myr_s:7.1f} Myr")

# 4) Dimensional check of the reionization ODE, Qdot = ndot/<n_H> - Q/t_rec
from sympy.physics.units import time as T, length as L
from sympy.physics.units.systems.si import dimsys_SI
ndot_d = 1/(T*L**3)          # [ndot]  = s^-1 cMpc^-3
nH_d   = 1/L**3              # [<n_H>] = cMpc^-3
print(f"[4] sympy dimensional check  [ndot/<n_H>] == [Q/t_rec] : "
      f"{dimsys_SI.equivalent_dims(ndot_d/nH_d, 1/T)}")
trec_d = 1/((L**3/T)*(1/L**3))   # 1/(alpha_B * n)
print(f"[4] sympy dimensional check  [t_rec] == [time]        : {dimsys_SI.equivalent_dims(trec_d, T)}")

# 5) ndot from fesc*xi_ion*rho_SFR (Robertson et al. 2015 eq. 1) with JWST-era values
#    xi_ion here in Hz erg^-1 must be paired with rho_UV (erg s^-1 Hz^-1 cMpc^-3)
#    Sanity: Robertson+15 fiducial log10 xi_ion = 53.14 [s^-1 / (Msun/yr)], fesc=0.2
xi_sfr = 10**53.14
for rho_sfr in (0.01, 0.02):   # Msun yr^-1 cMpc^-3, order of magnitude at z~6-7 (Madau & Dickinson 2014 compilation)
    print(f"[5] rho_SFR={rho_sfr}: ndot = {0.2*xi_sfr*rho_sfr:.2e} s^-1 cMpc^-3")
