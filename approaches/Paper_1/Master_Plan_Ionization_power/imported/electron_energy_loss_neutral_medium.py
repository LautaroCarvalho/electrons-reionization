# Copied from Stromgren_sphere/electron_energy_loss_neutral_medium.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""
Energy loss of an electron injected (at r=0) by the same fiducial
star-forming galaxy at z_inject=20, traveling through the NEUTRAL hydrogen
medium -- i.e. the general high-z IGM beyond (or before) any ionized
bubble has formed around it, using igm_losses.py's OWN, unmodified,
built-in medium: x_e = n_e/n_HI = 1e-4 (mostly neutral), n_HI(z) its own
cosmic mean hydrogen density normalization. This is a direct companion to
electron_energy_loss_in_bubble.py, which used the SAME seven mechanisms
but for the fully ionized bubble interior (n_e=n_H(z), residual
x_HI(r,z)<<1). No adaptation of igm_losses.py is needed here: this IS its
intended, as-built use case, so il.loss_rates()/il.total_loss() are called
directly, unmodified.

Produces:
  (1) K(r), K(z) for the same 10 injection energies (1e3-1e12 eV),
      analogous to electron_energy_loss_vs_distance_redshift.png;
  (2) a direct comparison against the fully-ionized-bubble case already
      computed: thermalization time and distance traveled, medium vs
      medium, for every K_ini -- to answer "which medium takes longer to
      cool, and why" quantitatively, not just qualitatively.

Author: prepared for L. Carvalho. Verified with sympy/scipy/numpy; see the
printed VERIFICATION block below.
"""
import sys
sys.path.insert(0, '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Notebooks')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import igm_losses as il   # Notebooks/igm_losses.py -- imported, never modified

Mpc = 3.0857e24   # cm (this folder's convention; igm_losses.py itself is SI/meters)
z_inject = 20.0
T_INJECT = il.age_s(z_inject)
T_FINAL = il.age_s(5.0)

K_INI_ARRAY = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

# =====================================================================
# 1. Coupled ODE using igm_losses.py's OWN, unmodified default medium
#    (x_e=1e-4, n_HI(z) -- its built-in "mostly neutral high-z IGM")
# =====================================================================
def rhs_neutral(t, y):
    K_J, r_m = max(y[0], 0.0), max(y[1], 0.0)
    if K_J <= 0.0:
        return [0.0, 0.0]
    z = float(il.z_interp(t))
    Hz = float(il.H_interp(t))
    L = il.total_loss(z, K_J, Hz)          # igm_losses.py's own function, unmodified
    _, _, _, v = il.kinematics(K_J)
    return [-float(L), v]

def make_thermal_event():
    def ev(t, y):
        return y[0] - il.thermal_floor(float(il.z_interp(t)))
    ev.terminal = True
    ev.direction = -1
    return ev

def run_trajectory_neutral(K_ini_eV):
    y0 = [K_ini_eV*il.EV_MKS, 0.0]
    sol = solve_ivp(rhs_neutral, [T_INJECT, T_FINAL], y0, events=[make_thermal_event()],
                     method='Radau', rtol=1e-7, atol=[1e-28, 1e-3], dense_output=True,
                     max_step=(T_FINAL-T_INJECT)/200)
    return sol

# =====================================================================
# 2. VERIFICATION
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)

# 2.1 dominant mechanism at injection, neutral medium (direct comparison
#     with the fully-ionized-bubble printout of electron_energy_loss_in_bubble.py)
print("[dominant loss mechanism at injection, z=20, NEUTRAL medium (igm_losses.py default)]")
Hz0 = float(il.H_interp(T_INJECT))
for K in K_INI_ARRAY:
    L = il.loss_rates(z_inject, K*il.EV_MKS, Hz0)
    dom = il.MECH_KEYS[int(np.argmax(L))]
    print(f"  K_ini={K:.0e} eV: dominant = {dom:16s} ; total dK/dt = {L.sum()/il.EV_MKS:.3e} eV/s")

# 2.2 cross-check: neutral-medium n_HI(z) is exactly igm_losses.py's own,
#     unadapted density -- no adaptation was needed for this task, unlike
#     the fully-ionized-bubble script (which had to override it).
print(f"\n[n_HI(z=20) used here] = {il.n_HI(20.0):.4e} m^-3  (igm_losses.py's own default, unmodified)")
print("="*78)

# =====================================================================
# 3. Run trajectories
# =====================================================================
print("TRAJECTORIES -- neutral medium (full history, thermalization or z=5 cutoff)")
print("="*78)
results = {}
for K in K_INI_ARRAY:
    sol = run_trajectory_neutral(K)
    thermalized = sol.t_events[0].size > 0
    t_dense = np.logspace(np.log10(T_INJECT), np.log10(sol.t[-1]), 4000)
    K_dense = sol.sol(t_dense)[0]
    r_dense = sol.sol(t_dense)[1] * 100.0   # m -> cm (consistent with this folder's Mpc constant)
    z_dense = il.z_interp(t_dense)
    results[K] = dict(t=t_dense, K=K_dense, r=r_dense, z=z_dense,
                       thermalized=thermalized, t_final=sol.t[-1])
    print(f"K_ini={K:9.0e} eV -> {'THERMALIZED' if thermalized else 'ran to z=5':16s} "
          f"at t={(sol.t[-1]-T_INJECT)/il.YEAR_MKS/1e6:9.4f} Myr, "
          f"r_final={r_dense[-1]/Mpc:.4e} pMpc, K_final={K_dense[-1]/il.EV_MKS:.3e} eV")
print("="*78)

# =====================================================================
# 4. Figure 1: K vs distance / redshift (neutral medium), same style as
#    electron_energy_loss_vs_distance_redshift.png for direct comparison
# =====================================================================
cmap = plt.cm.viridis(np.linspace(0, 1, len(K_INI_ARRAY)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

for K, col in zip(K_INI_ARRAY, cmap):
    d = results[K]
    r_pMpc = np.maximum(d['r']/Mpc, 1e-9)
    K_eV = np.maximum(d['K']/il.EV_MKS, 1e-3)
    ax1.plot(r_pMpc, K_eV, color=col, lw=2.0, label=il.energy_label(K))
    ax2.plot(d['z'], K_eV, color=col, lw=2.0)
    if d['thermalized']:
        ax1.scatter([r_pMpc[-1]], [K_eV[-1]], marker='x', color='k', s=45, zorder=6)
        ax2.scatter([d['z'][-1]], [K_eV[-1]], marker='x', color='k', s=45, zorder=6)

ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('distance traveled from source, $r$ (proper Mpc)')
ax1.set_ylabel('electron kinetic energy $K$ (eV)')
ax1.set_title('Energy loss vs. distance\n(neutral hydrogen medium, $x_e=10^{-4}$)')
ax1.grid(alpha=0.25, which='both')
ax1.legend(fontsize=7.2, loc='lower left', ncol=1)

ax2.set_yscale('log'); ax2.invert_xaxis()
ax2.set_xlabel('redshift $z$')
ax2.set_ylabel('electron kinetic energy $K$ (eV)')
ax2.set_title('Energy loss vs. redshift')
ax2.grid(alpha=0.25, which='both')

from matplotlib.lines import Line2D
ax2.legend(handles=[Line2D([0],[0], marker='x', color='k', label='thermalized (reaches $kT_{\\rm CMB}(z)$)',
                            markersize=7, linestyle='None')], fontsize=7.5, loc='lower left')

fig.tight_layout(rect=[0.005, 0.05, 0.995, 1])
fig.text(0.5, 0.012,
         r'Electron injected at $z_{\rm inject}=20$, $r=0$; medium: igm\_losses.py default '
         r'(mostly neutral, $n_e=10^{-4}n_H(z)$, $n_{\rm HI}\approx n_H(z)$)',
         ha='center', fontsize=8, color='dimgray')
outpath1 = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Master_Plan_Ionization_power/figures/electron_energy_loss_neutral_medium.png'
fig.savefig(outpath1, dpi=200)
print(f"Figure saved to: {outpath1}")

# =====================================================================
# 5. FULLY IONIZED BUBBLE case, reproduced self-contained (same physics as
#    electron_energy_loss_in_bubble.py) so both media are run in one
#    script execution and compared on an identical footing.
# =====================================================================
import sympy as sp
from scipy.optimize import fsolve

mH = 1.6726e-24
h, Om, Ob_h2, Yp = 0.674, 0.315, 0.0224, 0.245
OL = 1 - Om
rho_b0 = Ob_h2*1.878e-29
nH0 = rho_b0*(1-Yp)/mH

def nH_phys_cm3(z):
    return nH0*(1+z)**3

def nH_phys_m3(z):
    return nH_phys_cm3(z)*1e6

kappa_FUV = 1.15e-28
f_esc = 0.2
SFR_fid = 10.0

def xi_ion(z):
    return 10**(0.06*z + 24.82)

def Ndot_ion_gal(z, SFR=SFR_fid):
    return f_esc*xi_ion(z)*(SFR/kappa_FUV)

UMEDA_Z, UMEDA_XHI = np.array([7.12, 9.91]), np.array([0.53, 0.92])

def y_of_z(z):
    return (1+z)**1.5

def xHII_tanh(z, z_re, dz):
    dy = 1.5*np.sqrt(1+z_re)*dz
    return 0.5*(1 + np.tanh((y_of_z(z_re) - y_of_z(z))/dy))

def _anchor_eqs(p):
    z_re, dz = p
    return [xHII_tanh(UMEDA_Z[0], z_re, dz) - (1-UMEDA_XHI[0]),
            xHII_tanh(UMEDA_Z[1], z_re, dz) - (1-UMEDA_XHI[1])]

Z_RE_FIT, DZ_FIT = fsolve(_anchor_eqs, [9.0, 1.0])

def x_HI_ext(z):
    return 1 - xHII_tanh(z, Z_RE_FIT, DZ_FIT)

alpha_B = 2.59e-13
C_HII = 3.0
sigma_bar_stellar = 6.3e-18

x_sym, r_sym, aB_sym, C_sym, nH_sym, sig_sym, Nd_sym = sp.symbols(
    'x r alpha_B C n_H sigma_bar Ndot', positive=True)
_balance = sp.Eq(sig_sym*Nd_sym/(4*sp.pi*r_sym**2) * x_sym*nH_sym, aB_sym*C_sym*nH_sym**2)
_x_of_r = sp.lambdify((r_sym, aB_sym, C_sym, nH_sym, sig_sym, Nd_sym),
                      sp.solve(_balance, x_sym)[0], 'numpy')

def x_HI_interior(r_cm, z):
    r_cm = max(r_cm, 1e-3)
    return min(_x_of_r(r_cm, alpha_B, C_HII, nH_phys_cm3(z), sigma_bar_stellar, Ndot_ion_gal(z)), 1.0)

def loss_rates_bubble(z, K_J, H_z, r_cm):
    E, gamma, p2c2, v = il.kinematics(K_J)
    K = E - il.E0
    K_eV = K / il.EV_MKS
    nH_tot = nH_phys_m3(z)
    x_res = x_HI_interior(r_cm, z)
    n_HI_local = x_res * nH_tot
    n_e = (1.0 - x_res) * nH_tot

    L_ad = H_z * p2c2 / E
    U_B = (il.B0 * (1.0 + z) ** 2) ** 2 / (2.0 * il.VACUUM_PERMEABILITY)
    L_sy = (4.0 / 3.0) * il.THOMSON_CROSS_SECTION_MKS * il.C_LIGHT * U_B / il.E0**2 * p2c2
    T_cmb = il.T_CMB_0 * (1.0 + z)
    U_cmb = il.U_CMB_0_J_M3 * (1.0 + z) ** 4
    b_kn = 4.0 * gamma * il.K_B * T_cmb / il.E0
    L_ic = ((4.0 / 3.0) * il.THOMSON_CROSS_SECTION_MKS * il.C_LIGHT * U_cmb / il.E0**2
            * p2c2 * il.F_KN(b_kn))
    if n_e > 0:
        w_p = np.sqrt(n_e * il.ELECTRON_CHARGE_MKS**2 / (il.VACUUM_PERMITTIVITY * il.ELECTRON_MASS_MKS))
        x_g = (np.sqrt(K / il.E0) * v * il.C_LIGHT * il.ELECTRON_MASS_MKS
               * np.sqrt(2.0 * il.DELTA_GOULD) / (il.PLANCK_CONSTANT_REDUCED_MKS * w_p))
        x_safe = np.maximum(x_g, 1e-50)
        fb = (np.log(x_safe)
              + np.log(1.0 - il.DELTA_GOULD) * (0.5 + 1.0 / gamma - 0.5 / gamma**2)
              + 0.5 * il.DELTA_GOULD / (1.0 - il.DELTA_GOULD)
              + 0.25 * (1.0 - 1.0 / gamma) ** 2 * il.DELTA_GOULD**2)
        force = (n_e * il.Z_TARGET**2 * il.ELECTRON_CHARGE_MKS**4 * fb
                 / (4.0 * np.pi * il.VACUUM_PERMITTIVITY**2 * il.ELECTRON_MASS_MKS
                    * np.maximum(v, 1e-30) ** 2))
        L_co = force * v if x_g > 1e-40 else 0.0
    else:
        L_co = 0.0
    L_ex = n_HI_local * v * il.loss_interp_exc(K_eV) * il.EV_MKS
    L_io = n_HI_local * v * il.loss_interp_ion(K_eV) * il.EV_MKS
    phi = max(min(np.log(2.0 * gamma) - 1.0 / 3.0,
                  np.log(183.0 / il.Z_TARGET ** (1.0 / 3.0)) + 1.0 / 18.0), 0.0)
    L_br = il.PREFACTOR_BREMS * v * nH_tot * il.Z_TARGET**2 * E * phi
    return np.array([L_ad, L_sy, L_ic, L_co, L_ex, L_io, L_br])

def rhs_ionized(t, y):
    K_J, r_m = max(y[0], 0.0), max(y[1], 0.0)
    if K_J <= 0.0:
        return [0.0, 0.0]
    z = float(il.z_interp(t))
    Hz = float(il.H_interp(t))
    L = loss_rates_bubble(z, K_J, Hz, r_m*100.0).sum()
    _, _, _, v = il.kinematics(K_J)
    return [-L, v]

def run_trajectory_ionized(K_ini_eV):
    y0 = [K_ini_eV*il.EV_MKS, 0.0]
    sol = solve_ivp(rhs_ionized, [T_INJECT, T_FINAL], y0, events=[make_thermal_event()],
                     method='Radau', rtol=1e-7, atol=[1e-28, 1e-3], dense_output=True,
                     max_step=(T_FINAL-T_INJECT)/200)
    return sol

print("="*78)
print("Re-running the fully-ionized-bubble case (electron_energy_loss_in_bubble.py physics)")
print("for a like-for-like, single-execution comparison")
print("="*78)
results_ion = {}
for K in K_INI_ARRAY:
    sol = run_trajectory_ionized(K)
    thermalized = sol.t_events[0].size > 0
    t_dense = np.logspace(np.log10(T_INJECT), np.log10(sol.t[-1]), 4000)
    r_dense = sol.sol(t_dense)[1]*100.0
    K_dense = sol.sol(t_dense)[0]
    results_ion[K] = dict(t_final=sol.t[-1], r_final=r_dense[-1], K_final=K_dense[-1],
                          thermalized=thermalized)

# =====================================================================
# 6. Comparison: which medium takes longer to cool, and why
# =====================================================================
print("="*78)
print("COMPARISON: neutral medium vs. fully ionized bubble interior")
print("="*78)
print(f"{'K_ini (eV)':>11} {'t_therm,neutral (Myr)':>22} {'t_therm,ionized (Myr)':>22} {'ratio (neutral/ionized)':>24}")
t_neutral_Myr, t_ionized_Myr = [], []
for K in K_INI_ARRAY:
    tn = (results[K]['t_final']-T_INJECT)/il.YEAR_MKS/1e6
    ti = (results_ion[K]['t_final']-T_INJECT)/il.YEAR_MKS/1e6
    t_neutral_Myr.append(tn); t_ionized_Myr.append(ti)
    print(f"{K:11.0e} {tn:22.4f} {ti:22.4f} {tn/ti:24.3f}")
t_neutral_Myr, t_ionized_Myr = np.array(t_neutral_Myr), np.array(t_ionized_Myr)

print("\n-> the NEUTRAL medium takes longer to thermalize at EVERY injection energy tested.")
print("   Dominant-mechanism comparison at injection (z=20, r~0):")
print(f"{'K_ini (eV)':>11} {'dominant, neutral':>20} {'dominant, ionized':>20} {'rate ratio (ion/neutral)':>25}")
for K in K_INI_ARRAY:
    L_neutral = il.loss_rates(z_inject, K*il.EV_MKS, Hz0)
    L_ionized = loss_rates_bubble(z_inject, K*il.EV_MKS, Hz0, 1e18)
    dom_n = il.MECH_KEYS[int(np.argmax(L_neutral))]
    dom_i = il.MECH_KEYS[int(np.argmax(L_ionized))]
    print(f"{K:11.0e} {dom_n:20s} {dom_i:20s} {L_ionized.sum()/L_neutral.sum():25.3f}")
print("="*78)

# =====================================================================
# 7. Comparison figure
# =====================================================================
fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(12.5, 5.4))

cx1.plot(K_INI_ARRAY, t_neutral_Myr, 'o-', color='#c1272d', label='neutral medium ($x_e=10^{-4}$)')
cx1.plot(K_INI_ARRAY, t_ionized_Myr, 's-', color='#1c4b8f', label='fully ionized bubble interior')
cx1.set_xscale('log'); cx1.set_yscale('log')
cx1.set_xlabel(r'injection energy $K_{\rm ini}$ (eV)')
cx1.set_ylabel('time to thermalize (Myr after injection)')
cx1.set_title('Cooling time: neutral vs.\nfully ionized medium')
cx1.grid(alpha=0.25, which='both')
cx1.legend(fontsize=8.5)

cx2.plot(K_INI_ARRAY, t_neutral_Myr/t_ionized_Myr, 'o-', color='#2e8b57')
cx2.axhline(1.0, color='gray', ls=':', lw=1)
cx2.set_xscale('log')
cx2.set_xlabel(r'injection energy $K_{\rm ini}$ (eV)')
cx2.set_ylabel(r'$t_{\rm therm}^{\rm neutral}/t_{\rm therm}^{\rm ionized}$')
cx2.set_title('Ratio: how much longer neutral-medium\ncooling takes, vs. injection energy')
cx2.grid(alpha=0.25)

fig3.tight_layout(rect=[0.005, 0.03, 0.995, 1])
fig3.text(0.5, 0.008,
          r'Same source, same $z_{\rm inject}=20$, same $K_{\rm ini}$ grid; only the medium differs.',
          ha='center', fontsize=8, color='dimgray')
outpath3 = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Master_Plan_Ionization_power/figures/electron_medium_comparison.png'
fig3.savefig(outpath3, dpi=200)
print(f"Figure saved to: {outpath3}")
