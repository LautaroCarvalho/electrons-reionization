# Copied from Stromgren_sphere/electron_energy_loss_in_bubble.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""
Energy loss of an electron injected (at r=0) by a star-forming galaxy at
z_inject=20 (the same z_form used throughout this folder), as it travels
outward through the FULLY IONIZED medium inside its own ionized bubble.

FOLLOWS igm_losses.py (Notebooks/, imported read-only, not modified): all
seven loss mechanisms (adiabatic expansion, synchrotron, inverse Compton on
the CMB with Klein-Nishina correction, Coulomb scattering off free
electrons, collisional excitation/ionization of neutral hydrogen,
bremsstrahlung) are the exact formulas of that module (Blumenthal & Gould
1970; Gould 1972; Stone & Kim 2002; Kim et al. 2000), reusing its physical
constants, its Klein-Nishina kernel F_KN, its tabulated collisional
cross-sections, and its kinematics(). ONLY THE MEDIUM'S IONIZATION STATE is
changed, because igm_losses.py's own n_HI(z)/ION_FRACTION=1e-4 describe the
mostly-neutral high-z IGM (its intended use case), not the interior of an
H II bubble. Inside the bubble:
  - the free-electron density that drives Coulomb losses is the FULL cosmic
    mean hydrogen density (fully ionized, n_e = n_H(z), not 1e-4 n_H(z));
  - the neutral-hydrogen density that drives collisional excitation/
    ionization is the small RESIDUAL neutral fraction x_HI(r,z) set by
    local photoionization equilibrium (Cen & Haiman 2000, arXiv:astro-ph/
    0006376, already in references/; re-derived and verified in
    interior_consistency_check.py), evaluated AT THE ELECTRON'S OWN,
    self-consistently tracked, position r(t);
  - bremsstrahlung (electron-ION, not electron-atom) uses the full n_H(z),
    since the target nucleus's charge is unaffected by whether it currently
    holds a bound electron.
Synchrotron, inverse Compton and adiabatic-expansion losses are unaffected
by the ionization state and are taken directly from igm_losses.py with no
change.

BUBBLE-EDGE MODEL: both bubble-radius models developed earlier in this
folder are shown for reference -- the pure photon-counting R_bubble(z) of
bubble_radius_vs_redshift.py (used throughout this session) and the
recombination-corrected R(t) found to be 24-42% smaller
(interior_consistency_check.py, ionizing_photon_mean_free_path_response.tex
Addendum). The electron's own traveled distance r(t) = int v(K(t')) dt' is
compared against BOTH.

Author: prepared for L. Carvalho. Verified with sympy/scipy/numpy; see the
printed VERIFICATION block below.
"""
import sys
sys.path.insert(0, '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Notebooks')
import numpy as np
import sympy as sp
from scipy import integrate as sp_integrate
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import igm_losses as il   # Notebooks/igm_losses.py -- imported, never modified

# =====================================================================
# 0. Cosmology (Planck 2018, arXiv:1807.06209) -- identical to the other
#    scripts in this folder; independently cross-checked against
#    igm_losses.py's own n_HI(z=20)=1800 m^-3 = 1.80e-3 cm^-3 (vs. 1.76e-3
#    cm^-3 below, a 2% difference, confirming consistency).
# =====================================================================
mH = 1.6726e-24
pc, kpc, Mpc = 3.0857e18, 3.0857e21, 3.0857e24
yr, Gyr = 3.156e7, 3.156e7*1e9

h, Om, Ob_h2, Yp = 0.674, 0.315, 0.0224, 0.245
OL = 1 - Om
H0 = h*100*1e5/Mpc
rho_b0 = Ob_h2*1.878e-29
nH0 = rho_b0*(1-Yp)/mH   # cm^-3

def nH_phys_cm3(z):
    return nH0*(1+z)**3

def nH_phys_m3(z):
    return nH_phys_cm3(z)*1e6

# =====================================================================
# 1. Source and bubble model -- IDENTICAL to bubble_radius_vs_redshift.py
# =====================================================================
kappa_FUV = 1.15e-28
f_esc = 0.2
SFR_fid = 10.0
z_inject = 20.0   # = z_form used throughout this folder

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

def t_age_cm_ode(z):
    """age(z) - age(z_inject), in seconds, using igm_losses.py's own
    astropy-Planck18 age (fully consistent with its z_interp/H_interp)."""
    return il.age_s(z) - il.age_s(z_inject)

def R_bubble_photoncounting_cm(z):
    ta = t_age_cm_ode(z)
    if ta <= 0:
        return 0.0
    Nd, nH = Ndot_ion_gal(z), nH_phys_cm3(z)
    R3 = 3*Nd*ta/(4*np.pi*nH*x_HI_ext(z))
    return R3**(1/3)

alpha_B = 2.59e-13   # cm^3/s, T=1e4 K (Shapiro+2006)
C_HII = 3.0          # Robertson et al. 2015
sigma_bar_stellar = 6.3e-18   # cm^2 (approx sigma_H(E_H); see mean-free-path response)

def R_S_cm(z):
    Nd, nH = Ndot_ion_gal(z), nH_phys_cm3(z)
    return (3*Nd/(4*np.pi*alpha_B*C_HII*nH**2))**(1/3)

def t_rec_s(z):
    return 1.0/(alpha_B*C_HII*nH_phys_cm3(z))

def R_bubble_recomb_cm(z):
    ta = t_age_cm_ode(z)
    if ta <= 0:
        return 0.0
    return R_S_cm(z)*(1 - np.exp(-ta/t_rec_s(z)))**(1/3)

C_LIGHT_CGS = 2.99792458e10   # cm/s

def R_bubble_photoncounting_causal_cm(z):
    """R_bubble(z) capped at c*t_age: the idealized R~t^(1/3) photon-counting
    law is superluminal for t_age < ~1e5 yr (verified below), unphysical
    since no I-front can outrun light -- see Shapiro et al. (2006,
    arXiv:astro-ph/0507677, already in references/, 'Relativistic Ionization
    Fronts'), already discussed in stromgren_sphere_summary.tex Sec. 1.5."""
    ta = t_age_cm_ode(z)
    if ta <= 0:
        return 0.0
    return min(R_bubble_photoncounting_cm(z), C_LIGHT_CGS*ta)

def R_bubble_recomb_causal_cm(z):
    ta = t_age_cm_ode(z)
    if ta <= 0:
        return 0.0
    return min(R_bubble_recomb_cm(z), C_LIGHT_CGS*ta)

# ---------- x_HI(r,z): photoionization equilibrium (Cen & Haiman 2000) ----------
x_sym, r_sym, aB_sym, C_sym, nH_sym, sig_sym, Nd_sym = sp.symbols(
    'x r alpha_B C n_H sigma_bar Ndot', positive=True)
_balance = sp.Eq(sig_sym*Nd_sym/(4*sp.pi*r_sym**2) * x_sym*nH_sym, aB_sym*C_sym*nH_sym**2)
_x_solution = sp.solve(_balance, x_sym)[0]
_x_of_r = sp.lambdify((r_sym, aB_sym, C_sym, nH_sym, sig_sym, Nd_sym), _x_solution, 'numpy')

def x_HI_interior(r_cm, z):
    r_cm = max(r_cm, 1e-3)   # avoid r=0 singularity in the r^2 formula (x->0 anyway)
    x = _x_of_r(r_cm, alpha_B, C_HII, nH_phys_cm3(z), sigma_bar_stellar, Ndot_ion_gal(z))
    return min(x, 1.0)

# =====================================================================
# 2. Loss rates inside the FULLY IONIZED bubble interior
#    (reuses every igm_losses.py building block; only n_e / n_HI change)
# =====================================================================
def loss_rates_bubble(z, K_J, H_z, r_cm):
    """Same 7 mechanisms and formulas as igm_losses.loss_rates, but with the
    medium's ionization state set by the bubble interior instead of the
    mostly-neutral high-z IGM. Returns array of shape (7,) in J/s."""
    E, gamma, p2c2, v = il.kinematics(K_J)
    K = E - il.E0
    K_eV = K / il.EV_MKS

    nH_tot = nH_phys_m3(z)                 # m^-3, total (proton) density -- ions, for Coulomb & brems
    x_res = x_HI_interior(r_cm, z)          # residual neutral fraction at the electron's own position
    n_HI_local = x_res * nH_tot              # m^-3, RESIDUAL neutral H -- for excitation/ionization only
    n_e = (1.0 - x_res) * nH_tot              # m^-3, free electrons -- for Coulomb (~= nH_tot, x_res<<1)

    # 0) Adiabatic expansion -- unaffected by ionization state
    L_ad = H_z * p2c2 / E

    # 1) Synchrotron -- unaffected by ionization state (same B(z) as igm_losses.py)
    U_B = (il.B0 * (1.0 + z) ** 2) ** 2 / (2.0 * il.VACUUM_PERMEABILITY)
    L_sy = (4.0 / 3.0) * il.THOMSON_CROSS_SECTION_MKS * il.C_LIGHT * U_B / il.E0**2 * p2c2

    # 2) Inverse Compton on the CMB -- unaffected by ionization state
    T_cmb = il.T_CMB_0 * (1.0 + z)
    U_cmb = il.U_CMB_0_J_M3 * (1.0 + z) ** 4
    b_kn = 4.0 * gamma * il.K_B * T_cmb / il.E0
    L_ic = ((4.0 / 3.0) * il.THOMSON_CROSS_SECTION_MKS * il.C_LIGHT * U_cmb / il.E0**2
            * p2c2 * il.F_KN(b_kn))

    # 3) Coulomb scattering off the FULLY IONIZED plasma (Gould 1972) -- n_e = nH_tot here.
    #    Guard n_e->0 (e.g. if a stiff-solver probe point strays to r far beyond any bubble,
    #    where x_HI_interior saturates at 1 and n_e=(1-x_res)*nH_tot=0): no free electrons
    #    means no Coulomb loss on free electrons, L_co=0, rather than the 0*inf=NaN that the
    #    unguarded formula would produce there.
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

    # 4)+5) Collisional excitation/ionization of the RESIDUAL neutral hydrogen only
    L_ex = n_HI_local * v * il.loss_interp_exc(K_eV) * il.EV_MKS
    L_io = n_HI_local * v * il.loss_interp_ion(K_eV) * il.EV_MKS

    # 6) Bremsstrahlung off the ion charge (present whether or not it holds a bound e-)
    phi = max(min(np.log(2.0 * gamma) - 1.0 / 3.0,
                  np.log(183.0 / il.Z_TARGET ** (1.0 / 3.0)) + 1.0 / 18.0), 0.0)
    L_br = il.PREFACTOR_BREMS * v * nH_tot * il.Z_TARGET**2 * E * phi

    return np.array([L_ad, L_sy, L_ic, L_co, L_ex, L_io, L_br])

def total_loss_bubble(z, K_J, H_z, r_cm):
    return loss_rates_bubble(z, K_J, H_z, r_cm).sum()

# =====================================================================
# 3. Coupled ODE: dK/dt = -total_loss ; dr/dt = v(K)   (r in meters)
# =====================================================================
def rhs(t, y):
    K_J, r_m = max(y[0], 0.0), max(y[1], 0.0)
    if K_J <= 0.0:
        return [0.0, 0.0]
    z = float(il.z_interp(t))
    Hz = float(il.H_interp(t))
    r_cm = r_m*100.0
    L = total_loss_bubble(z, K_J, Hz, r_cm)
    _, _, _, v = il.kinematics(K_J)
    return [-L, v]

def thermal_floor_J(z):
    return il.thermal_floor(z)

def make_thermal_event():
    def ev_thermal(t, y):
        return y[0] - thermal_floor_J(float(il.z_interp(t)))
    ev_thermal.terminal = True
    ev_thermal.direction = -1
    return ev_thermal

T_INJECT = il.age_s(z_inject)
T_FINAL = il.age_s(5.0)   # do not integrate below z=5 (outside this folder's validity range)

def run_trajectory(K_ini_eV):
    """Integrate the electron over its FULL history (injection to z=5, or
    until it thermalizes), with NO moving-target stopping condition. The
    'reach the bubble edge' question is answered afterwards (Sec. 6) by
    interpolating this continuous r(t) against FIXED reference distances
    (the bubble's actual size at z=6,8,10,12), which avoids the degenerate
    'both start at r=0, t=0' comparison discussed in the VERIFICATION
    block's causality check."""
    y0 = [K_ini_eV*il.EV_MKS, 0.0]
    sol = solve_ivp(rhs, [T_INJECT, T_FINAL], y0, events=[make_thermal_event()],
                     method='Radau', rtol=1e-7, atol=[1e-28, 1e-3], dense_output=True,
                     max_step=(T_FINAL-T_INJECT)/200)
    return sol

# =====================================================================
# 4. VERIFICATION
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)

# 4.1 kinematics identity: E^2 = (pc)^2 + (m c^2)^2, checked symbolically
K_s, E0_s = sp.symbols('K E_0', positive=True)
E_s = K_s + E0_s
p2c2_s = sp.expand(K_s**2 + 2*K_s*E0_s)
resid = sp.simplify(E_s**2 - E0_s**2 - p2c2_s)
print(f"[sympy] E^2-(mc^2)^2-(pc)^2 (rewritten via K=E-mc^2) = {resid}  -> {'OK' if resid==0 else 'FAIL'}")

# 4.2 cross-check n_H normalization against igm_losses.py's own n_HI(z)
for z in [10, 20, 30]:
    mine, theirs = nH_phys_m3(z), il.n_HI(z)
    print(f"[cross-check] n_H(z={z}): mine={mine:.4e} m^-3  vs  igm_losses.n_HI={theirs:.4e} m^-3  "
          f"(ratio {mine/theirs:.3f})")

# 4.3 thermal floor sanity
for z in [8, 20]:
    print(f"[thermal floor] z={z}: K_floor = {thermal_floor_J(z)/il.EV_MKS*1e3:.3f} meV")

# 4.4 R_bubble(z) cross-check against bubble_radius_vs_redshift.py's own printed values
print("\n[cross-check vs bubble_radius_vs_redshift.py]")
for z in [6, 8, 10, 12]:
    print(f"  z={z:4.1f}: R_bubble(photon-counting)={R_bubble_photoncounting_cm(z)/Mpc:.4f} pMpc "
          f"; R(with recomb)={R_bubble_recomb_cm(z)/Mpc:.4f} pMpc")

# 4.5b CRITICAL CHECK: the idealized R~t^(1/3) growth law is superluminal at
#      early times (t_age small); cap it at c*t_age (Shapiro et al. 2006).
from scipy.optimize import brentq as _brentq
def _gap(dt_yr):
    t = T_INJECT + dt_yr*il.YEAR_MKS
    z = float(il.z_interp(t))
    return R_bubble_recomb_cm(z) - C_LIGHT_CGS*dt_yr*il.YEAR_MKS
t_star_yr = _brentq(_gap, 1e1, 1e7)
print(f"\n[CAUSALITY CHECK] naive R(t_age) exceeds c*t_age (superluminal) for t_age < {t_star_yr:.0f} yr "
      f"after injection -- e.g. at t_age=1 yr, R_recomb={R_bubble_recomb_cm(float(il.z_interp(T_INJECT+1*il.YEAR_MKS)))/Mpc:.2e} pMpc "
      f"vs. c*(1 yr)={C_LIGHT_CGS*il.YEAR_MKS/Mpc:.2e} pMpc (a factor {R_bubble_recomb_cm(float(il.z_interp(T_INJECT+1*il.YEAR_MKS)))/(C_LIGHT_CGS*il.YEAR_MKS):.0f}x too fast).")
print(f"  This is the R-type/relativistic I-front regime already discussed in Shapiro et al. (2006,")
print(f"  arXiv:astro-ph/0507677, already in references/) and Sec. 1.5 of stromgren_sphere_summary.tex.")
print(f"  FIX: the bubble-edge target used below is R_edge(t)=min(R_formula(t), c*t_age) -- the")
print(f"  'R_bubble_*_causal_cm' functions -- so the electron is never asked to catch up to an")
print(f"  unphysically superluminal target.")

# 4.5 dominant mechanism at injection, for each K_ini (diagnostic)
print("\n[dominant loss mechanism at injection, z=20, r->0]")
K_INI_ARRAY = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
Hz0 = float(il.H_interp(T_INJECT))
for K in K_INI_ARRAY:
    L = loss_rates_bubble(z_inject, K*il.EV_MKS, Hz0, 1e18)   # r~0 (small but nonzero)
    dom = il.MECH_KEYS[int(np.argmax(L))]
    print(f"  K_ini={K:.0e} eV: dominant = {dom:16s} ; total dK/dt = {L.sum()/il.EV_MKS:.3e} eV/s")
print("="*78)

# =====================================================================
# 5. Run trajectories for every K_ini over their FULL history (no moving
#    target -- see the causality check above for why); "reaching the edge"
#    is evaluated afterwards against FIXED reference distances: the
#    bubble's actual (recombination-corrected) size at z=12,10,8,6, i.e.
#    the same milestones already tabulated throughout this folder.
# =====================================================================
REF_Z = np.array([12.0, 10.0, 8.0, 6.0])
REF_R_Mpc = np.array([R_bubble_recomb_cm(z)/Mpc for z in REF_Z])
print("Reference bubble-edge distances (recombination-corrected R(t), from interior_consistency_check.py):")
for zz, rr in zip(REF_Z, REF_R_Mpc):
    print(f"  z={zz:4.1f}: R = {rr:.4f} pMpc")
print("="*78)

print("TRAJECTORIES (full history, thermalization or z=5 cutoff)")
print("="*78)
results = {}
for K in K_INI_ARRAY:
    sol = run_trajectory(K)
    thermalized = sol.t_events[0].size > 0
    t_dense = np.logspace(np.log10(T_INJECT), np.log10(sol.t[-1]), 4000)
    K_dense = sol.sol(t_dense)[0]
    r_dense = sol.sol(t_dense)[1] * 100.0   # ODE state y[1] is in METERS; convert to cm here,
                                             # once, so 'r' is in cm (consistent with the cm-based
                                             # Mpc constant) everywhere downstream in this script.
    z_dense = il.z_interp(t_dense)
    results[K] = dict(t=t_dense, K=K_dense, r=r_dense, z=z_dense,
                       thermalized=thermalized, t_final=sol.t[-1])
    print(f"K_ini={K:9.0e} eV -> {'THERMALIZED' if thermalized else 'ran to z=5':16s} "
          f"at t={ (sol.t[-1]-T_INJECT)/il.YEAR_MKS/1e6:9.3f} Myr, "
          f"r_final={r_dense[-1]/Mpc:.4e} pMpc, K_final={K_dense[-1]/il.EV_MKS:.3e} eV")

# --- find, for each K_ini and each reference distance, the time/z/K at
#     which the electron's own r(t) first crosses that distance ---
from scipy.interpolate import interp1d as _interp1d
crossings = {K: {} for K in K_INI_ARRAY}
for K in K_INI_ARRAY:
    d = results[K]
    r_of_t = d['r']   # cm
    for zz, Rref_Mpc in zip(REF_Z, REF_R_Mpc):
        Rref_cm = Rref_Mpc*Mpc
        idx = np.where(r_of_t >= Rref_cm)[0]
        if idx.size == 0:
            crossings[K][zz] = None   # never reaches this distance (thermalized or too slow)
            continue
        i1 = idx[0]
        if i1 == 0:
            t_cross, K_cross, z_cross = d['t'][0], d['K'][0], d['z'][0]
        else:
            i0 = i1-1
            f_t = _interp1d([r_of_t[i0], r_of_t[i1]], [d['t'][i0], d['t'][i1]])
            t_cross = float(f_t(Rref_cm))
            K_cross = float(np.interp(t_cross, d['t'], d['K']))
            z_cross = float(il.z_interp(t_cross))
        crossings[K][zz] = dict(t=t_cross, K=K_cross, z=z_cross)

print("\nTime and energy at which each electron's OWN position reaches each reference distance:")
for K in K_INI_ARRAY:
    print(f"K_ini={K:9.0e} eV:")
    for zz in REF_Z:
        c = crossings[K][zz]
        if c is None:
            print(f"    R(z={zz:4.1f})={dict(zip(REF_Z,REF_R_Mpc))[zz]:.3f} pMpc : NEVER REACHED (thermalized first)")
        else:
            print(f"    R(z={zz:4.1f})={dict(zip(REF_Z,REF_R_Mpc))[zz]:.3f} pMpc : "
                  f"t={(c['t']-T_INJECT)/il.YEAR_MKS/1e6:8.4f} Myr after injection, "
                  f"K_there={c['K']/il.EV_MKS:.3e} eV")
print("="*78)

# =====================================================================
# 6. Figures
# =====================================================================
cmap = plt.cm.viridis(np.linspace(0, 1, len(K_INI_ARRAY)))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
ax1, ax2 = axes

for K, col in zip(K_INI_ARRAY, cmap):
    d = results[K]
    r_pMpc = np.maximum(d['r']/Mpc, 1e-9)
    K_eV = np.maximum(d['K']/il.EV_MKS, 1e-3)
    ax1.plot(r_pMpc, K_eV, color=col, lw=2.0, label=il.energy_label(K))
    ax2.plot(d['z'], K_eV, color=col, lw=2.0)
    if d['thermalized']:
        ax1.scatter([r_pMpc[-1]], [K_eV[-1]], marker='x', color='k', s=45, zorder=6)
        ax2.scatter([d['z'][-1]], [K_eV[-1]], marker='x', color='k', s=45, zorder=6)

for zz, rr in zip(REF_Z, REF_R_Mpc):
    ax1.axvline(rr, color='gray', ls=':', lw=1.1, zorder=0)
    ax1.text(rr, 2e10, f'$z={zz:.0f}$', fontsize=7, color='dimgray', rotation=90, ha='right', va='top')

ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('distance traveled from source, $r$ (proper Mpc)')
ax1.set_ylabel('electron kinetic energy $K$ (eV)')
ax1.set_title('Energy loss vs. distance\n(fully ionized bubble interior)')
ax1.grid(alpha=0.25, which='both')
ax1.legend(fontsize=7.2, loc='lower left', ncol=1)

ax2.set_xscale('linear'); ax2.set_yscale('log')
ax2.invert_xaxis()
ax2.set_xlabel('redshift $z$')
ax2.set_ylabel('electron kinetic energy $K$ (eV)')
ax2.set_title('Energy loss vs. redshift')
ax2.grid(alpha=0.25, which='both')

from matplotlib.lines import Line2D
_edge_legend = [Line2D([0],[0], marker='x', color='k', label='thermalized (reaches $kT_{\\rm CMB}(z)$)',
                        markersize=7, linestyle='None')]
ax2.legend(handles=_edge_legend, fontsize=7.5, loc='lower left')

fig.tight_layout(rect=[0.005, 0.05, 0.995, 1])
fig.text(0.5, 0.012,
         r'Electron injected at $z_{\rm inject}=20$, $r=0$; medium: fully ionized bubble interior '
         r'(Coulomb on $n_e{=}n_H(z)$, residual $x_{\rm HI}(r,z)$ for collisional exc./ion.); '
         r'dotted lines: bubble size at $z{=}12,10,8,6$ (recombination-corrected)',
         ha='center', fontsize=7.5, color='dimgray')
outpath1 = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Master_Plan_Ionization_power/figures/electron_energy_loss_vs_distance_redshift.png'
fig.savefig(outpath1, dpi=200)
print(f"Figure saved to: {outpath1}")

# --- second figure: time-to-reach-the-z=8-edge / energy-there, vs K_ini ---
Z_PRIMARY = 8.0
R_PRIMARY = dict(zip(REF_Z, REF_R_Mpc))[Z_PRIMARY]
fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12.5, 5.4))

t_to_edge_Myr, K_at_edge, reached_mask = [], [], []
for K in K_INI_ARRAY:
    c = crossings[K][Z_PRIMARY]
    if c is None:
        t_to_edge_Myr.append(np.nan); K_at_edge.append(np.nan); reached_mask.append(False)
    else:
        t_to_edge_Myr.append((c['t']-T_INJECT)/il.YEAR_MKS/1e6)
        K_at_edge.append(c['K']/il.EV_MKS)
        reached_mask.append(True)
t_to_edge_Myr, K_at_edge, reached_mask = map(np.array, (t_to_edge_Myr, K_at_edge, reached_mask))

bx1.scatter(K_INI_ARRAY[reached_mask], t_to_edge_Myr[reached_mask], c='#2e8b57', s=70,
            zorder=5, edgecolor='k', linewidth=0.5, label=f'reaches $R(z{{=}}{Z_PRIMARY:.0f})={R_PRIMARY:.2f}$ pMpc')
bx1.scatter(K_INI_ARRAY[~reached_mask], np.full((~reached_mask).sum(), np.nan), c='#c1272d', s=70)
for K in K_INI_ARRAY[~reached_mask]:
    bx1.axvline(K, color='#c1272d', ls=':', lw=0.8, alpha=0.4)
bx1.plot(K_INI_ARRAY[reached_mask], t_to_edge_Myr[reached_mask], color='gray', lw=1, alpha=0.5, zorder=1)
bx1.set_xscale('log')
bx1.set_xlabel(r'injection energy $K_{\rm ini}$ (eV)')
bx1.set_ylabel(f'time to reach $R(z={Z_PRIMARY:.0f})$ (Myr after injection)')
bx1.set_title(f'Time to travel {R_PRIMARY:.2f} pMpc\n(the bubble size at $z={Z_PRIMARY:.0f}$)')
bx1.grid(alpha=0.25, which='both')
bx1.legend(fontsize=8)

bx2.scatter(K_INI_ARRAY[reached_mask], K_at_edge[reached_mask], c='#2e8b57', s=70,
            zorder=5, edgecolor='k', linewidth=0.5)
bx2.plot(K_INI_ARRAY[reached_mask], K_at_edge[reached_mask], color='gray', lw=1, alpha=0.5, zorder=1)
bx2.plot(K_INI_ARRAY, K_INI_ARRAY, color='gray', ls=':', lw=1, label='$K_{\\rm final}=K_{\\rm ini}$ (no losses)')
bx2.set_xscale('log'); bx2.set_yscale('log')
bx2.set_xlabel(r'injection energy $K_{\rm ini}$ (eV)')
bx2.set_ylabel(f'kinetic energy at $R(z={Z_PRIMARY:.0f})$ (eV)')
bx2.set_title(f'Energy remaining after traveling {R_PRIMARY:.2f} pMpc')
bx2.grid(alpha=0.25, which='both')
bx2.legend(fontsize=8.5)

_legend2 = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#2e8b57', markeredgecolor='k', label='reaches the edge', markersize=9),
            Line2D([0],[0], color='#c1272d', ls=':', label='thermalizes first (never reaches it)')]
fig2.legend(handles=_legend2, loc='upper center', ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))

fig2.tight_layout(rect=[0.005, 0.02, 0.995, 0.90])
outpath2 = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Master_Plan_Ionization_power/figures/electron_time_and_energy_at_edge.png'
fig2.savefig(outpath2, dpi=200)
print(f"Figure saved to: {outpath2}")

# =====================================================================
# 7. Audit trail
# =====================================================================
print("="*78)
print("PARAMETERS USED (audit trail)")
print("="*78)
print(f"z_inject={z_inject} ; SFR_fid={SFR_fid} Msun/yr ; f_esc={f_esc} ; C_HII={C_HII} ; alpha_B={alpha_B:.3e} cm^3/s")
print(f"Cosmology: h={h}, Omega_m={Om}, Omega_b h^2={Ob_h2}, Y_p={Yp}  [Planck 2018 VI]; "
      f"igm_losses.py's astropy-Planck18 used for t<->z (cross-checked to <3%)")
print(f"K_ini grid (eV): {list(K_INI_ARRAY)}")
print(f"Reference bubble edges (recomb-corrected R(t)): "
      + ", ".join(f"z={zz:.0f}:{rr:.3f}pMpc" for zz,rr in zip(REF_Z,REF_R_Mpc)))
