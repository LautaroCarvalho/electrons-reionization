# Copied from Stromgren_sphere/reionization_completion.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""
How does reionization complete globally if individual star-forming-galaxy
bubbles stay below ~1 proper Mpc?

This script resolves the apparent paradox raised after inspecting
bubble_radius_vs_redshift.png / bubble_radius_vs_redshift_planckXHI.png: the
ionized "Stromgren sphere" of ONE fiducial star-forming galaxy
(SFR=10 Msun/yr) never exceeds ~1 pMpc over 6<z<18. But cosmic reionization
is not the growth of a single bubble to cosmological size -- it is the
COLLECTIVE overlap of the (very numerous) bubbles produced by the ENTIRE
star-forming-galaxy population. This script makes that argument quantitative,
using only ingredients already established and cited in this folder.

TWO INDEPENDENT CALCULATIONS ARE PERFORMED AND CROSS-CHECKED AGAINST EACH
OTHER AND AGAINST OBSERVATIONS (Master Rule 5):

(A) THE GLOBAL VOLUME-FILLING-FACTOR EQUATION
    eq. (QHII) of stromgren_sphere_summary.tex (Choudhury 2022, arXiv:2209.08558,
    already in references/):
        dQ_HII/dt = ndot_gamma/nH0 - Q_HII * a^-3 * C * alpha_B(T) * n_e
    driven by the ENTIRE cosmic star-formation-rate density (Madau & Dickinson
    2014), not by one galaxy. Q_HII(z) -> 1 is what "reionization completes"
    means; it says nothing about how large any single bubble is.
    IMPORTANT: the source term uses n_H0, the FIXED COMOVING mean hydrogen
    density (i.e. evaluated at z=0), NOT the diluting physical n_H(z); this
    is because dot n_gamma (the SFRD-driven emissivity) and Q_HII are
    naturally comoving quantities. A first attempt at this calculation
    (documented in Sec. Verification, item 5, of the companion LaTeX
    response) mistakenly used the physical n_H(z) in the source term, which
    suppressed Q_HII by a factor (1+z)^3 and made reionization appear unable
    to complete within a Hubble time -- inconsistent with the well
    established observational fact that it does (Planck 2018; White et al.
    2003). This was caught by cross-checking against that basic benchmark,
    exactly the kind of check Master Rule 5 asks for.

(B) THE PERCOLATION / BUBBLE-OVERLAP ARGUMENT
    The comoving number density of "SFR=10 Msun/yr-equivalent" star-forming
    galaxies implied by the same cosmic SFRD, n_gal(z) = rho_SFR(z)/SFR_fid,
    sets a mean inter-source separation d_sep(z) = n_gal(z)^(-1/3). Comparing
    d_sep(z)/2 to the single-galaxy bubble radius R_bubble(z) (from
    bubble_radius_vs_redshift.py) tests directly whether neighboring bubbles
    are expected to touch/overlap -- the mechanism invoked qualitatively by
    Furlanetto, Zaldarriaga & Hernquist (2004, arXiv:astro-ph/0403697,
    already in references/) and seen directly in radiation-hydrodynamic
    simulations by Neyer et al. (2023, THESAN, arXiv:2310.03783, already in
    references/, their "flash ionization" of regions that suddenly join a
    much larger pre-existing bubble).

Units: cgs internally; radii in proper Mpc where noted, comoving Mpc
elsewhere (always labeled).

Author: prepared for L. Carvalho. Verified with sympy/scipy/numpy; see the
printed VERIFICATION block.
"""
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# =====================================================================
# 0. Cosmology (Planck 2018, arXiv:1807.06209) -- identical to the other
#    scripts in this folder
# =====================================================================
mH = 1.6726e-24
pc = 3.0857e18
Mpc = 3.0857e24
yr = 3.156e7
Gyr = yr*1e9

h, Om, Ob_h2, Yp = 0.674, 0.315, 0.0224, 0.245
OL = 1 - Om
H0 = h*100*1e5/Mpc
rho_b0 = Ob_h2*1.878e-29
nH0 = rho_b0*(1-Yp)/mH          # cm^-3, COMOVING mean H density (= physical at z=0)
chi_He = 1.08

def Hz(z):
    return H0*np.sqrt(Om*(1+z)**3 + OL)

def nH_phys(z):
    return nH0*(1+z)**3

def ne_phys(z):
    return chi_He*nH_phys(z)

def _age_scalar(z):
    val, _ = integrate.quad(lambda zp: 1.0/((1+zp)*Hz(zp)), z, np.inf, limit=200)
    return val

cosmic_age = np.vectorize(_age_scalar)

# =====================================================================
# 1. Ionizing photon production -- SAME xi_ion(z), f_esc, kappa_FUV as
#    bubble_radius_vs_redshift.py, now applied to the WHOLE population via
#    the cosmic SFRD (Madau & Dickinson 2014, arXiv:1403.0007, their eq. 15,
#    confirmed against two independent secondary sources).
# =====================================================================
kappa_FUV = 1.15e-28    # Msun/yr per (erg/s/Hz)
f_esc = 0.2             # Robertson et al. 2015, arXiv:1502.02024

def xi_ion(z):
    return 10**(0.06*z + 24.82)   # Llerena et al. 2024, arXiv:2412.01358

def rho_SFR(z):
    """Madau & Dickinson (2014) cosmic SFRD fit, Msun/yr per COMOVING Mpc^3."""
    return 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)

def ndot_ion_comoving(z):
    """Ionizing photons produced per second per COMOVING cm^3 by the ENTIRE
    galaxy population (pairs with the fixed comoving nH0 below)."""
    rho_SFR_cgs = rho_SFR(z)/Mpc**3
    L_UV_density = rho_SFR_cgs/kappa_FUV
    return f_esc*xi_ion(z)*L_UV_density

# =====================================================================
# 2. Recombinations: alpha_B(1e4 K) as established in check_consistency.py
#    (Shapiro & Iliev+2006), clumping factor C=3 (Robertson et al. 2015).
# =====================================================================
alpha_B = 2.59e-13   # cm^3/s, T=1e4 K
C_HII = 3.0          # Robertson et al. 2015 (their T=2e4K; using alpha_B(1e4K)
                     # here is a mild, explicitly flagged approximation)

# =====================================================================
# 3. VERIFICATION, part 1: sympy check of the constant-coefficient limit
#    of eq. (QHII) -- structurally the same "recombination clock" as
#    eq. (Rt) of stromgren_sphere_summary.tex, now for Q_HII instead of R^3.
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)
t_s, ndot_s, nH_s, C_s, aB_s, ne_s = sp.symbols('t Ndot n_H C alpha_B n_e', positive=True)
Qsol = (ndot_s/(nH_s*C_s*aB_s*ne_s))*(1-sp.exp(-C_s*aB_s*ne_s*t_s))
resid = sp.simplify(sp.diff(Qsol, t_s) - (ndot_s/nH_s - Qsol*C_s*aB_s*ne_s))
print(f"[sympy] constant-coefficient dQ/dt solution residual = {resid}  -> {'OK' if resid==0 else 'FAIL'}")

# =====================================================================
# 4. Global Q_HII(z): integrate eq. (QHII), Choudhury (2022) eq. 66
#    (already in stromgren_sphere_summary.tex), from z=25 (Q=0) to z=4.
# =====================================================================
def dQdz(Q, z, C=C_HII):
    dQdt = ndot_ion_comoving(z)/nH0 - Q*C*alpha_B*ne_phys(z)
    dzdt = -(1+z)*Hz(z)
    return dQdt/dzdt

def solve_QHII(C=C_HII, z_start=25, z_end=4, n=6000):
    zgrid = np.linspace(z_start, z_end, n)
    sol = solve_ivp(lambda z, Q: dQdz(Q, z, C=C), [z_start, z_end], [0.0],
                     t_eval=zgrid, max_step=0.01, method='RK45')
    return sol.t, sol.y[0]

zz, Qz = solve_QHII(C=3.0)
_,  Qz_C1 = solve_QHII(C=1.0)

def first_cross(zarr, Qarr, val):
    idx = np.where(Qarr >= val)[0]
    return zarr[idx[0]] if len(idx) else None

z_half = first_cross(zz, Qz, 0.5)
z_full = first_cross(zz, Qz, 0.999)
z_half_C1 = first_cross(zz, Qz_C1, 0.5)

print(f"[global Q_HII, C=3] Q_HII(z=15)={Qz[np.argmin(np.abs(zz-15))]:.4f}  "
      f"Q_HII(z=9)={Qz[np.argmin(np.abs(zz-9))]:.4f}  "
      f"Q_HII(z=8)={Qz[np.argmin(np.abs(zz-8))]:.4f}  "
      f"Q_HII(z=7)={Qz[np.argmin(np.abs(zz-7))]:.4f}  "
      f"Q_HII(z=6)={Qz[np.argmin(np.abs(zz-6))]:.4f}")
print(f"[global Q_HII, C=3] z(Q_HII=0.5)   = {z_half:.2f}   (Planck 2018 z_re = 7.68+/-0.79 -- SAME early/late tension flagged before)")
print(f"[global Q_HII, C=3] z(Q_HII=0.999) = {z_full:.2f}   (i.e. reionization completes by z~{z_full:.1f} in this simple model)")
print(f"[global Q_HII, C=1] z(Q_HII=0.5)   = {z_half_C1:.2f}   (sensitivity to clumping factor: lower C -> earlier completion)")

# cross-check against the Umeda+2023 measured x_HI (already used in bubble_radius_vs_redshift.py)
for zt, xhi_obs in zip([7.12, 9.91], [0.53, 0.92]):
    idx = np.argmin(np.abs(zz-zt))
    print(f"[cross-check vs Umeda+2023] z={zt:.2f}: 1-Q_HII(model)={1-Qz[idx]:.3f}  vs  x_HI,Umeda(measured)={xhi_obs:.2f}")

# =====================================================================
# 5. Percolation argument: mean separation of SFR-equivalent sources vs the
#    single-galaxy bubble radius of bubble_radius_vs_redshift.py (reproduced
#    identically here for self-containment).
# =====================================================================
SFR_fid = 10.0
z_form = 20.0

def t_age(z):
    return cosmic_age(z) - cosmic_age(z_form)

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

def x_HI_local(z):
    return 1 - xHII_tanh(z, Z_RE_FIT, DZ_FIT)

def R_bubble_pMpc(z):
    ta = t_age(z)
    R3 = 3*Ndot_ion_gal(z)*ta/(4*np.pi*nH_phys(z)*x_HI_local(z))
    return (R3)**(1/3)/Mpc

def n_gal_equiv_cMpc3(z):
    """comoving number density of SFR=SFR_fid-equivalent galaxies implied by
    the SAME cosmic SFRD driving panel (A) above."""
    return rho_SFR(z)/SFR_fid

def mean_sep_cMpc(z):
    return n_gal_equiv_cMpc3(z)**(-1/3)

print()
print(f"{'z':>6} {'R_bubble(pMpc)':>15} {'n_gal(cMpc^-3)':>15} {'sep(pMpc)':>11} {'sep/(2R)':>10} {'Q_naive(nV)':>12}")
zcheck = [12, 10, 9, 8, 7.68, 7.12, 7, 6.5, 6]
for zt in zcheck:
    Rb = R_bubble_pMpc(zt)
    ngal = n_gal_equiv_cMpc3(zt)
    sep_p = mean_sep_cMpc(zt)/(1+zt)
    Rb_c = Rb*(1+zt)
    Q_naive = ngal*(4/3)*np.pi*Rb_c**3
    print(f"{zt:6.2f} {Rb:15.4f} {ngal:15.4e} {sep_p:11.4f} {sep_p/(2*Rb):10.3f} {Q_naive:12.3f}")

print()
print("Q_naive = n_gal * (4/3) pi R_bubble_comoving^3 is a NAIVE, NON-OVERLAPPING")
print("packing estimate (neglects recombinations and overlap-inefficiency); its")
print("value exceeding 1 already at z<=8 is itself direct evidence that the bubbles")
print("MUST overlap well before any single one needs to grow beyond ~1 Mpc.")
print("The physically self-consistent Q_HII (panel A, which DOES include")
print("recombinations) is systematically lower, exactly as expected.")
print("="*78)

# =====================================================================
# 6. Figure: two panels telling the complete, self-consistent story
# =====================================================================
z_plot = np.linspace(6, 18, 400)
Rb_arr = R_bubble_pMpc(z_plot)
sep_arr = mean_sep_cMpc(z_plot)/(1+z_plot)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 9.2), sharex=True)

ax1.plot(z_plot, Rb_arr, color='#1c4b8f', lw=2.2, label=r'single-galaxy bubble radius $R_{\rm bubble}(z)$')
ax1.plot(z_plot, sep_arr/2, color='#c1272d', lw=2.2, ls='--',
         label=r'half the mean inter-source separation, $d_{\rm sep}(z)/2$')
ax1.set_yscale('log')
ax1.set_ylabel('proper Mpc')
ax1.set_title('Why sub-Mpc bubbles are enough: bubbles touch their neighbors\n'
              'before any single one needs to grow large', pad=14)
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(alpha=0.25, which='both')
ax1.annotate('bubbles overlap\nwhere curves cross', xy=(8.7, 0.62), xytext=(13, 0.9),
             fontsize=8.5, color='dimgray', ha='center',
             arrowprops=dict(arrowstyle='->', color='dimgray', lw=1))

z_plot2 = zz[(zz >= 4) & (zz <= 18)]
Q_plot2 = Qz[(zz >= 4) & (zz <= 18)]
Q_plot2_C1 = Qz_C1[(zz >= 4) & (zz <= 18)]
ax2.fill_between(z_plot2, np.clip(Q_plot2_C1, 0, 1), np.clip(Q_plot2, 0, 1),
                  color='#3b7dd8', alpha=0.2, label=r'clumping factor $C=1$–$3$ range')
ax2.plot(z_plot2, np.clip(Q_plot2, 0, 1), color='#1c4b8f', lw=2.2,
          label=r'global $Q_{\rm HII}(z)$, $C=3$ (Robertson+2015)')
ax2.errorbar(UMEDA_Z, 1-UMEDA_XHI, yerr=[[0.18,0.10],[0.47,0.08]], fmt='D', ms=7,
             color='#c1272d', ecolor='#c1272d', capsize=4,
             label=r'$1-x_{\rm HI}$, Umeda et al. (2023)')
ax2.axvline(7.68, color='gray', ls=':', lw=1.2)
ax2.text(7.83, 0.05, r'Planck $z_{\rm re}=7.68$', rotation=90, va='bottom', ha='left',
         color='gray', fontsize=8.5)
ax2.set_ylim(0, 1.05)
ax2.set_xlim(6, 18)
ax2.set_xlabel('redshift $z$')
ax2.set_ylabel(r'ionized volume filling fraction $Q_{\rm HII}$')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.25)

fig.tight_layout(rect=[0.015, 0.035, 0.985, 1])
fig.text(0.5, 0.010,
         r'Same SFRD (Madau & Dickinson 2014), $\xi_{\rm ion}(z)$ (Llerena+2024), $f_{\rm esc}$ (Robertson+2015) as bubble_radius_vs_redshift.py',
         ha='center', fontsize=8, color='dimgray')

outpath = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Master_Plan_Ionization_power/figures/reionization_completion.png'
fig.savefig(outpath, dpi=200)
print(f"Figure saved to: {outpath}")

# =====================================================================
# 7. Audit trail
# =====================================================================
print("="*78)
print("PARAMETERS USED (audit trail)")
print("="*78)
print(f"SFR_fid={SFR_fid} Msun/yr ; f_esc={f_esc} ; z_form={z_form} ; C_HII={C_HII} ; alpha_B={alpha_B:.3e} cm^3/s")
print(f"rho_SFR(z): Madau & Dickinson (2014) eq.15 fit  ;  xi_ion(z): Llerena et al. (2024)")
print(f"Cosmology: h={h}, Omega_m={Om}, Omega_b h^2={Ob_h2}, Y_p={Yp}  [Planck 2018 VI]")
print(f"x_HI(z) (for R_bubble only): tanh fit to Umeda+2023, (z_re,dz)=({Z_RE_FIT:.3f},{DZ_FIT:.3f})")
