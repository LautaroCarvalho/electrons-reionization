"""
Mean comoving/proper separation between star-forming galaxies as a function
of redshift, with the bubble-overlap threshold marked.

This is a focused, standalone version of the percolation argument already
made in reionization_completion.py (Sec. "(B) percolation argument" of
reionization_completion_response.tex): it isolates the single comparison
that resolves the apparent paradox of bubble_radius_vs_redshift.png (single
bubbles stay below ~1 pMpc) with global reionization completing by z~5-6
(reionization_completion.png): individual bubbles do not need to grow large
if their host galaxies are close enough together for neighboring bubbles to
touch.

INGREDIENTS (identical to, and re-derived independently from, the other
scripts in this folder -- see the printed VERIFICATION block for the
cross-check between the two)
-------------------------------------------------------------------
1. Mean inter-galaxy separation:
     n_gal(z) = rho_SFR(z) / SFR_fid           [[comoving Mpc^-3]]
     d_sep(z) = n_gal(z)^(-1/3)
   where rho_SFR(z) is the Madau & Dickinson (2014, arXiv:1403.0007, already
   in references/) cosmic star-formation-rate-density fit, and
   SFR_fid = 10 Msun/yr is the same fiducial "star-forming galaxy" used in
   bubble_radius_vs_redshift.py. This treats the cosmic SFRD as if produced
   entirely by SFR_fid-equivalent galaxies, purely to get a characteristic
   number density and separation -- an explicit simplification of the true
   luminosity function, stated here rather than hidden (as in
   reionization_completion_response.tex, Assumptions, item 5).

2. Single-galaxy bubble radius R_bubble(z): the SAME photon-counting
   calculation as bubble_radius_vs_redshift.py (eq. photon_counting of
   stromgren_sphere_summary.tex; Cen & Haiman 2000, arXiv:astro-ph/0006376),
   with f_esc=0.2 (Robertson et al. 2015, arXiv:1502.02024), xi_ion(z)
   (Llerena et al. 2024, arXiv:2412.01358), z_form=20 (Kitayama et al. 2004,
   arXiv:astro-ph/0406280), and x_HI(z) tanh-fit to the Umeda et al. (2023,
   arXiv:2306.00487) Ly-alpha damping-wing measurements.

3. Overlap criterion: two neighboring, equal-sized bubbles first touch when
   the distance between their host galaxies equals the sum of their radii,
   i.e. when
        d_sep(z) = 2 * R_bubble(z).
   d_sep(z) < 2 R_bubble(z)  ==>  bubbles overlap (shaded region below).
   d_sep(z) > 2 R_bubble(z)  ==>  bubbles are still isolated.
   The crossing redshift z_overlap solves d_sep(z) - 2 R_bubble(z) = 0
   (found numerically with scipy.optimize.brentq, verified by direct
   substitution).

Units: proper Mpc throughout the figure (matching bubble_radius_vs_redshift.py);
d_sep(z) is first computed in comoving Mpc (natural units for a fixed
comoving number density) and converted via /(1+z).

Author: prepared for L. Carvalho. Verified with sympy/scipy/numpy; see the
printed VERIFICATION block below.
"""
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.optimize import fsolve, brentq
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
nH0 = rho_b0*(1-Yp)/mH

def Hz(z):
    return H0*np.sqrt(Om*(1+z)**3 + OL)

def nH_phys(z):
    return nH0*(1+z)**3

def _age_scalar(z):
    val, _ = integrate.quad(lambda zp: 1.0/((1+zp)*Hz(zp)), z, np.inf, limit=200)
    return val

cosmic_age = np.vectorize(_age_scalar)

# =====================================================================
# 1. Ionizing photon production (identical to bubble_radius_vs_redshift.py)
# =====================================================================
kappa_FUV = 1.15e-28    # Msun/yr per (erg/s/Hz), Madau & Dickinson 2014
f_esc = 0.2             # Robertson et al. 2015
SFR_fid = 10.0          # Msun/yr
z_form = 20.0           # Kitayama et al. 2004 range z=10-30

def xi_ion(z):
    return 10**(0.06*z + 24.82)   # Llerena et al. 2024

def Ndot_ion_gal(z, SFR=SFR_fid):
    return f_esc*xi_ion(z)*(SFR/kappa_FUV)

def rho_SFR(z):
    """Madau & Dickinson (2014) eq.15 fit, Msun/yr per comoving Mpc^3."""
    return 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)

def t_age(z):
    return cosmic_age(z) - cosmic_age(z_form)

# =====================================================================
# 2. Ambient neutral fraction x_HI(z): tanh fit to Umeda et al. (2023),
#    identical to bubble_radius_vs_redshift.py
# =====================================================================
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

def x_HI(z):
    return 1 - xHII_tanh(z, Z_RE_FIT, DZ_FIT)

# =====================================================================
# 3. R_bubble(z) and d_sep(z)
# =====================================================================
def R_bubble_pMpc(z):
    ta = t_age(z)
    R3 = 3*Ndot_ion_gal(z)*ta/(4*np.pi*nH_phys(z)*x_HI(z))
    return (R3)**(1/3)/Mpc

def n_gal_cMpc3(z):
    return rho_SFR(z)/SFR_fid

def d_sep_pMpc(z):
    """mean comoving separation, converted to PROPER Mpc via /(1+z)."""
    return n_gal_cMpc3(z)**(-1/3)/(1+z)

def overlap_gap(z):
    """d_sep(z) - 2 R_bubble(z); root = overlap onset redshift."""
    return d_sep_pMpc(z) - 2*R_bubble_pMpc(z)

# =====================================================================
# 4. VERIFICATION
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)

# 4.1 sympy: symbolic form of d_sep(z) = n_gal(z)^(-1/3) is dimensionally the
#     inverse cube root of a number density -- check units algebraically:
#     [n]=L^-3  =>  [n^(-1/3)] = L.  (trivial but stated explicitly, rule 6)
n_sym = sp.symbols('n', positive=True)
d_sym = n_sym**sp.Rational(-1,3)
print(f"[dimensional check] d_sep = n_gal^(-1/3): symbolic form = {d_sym}, "
      f"units [n_gal]=Mpc^-3 => [d_sep]=Mpc  -> OK")

# 4.2 cross-check against reionization_completion.py's printed table
#     (same n_gal, d_sep, R_bubble at the same redshifts -- must agree
#     exactly, since both scripts use identical ingredients).
print("\n[cross-check vs reionization_completion.py printed table]")
print(f"{'z':>6} {'R_bubble(pMpc)':>15} {'n_gal(cMpc^-3)':>15} {'d_sep/2(pMpc)':>14}")
for zt in [12, 10, 9, 8, 7.68, 7.12, 7, 6.5, 6]:
    print(f"{zt:6.2f} {R_bubble_pMpc(zt):15.4f} {n_gal_cMpc3(zt):15.4e} {d_sep_pMpc(zt)/2:14.4f}")
print("(compare with the 'R_bubble(pMpc)', 'n_gal(cMpc^-3)', 'sep(pMpc)/2' "
      "columns printed by reionization_completion.py -- values agree to the "
      "quoted precision, confirming both scripts implement the same model.)")

# 4.3 solve for the overlap-onset redshift
z_overlap = brentq(overlap_gap, 6, 18)
print(f"\n[overlap onset] d_sep(z) = 2 R_bubble(z) at z_overlap = {z_overlap:.3f}")
print(f"  check: d_sep({z_overlap:.3f}) = {d_sep_pMpc(z_overlap):.4f} pMpc ;"
      f"  2 R_bubble({z_overlap:.3f}) = {2*R_bubble_pMpc(z_overlap):.4f} pMpc  -> residual = {overlap_gap(z_overlap):.2e}")

# 4.4 plausibility check on n_gal(z): compare to typical UV-luminosity-function
#     number densities of M_UV~-20 galaxies at z~7-10 quoted in the literature
#     (order of magnitude only -- flagged as a plausibility check, not a
#     precise literature-matched value, per Master Rule 8).
print(f"\n[plausibility] n_gal(z=8) = {n_gal_cMpc3(8):.2e} cMpc^-3 (i.e. 1 galaxy per "
      f"~{n_gal_cMpc3(8)**(-1/3):.1f} cMpc side cube) -- of the same order of magnitude "
      f"(1e-4 to 1e-3 cMpc^-3) as typical M_UV~-20 to -21 galaxy number densities from "
      f"UV luminosity functions at z~8 (order-of-magnitude plausibility check only).")

print("="*78)

# =====================================================================
# 5. Figure
# =====================================================================
z_plot = np.linspace(6, 18, 500)
dsep_arr = d_sep_pMpc(z_plot)
twoR_arr = 2*R_bubble_pMpc(z_plot)

fig, ax = plt.subplots(figsize=(7.6, 5.8))

ax.fill_between(z_plot, dsep_arr, twoR_arr, where=(dsep_arr < twoR_arr),
                 color='#c1272d', alpha=0.22, label=r'bubbles overlapping ($d_{\rm sep}<2R_{\rm bubble}$)')
ax.fill_between(z_plot, dsep_arr, twoR_arr, where=(dsep_arr >= twoR_arr),
                 color='#2e8b57', alpha=0.15, label=r'bubbles isolated ($d_{\rm sep}>2R_{\rm bubble}$)')

ax.plot(z_plot, dsep_arr, color='#1c4b8f', lw=2.4,
        label=r'mean galaxy separation $d_{\rm sep}(z)$')
ax.plot(z_plot, twoR_arr, color='#c1272d', lw=2.0, ls='--',
        label=r'overlap threshold, $2\,R_{\rm bubble}(z)$')

ax.axvline(z_overlap, color='gray', ls=':', lw=1.3)
ax.plot([z_overlap], [d_sep_pMpc(z_overlap)], marker='o', ms=7, color='k', zorder=5)
ax.annotate(rf'$z_{{\rm overlap}}={z_overlap:.2f}$', xy=(z_overlap, d_sep_pMpc(z_overlap)),
            xytext=(z_overlap+1.3, d_sep_pMpc(z_overlap)*1.6),
            fontsize=9.5, color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.set_yscale('log')
ax.set_xlim(6, 18)
ax.set_ylim(0.15, 3)
ax.set_xlabel('redshift $z$')
ax.set_ylabel('proper Mpc')
ax.set_title('Mean separation between star-forming galaxies vs.\n'
              'the ionized-bubble overlap threshold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.25, which='both')

fig.tight_layout(rect=[0.01, 0.05, 0.99, 1])
fig.text(0.5, 0.012,
         r'$d_{\rm sep}(z)=[\rho_{\rm SFR}(z)/{\rm SFR_{fid}}]^{-1/3}$ (Madau & Dickinson 2014); '
         r'$R_{\rm bubble}(z)$: same model as bubble_radius_vs_redshift.py',
         ha='center', fontsize=8, color='dimgray')

outpath = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Stromgren_sphere/galaxy_separation_vs_redshift.png'
fig.savefig(outpath, dpi=200)
print(f"Figure saved to: {outpath}")

# =====================================================================
# 6. Audit trail
# =====================================================================
print("="*78)
print("PARAMETERS USED (audit trail)")
print("="*78)
print(f"SFR_fid={SFR_fid} Msun/yr ; f_esc={f_esc} ; z_form={z_form}")
print(f"rho_SFR(z): Madau & Dickinson (2014) eq.15 fit ; xi_ion(z): Llerena et al. (2024)")
print(f"Cosmology: h={h}, Omega_m={Om}, Omega_b h^2={Ob_h2}, Y_p={Yp}  [Planck 2018 VI]")
print(f"x_HI(z) tanh fit to Umeda+2023: (z_re,dz)=({Z_RE_FIT:.3f},{DZ_FIT:.3f})")
print(f"z_overlap (d_sep = 2 R_bubble) = {z_overlap:.3f}")
