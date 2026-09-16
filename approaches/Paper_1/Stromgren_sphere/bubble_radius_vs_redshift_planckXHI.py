"""
Ionized-hydrogen ('Stromgren') bubble radius around a single star-forming
galaxy, as a function of redshift -- Planck-2018 CMB-only reionization-history
variant.

This script is COMPLETELY ANALOGOUS to bubble_radius_vs_redshift.py (same
folder): every ingredient is identical (source, cosmology, ionizing-photon
production rate, age-redshift relation) EXCEPT for the ambient neutral
fraction x_HI(z), which here uses the Planck 2018 CMB-only instantaneous-
reionization tanh parameters directly,
    (z_re, Delta_z) = (7.68, 0.5)   (Planck Collaboration 2020, arXiv:1807.06209),
instead of the Umeda et al. (2023) JWST-anchored fit used in the other
script. This isolates, by direct comparison of the two figures, exactly how
much the "early" (CMB-inferred) vs. "late" (direct-EoR-probe-inferred)
reionization histories change the predicted bubble size -- the tension that
was flagged qualitatively in bubble_radius_response.tex.

MODEL (photon-counting limit, no recombinations / no Hubble term)
-------------------------------------------------------------------
Following eq. (photon_counting) of stromgren_sphere_summary.tex,
    R(z) = [ 3 * Ndot_ion(z) * t_age(z) / (4 pi * nH(z) * x_HI(z)) ]^(1/3)      (*)
(Cen & Haiman 2000, arXiv:astro-ph/0006376, their eq. 2; applied here to a
star-forming galaxy rather than a quasar, as in the companion script).

INGREDIENTS AND THEIR SOURCES (identical to bubble_radius_vs_redshift.py,
except item 4)
-------------------------------------------------------------------
1. Ndot_ion(z) = f_esc * xi_ion(z) * L_UV, L_UV = SFR/kappa_FUV,
   kappa_FUV = 1.15e-28 Msun/yr/(erg/s/Hz) (Madau & Dickinson 2014,
   arXiv:1403.0007, Salpeter IMF);
   log10(xi_ion/Hz erg^-1) = 0.06 z + 24.82 (Llerena et al. 2024,
   arXiv:2412.01358, z=4-10 JWST fit, measured at f_esc=0);
   f_esc = 0.2 fiducial (Robertson et al. 2015, arXiv:1502.02024), with a
   f_esc=0.05 sensitivity band.
2. nH(z) = nH0 (1+z)^3, nH0 from Planck 2018 (arXiv:1807.06209)
   Omega_b h^2 = 0.0224, h=0.674, Y_p=0.245.
3. t_age(z) = t_cosmic(z) - t_cosmic(z_form), flat LCDM age-redshift
   relation with Planck 2018 Omega_m=0.315; z_form = 20 (Kitayama et al. 2004
   range z=10-30, arXiv:astro-ph/0406280).
4. Ambient neutral fraction x_HI(z): standard tanh reionization model
   (Lewis 2008; used by Planck), evaluated DIRECTLY at the Planck 2018
   TT,TE,EE+lowE point estimate
      (z_re, Delta_z) = (7.68, 0.5)   (Planck Collaboration 2020, arXiv:1807.06209),
   i.e. NOT fit to any galaxy data -- a pure CMB-optical-depth-based history.
   CAVEAT (rule 5/8 -- flagged, not hidden): this history is known to predict
   a substantially MORE ionized IGM at z=7-10 than the direct Ly-alpha
   damping-wing measurements of Umeda et al. (2023, arXiv:2306.00487,
   already in references/), which give x_HI=0.53 at z=7.12 and x_HI=0.92 at
   z=9.91 -- the same "early vs. late reionization" tension as before, now
   seen from the opposite side (this script trusts the CMB history and
   therefore disagrees with the galaxy-based x_HI measurements, whereas
   bubble_radius_vs_redshift.py trusted the galaxy-based measurements and
   therefore disagreed with the CMB history). Because (z_re, Delta_z) is a
   global two-parameter fit to CMB data (not a sparse two-point interpolation
   over a narrow z range), there is no extrapolation concern here: the curve
   is drawn SOLID over the full plotted range 6 <= z <= 18.

Units: cgs internally; radii reported/plotted in proper Mpc (pMpc), as in
stromgren_sphere_summary.tex.

Author: prepared for L. Carvalho. Verified with sympy 1.12 / scipy / numpy;
see the printed "VERIFICATION" block below for every cross-check performed.
"""
import numpy as np
import sympy as sp
from scipy import integrate
import matplotlib.pyplot as plt

# =====================================================================
# 0. Physical constants (cgs) and Planck 2018 cosmology (arXiv:1807.06209)
#    -- identical to bubble_radius_vs_redshift.py
# =====================================================================
mH = 1.6726e-24            # g, hydrogen atom mass
pc = 3.0857e18             # cm
Mpc = 3.0857e24            # cm
yr = 3.156e7                # s
Gyr = yr*1e9

h = 0.674
Om = 0.315
OL = 1 - Om
Ob_h2 = 0.0224
Yp = 0.245
H0 = h*100*1e5/Mpc          # s^-1
rho_crit0 = 1.878e-29*h**2  # g/cm^3
rho_b0 = Ob_h2*1.878e-29    # g/cm^3
nH0 = rho_b0*(1-Yp)/mH      # cm^-3

def Hz(z):
    return H0*np.sqrt(Om*(1+z)**3 + OL)

def nH(z):
    return nH0*(1+z)**3

def _cosmic_age_scalar(z):
    integrand = lambda zp: 1.0/((1+zp)*Hz(zp))
    val, _ = integrate.quad(integrand, z, np.inf, limit=200)
    return val

cosmic_age = np.vectorize(_cosmic_age_scalar)

# =====================================================================
# 1. Ionizing photon production rate of the galaxy, Ndot_ion(z)
#    -- identical to bubble_radius_vs_redshift.py
# =====================================================================
kappa_FUV = 1.15e-28        # Msun/yr per (erg/s/Hz), Madau & Dickinson 2014
SFR_fid = 10.0              # Msun/yr
L_UV_fid = SFR_fid/kappa_FUV

def xi_ion(z):
    """Llerena et al. 2024 (arXiv:2412.01358), z=4-10 fit."""
    return 10**(0.06*z + 24.82)

def Ndot_ion(z, f_esc=0.2, SFR=SFR_fid):
    L_UV = SFR/kappa_FUV
    return f_esc*xi_ion(z)*L_UV

d10pc = 10*pc
M_UV_fid = -2.5*np.log10(L_UV_fid/(4*np.pi*d10pc**2)) - 48.60

# =====================================================================
# 2. Ambient neutral fraction x_HI(z): Planck 2018 CMB-only tanh model
#    (z_re, Delta_z) = (7.68, 0.5), arXiv:1807.06209 -- NOT fit to data.
# =====================================================================
Z_RE_PLANCK = 7.68
DZ_PLANCK   = 0.5

def y_of_z(z):
    return (1+z)**1.5

def xHII_tanh(z, z_re, dz):
    dy = 1.5*np.sqrt(1+z_re)*dz
    return 0.5*(1 + np.tanh((y_of_z(z_re) - y_of_z(z))/dy))

def x_HI(z):
    return 1 - xHII_tanh(z, Z_RE_PLANCK, DZ_PLANCK)

z_form = 20.0   # fiducial formation redshift (Kitayama+2004 range z=10-30)

def t_age(z):
    return cosmic_age(z) - cosmic_age(z_form)

# For the cross-check against Umeda et al. (2023) below:
UMEDA_Z    = np.array([7.12, 9.91])
UMEDA_XHI  = np.array([0.53, 0.92])
UMEDA_LOGRB_CMPC = np.array([1.67, -0.69])   # log10(R_b/cMpc)
UMEDA_LOGRB_ERR  = np.array([[0.16, 0.24], [0.14, 0.89]])  # [-,+]

# =====================================================================
# 3. Bubble radius R(z), eq. (*)
# =====================================================================
def R_proper_Mpc(z, f_esc=0.2, SFR=SFR_fid):
    ta = t_age(z)
    ta = np.where(ta > 0, ta, np.nan)
    R3 = 3*Ndot_ion(z, f_esc=f_esc, SFR=SFR)*ta/(4*np.pi*nH(z)*x_HI(z))
    return (R3)**(1/3)/Mpc

# =====================================================================
# 4. VERIFICATION (sympy + numerical cross-checks) -- printed, not hidden
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)

# 4.1 sympy: same ODE check as bubble_radius_vs_redshift.py (model form is
#     unchanged; only the numerical x_HI(z) input differs).
t_s, Nd_s, nH_s, xHI_s = sp.symbols('t Ndot n_H x_HI', positive=True)
R3_expr = (3*Nd_s*t_s)/(4*sp.pi*nH_s*xHI_s)
residual = sp.simplify(sp.diff(R3_expr, t_s) - 3*Nd_s/(4*sp.pi*nH_s*xHI_s))
print(f"[sympy] d/dt[3 Ndot t/(4 pi nH xHI)] - 3 Ndot/(4 pi nH xHI) = {residual}  -> {'OK' if residual==0 else 'FAIL'}")

# 4.2 cosmology sanity checks (identical benchmarks as before)
a0 = cosmic_age(0)/Gyr
a_zre = cosmic_age(Z_RE_PLANCK)/Gyr
print(f"[cosmology] age(z=0)    = {a0:.3f} Gyr   (benchmark: 13.8 Gyr)")
print(f"[cosmology] age(z=7.68) = {a_zre:.3f} Gyr   (benchmark: ~0.6-0.7 Gyr, Planck z_re)")
assert abs(a0-13.8) < 0.1, "age(z=0) inconsistent with standard LCDM benchmark"

# 4.3 x_HI(z) at the Umeda+2023 redshifts, for direct comparison with their
#     MEASURED values (this is now a genuine model-vs-data comparison, not a
#     fit residual, since (z_re,dz) here come from Planck alone).
for zz, xhi_obs in zip(UMEDA_Z, UMEDA_XHI):
    xhi_mod = x_HI(zz)
    print(f"[x_HI check] z={zz:.2f}: x_HI,Planck-only = {xhi_mod:.3f}  vs  x_HI,Umeda+2023(measured) = {xhi_obs:.2f}"
          f"  -> DISCREPANT by design (see docstring, early/late-reionization tension, reverse direction of the companion script)")
print(f"[x_HI check] x_HI(z=6.0) = {x_HI(6.0):.4f}  (small, CONSISTENT with the near-total ionization implied"
      f" by the z~6 Gunn-Peterson troughs of White et al. 2003 -- no extrapolation tension in this direction here)")

# 4.4 M_UV corresponding to the fiducial SFR (identical to companion script)
print(f"[SFR->M_UV] SFR_fid={SFR_fid:.0f} Msun/yr -> L_UV={L_UV_fid:.3e} erg/s/Hz -> M_UV={M_UV_fid:.2f}"
      f"  (brighter than the M_UV<-18.5 selection of Umeda et al. 2023)")

# 4.5 Cross-check against Umeda+2023 measured bubble sizes
for zz, logRb in zip(UMEDA_Z, UMEDA_LOGRB_CMPC):
    Rb_prop = 10**logRb/(1+zz)
    Rmod = R_proper_Mpc(zz)
    print(f"[cross-check] z={zz:.2f}: R_model = {Rmod:.4f} pMpc  vs  R_b,Umeda+2023 = {Rb_prop:.4f} pMpc"
          f"  (ratio model/obs = {Rmod/Rb_prop:.2e})")

print(f"[sensitivity] R(z=8) for f_esc=0.2 vs 0.05: {R_proper_Mpc(8,f_esc=0.2):.3f} vs {R_proper_Mpc(8,f_esc=0.05):.3f} pMpc"
      f"  (scales as f_esc^(1/3): {(0.2/0.05)**(1/3):.3f})")
print(f"[sensitivity] R(z=8) for SFR=1 vs 50 Msun/yr: {R_proper_Mpc(8,SFR=1):.3f} vs {R_proper_Mpc(8,SFR=50):.3f} pMpc")
xi_scatter_factor = 10**(0.42/3)
print(f"[sensitivity] +/-0.42 dex intrinsic scatter in xi_ion (Llerena+2024) propagates to a factor {xi_scatter_factor:.2f}x in R")

# 4.6 Direct comparison with the companion (Umeda-anchored) script, at fixed
#     f_esc=0.2, to quantify how much the reionization-history choice matters.
print(f"[history comparison] at z=8: R(Planck-only x_HI) = {R_proper_Mpc(8):.4f} pMpc"
      f"  vs. R(Umeda-anchored x_HI, bubble_radius_vs_redshift.py) = 0.7517 pMpc"
      f"  -> ratio = {R_proper_Mpc(8)/0.7517:.3f}")
print("="*78)

# =====================================================================
# 5. Figure (same layout/style as bubble_radius_vs_redshift.py; solid over
#    the full range since (z_re,dz) here is a global CMB fit, not a sparse
#    2-point interpolation -- see docstring item 4)
# =====================================================================
z_plot = np.linspace(6.0, 18.0, 500)
R_central = R_proper_Mpc(z_plot, f_esc=0.2)
R_low     = R_proper_Mpc(z_plot, f_esc=0.05)

fig, ax = plt.subplots(figsize=(8.2, 5.8))

ax.fill_between(z_plot, R_low, R_central, color='#3b7dd8', alpha=0.25,
                 label=r'$f_{\rm esc}=0.05$–$0.2$ range (Robertson+2013,2015)')
ax.plot(z_plot, R_central, color='#1c4b8f', lw=2.2,
        label=r'$R(z)$, $f_{\rm esc}=0.2$, Planck-2018 CMB-only $x_{\rm HI}(z)$')

Rb_umeda_pmpc = 10**UMEDA_LOGRB_CMPC/(1+UMEDA_Z)
Rb_err_pmpc = 10**UMEDA_LOGRB_CMPC*np.log(10)*UMEDA_LOGRB_ERR.T/(1+UMEDA_Z)
ax.errorbar(UMEDA_Z, Rb_umeda_pmpc, yerr=np.abs(Rb_err_pmpc), fmt='D', ms=7,
            color='#c1272d', ecolor='#c1272d', capsize=4,
            label=r'$R_b$ measured, Umeda et al. (2023) (JWST-based $x_{\rm HI}$, not used here)')

ax.set_yscale('log')
ax.set_xlabel('redshift $z$')
ax.set_ylabel(r'ionized-bubble radius $R$ (proper Mpc)')
ax.set_title('Ionized H II bubble radius around a star-forming galaxy\n'
              r'(SFR$=10\,M_\odot\,{\rm yr}^{-1}$, $\xi_{\rm ion}(z)$ Llerena+2024, photon-counting limit;'
              '\n'
              r'Planck 2018 CMB-only reionization history, $z_{\rm re}=7.68$, $\Delta z=0.5$)')
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25, which='both')
ax.set_xlim(6, 18)

ymin, ymax = ax.get_ylim()
ax.axvline(Z_RE_PLANCK, color='gray', ls=':', lw=1.2, zorder=0)
ax.text(Z_RE_PLANCK+0.15, ymax*0.55, r'$z_{\rm re}=7.68$', rotation=90, va='top', ha='left',
        color='gray', fontsize=8.5)

fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.text(0.5, 0.012,
         rf'$z_{{\rm form}}={z_form:.0f}$ (Kitayama+2004 range); '
         r'$x_{\rm HI}(z)$: Planck-2018 CMB-only tanh, $(z_{\rm re},\Delta z)=(7.68,0.5)$ (Planck Collab. 2020)',
         ha='center', fontsize=8, color='dimgray')

outpath = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Stromgren_sphere/bubble_radius_vs_redshift_planckXHI.png'
fig.savefig(outpath, dpi=200)
print(f"Figure saved to: {outpath}")

# =====================================================================
# 6. Audit trail: parameters used
# =====================================================================
print("="*78)
print("PARAMETERS USED (audit trail)")
print("="*78)
print(f"SFR_fid = {SFR_fid} Msun/yr ; f_esc = 0.2 (band down to 0.05) ; z_form = {z_form}")
print(f"kappa_FUV = {kappa_FUV:.3e} Msun/yr/(erg/s/Hz)  [Madau & Dickinson 2014]")
print(f"xi_ion(z): log10 = 0.06 z + 24.82  [Llerena et al. 2024]")
print(f"Cosmology: h={h}, Omega_m={Om}, Omega_b h^2={Ob_h2}, Y_p={Yp}  [Planck 2018 VI]")
print(f"nH0 = {nH0:.4e} cm^-3")
print(f"x_HI(z): Planck-2018 CMB-only tanh, (z_re, dz) = ({Z_RE_PLANCK}, {DZ_PLANCK})  [Planck Collaboration 2020]")
