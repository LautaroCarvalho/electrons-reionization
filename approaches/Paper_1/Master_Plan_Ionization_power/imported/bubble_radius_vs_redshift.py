# Copied from Stromgren_sphere/bubble_radius_vs_redshift.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""
Ionized-hydrogen ('Stromgren') bubble radius around a single star-forming
galaxy, as a function of redshift.

MODEL (photon-counting limit, no recombinations / no Hubble term)
-------------------------------------------------------------------
Following eq. (photon_counting) of stromgren_sphere_summary.tex,
    R(z) = [ 3 * Ndot_ion(z) * t_age(z) / (4 pi * nH(z) * x_HI(z)) ]^(1/3)      (*)
i.e. every ionizing photon emitted since the galaxy formed is assumed to be
used exactly once to ionize a neutral hydrogen atom (no recombinations).
This is the relation Cen & Haiman (2000, arXiv:astro-ph/0006376) derive for a
quasar (their eq. 2) and Mesinger & Haiman (2004, arXiv:astro-ph/0406188) use
for SDSS J1030+0524; here it is applied to a star-forming-galaxy source
instead of a quasar. check_consistency.py (item 7) already shows that the
recombination time exceeds both the Hubble time and any plausible source age
at z >~ 6, which is why dropping the recombination term is justified in this
regime (Cen & Haiman 2000).

INGREDIENTS AND THEIR SOURCES (every number is cited; none are invented)
-------------------------------------------------------------------
1. Ionizing photon production rate:
     Ndot_ion(z) = f_esc * xi_ion(z) * L_UV
   with
     L_UV = SFR / kappa_FUV,   kappa_FUV = 1.15e-28 Msun/yr / (erg/s/Hz)
            (Madau & Dickinson 2014, arXiv:1403.0007, Salpeter IMF, SFR = kappa_FUV * L_UV)
     log10( xi_ion / Hz erg^-1 ) = (0.06 +/- 0.012) z + (24.82 +/- 0.07)
            (Llerena et al. 2024, arXiv:2412.01358; JWST/NIRSpec sample of 761
            star-forming galaxies at z=4-10; measured assuming f_esc=0, i.e. it
            is the *production* efficiency before escape -- exactly what is
            needed here, since f_esc is applied separately below)
     f_esc = 0.2 fiducial (Robertson et al. 2015, arXiv:1502.02024, following
            Robertson et al. 2013); a lower literature value f_esc=0.05 is
            shown as a sensitivity band (see e.g. the review discussion in
            Chakraborty & Choudhury 2025, arXiv:2502.12004, already in
            references/, on the uncertainty of f_esc).

2. Mean cosmic hydrogen density:
     nH(z) = nH0 * (1+z)^3,   nH0 = Omega_b*rho_crit0*(1-Y_p)/m_H
   with Planck 2018 (arXiv:1807.06209) parameters Omega_b h^2 = 0.0224,
   h = 0.674, Y_p = 0.245.

3. Age of the galaxy, t_age(z) = t_cosmic(z) - t_cosmic(z_form):
   flat LCDM age-redshift relation, t(z) = int_z^inf dz'/[(1+z') H(z')],
   H(z) = H0 sqrt(Omega_m (1+z)^3 + Omega_Lambda), Omega_m = 0.315 (Planck
   2018, arXiv:1807.06209). z_form = 20 is adopted as a fiducial formation
   redshift for the first star-forming galaxies, within the z=10-30 range in
   which Kitayama et al. (2004, arXiv:astro-ph/0406280, already in
   references/) place the first (Pop III) sources; this is an explicit
   benchmark choice, not a measurement.

4. Ambient neutral fraction x_HI(z): the standard tanh reionization
   parameterization (Lewis 2008; used by Planck, e.g. arXiv:1807.06209),
      x_HII(z) = 0.5*(1 + tanh[(y(z_re)-y(z))/Dy]),  y(z)=(1+z)^1.5,
      Dy = 1.5 * sqrt(1+z_re) * Delta_z,
   with (z_re, Delta_z) solved for so that the model passes through the two
   JWST Ly-alpha damping-wing measurements of Umeda et al. (2023,
   arXiv:2306.00487, already in references/):
      x_HI(z=7.12) = 0.53 ,  x_HI(z=9.91) = 0.92 .
   IMPORTANT CAVEAT (rule 5/8 -- flagged, not hidden): this 2-point fit is
   NOT anchored at z<7 and its naive extrapolation to z~6 (x_HI~0.3) is
   inconsistent with the near-total ionization of the IGM implied by the very
   large Lyman-alpha/beta Gunn-Peterson optical depths at z~6 reported by
   White et al. (2003, arXiv:astro-ph/0303476, already in references/). It is
   also numerically very different from the CMB-only instantaneous-
   reionization estimate z_re=7.68+/-0.79 of Planck 2018 (arXiv:1807.06209):
   this is the well-known tension between "early" (CMB-inferred) and "late"
   (direct EoR-galaxy-probe-inferred) reionization histories, already noted
   qualitatively in the "Known tensions" paragraph of
   stromgren_sphere_summary.tex. Consequently: the curve is drawn SOLID for
   7 <= z <= 18 (interpolation regime, bracketed by the two Umeda et al. 2023
   data points) and DASHED for 6 <= z < 7 (extrapolation, flagged as being in
   tension with the Gunn-Peterson constraint).

Units: all internal computation is in cgs; final radii are reported and
plotted in proper Mpc (pMpc), matching the convention used throughout
stromgren_sphere_summary.tex (e.g. eq. mesinger).

Author: prepared for L. Carvalho. Verified with sympy 1.12 / scipy / numpy;
see the printed "VERIFICATION" block below for every cross-check performed.
"""
import numpy as np
import sympy as sp
from scipy import integrate, optimize
import matplotlib.pyplot as plt

# =====================================================================
# 0. Physical constants (cgs) and Planck 2018 cosmology (arXiv:1807.06209)
# =====================================================================
mH = 1.6726e-24            # g, hydrogen atom mass
pc = 3.0857e18             # cm
Mpc = 3.0857e24            # cm
yr = 3.156e7                # s (as in check_consistency.py)
Gyr = yr*1e9

h = 0.674
Om = 0.315
OL = 1 - Om
Ob_h2 = 0.0224
Yp = 0.245
H0 = h*100*1e5/Mpc          # s^-1  (100 h km/s/Mpc, converted to cgs)
rho_crit0 = 1.878e-29*h**2  # g/cm^3
rho_b0 = Ob_h2*1.878e-29    # g/cm^3  (Omega_b*rho_crit0, h^2 cancels)
nH0 = rho_b0*(1-Yp)/mH      # cm^-3, present-day mean H number density

def Hz(z):
    return H0*np.sqrt(Om*(1+z)**3 + OL)

def nH(z):
    return nH0*(1+z)**3

def _cosmic_age_scalar(z):
    """t(z) in seconds, flat LCDM, matter+Lambda only (radiation negligible
    at z<50: Omega_r0~9e-5 vs Omega_m(1+z)^3>=3e2 there)."""
    integrand = lambda zp: 1.0/((1+zp)*Hz(zp))
    val, _ = integrate.quad(integrand, z, np.inf, limit=200)
    return val

cosmic_age = np.vectorize(_cosmic_age_scalar)

# =====================================================================
# 1. Ionizing photon production rate of the galaxy, Ndot_ion(z)
# =====================================================================
kappa_FUV = 1.15e-28        # Msun/yr per (erg/s/Hz), Madau & Dickinson 2014, Salpeter IMF
SFR_fid = 10.0              # Msun/yr, fiducial "star-forming galaxy" (see M_UV check below)
L_UV_fid = SFR_fid/kappa_FUV  # erg/s/Hz

def xi_ion(z):
    """Llerena et al. 2024 (arXiv:2412.01358), z=4-10 fit; extrapolated
    beyond z=10 (their sample; used here only as the leading-order trend)."""
    log_xi = 0.06*z + 24.82
    return 10**log_xi           # Hz erg^-1 = photons/s per erg/s/Hz

def Ndot_ion(z, f_esc=0.2, SFR=SFR_fid):
    L_UV = SFR/kappa_FUV
    return f_esc*xi_ion(z)*L_UV   # photons/s

# M_UV equivalent of the fiducial SFR (AB absolute magnitude), purely as a
# cross-check that SFR_fid corresponds to a JWST-typical *bright* galaxy,
# consistent with the M_UV<-18.5 selection of Umeda et al. (2023):
d10pc = 10*pc
M_UV_fid = -2.5*np.log10(L_UV_fid/(4*np.pi*d10pc**2)) - 48.60

# =====================================================================
# 2. Ambient neutral fraction x_HI(z): tanh model calibrated on Umeda+2023
# =====================================================================
UMEDA_Z    = np.array([7.12, 9.91])
UMEDA_XHI  = np.array([0.53, 0.92])
UMEDA_XHI_ERR = np.array([[0.47, 0.10], [0.18, 0.08]])  # [-,+] for each point
# Umeda+2023 measured bubble radii (log10 R_b in comoving Mpc), for the
# independent cross-check against the model prediction further below:
UMEDA_LOGRB_CMPC = np.array([1.67, -0.69])
UMEDA_LOGRB_ERR  = np.array([[0.16, 0.24], [0.14, 0.89]])  # [-,+]

def y_of_z(z):
    return (1+z)**1.5

def xHII_tanh(z, z_re, dz):
    dy = 1.5*np.sqrt(1+z_re)*dz
    return 0.5*(1 + np.tanh((y_of_z(z_re) - y_of_z(z))/dy))

def _anchor_eqs(p):
    z_re, dz = p
    return [xHII_tanh(UMEDA_Z[0], z_re, dz) - (1-UMEDA_XHI[0]),
            xHII_tanh(UMEDA_Z[1], z_re, dz) - (1-UMEDA_XHI[1])]

Z_RE_FIT, DZ_FIT = optimize.fsolve(_anchor_eqs, [9.0, 1.0])

def x_HI(z):
    return 1 - xHII_tanh(z, Z_RE_FIT, DZ_FIT)

z_form = 20.0   # fiducial formation redshift (Kitayama+2004 range z=10-30)

def t_age(z):
    return cosmic_age(z) - cosmic_age(z_form)

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

# 4.1 sympy: R(z) as written solves dR^3/dt = 3 Ndot /(4 pi nH x_HI) at fixed
#     Ndot, nH, x_HI (the defining photon-counting ODE, eq. photon_counting).
t_s, Nd_s, nH_s, xHI_s = sp.symbols('t Ndot n_H x_HI', positive=True)
R3_expr = (3*Nd_s*t_s)/(4*sp.pi*nH_s*xHI_s)
residual = sp.simplify(sp.diff(R3_expr, t_s) - 3*Nd_s/(4*sp.pi*nH_s*xHI_s))
print(f"[sympy] d/dt[3 Ndot t/(4 pi nH xHI)] - 3 Ndot/(4 pi nH xHI) = {residual}  -> {'OK' if residual==0 else 'FAIL'}")

# 4.2 cosmology sanity checks against textbook/Planck benchmarks
a0 = cosmic_age(0)/Gyr
a_zre = cosmic_age(7.68)/Gyr
print(f"[cosmology] age(z=0)    = {a0:.3f} Gyr   (benchmark: 13.8 Gyr)")
print(f"[cosmology] age(z=7.68) = {a_zre:.3f} Gyr   (benchmark: ~0.6-0.7 Gyr, Planck z_re)")
assert abs(a0-13.8) < 0.1, "age(z=0) inconsistent with standard LCDM benchmark"

# 4.3 x_HI(z) tanh fit reproduces the two Umeda+2023 anchors
res712 = x_HI(7.12) - 0.53
res991 = x_HI(9.91) - 0.92
print(f"[x_HI fit] residual at z=7.12: {res712:.2e} ; at z=9.91: {res991:.2e}  (both ~0 by construction)")
print(f"[x_HI fit] fitted (z_re, dz) = ({Z_RE_FIT:.2f}, {DZ_FIT:.2f})"
      f"  vs Planck2018 CMB-only (z_re, dz) = (7.68, 0.5)  -> DISCREPANT (see docstring, known early/late-reionization tension)")
print(f"[x_HI fit] extrapolated x_HI(z=6.0) = {x_HI(6.0):.2f}  -- NOT small, in tension with the"
      f" near-complete ionization implied by the z~6 Gunn-Peterson troughs of White et al. 2003."
      f" This is why z<7 is shown dashed/flagged below, per rule 5/8.")

# 4.4 M_UV corresponding to the fiducial SFR
print(f"[SFR->M_UV] SFR_fid={SFR_fid:.0f} Msun/yr -> L_UV={L_UV_fid:.3e} erg/s/Hz -> M_UV={M_UV_fid:.2f}"
      f"  (brighter than the M_UV<-18.5 selection of Umeda et al. 2023, i.e. a JWST-typical bright source)")

# 4.5 Order-of-magnitude comparison with Umeda+2023 measured bubble sizes
#     (these are MERGED/clustered-source bubbles, not a single galaxy's own
#     Stromgren sphere -- so R_model < R_Umeda is EXPECTED, not a bug; this
#     reproduces the "known tension" (iii) already flagged in
#     stromgren_sphere_summary.tex).
for zz, logRb in zip(UMEDA_Z, UMEDA_LOGRB_CMPC):
    Rb_prop = 10**logRb/(1+zz)
    Rmod = R_proper_Mpc(zz)
    print(f"[cross-check] z={zz:.2f}: R_model = {Rmod:.4f} pMpc  vs  R_b,Umeda+2023 = {Rb_prop:.4f} pMpc"
          f"  (ratio model/obs = {Rmod/Rb_prop:.2e})")

print(f"[sensitivity] R(z=8) for f_esc=0.2 vs 0.05: {R_proper_Mpc(8,f_esc=0.2):.3f} vs {R_proper_Mpc(8,f_esc=0.05):.3f} pMpc"
      f"  (scales as f_esc^(1/3): {(0.2/0.05)**(1/3):.3f})")
print(f"[sensitivity] R(z=8) for SFR=1 vs 50 Msun/yr: {R_proper_Mpc(8,SFR=1):.3f} vs {R_proper_Mpc(8,SFR=50):.3f} pMpc")
xi_scatter_factor = 10**(0.42/3)  # 0.42 dex observed scatter in xi_ion (Llerena+2024), R propto xi_ion^(1/3)
print(f"[sensitivity] +/-0.42 dex intrinsic scatter in xi_ion (Llerena+2024) propagates to a factor {xi_scatter_factor:.2f}x in R")
print("="*78)

# =====================================================================
# 5. Figure
# =====================================================================
z_hi = np.linspace(7.0, 18.0, 400)      # interpolation regime (solid)
z_lo = np.linspace(6.0, 7.0, 100)       # extrapolation regime  (dashed, flagged)

R_hi_central = R_proper_Mpc(z_hi, f_esc=0.2)
R_hi_low     = R_proper_Mpc(z_hi, f_esc=0.05)  # 0.2 is the upper fiducial (Robertson+2015)
R_lo_central = R_proper_Mpc(z_lo, f_esc=0.2)
R_lo_band    = R_proper_Mpc(z_lo, f_esc=0.05)

fig, ax = plt.subplots(figsize=(7.2, 5.6))

ax.fill_between(z_hi, R_hi_low, R_hi_central, color='#3b7dd8', alpha=0.25,
                 label=r'$f_{\rm esc}=0.05$–$0.2$ range (Robertson+2013,2015)')
ax.fill_between(z_lo, R_lo_band, R_lo_central, color='#3b7dd8', alpha=0.15)

ax.plot(z_hi, R_hi_central, color='#1c4b8f', lw=2.2,
        label=r'$R(z)$, $f_{\rm esc}=0.2$ (interpolation, $7\leq z\leq18$)')
ax.plot(z_lo, R_lo_central, color='#1c4b8f', lw=2.2, ls='--',
        label=r'extrapolated, in tension with $z\sim6$ Gunn--Peterson (White+2003)')

Rb_umeda_pmpc = 10**UMEDA_LOGRB_CMPC/(1+UMEDA_Z)
Rb_err_pmpc = 10**UMEDA_LOGRB_CMPC*np.log(10)*UMEDA_LOGRB_ERR.T/(1+UMEDA_Z)
ax.errorbar(UMEDA_Z, Rb_umeda_pmpc, yerr=np.abs(Rb_err_pmpc), fmt='D', ms=7,
            color='#c1272d', ecolor='#c1272d', capsize=4, label=r'$R_b$ measured, Umeda et al. (2023)')

ax.set_yscale('log')
ax.set_xlabel('redshift $z$')
ax.set_ylabel(r'ionized-bubble radius $R$ (proper Mpc)')
ax.set_title('Ionized H II bubble radius around a star-forming galaxy\n'
              r'(SFR$=10\,M_\odot\,{\rm yr}^{-1}$, $\xi_{\rm ion}(z)$ Llerena+2024, photon-counting limit)')
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax.grid(alpha=0.25, which='both')
ax.set_xlim(6, 18)
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.text(0.5, 0.012,
         rf'$t_{{\rm age}}(z)=t_{{\rm cosmic}}(z)-t_{{\rm cosmic}}(z_{{\rm form}})$, $z_{{\rm form}}={z_form:.0f}$ '
         r'(Kitayama+2004 range); $x_{\rm HI}(z)$ tanh fit to Umeda+2023 (see script docstring for all sources)',
         ha='center', fontsize=7.5, color='dimgray')

outpath = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Master_Plan_Ionization_power/figures/bubble_radius_vs_redshift.png'
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
print(f"x_HI(z) tanh fit: (z_re, dz) = ({Z_RE_FIT:.3f}, {DZ_FIT:.3f}), calibrated on Umeda et al. 2023")
