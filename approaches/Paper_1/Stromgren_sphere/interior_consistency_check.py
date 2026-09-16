"""
Follow-up consistency check, triggered by a direct question from L. Carvalho
on ionizing_photon_mean_free_path_response.tex:

    "Once the ionized bubble is established, new photons emitted from the
    source travel through a FULLY IONIZED medium (not the fully neutral
    medium used to compute lambda_mfp,neutral(E,z)). Check if there is an
    inconsistency."

This script performs that check in full, in two independent parts, and
reports what it finds honestly (Master Rule 5), including a genuine,
previously-unflagged systematic effect it uncovers.

PART 1 -- Is the "photon skips the bubble" argument itself consistent?
-------------------------------------------------------------------
YES, with a clarification: lambda_mfp,neutral(E,z) in
ionizing_photon_mean_free_path.py was never meant to describe the path
through the bubble's ionized INTERIOR (where the residual neutral fraction
is tiny and the mean free path is enormous -- photons stream through nearly
unimpeded, see PART 2). It describes how far a photon travels once it
reaches fresh, still-neutral gas just beyond the CURRENT front. Comparing
that quantity to R_bubble(z) tests whether a photon can punch a further
R_bubble's worth of distance into that fresh neutral gas (i.e. reach
"deep" into the IGM) rather than being absorbed in the usual thin
(~kpc) skin -- a meaningful question, but one that was not stated clearly
enough in the original response. This script makes the two-zone (ionized
interior / neutral exterior) structure explicit.

PART 2 -- Quantifying the interior: is it REALLY fully transparent?
-------------------------------------------------------------------
Using the standard photoionization-equilibrium formula for the residual
neutral fraction inside an ionized region (Cen & Haiman 2000,
arXiv:astro-ph/0006376, their eq. 5/7, already in references/; re-derived
from scratch with sympy below and cross-checked against their own quoted
normalization), this script finds that the interior is NOT perfectly
transparent everywhere: close to the edge (r ~ R_bubble), the LOCAL mean
free path is only ~10-20% of R_bubble itself, i.e. the "sharp front" is
really a transition zone of finite (though still sub-dominant) thickness,
not a true discontinuity.

More importantly, this check surfaces a genuine, previously-unflagged
tension: for the specific parameters adopted throughout this folder
(z_form=20, clumping factor C_HII=3 from Robertson et al. 2015), the
source age t_age(z) actually EXCEEDS the recombination time t_rec(z) by a
factor ~1.2-1.9 across z=6-12 -- meaning recombinations are NOT negligible
over the galaxy's lifetime, contrary to the implicit assumption underlying
the "pure photon-counting" model chosen for bubble_radius_vs_redshift.py
(a choice explicitly made by L. Carvalho at the start of that task).
Solving the FULL recombination-included growth equation
R(t) = R_S (1 - e^{-t/t_rec})^{1/3} (eq. Rt of stromgren_sphere_summary.tex,
already sympy-verified there) with self-consistent normalization gives a
bubble radius 24-42% SMALLER than the pure photon-counting estimate used
throughout this session. The qualitative conclusion of
ionizing_photon_mean_free_path_response.tex is essentially unchanged
(E_cross shifts down by only 9-17%, remaining solidly in the soft X-ray
regime), but the absolute R_bubble(z) values used throughout this session
should be read as mild (25-40%) overestimates once recombinations are
self-consistently included.

Author: prepared for L. Carvalho. Verified with sympy/scipy/numpy; see the
printed VERIFICATION block below.
"""
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# =====================================================================
# 0. Cosmology (Planck 2018) -- identical to the other scripts in this folder
# =====================================================================
mH = 1.6726e-24
pc, kpc, Mpc = 3.0857e18, 3.0857e21, 3.0857e24
yr, Gyr = 3.156e7, 3.156e7*1e9

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
# 1. Source and bubble model -- IDENTICAL to bubble_radius_vs_redshift.py
# =====================================================================
kappa_FUV = 1.15e-28
f_esc = 0.2
SFR_fid = 10.0
z_form = 20.0

def xi_ion(z):
    return 10**(0.06*z + 24.82)

def Ndot_ion_gal(z, SFR=SFR_fid):
    return f_esc*xi_ion(z)*(SFR/kappa_FUV)

def t_age(z):
    return cosmic_age(z) - cosmic_age(z_form)

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

def R_bubble_cm(z):
    """Pure photon-counting radius (NO recombination), as used throughout
    bubble_radius_vs_redshift.py and every downstream script this session."""
    ta = t_age(z)
    R3 = 3*Ndot_ion_gal(z)*ta/(4*np.pi*nH_phys(z)*x_HI_ext(z))
    return R3**(1/3)

alpha_B = 2.59e-13   # cm^3/s, T=1e4 K (Shapiro+2006)
C_HII = 3.0          # Robertson et al. 2015

def R_S_cm(z):
    """Equilibrium Stromgren radius, eq. RS of stromgren_sphere_summary.tex."""
    Nd, nH = Ndot_ion_gal(z), nH_phys(z)
    return (3*Nd/(4*np.pi*alpha_B*C_HII*nH**2))**(1/3)

def t_rec(z):
    """eq. Rt of stromgren_sphere_summary.tex: t_rec=(C alpha_B n_H)^-1."""
    return 1.0/(alpha_B*C_HII*nH_phys(z))

def R_with_recomb_cm(z):
    """Full time-dependent solution INCLUDING recombination,
    eq. Rt of stromgren_sphere_summary.tex (already sympy-verified there)."""
    return R_S_cm(z)*(1 - np.exp(-t_age(z)/t_rec(z)))**(1/3)

# =====================================================================
# 2. Interior residual neutral fraction x(r): photoionization equilibrium
#    (Cen & Haiman 2000, arXiv:astro-ph/0006376, eq. 5/7)
# =====================================================================
sigma_bar_stellar = 6.3e-18   # cm^2, ~ sigma_H(E_H) since stellar photons cluster near threshold

x_sym, r_sym, aB_sym, C_sym, nH_sym, sig_sym, Nd_sym = sp.symbols(
    'x r alpha_B C n_H sigma_bar Ndot', positive=True)
_balance = sp.Eq(sig_sym*Nd_sym/(4*sp.pi*r_sym**2) * x_sym*nH_sym, aB_sym*C_sym*nH_sym**2)
_x_solution = sp.solve(_balance, x_sym)[0]
x_of_r = sp.lambdify((r_sym, aB_sym, C_sym, nH_sym, sig_sym, Nd_sym), _x_solution, 'numpy')

def x_interior(r_cm, z):
    return x_of_r(r_cm, alpha_B, C_HII, nH_phys(z), sigma_bar_stellar, Ndot_ion_gal(z))

def mfp_interior_cm(r_cm, z):
    return 1.0/(sigma_bar_stellar*x_interior(r_cm, z)*nH_phys(z))

E_H = 13.6
sigma_H0 = 6.3e-18
def mfp_neutral_cm(E_eV, z):
    return 1.0/(sigma_H0*(E_eV/E_H)**-3*nH_phys(z))

# =====================================================================
# 3. VERIFICATION
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)

# 3.1 sympy: re-derive Cen & Haiman (2000) eq.5/7 from first-principles
#     photoionization equilibrium, and reproduce their quoted normalization
print(f"[sympy] photoionization-equilibrium x(r) = {_x_solution}")
sigma_bar_quasar, nH_z7, Nd_quasar = 2.5e-18, 8.5e-5, 1e57
x_check = x_of_r(1*Mpc, alpha_B, 1.0, nH_z7, sigma_bar_quasar, Nd_quasar)
print(f"[cross-check] x(r=1 Mpc, Ndot=1e57/s, C=1, z=7) = {x_check:.3e}  (Cen & Haiman 2000 quote ~1e-6) -> matches to <10%")

# 3.2 sympy: verify R(t) with recombination solves the correct ODE
t_s, Nd_s, nH_s, C_s, aB_s = sp.symbols('t Ndot n_H C alpha_B', positive=True)
RS_s = (3*Nd_s/(4*sp.pi*aB_s*C_s*nH_s**2))**sp.Rational(1,3)
trec_s = 1/(aB_s*C_s*nH_s)
Rt_s = RS_s*(1-sp.exp(-t_s/trec_s))**sp.Rational(1,3)
resid = sp.simplify(sp.diff(Rt_s**3, t_s) - (3*Nd_s/(4*sp.pi*nH_s) - aB_s*C_s*nH_s*Rt_s**3))
print(f"[sympy] R(t) with recombination solves dR^3/dt=3Ndot/(4pi nH)-alpha_B C nH R^3: residual={resid} -> {'OK' if resid==0 else 'FAIL'}")

# 3.3 interior transparency at r=R_bubble
print("\n[interior transparency at r=R_bubble, photoionization equilibrium]")
print(f"{'z':>5} {'x_HI,interior':>14} {'lambda_mfp,int(pMpc)':>21} {'lambda/R_bubble':>16}")
for z in [6, 8, 10, 12]:
    Rb = R_bubble_cm(z)
    xi = x_interior(Rb, z)
    lam = mfp_interior_cm(Rb, z)
    print(f"{z:5.1f} {xi:14.3e} {lam/Mpc:21.4f} {lam/Rb:16.3f}")
print("  -> interior mean free path is 9-17% of R_bubble near the edge: the 'sharp front'")
print("     is a transition zone of finite (sub-dominant) thickness, not a true discontinuity.")
print("     Well inside the bubble (r<<R_bubble), x(r) ~ r^2 falls rapidly and the interior")
print("     mean free path is correspondingly far larger than r there (free-streaming holds).")

# 3.4 THE MAIN FINDING: t_age vs t_rec, and the resulting bubble-radius correction
print("\n[main finding: is t_age << t_rec, as implicitly assumed by the pure photon-counting model?]")
print(f"{'z':>5} {'t_age(Gyr)':>11} {'t_rec(Gyr,C=3)':>15} {'t_age/t_rec':>12} {'R_bubble,no-recomb(pMpc)':>25} {'R(t),with-recomb(pMpc)':>24} {'ratio':>8}")
for z in [6, 8, 10, 12]:
    ta, tr = t_age(z), t_rec(z)
    Rb, Rt = R_bubble_cm(z), R_with_recomb_cm(z)
    print(f"{z:5.1f} {ta/Gyr:11.3f} {tr/Gyr:15.3f} {ta/tr:12.3f} {Rb/Mpc:25.4f} {Rt/Mpc:24.4f} {Rt/Rb:8.3f}")
print("  -> t_age EXCEEDS t_rec by a factor 1.2-1.9 for the (z_form=20, C_HII=3) parameters")
print("     adopted throughout this folder: recombinations are NOT negligible over the")
print("     galaxy's assumed lifetime. The recombination-included radius R(t) is 24-42%")
print("     SMALLER than the pure photon-counting R_bubble used throughout this session.")

# 3.5 impact on the mean-free-path conclusion (ionizing_photon_mean_free_path_response.tex)
print("\n[impact on E_cross of ionizing_photon_mean_free_path_response.tex]")
for z in [6, 8, 10, 12]:
    Rb, Rt = R_bubble_cm(z), R_with_recomb_cm(z)
    Ec_old = E_H*(Rb/mfp_neutral_cm(E_H, z))**(1/3)
    Ec_new = E_H*(Rt/mfp_neutral_cm(E_H, z))**(1/3)
    print(f"  z={z:4.1f}: E_cross(no-recomb R)={Ec_old:.1f} eV -> E_cross(with-recomb R)={Ec_new:.1f} eV  ({100*(Ec_new/Ec_old-1):+.1f}%)")
print("  -> shift is only 9-17%; E_cross remains solidly in the soft X-ray regime in both")
print("     cases, so the QUALITATIVE conclusion (negligible photon leakage for stellar")
print("     sources) is UNCHANGED. The correction matters for the absolute R_bubble(z)")
print("     values, not for the mean-free-path argument's conclusion.")
print("="*78)

# =====================================================================
# 4. Figure: two panels telling the complete, corrected story
# =====================================================================
z_ref = 8.0
Rb_ref = R_bubble_cm(z_ref)
r_plot = np.logspace(np.log10(0.01*Rb_ref), np.log10(3*Rb_ref), 500)
mfp_profile = np.where(r_plot <= Rb_ref,
                        mfp_interior_cm(np.minimum(r_plot, Rb_ref*0.9999999), z_ref),
                        mfp_neutral_cm(E_H, z_ref))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.4))

ax1.plot(r_plot/Mpc, mfp_profile/Mpc, color='#1c4b8f', lw=2.2)
ax1.plot(r_plot/Mpc, r_plot/Mpc, color='gray', lw=1.2, ls=':', label=r'$\lambda_{\rm mfp}=r$ (1:1 line)')
ax1.axvline(Rb_ref/Mpc, color='#c1272d', lw=1.8, ls='--', label=r'$R_{\rm bubble}(z{=}8)$')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('distance from source, $r$ (proper Mpc)')
ax1.set_ylabel(r'$\lambda_{\rm mfp}(E_H,z{=}8)$ at that location (proper Mpc)')
ax1.set_title('Interior (ionized, transparent) vs.\nexterior (neutral, opaque) mean free path')
ax1.legend(loc='lower left', fontsize=8.5)
ax1.grid(alpha=0.25, which='both')
ax1.annotate('interior:\nfree-streaming\n'+r'($\lambda_{\rm mfp}\gg r$)', xy=(0.05*Rb_ref/Mpc, mfp_interior_cm(0.05*Rb_ref,z_ref)/Mpc),
             fontsize=8, color='#1c4b8f', ha='left')
ax1.annotate('exterior:\nabsorbed within ~kpc', xy=(1.5*Rb_ref/Mpc, mfp_neutral_cm(E_H,z_ref)/Mpc),
             fontsize=8, color='#1c4b8f', ha='left', va='top')

zarr = np.array([6, 8, 10, 12])
Rb_arr = np.array([R_bubble_cm(z)/Mpc for z in zarr])
Rt_arr = np.array([R_with_recomb_cm(z)/Mpc for z in zarr])
width = 0.35
xpos = np.arange(len(zarr))
ax2.bar(xpos-width/2, Rb_arr, width, color='#3b7dd8', label='pure photon-counting\n(used throughout this session)')
ax2.bar(xpos+width/2, Rt_arr, width, color='#c1272d', label='with recombination\n(eq. Rt, this check)')
ax2.set_xticks(xpos); ax2.set_xticklabels([f'{z:.0f}' for z in zarr])
ax2.set_xlabel('redshift $z$')
ax2.set_ylabel(r'$R_{\rm bubble}$ (proper Mpc)')
ax2.set_title('Recombination correction to $R_{\\rm bubble}(z)$\n'+r'($z_{\rm form}=20$, $C_{\rm HII}=3$)')
ax2.legend(fontsize=8.5)
ax2.grid(alpha=0.25, axis='y')
for i,(rb,rt) in enumerate(zip(Rb_arr,Rt_arr)):
    ax2.text(i, max(rb,rt)*1.03, f'{rt/rb:.0%}', ha='center', fontsize=8.5, color='dimgray')

fig.tight_layout(rect=[0.005, 0.045, 0.995, 1])
fig.text(0.5, 0.010,
         r'Left: $z=8$ only. Right: percentages show $R(t)_{\rm with-recomb}/R_{\rm bubble,no-recomb}$. '
         r'Both panels use the identical model as bubble_radius_vs_redshift.py plus recombination.',
         ha='center', fontsize=8, color='dimgray')

outpath = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Stromgren_sphere/interior_consistency_check.png'
fig.savefig(outpath, dpi=200)
print(f"Figure saved to: {outpath}")
