"""
What fraction of a star-forming galaxy's ionizing photons "skip" the local
bubble and travel deeper into the still-neutral IGM, rather than being
absorbed at (and so driving the expansion of) the ionization front?

This question is answered by the PHOTON MEAN FREE PATH in a neutral medium,
lambda_mfp(E,z), a standard quantity already discussed (with the exact
formula used below) in Choudhury (2022, arXiv:2209.08558, already in
references/, their eq. 46 and Sec. "Local absorption"), and used throughout
the reionization literature to justify (or not) treating the IGM as a
sharp-fronted, two-phase (ionized/neutral) medium -- see also Miralda-Escude,
Haehnelt & Rees (1998, arXiv:astro-ph/9812306, already in references/, their
Sec. 3.1 "The mean free path of the ionizing photons") and Gnedin & Madau
(2022, arXiv:2208.02260, already in references/, their Sec. 2.5 and Fig. 7).

PHYSICS
-------------------------------------------------------------------
The hydrogen photoionization cross section falls steeply with photon energy,
    sigma_H(E) = sigma_H(E_H) * (E/E_H)^-3 ,   sigma_H(E_H) = 6.3e-18 cm^2
(Choudhury 2022, eq. 46; E_H = 13.6 eV is the Lyman limit). In a neutral
medium of physical hydrogen density n_H(z), a photon of energy E therefore
has mean free path
    lambda_mfp(E,z) = 1 / [sigma_H(E) * n_H(z)]  ~  E^3 .
Because lambda_mfp ~ E^3, a photon only needs to be modestly harder than the
Lyman limit to travel MUCH further before being absorbed. The question of
"how much of a source's photon budget reaches beyond its own bubble" is
therefore a question about how hard the emitted spectrum is, compared to the
energy E_cross(z) at which lambda_mfp(E_cross,z) equals the bubble's own
radius R_bubble(z) (from bubble_radius_vs_redshift.py).

RESULT (see VERIFICATION block for every number)
-------------------------------------------------------------------
At z=6-12, lambda_mfp(13.6 eV, z) is only ~0.1-0.8 kpc -- matching
Choudhury (2022)'s statement that this scale is "much smaller than any
cosmologically relevant length scale" -- while R_bubble(z) is ~0.4-1.3 pMpc
(3-4 orders of magnitude larger). Photons need E_cross~160-200 eV (soft
X-rays) to have a neutral-IGM mean free path as large as the bubble itself.
For a normal STELLAR ionizing spectrum (thermal, cut off in the Wien tail
well below ~100 eV for O/B-star effective temperatures), essentially none of
the photons reach that energy, so essentially none of them skip the bubble:
almost the entire photon budget is absorbed at/near the ionization front (or
by residual Lyman-limit systems inside the bubble), consistent with the
zero-leakage "photon-counting" model used throughout this folder. For a
QUASAR-like power-law spectrum (Cen & Haiman 2000, arXiv:astro-ph/0006376,
already in references/, L_nu ~ nu^-1.8), the same calculation gives a
small but non-negligible ~1% of ionizing photons (by number) able to reach
beyond the local bubble -- illustrating why hard (AGN) sources, unlike
star-forming galaxies, are treated separately in the reionization literature
(explicitly flagged as a distinct regime in Choudhury 2022's footnote on
X-ray sources).

Author: prepared for L. Carvalho. Verified with sympy/numpy; see the
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
# 1. Photoionization cross section and neutral-medium mean free path
#    (Choudhury 2022, eq. 46)
# =====================================================================
E_H = 13.6          # eV, Lyman-limit energy
sigma_H0 = 6.3e-18  # cm^2, hydrogenic cross section at E_H

def sigma_H(E_eV):
    return sigma_H0*(E_eV/E_H)**-3

def mfp_neutral_cm(E_eV, z):
    return 1.0/(sigma_H(E_eV)*nH_phys(z))

# =====================================================================
# 2. Single-galaxy bubble radius R_bubble(z) -- IDENTICAL model to
#    bubble_radius_vs_redshift.py (reproduced here for self-containment)
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

def x_HI(z):
    return 1 - xHII_tanh(z, Z_RE_FIT, DZ_FIT)

def R_bubble_cm(z):
    ta = t_age(z)
    R3 = 3*Ndot_ion_gal(z)*ta/(4*np.pi*nH_phys(z)*x_HI(z))
    return R3**(1/3)

def E_cross(z):
    """Energy at which lambda_mfp,neutral(E,z) = R_bubble(z)."""
    Rb = R_bubble_cm(z)
    return E_H*(Rb/mfp_neutral_cm(E_H, z))**(1/3)

# =====================================================================
# 3. VERIFICATION
# =====================================================================
print("="*78)
print("VERIFICATION")
print("="*78)

# 3.1 sympy: symbolic form and dimensional check of lambda_mfp(E,z) ~ E^3
E_s, EH_s, s0_s, n_s = sp.symbols('E E_H sigma0 n', positive=True)
sigma_expr = s0_s*(E_s/EH_s)**-3
mfp_expr = sp.simplify(1/(sigma_expr*n_s))
print(f"[sympy] lambda_mfp(E) = {mfp_expr}  (explicit E^3 scaling; [sigma0]=cm^2, [n]=cm^-3 => [lambda]=cm, OK)")

# 3.2 reproduce Choudhury (2022)'s stated "~kpc" mean free path at the Lyman
#     limit for z~5-20
print("\n[cross-check vs Choudhury 2022, 'mean free path ~kpc at z~5-20']")
for z in [5, 8, 12, 20]:
    print(f"  z={z:4.1f}: lambda_mfp(13.6 eV) = {mfp_neutral_cm(E_H,z)/kpc:.3f} kpc")

# 3.3 crossover energy E_cross(z) vs R_bubble(z)
print("\n[crossover energy: lambda_mfp(E_cross,z) = R_bubble(z)]")
for z in [6, 8, 10, 12]:
    Rb, Ec = R_bubble_cm(z)/Mpc, E_cross(z)
    print(f"  z={z:4.1f}: R_bubble={Rb:.4f} pMpc ; E_cross={Ec:.1f} eV ({Ec/E_H:.2f} x Lyman limit)"
          f"  -- soft X-ray/EUV regime")

# 3.4 fraction of photons above E_cross for a QUASAR-like power-law spectrum
#     (Cen & Haiman 2000: L_nu ~ nu^-1.8 => photon number dN/dE ~ E^-2.8)
p_photon = 2.8
z_ref = 8.0
Ec8 = E_cross(z_ref)
frac_agn = (Ec8/E_H)**(1-p_photon)
print(f"\n[AGN-like power-law spectrum, Cen & Haiman 2000 slope L_nu~nu^-1.8 => dN/dE~E^-{p_photon}]")
print(f"  fraction of ionizing photons (by number) with E>E_cross={Ec8:.1f} eV at z={z_ref:.0f}:"
      f"  (E_cross/E_H)^(1-p) = {frac_agn:.4f}  ~ {frac_agn*100:.2f}%")
print("  i.e. a SMALL BUT NON-NEGLIGIBLE fraction of quasar-like ionizing photons can reach")
print("  beyond the local bubble -- this is why AGN/quasar sources are treated as a distinct")
print("  regime in the reionization literature (Choudhury 2022, footnote on X-ray sources).")

print(f"\n[stellar ionizing spectrum] O/B-star photospheres are thermal with T_eff~3-5x10^4 K")
print(f"  (kT~3-4 eV); suppressing the emitted photon flux at E_cross~{Ec8:.0f} eV requires reaching")
print(f"  ~{Ec8/4:.0f}-{Ec8/3:.0f} kT into the Wien tail, i.e. a suppression factor ~exp(-{Ec8/4:.0f} to -{Ec8/3:.0f})"
      f" -- utterly negligible compared to the already-small {frac_agn*100:.1f}% AGN value.")
print("  This is standard blackbody/stellar-atmosphere physics (Wien tail), not a fitted number;")
print("  a precise value would require full stellar population-synthesis spectra (e.g. BPASS,")
print("  Starburst99), which are not among the references downloaded for this project (flagged,")
print("  Master Rule 8) -- the qualitative conclusion (negligible) is robust regardless.")

print("="*78)

# =====================================================================
# 4. Figure
# =====================================================================
E_plot = np.logspace(np.log10(13.6), np.log10(5000), 400)
z_ref = 8.0
mfp_plot_Mpc = mfp_neutral_cm(E_plot, z_ref)/Mpc
Rb_Mpc = R_bubble_cm(z_ref)/Mpc
Ec = E_cross(z_ref)

fig, ax = plt.subplots(figsize=(7.6, 5.8))

ax.axvspan(13.6, 54.4, color='#f4b400', alpha=0.18, label='typical stellar ionizing photons (H I–He I)')
ax.plot(E_plot, mfp_plot_Mpc, color='#1c4b8f', lw=2.3,
        label=r'$\lambda_{\rm mfp}(E,z{=}8)$ in neutral gas')
ax.axhline(Rb_Mpc, color='#c1272d', lw=2.0, ls='--',
           label=rf'$R_{{\rm bubble}}(z{{=}}8)={Rb_Mpc:.2f}\,$pMpc')
ax.axvline(Ec, color='gray', ls=':', lw=1.3)
ax.plot([Ec], [Rb_Mpc], marker='o', ms=7, color='k', zorder=5)
ax.annotate(rf'$E_{{\rm cross}}={Ec:.0f}$ eV' '\n(soft X-ray)', xy=(Ec, Rb_Mpc),
            xytext=(Ec*2.2, Rb_Mpc*0.15), fontsize=9.5,
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('photon energy $E$ (eV)')
ax.set_ylabel(r'$\lambda_{\rm mfp}$ in neutral gas / $R_{\rm bubble}$ (proper Mpc)')
ax.set_title('Only photons harder than ${\\sim}160$–$200$ eV can outrun\n'
             'their own ionization front')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.25, which='both')

fig.tight_layout(rect=[0.01, 0.05, 0.99, 1])
fig.text(0.5, 0.012,
         r'$\lambda_{\rm mfp}(E,z)=[\sigma_H(E_H)(E/E_H)^{-3}n_H(z)]^{-1}$ (Choudhury 2022); '
         r'$R_{\rm bubble}(z)$: same model as bubble_radius_vs_redshift.py',
         ha='center', fontsize=8, color='dimgray')

outpath = '/home/byaku/Desktop/Doctorado-Trabajo/Paper_1/Stromgren_sphere/ionizing_photon_mean_free_path.png'
fig.savefig(outpath, dpi=200)
print(f"Figure saved to: {outpath}")

# =====================================================================
# 5. Audit trail
# =====================================================================
print("="*78)
print("PARAMETERS USED (audit trail)")
print("="*78)
print(f"sigma_H0={sigma_H0} cm^2 at E_H={E_H} eV [Choudhury 2022 eq.46]")
print(f"SFR_fid={SFR_fid} Msun/yr ; f_esc={f_esc} ; z_form={z_form}  [same as bubble_radius_vs_redshift.py]")
print(f"Cosmology: h={h}, Omega_m={Om}, Omega_b h^2={Ob_h2}, Y_p={Yp}  [Planck 2018 VI]")
print(f"E_cross(z=8) = {Ec:.2f} eV ; AGN-spectrum fraction above E_cross = {frac_agn*100:.2f}%")
