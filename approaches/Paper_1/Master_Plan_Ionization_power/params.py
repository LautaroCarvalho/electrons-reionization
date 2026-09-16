"""
Central parameter module for Master_Plan_Ionization_power.

Every physical/numerical input used anywhere in this project is defined ONCE,
here, as a named module-level value with its cited source. Every other script
does `import params as P` and reads values from here -- no physical constant
is hardcoded a second time downstream. This lets a future run with different
parameter choices be a one-line edit (or a call to `override()` from a driver
script) rather than a multi-file search-and-replace.

See Master_Plan_Ionization_power/master_plan_response.tex, Section "Parameter
table", for the human-readable version of this table with full citations.
"""
import types

# ---------------------------------------------------------------------
# Cosmology (Planck 2018, arXiv:1807.06209) -- identical to Stromgren_sphere/
# ---------------------------------------------------------------------
H_LITTLE = 0.674
OMEGA_M = 0.315
OMEGA_L = 1 - OMEGA_M
OMEGA_B_H2 = 0.0224
Y_HE = 0.245                       # primordial He mass fraction

# ---------------------------------------------------------------------
# Physical constants (cgs)
# ---------------------------------------------------------------------
M_H_G = 1.6726e-24                 # g, hydrogen atom mass
PC_CM = 3.0857e18
MPC_CM = 3.0857e24
YR_S = 3.156e7
GYR_S = YR_S * 1e9
M_SUN_G = 1.989e33
ERG_PER_EV = 1.602176634e-12

# ---------------------------------------------------------------------
# Fiducial star-forming galaxy (Stromgren_sphere/bubble_radius_vs_redshift.py)
# ---------------------------------------------------------------------
SFR_FID_MSUN_YR = 10.0             # M_sun/yr fiducial SFR
F_ESC = 0.2                        # Robertson et al. 2015, arXiv:1502.02024
KAPPA_FUV = 1.15e-28                # Msun/yr per (erg/s/Hz), Madau & Dickinson 2014
XI_ION_LOG10_SLOPE = 0.06           # Llerena et al. 2024 (arXiv:2412.01358)
XI_ION_LOG10_INTERCEPT = 24.82
Z_FORM = 20.0                       # fiducial galaxy formation redshift
Z_INJECT = Z_FORM                   # CR-electron injection snapshot redshift = z_form

# ---------------------------------------------------------------------
# CR-electron injection spectrum (this planning discussion; see Section A.3
# of master_plan_response.tex for the full derivation and citations)
# ---------------------------------------------------------------------
DSA_INDEX_P = 2.0                   # canonical DSA index, Park et al. 2015, PRL 114
K_MIN_EV = 1.0e2                    # eV, low-energy edge (user-adopted: unbroken
                                     # power law extrapolated to here, no injection
                                     # cutoff modeled -- see caveat in Section A.3)
K_MAX_EV = 1.0e12                   # eV (1 TeV)

EPS_CR_LOW = 0.10                   # total CR (hadronic+leptonic) efficiency,
EPS_CR_HIGH = 0.20                  # fraction of E_SN -- Blasi 2013, A&ARv 21, 70,
                                     # arXiv:1206.2363
K_EP_RATIO = 1.0e-2                 # electron/proton NUMBER ratio at injection,
                                     # Park, Caprioli & Spitkovsky 2015, PRL 114
                                     # (already adopted in manuscript.tex l.185)
XI_CR_E_LOW = EPS_CR_LOW * K_EP_RATIO      # ~1e-3, fraction of E_SN into CR electrons
XI_CR_E_HIGH = EPS_CR_HIGH * K_EP_RATIO    # ~2e-3
XI_CR_E_FID = 0.5 * (XI_CR_E_LOW + XI_CR_E_HIGH)   # fiducial point value, ~1.5e-3

E_SN_ERG = 1.0e51                   # canonical core-collapse SN kinetic energy
K_CC_PER_MSUN = 0.007                # core-collapse SNe per Msun formed, Salpeter
                                      # IMF 8-50 Msun progenitors; Stockholm VIMOS
                                      # Supernova Survey, arXiv:1206.6897 and refs
                                      # therein

# ---------------------------------------------------------------------
# Recombination physics (point A.2)
# ---------------------------------------------------------------------
# Case-B recombination coefficient, Hui & Gnedin (1997), ApJ 292, 27 fit form:
#   alpha_B(T) = 2.753e-13 * lam^1.5 / (1+(lam/2.74)^0.407)^2.242   cm^3/s
#   lam = 315614 / T[K]
ALPHA_B_HG97_A = 2.753e-14
ALPHA_B_HG97_LAM0 = 315614.0
ALPHA_B_HG97_B = 2.74
ALPHA_B_HG97_C = 0.407
ALPHA_B_HG97_D = 2.242
T_HII_FID_K = 1.0e4                  # fiducial photoionized bubble-interior temperature
CLUMPING_C_HII = 3.0                 # Robertson et al. 2015 fiducial clumping factor

# ---------------------------------------------------------------------
# Cascade / heat-lock constants (manuscript.tex, already-verified numbers)
# ---------------------------------------------------------------------
HEAT_PER_IONIZATION_EV_LOW = 6.0
HEAT_PER_IONIZATION_EV_HIGH = 9.6
THETA_LOCK_K_LOW = 4.3e4              # Theta in N_ion/n_H = Delta_T / Theta
THETA_LOCK_K_HIGH = 6.9e4
PARTICLES_PER_H_ATOM = 1.081          # chi_He-corrected particle count per H atom

# ---------------------------------------------------------------------
# IGM / cascade medium (igm_losses.py defaults, neutral background)
# ---------------------------------------------------------------------
ION_FRACTION_NEUTRAL = 1.0e-4          # x_e = n_e/n_H in the "neutral" IGM used
                                        # for the CR-electron cascade (igm_losses.py
                                        # default; same convention as manuscript.tex)
Z_THERMAL_FLOOR = 5.5                  # cascade integration floor redshift
                                        # (matches cascade_yield.py z_f default)

# ---------------------------------------------------------------------
# Derived convenience values
# ---------------------------------------------------------------------
H0_S = H_LITTLE * 100 * 1e5 / MPC_CM   # s^-1
RHO_CRIT0_G_CM3 = 1.878e-29 * H_LITTLE**2
RHO_B0_G_CM3 = OMEGA_B_H2 * 1.878e-29
NH0_CM3 = RHO_B0_G_CM3 * (1 - Y_HE) / M_H_G   # present-day mean H number density


def alpha_B(T_K):
    """Case-B recombination coefficient, Hui & Gnedin (1997) fit. cm^3/s."""
    lam = ALPHA_B_HG97_LAM0 / T_K
    return (ALPHA_B_HG97_A * lam**1.5
            / (1.0 + (lam / ALPHA_B_HG97_B)**ALPHA_B_HG97_C)**ALPHA_B_HG97_D)


def xi_cr_e(eps_cr=None, k_ep=None):
    """Fraction of E_SN channeled into CR electrons. Defaults to the fiducial
    point estimate; pass eps_cr/k_ep to explore the literature range or a
    user-supplied value (see Section A.3)."""
    if eps_cr is None:
        eps_cr = 0.5 * (EPS_CR_LOW + EPS_CR_HIGH)
    if k_ep is None:
        k_ep = K_EP_RATIO
    return eps_cr * k_ep


def override(**kwargs):
    """Return a lightweight namespace = this module's values, with `kwargs`
    overridden -- lets a driver script do
        p = params.override(XI_CR_E_FID=3e-3)
    to explore the impact of one parameter without touching this file."""
    ns = types.SimpleNamespace(**{k: v for k, v in globals().items()
                                   if k.isupper()})
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


if __name__ == "__main__":
    print(f"NH0 = {NH0_CM3:.6e} cm^-3")
    print(f"XI_CR_E range = [{XI_CR_E_LOW:.3e}, {XI_CR_E_HIGH:.3e}], fiducial {XI_CR_E_FID:.3e}")
    print(f"alpha_B(1e4 K) = {alpha_B(1e4):.6e} cm^3/s")
    print(f"alpha_B(1e4+200 K) = {alpha_B(1e4+200):.6e} cm^3/s "
          f"(fractional change {(alpha_B(1e4)-alpha_B(1e4+200))/alpha_B(1e4)*100:.2f}%)")
