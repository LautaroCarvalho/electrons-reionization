"""
Shared cosmology / photon-counting Stromgren-radius functions for this
project, reimplemented cleanly (no import-time side effects, no plotting)
from imported/bubble_radius_vs_redshift.py and imported/reionization_completion.py
so they can be safely imported by multiple downstream scripts. Every constant
comes from params.py.

Verified in __main__ against a captured run of the original
imported/bubble_radius_vs_redshift.py (see verify_against_original()) --
agreement to < 1e-6 relative at all test redshifts (Master Rule 5).
"""
import numpy as np
from scipy import integrate, optimize

import params as P

MPC_CM = P.MPC_CM


def Hz(z):
    return P.H0_S * np.sqrt(P.OMEGA_M * (1 + z)**3 + P.OMEGA_L)


def nH_comoving(z):
    """Physical mean H number density at redshift z, cm^-3 (grows as (1+z)^3;
    same convention as bubble_radius_vs_redshift.py's nH(z))."""
    return P.NH0_CM3 * (1 + z)**3


def _cosmic_age_scalar(z):
    integrand = lambda zp: 1.0 / ((1 + zp) * Hz(zp))
    val, _ = integrate.quad(integrand, z, np.inf, limit=200)
    return val


cosmic_age = np.vectorize(_cosmic_age_scalar)


def t_age(z, z_form=P.Z_FORM):
    """Time elapsed since the galaxy formed at z_form, seconds."""
    return cosmic_age(z) - cosmic_age(z_form)


def xi_ion(z):
    """Llerena et al. 2024 (arXiv:2412.01358) fit. photons/s per erg/s/Hz."""
    return 10**(P.XI_ION_LOG10_SLOPE * z + P.XI_ION_LOG10_INTERCEPT)


def Ndot_ion(z, f_esc=P.F_ESC, SFR=P.SFR_FID_MSUN_YR):
    L_UV = SFR / P.KAPPA_FUV
    return f_esc * xi_ion(z) * L_UV  # photons/s


# --- x_HI(z): tanh model calibrated on Umeda et al. 2023 (identical anchors
#     to bubble_radius_vs_redshift.py) ---
_UMEDA_Z = np.array([7.12, 9.91])
_UMEDA_XHI = np.array([0.53, 0.92])


def _y_of_z(z):
    return (1 + z)**1.5


def _xHII_tanh(z, z_re, dz):
    dy = 1.5 * np.sqrt(1 + z_re) * dz
    return 0.5 * (1 + np.tanh((_y_of_z(z_re) - _y_of_z(z)) / dy))


def _anchor_eqs(p):
    z_re, dz = p
    return [_xHII_tanh(_UMEDA_Z[0], z_re, dz) - (1 - _UMEDA_XHI[0]),
            _xHII_tanh(_UMEDA_Z[1], z_re, dz) - (1 - _UMEDA_XHI[1])]


_Z_RE_FIT, _DZ_FIT = optimize.fsolve(_anchor_eqs, [9.0, 1.0])


def x_HI(z):
    return 1 - _xHII_tanh(z, _Z_RE_FIT, _DZ_FIT)


def R_UV_proper_Mpc(z, f_esc=P.F_ESC, SFR=P.SFR_FID_MSUN_YR, z_form=P.Z_FORM):
    """Baseline photon-counting Stromgren radius, proper Mpc (eq. photon_counting)."""
    ta = t_age(z, z_form=z_form)
    ta = np.where(ta > 0, ta, np.nan)
    R3 = 3 * Ndot_ion(z, f_esc=f_esc, SFR=SFR) * ta / (4 * np.pi * nH_comoving(z) * x_HI(z))
    return (R3)**(1 / 3) / MPC_CM


def R_UV_alphaB_modified_Mpc(z, delta_T_K, f_esc=P.F_ESC, SFR=P.SFR_FID_MSUN_YR,
                              z_form=P.Z_FORM, T0_K=P.T_HII_FID_K):
    """R_UV(z) with the recombination-sink term modified by CR pre-heating of
    the bubble-interior gas, T0_K -> T0_K + delta_T_K (point A.2: heating
    LOWERS alpha_B, so R grows -- opposite sign from a naive 'CR heating
    hurts the bubble' guess). Implemented via the equilibrium-with-recombination
    solution R^3 (1 - exp(-t/t_rec)) analogue is beyond the pure photon-counting
    limit used elsewhere in this project; here we apply the sink-suppression
    multiplicatively through the alpha_B(T) ratio, i.e. R_mod^3/R_base^3 =
    alpha_B(T0)/alpha_B(T0+deltaT) approximately, valid to the extent that the
    recombination term acts as a simple multiplicative sink on the ionized
    volume (Case-B, optically thin, single-zone approximation stated
    explicitly here as an assumption)."""
    ratio = P.alpha_B(T0_K) / P.alpha_B(T0_K + delta_T_K)
    return R_UV_proper_Mpc(z, f_esc=f_esc, SFR=SFR, z_form=z_form) * ratio**(1 / 3)


def verify_against_original():
    """Cross-check against a captured run of imported/bubble_radius_vs_redshift.py
    (Master Rule 5). Reference values captured 2026-09-06 via a subprocess run
    of the original module at SFR=10, f_esc=0.2, z_form=20."""
    ref = {
        6: (1.3316858215051477, 6.513172574435012e-05, 0.32929955962443747),
        7: (0.9707944642725886, 9.722286758340311e-05, 0.5070333799304654),
        8: (0.7521077985017156, 0.00013842865325840013, 0.6932622203494421),
        9: (0.6167547476628635, 0.0001898884132488342, 0.8385856274578705),
        10: (0.5277238455378894, 0.00025274147803419834, 0.9257246626100244),
        12: (0.41357419207301865, 0.0004171848439076887, 0.9877920043661645),
        15: (0.30054822219046573, 0.0007777829406672249, 0.9994344986728332),
        18: (0.1990598017582474, 0.0013024446264737538, 0.9999808959463008),
    }
    print("Cross-check: stromgren_common vs imported/bubble_radius_vs_redshift.py")
    print(f"{'z':>4}  {'R rel.err':>12}  {'nH rel.err':>12}  {'xHI rel.err':>12}")
    worst = 0.0
    for z, (R_ref, nH_ref, xHI_ref) in ref.items():
        R = R_UV_proper_Mpc(z)
        nH_v = nH_comoving(z)
        xHI_v = x_HI(z)
        eR = abs(R - R_ref) / R_ref
        enH = abs(nH_v - nH_ref) / nH_ref
        exHI = abs(xHI_v - xHI_ref) / xHI_ref
        worst = max(worst, eR, enH, exHI)
        print(f"{z:4d}  {eR:12.3e}  {enH:12.3e}  {exHI:12.3e}")
    status = "OK (< 1e-6)" if worst < 1e-6 else f"FAIL (worst={worst:.2e})"
    print(f"-> {status}")
    return worst


if __name__ == "__main__":
    worst = verify_against_original()
    assert worst < 1e-6, "stromgren_common disagrees with the original module -- see Master Rule 5"
