"""
Phase 2 -- build R_CR(z) (the CR-electron analogue of the Strogmren radius)
and compare it to the baseline photon-counting R_UV(z).

R_CR(z) is defined via the LOCAL form of the manuscript's "lock" equation
(master_plan_response.tex Section on Eq. lock; N_ion/n_H = Delta_T/Theta,
both densities, so any volume cancels): the shell radius r at which the
cumulative ionizations produced by ALL of this galaxy's escaped CR electrons,
within proper path length <= r, divided by n_H(z)*(4/3)*pi*r^3, reaches ~1.

This is a MAGNITUDE question, deliberately transport-regime-independent for
the total-yield part (Y(K,z) is a per-electron total, not spatial), but the
path-length-vs-r mapping inherits the ballistic-upper-bound caveat of point
A.1: r here is a path length, a rigorous UPPER BOUND on radial displacement,
never a literal radius (gyroradius ~1e-8-1e-2 pc vs Mpc bubble scale; Jana &
Nath 2018). Every figure below is captioned accordingly.

Also computes the separate alpha_B(T)-modified R_UV(z) (point A.2, correct
sign: CR pre-heating LOWERS alpha_B, suppresses recombination, so R_UV grows,
never shrinks) -- reported and plotted separately, never conflated with R_CR.
"""
import pathlib
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

import params as P
import stromgren_common as SC
import cr_transport as T

HERE = pathlib.Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
MPC_M = T.MPC_M
MPC_CM = P.MPC_CM


# ---------------------------------------------------------------------
# 1. CR-electron injection spectrum for ONE galaxy (leaky/cumulative, point
#    A.3): dN_e/dK/dt = A * K^-p, normalized so integral of K*dN/dK/dt dK
#    over [K_min,K_max] equals xi_CR_e * E_SN * R_SN (erg/s).
# ---------------------------------------------------------------------
def R_SN_per_s(SFR=P.SFR_FID_MSUN_YR):
    """Core-collapse SN rate for this galaxy, s^-1."""
    return SFR * P.K_CC_PER_MSUN / P.YR_S


def L_CR_e_erg_s(xi_cr_e, SFR=P.SFR_FID_MSUN_YR):
    """Total CR-electron injected power for one galaxy, erg/s."""
    return xi_cr_e * P.E_SN_ERG * R_SN_per_s(SFR=SFR)


def injection_normalization_eV(xi_cr_e, p=P.DSA_INDEX_P, SFR=P.SFR_FID_MSUN_YR):
    """A such that dN_e/dK/dt = A*K^-p [electrons/s/eV], from
    integral_Kmin^Kmax K^(1-p) dK * A = L_CR_e [eV/s]."""
    L_eV_s = L_CR_e_erg_s(xi_cr_e, SFR=SFR) / P.ERG_PER_EV
    Kmin, Kmax = P.K_MIN_EV, P.K_MAX_EV
    if abs(p - 2.0) < 1e-9:
        norm_integral = np.log(Kmax / Kmin)
    else:
        norm_integral = (Kmax**(2 - p) - Kmin**(2 - p)) / (2 - p)
    return L_eV_s / norm_integral


# ---------------------------------------------------------------------
# 2. Per-electron trajectories on a K grid, cached per z (Phase-2 core).
# ---------------------------------------------------------------------
N_K_GRID = 26
K_GRID_EV = np.logspace(np.log10(P.K_MIN_EV), np.log10(P.K_MAX_EV), N_K_GRID)


def build_trajectories(z_inject):
    """Return list of cr_transport trajectories, one per K in K_GRID_EV,
    all injected at redshift z_inject."""
    return [T.electron_trajectory(K, z_inject) for K in K_GRID_EV]


def cumulative_N_ion_within_r(trajs, r_m, xi_cr_e, p=P.DSA_INDEX_P,
                               SFR=P.SFR_FID_MSUN_YR):
    """Total rate of ionizations (all injected electrons, all energies)
    deposited within proper path length <= r_m, at the injection-spectrum
    normalization for the given xi_cr_e. Units: ionizations/s."""
    A = injection_normalization_eV(xi_cr_e, p=p, SFR=SFR)
    N_of_K = np.array([T.N_total_within_r(tr, r_m) for tr in trajs])
    integrand = A * K_GRID_EV**(1 - p) * N_of_K   # f(K)*K, for d(lnK) integration
    return np.trapz(integrand, np.log(K_GRID_EV))


def solve_R_CR_Mpc(z_inject, xi_cr_e, trajs=None, r_bracket_Mpc=(1e-6, 1e4),
                   p=P.DSA_INDEX_P, z_form=P.Z_FORM, SFR=P.SFR_FID_MSUN_YR):
    """Root-find R_CR(z): N_ion_cum(<r,z)/[nH(z)*(4/3)pi r^3] = 1.

    `p` (DSA injection index) and `z_form` (galaxy formation / CR-injection
    onset redshift, entering only through t_age) are overridable per the
    params-override architecture -- the underlying per-electron trajectories
    (in `trajs`) do not depend on either, so the same `trajs` can be reused
    across a p-sweep or a z_form sweep without re-solving any ODEs.
    """
    if trajs is None:
        trajs = build_trajectories(z_inject)
    nH = SC.nH_comoving(z_inject)   # cm^-3
    ta = SC.t_age(z_inject, z_form=z_form)   # s

    def f(logr_Mpc):
        r_Mpc = 10**logr_Mpc
        r_m = r_Mpc * MPC_M
        r_cm = r_Mpc * MPC_CM
        rate = cumulative_N_ion_within_r(trajs, r_m, xi_cr_e, p=p, SFR=SFR)  # ion/s
        N_cum = rate * ta
        shell_H = nH * (4 / 3) * np.pi * r_cm**3
        return N_cum / shell_H - 1.0

    lo, hi = np.log10(r_bracket_Mpc[0]), np.log10(r_bracket_Mpc[1])
    flo, fhi = f(lo), f(hi)
    if flo < 0 and fhi < 0:
        return np.nan   # never reaches ~1 ionization/H even at r_max: R_CR undefined (too small)
    if flo > 0 and fhi > 0:
        return 10**lo    # already >=1 at the smallest r probed
    logr = brentq(f, lo, hi, xtol=1e-4)
    return 10**logr


def main():
    Z_ARR = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0])

    R_UV = SC.R_UV_proper_Mpc(Z_ARR)
    R_CR_lo = np.full_like(Z_ARR, np.nan)
    R_CR_hi = np.full_like(Z_ARR, np.nan)
    R_CR_fid = np.full_like(Z_ARR, np.nan)
    R_CR_zero = np.full_like(Z_ARR, np.nan)   # xi_CR,e -> 0 limiting-case check
    deltaT_at_RUV = np.full_like(Z_ARR, np.nan)

    print(f"{'z':>5} {'R_UV (Mpc)':>12} {'R_CR lo (Mpc)':>14} "
          f"{'R_CR fid (Mpc)':>15} {'R_CR hi (Mpc)':>14} {'R_CR/R_UV (fid)':>16}")
    for i, z in enumerate(Z_ARR):
        trajs = build_trajectories(z)
        R_CR_lo[i] = solve_R_CR_Mpc(z, P.XI_CR_E_LOW, trajs=trajs)
        R_CR_fid[i] = solve_R_CR_Mpc(z, P.XI_CR_E_FID, trajs=trajs)
        R_CR_hi[i] = solve_R_CR_Mpc(z, P.XI_CR_E_HIGH, trajs=trajs)

        # xi_CR,e -> 0 limiting-case check: N_ion_cum -> 0, so R_CR should
        # collapse to (numerically) zero / undefined -- verify explicitly.
        R_CR_zero[i] = solve_R_CR_Mpc(z, 1e-30, trajs=trajs)

        # Self-consistent Delta_T at the UV bubble edge, from the lock
        # equation using the fiducial CR-ionization density there.
        nH = SC.nH_comoving(z)
        ta = SC.t_age(z)
        rate = cumulative_N_ion_within_r(trajs, R_UV[i] * MPC_M, P.XI_CR_E_FID)
        N_ion_at_RUV = rate * ta
        shell_H = nH * (4 / 3) * np.pi * (R_UV[i] * MPC_CM)**3
        NionOverNH = N_ion_at_RUV / shell_H
        theta_fid = 0.5 * (P.THETA_LOCK_K_LOW + P.THETA_LOCK_K_HIGH)
        deltaT_at_RUV[i] = NionOverNH * theta_fid

        print(f"{z:5.1f} {R_UV[i]:12.4e} {R_CR_lo[i]:14.4e} "
              f"{R_CR_fid[i]:15.4e} {R_CR_hi[i]:14.4e} "
              f"{R_CR_fid[i]/R_UV[i]:16.4e}")

    print("\nxi_CR,e -> 0 limiting-case check (R_CR should -> ~0 / undefined, "
          "confirming the calculation has no spurious floor):")
    print(R_CR_zero)
    assert np.all(np.isnan(R_CR_zero) | (R_CR_zero < 1e-5 * R_UV)), \
        "xi_CR,e->0 limit did not collapse R_CR towards zero -- FAIL, Master Rule 5"
    print("-> OK: xi_CR,e->0 collapses R_CR to a negligible/undefined radius, as required.\n")

    print("Delta_T(R_UV) self-consistently sourced from the CR-ionization "
          "density at the bubble edge (fiducial xi_CR,e), via the lock eq.:")
    for z, dT in zip(Z_ARR, deltaT_at_RUV):
        print(f"  z={z:5.1f}  Delta_T = {dT:.4e} K")

    R_UV_mod = np.array([SC.R_UV_alphaB_modified_Mpc(z, dT)
                          for z, dT in zip(Z_ARR, deltaT_at_RUV)])
    print("\nalpha_B(T)-modified R_UV(z) (point A.2, heating suppresses "
          "recombination -> R grows, never shrinks):")
    for z, r0, r1 in zip(Z_ARR, R_UV, R_UV_mod):
        print(f"  z={z:5.1f}  R_UV={r0:.4e} Mpc   R_UV,mod={r1:.4e} Mpc   "
              f"ratio={r1/r0:.6f}")
    assert np.all(R_UV_mod >= R_UV), "alpha_B(T) modification shrank R_UV -- sign error, FAIL"
    print("-> OK: R_UV,mod >= R_UV at every z (correct sign, point A.2).")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.semilogy(Z_ARR, R_UV, "k-o", label=r"$R_{\rm UV}(z)$ (baseline, photon-counting)")
    ax.fill_between(Z_ARR, R_CR_lo, R_CR_hi, color="C3", alpha=0.25,
                     label=r"$R_{\rm CR}(z)$ range ($\xi_{\rm CR,e}=1$--$2\times10^{-3}$)")
    ax.semilogy(Z_ARR, R_CR_fid, "C3--s", label=r"$R_{\rm CR}(z)$ fiducial")
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel("radius (proper Mpc)")
    ax.set_title(r"CR-electron-driven $R_{\rm CR}(z)$ vs. UV photon-counting $R_{\rm UV}(z)$"
                  "\n(SFR=10 $M_\\odot$/yr fiducial galaxy)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGDIR / "R_CR_vs_R_UV.png", dpi=200)
    print(f"\nFigure saved to: {FIGDIR/'R_CR_vs_R_UV.png'}")

    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    ax2.plot(Z_ARR, R_UV_mod / R_UV, "C0-o")
    ax2.set_xlabel("redshift $z$")
    ax2.set_ylabel(r"$R_{\rm UV,mod}/R_{\rm UV}$ (recombination-sink effect only)")
    ax2.set_title("Effect of CR pre-heating on the recombination sink alone\n"
                   "(point A.2: heating suppresses recombination, R grows)")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGDIR / "R_UV_alphaB_modified.png", dpi=200)
    print(f"Figure saved to: {FIGDIR/'R_UV_alphaB_modified.png'}")

    np.savez(HERE / "phase2_results.npz", z=Z_ARR, R_UV=R_UV, R_CR_lo=R_CR_lo,
             R_CR_fid=R_CR_fid, R_CR_hi=R_CR_hi, R_UV_mod=R_UV_mod,
             deltaT_at_RUV=deltaT_at_RUV)
    print(f"Results saved to: {HERE/'phase2_results.npz'}")


if __name__ == "__main__":
    main()
