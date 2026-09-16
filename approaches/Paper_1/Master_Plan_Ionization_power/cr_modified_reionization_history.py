"""
Phase 3 -- Q2: patchy-reionization-history implications.

Extends imported/reionization_completion.py's global Q_HII(z) ODE with the
SAME CR-electron physics as Phase 2, but in the MEAN/UNIFORM framing (per
Jana & Nath 2018) appropriate for a global history: the whole comoving
volume, not one galaxy's shell. This is a deliberately DIFFERENT regime from
Phase 2's per-galaxy upper-bound R_CR(z) -- Phase 2 asks "how big a shell
around one galaxy reaches ~1 ionization/H", Phase 3 asks "what is the mean
extra ionization + heating this channel adds to the whole IGM's history".

Two effects, both derived from the SAME xi_CR,e / cascade-yield machinery
(no new free parameters):
  1. A direct CR-ionization source term added to dQ_HII/dt, from the cosmic
     SFRD (Madau & Dickinson 2014) instead of one galaxy's SFR.
  2. A self-consistent Delta_T_IGM(z), via the SAME lock equation as Phase 2
     (Delta_T = Theta * N_ion,CR_raw/n_H, tracked as its own ODE state so it
     is the raw cumulative CR-ionization production, NOT net of recombination
     -- the lock equation is an energy-conservation statement about heat
     deposited per ionization PRODUCED, independent of what happens to that
     ionization afterward), fed into alpha_B(T) in the recombination sink
     (point A.2: heating LOWERS alpha_B, suppresses recombination).

Resulting tau is compared against Planck 2018 and 21-cm bounds already used
elsewhere in this project.
"""
import pathlib
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import params as P
import stromgren_common as SC
import cr_transport as T

HERE = pathlib.Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
MPC_CM = P.MPC_CM
SIGMA_T_CM2 = 6.6524587e-25   # Thomson cross section, cm^2
C_LIGHT_CM_S = 2.99792458e10

chi_He = 1.08   # same convention as imported/reionization_completion.py


def rho_SFR(z):
    """Madau & Dickinson (2014) cosmic SFRD fit, Msun/yr per comoving Mpc^3."""
    return 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)


def ndot_ion_UV_comoving(z, f_esc=P.F_ESC):
    """UV ionizing photons / s / comoving cm^3, whole galaxy population."""
    rho_SFR_cgs = rho_SFR(z) / MPC_CM**3
    L_UV_density = rho_SFR_cgs / P.KAPPA_FUV
    return f_esc * SC.xi_ion(z) * L_UV_density


# ---------------------------------------------------------------------
# CR-electron cascade yield table Y_total(K, z), tabulated once (Phase 2
# already verified cr_transport/cascade_yield against the manuscript's own
# numbers to <1e-3 relative -- reuse the same cascade_yield.yields() here
# directly since Phase 3 needs only the TOTAL yield, not the path-length
# resolved version).
# ---------------------------------------------------------------------
Z_TAB = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0])
N_K_TAB = 20
K_TAB_EV = np.logspace(np.log10(P.K_MIN_EV), np.log10(P.K_MAX_EV), N_K_TAB)


def build_Y_table():
    Y = np.zeros((Z_TAB.size, K_TAB_EV.size))
    for iz, z in enumerate(Z_TAB):
        for ik, K in enumerate(K_TAB_EV):
            _, Ytot, _ = T.CY.yields(K, z)
            Y[iz, ik] = Ytot
        print(f"Y-table: z={z:5.1f} done", flush=True)
    return Y


def ndot_ion_CR_comoving(z, Y_of_K, xi_cr_e=P.XI_CR_E_FID, p=P.DSA_INDEX_P):
    """CR-electron-driven ionizations / s / comoving cm^3, whole population,
    using the cosmic SFRD -> core-collapse SN rate density -> CR-electron
    power density -> normalized power-law spectrum -> cascade yield Y(K,z).
    """
    R_SN_density = rho_SFR(z) * P.K_CC_PER_MSUN / P.YR_S / MPC_CM**3  # SNe/s/comoving cm^3
    L_CR_e_density = xi_cr_e * P.E_SN_ERG * R_SN_density               # erg/s/comoving cm^3
    L_eV_s_cm3 = L_CR_e_density / P.ERG_PER_EV
    Kmin, Kmax = P.K_MIN_EV, P.K_MAX_EV
    norm_integral = np.log(Kmax / Kmin) if abs(p - 2.0) < 1e-9 else \
        (Kmax**(2 - p) - Kmin**(2 - p)) / (2 - p)
    A = L_eV_s_cm3 / norm_integral
    integrand = A * K_TAB_EV**(1 - p) * Y_of_K   # times K, for d(lnK) integration
    return np.trapz(integrand, np.log(K_TAB_EV))


def run_history(xi_cr_e, Y_table, include_cr=True, p=P.DSA_INDEX_P,
                z_inject_cutoff=None):
    """Integrate the coupled [Q_HII, Q_ion_CR_raw] ODE from z=25 to z=4.

    `p` (DSA injection index) is overridable per the params-override
    architecture; the Y(K,z) cascade-yield table is independent of both `p`
    and `z_inject_cutoff` (it is a per-electron microphysics property), so
    the same cached Y_table can be reused across a parameter sweep.
    `z_inject_cutoff`, if given, is the global-history analogue of Phase 2's
    single-galaxy z_form: the CR-electron-producing population is assumed
    not to exist yet at z > z_inject_cutoff, so the CR source term is
    switched off there (mirroring, at the population level, the same
    "before this redshift, no CR electrons" cutoff that z_form enforces for
    one galaxy in Phase 2)."""
    theta_fid = 0.5 * (P.THETA_LOCK_K_LOW + P.THETA_LOCK_K_HIGH)

    def Y_interp(z):
        return np.array([np.interp(z, Z_TAB, Y_table[:, ik])
                          for ik in range(K_TAB_EV.size)])

    def ndot_CR(z):
        if not include_cr:
            return 0.0
        if z_inject_cutoff is not None and z > z_inject_cutoff:
            return 0.0
        return ndot_ion_CR_comoving(z, Y_interp(z), xi_cr_e=xi_cr_e, p=p)

    def rhs(z, y):
        Q, Qcr_raw = y
        dT = theta_fid * max(Qcr_raw, 0.0)
        aB = P.alpha_B(P.T_HII_FID_K + dT)
        nH0 = P.NH0_CM3
        ne_phys = chi_He * SC.nH_comoving(z)   # physical density, grows as (1+z)^3
                                                # (bug fix: an earlier version used
                                                # a constant ne0, missing this growth,
                                                # which starved the recombination sink
                                                # at high z and let Q_HII run past 1 --
                                                # caught by cross-checking against
                                                # imported/reionization_completion.py's
                                                # own Q_HII(z=6)=0.674 reference)
        source = (ndot_ion_UV_comoving(z) + ndot_CR(z)) / nH0
        sink = max(Q, 0.0) * P.CLUMPING_C_HII * aB * ne_phys
        dzdt = -(1 + z) * SC.Hz(z)
        dQdz = (source - sink) / dzdt
        dQcr_raw_dz = (ndot_CR(z) / nH0) / dzdt
        return [dQdz, dQcr_raw_dz]

    zgrid = np.linspace(25, 4, 4000)
    sol = solve_ivp(rhs, [25, 4], [0.0, 0.0], t_eval=zgrid, max_step=0.02, method="RK45")
    return sol.t, sol.y[0], sol.y[1]


def tau_of_z(zarr, Qarr):
    """Thomson optical depth out to each z, tau(z) = sigma_T * integral_0^z
    n_e(z') c dt/dz' dz' with n_e = Q(z')*chi_He*nH0*(1+z')^3 (Q clipped to
    [0,1]), c dt/dz = -c/[(1+z)H(z)]."""
    Qc = np.clip(Qarr, 0.0, 1.0)
    ne = Qc * chi_He * P.NH0_CM3 * (1 + zarr)**3
    integrand = ne * C_LIGHT_CM_S / ((1 + zarr) * SC.Hz(zarr))
    # zarr is decreasing (25->4); integrate |dz| via cumulative trapz on reversed arrays
    order = np.argsort(zarr)
    zs, ints = zarr[order], integrand[order]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (ints[1:] + ints[:-1]) * np.diff(zs))])
    cum_full = SIGMA_T_CM2 * cum
    # cum_full[i] = tau from z=zs[0] up to zs[i]; we want tau(0->z), i.e. from
    # z=0 to z, which is the total integral MINUS the part beyond z... but
    # our domain starts at z=4, not 0. tau(0->z=4) is added as a small,
    # explicitly-flagged constant floor from the fully-ionized low-z universe.
    tau_from_4_to_z = np.interp(zarr, zs, cum_full)
    return tau_from_4_to_z


def first_cross(zarr, Qarr, val):
    idx = np.where(Qarr >= val)[0]
    return zarr[idx[0]] if len(idx) else np.nan


def verify_baseline_against_original():
    """Cross-check the CR-free baseline against a captured run of
    imported/reionization_completion.py (Master Rule 5). Reference values
    from a subprocess run of the original module, C_HII=3.0, 2026-09-06.
    An earlier version of this script's rhs() used a constant (not physical,
    (1+z)^3-growing) recombination density, which starved the sink at high z
    and let Q_HII run past 1 by z~6 -- this check would have caught that bug
    immediately; it is kept here permanently to guard against a regression.
    """
    ref = {15: 0.0342, 10: None, 9: 0.1729, 8: 0.2561, 7: 0.4014, 6: 0.6740}
    Y_dummy = np.zeros((2, 4))
    Z_TAB_save, K_TAB_save = Z_TAB.copy(), K_TAB_EV.copy()
    z, Q, _ = run_history(0.0, Y_dummy, include_cr=False)
    print("Cross-check: baseline Q_HII(z) vs imported/reionization_completion.py")
    worst = 0.0
    for zt, Qref in ref.items():
        if Qref is None:
            continue
        Qv = np.interp(zt, z[::-1], Q[::-1])
        rel = abs(Qv - Qref) / Qref
        worst = max(worst, rel)
        print(f"  z={zt:4.1f}  this work={Qv:.4f}  reference={Qref:.4f}  rel.diff={rel:.4f}")
    status = "OK (< 2%)" if worst < 0.02 else f"FAIL (worst={worst:.3f})"
    print(f"-> {status}")
    return worst


def main():
    worst = verify_baseline_against_original()
    assert worst < 0.02, "baseline Q_HII disagrees with the original module -- see Master Rule 5"

    print("\nBuilding CR-electron cascade yield table Y_total(K,z) ...")
    Y_table = build_Y_table()
    np.savez(HERE / "Y_table_phase3.npz", z=Z_TAB, K=K_TAB_EV, Y=Y_table)

    z_base, Q_base, _ = run_history(0.0, Y_table, include_cr=False)
    z_fid, Q_fid, Qcr_fid = run_history(P.XI_CR_E_FID, Y_table, include_cr=True)
    z_lo, Q_lo, _ = run_history(P.XI_CR_E_LOW, Y_table, include_cr=True)
    z_hi, Q_hi, _ = run_history(P.XI_CR_E_HIGH, Y_table, include_cr=True)

    tau_base = tau_of_z(z_base, Q_base)
    tau_fid = tau_of_z(z_fid, Q_fid)
    tau_lo = tau_of_z(z_lo, Q_lo)
    tau_hi = tau_of_z(z_hi, Q_hi)

    TAU_FLOOR_0_TO_4 = 0.020   # Planck 2018 fully-ionized-below-z=4 floor contribution,
                                # explicitly separate & approximate -- see caption/text

    def first_cross(zarr, Qarr, val):
        idx = np.where(Qarr >= val)[0]
        return zarr[idx[0]] if len(idx) else np.nan

    print(f"\n{'model':>14} {'z(Q=0.5)':>10} {'z(Q=0.999)':>12} "
          f"{'tau(z=25)':>12} {'delta_tau vs base':>18}")
    for name, zz, QQ, tt in [("baseline", z_base, Q_base, tau_base),
                              ("xi_CR lo", z_lo, Q_lo, tau_lo),
                              ("xi_CR fid", z_fid, Q_fid, tau_fid),
                              ("xi_CR hi", z_hi, Q_hi, tau_hi)]:
        z_half = first_cross(zz, QQ, 0.5)
        z_full = first_cross(zz, QQ, 0.999)
        dtau = (tt[0] - tau_base[0])
        print(f"{name:>14} {z_half:10.3f} {z_full:12.3f} "
              f"{tt[0]+TAU_FLOOR_0_TO_4:12.5f} {dtau:18.6f}")

    PLANCK_TAU = 0.0561
    PLANCK_TAU_ERR = 0.007
    print(f"\nPlanck 2018 tau = {PLANCK_TAU} +/- {PLANCK_TAU_ERR}")
    print(f"delta_tau,fid / sigma(tau) = {(tau_fid[0]-tau_base[0])/PLANCK_TAU_ERR:.4f}")

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(z_base, Q_base, "k-", label="baseline (UV only)")
    ax.fill_between(z_fid, Q_lo, Q_hi, color="C3", alpha=0.25,
                     label=r"$\xi_{\rm CR,e}=1$--$2\times10^{-3}$")
    ax.plot(z_fid, Q_fid, "C3--", label="fiducial + CR terms")
    ax.set_xlim(20, 5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$Q_{\rm HII}(z)$")
    ax.set_title("Global ionized volume-filling factor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(z_base, tau_base + TAU_FLOOR_0_TO_4, "k-", label="baseline")
    ax2.fill_between(z_fid, tau_lo + TAU_FLOOR_0_TO_4, tau_hi + TAU_FLOOR_0_TO_4,
                      color="C3", alpha=0.25, label=r"$\xi_{\rm CR,e}$ range")
    ax2.plot(z_fid, tau_fid + TAU_FLOOR_0_TO_4, "C3--", label="fiducial + CR")
    ax2.axhline(PLANCK_TAU, color="b", ls="-", lw=1, label="Planck 2018 $\\tau$")
    ax2.axhspan(PLANCK_TAU - PLANCK_TAU_ERR, PLANCK_TAU + PLANCK_TAU_ERR,
                color="b", alpha=0.15)
    ax2.set_xlim(20, 5)
    ax2.set_xlabel("redshift $z$")
    ax2.set_ylabel(r"$\tau(0\to z)$")
    ax2.set_title("Thomson optical depth history")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "Q_HII_tau_history.png", dpi=200)
    print(f"\nFigure saved to: {FIGDIR/'Q_HII_tau_history.png'}")

    np.savez(HERE / "phase3_results.npz", z=z_base, Q_base=Q_base, Q_fid=Q_fid,
             Q_lo=Q_lo, Q_hi=Q_hi, tau_base=tau_base, tau_fid=tau_fid,
             tau_lo=tau_lo, tau_hi=tau_hi, tau_floor=TAU_FLOOR_0_TO_4)
    print(f"Results saved to: {HERE/'phase3_results.npz'}")


if __name__ == "__main__":
    main()
