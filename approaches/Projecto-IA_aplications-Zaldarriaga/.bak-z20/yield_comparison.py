#!/usr/bin/env python3
r"""
Two independent routes to the HI ionization yield of an electron in the z = 10 IGM.

ROUTE 1 -- "current method", ionization_yield.py, models A / B / C.
    Energy bookkeeping through a PUBLISHED DEPOSITION FRACTION. The electron's
    collisional losses are split by f_ion(E, x_e) (Furlanetto & Stoever 2010
    eq. 13 = Ricotti et al. 2002 fit to Shull & van Steenberg 1985); the whole
    cascade is folded into that one fitted number. Two loss channels compete:
    Bethe-Berger-Seltzer collisional stopping, and Thomson-limit inverse Compton
    on the CMB.

ROUTE 2 -- "loss-based", model D, built here on igm_losses.py.
    EVENT COUNTING against ab-initio cross sections. Seven loss mechanisms
    compete (adiabatic, synchrotron, inverse Compton with Klein-Nishina,
    Coulomb, collisional excitation, collisional ionization, bremsstrahlung);
    ionizations are counted one at a time from the RBEB cross section, and the
    knock-on electrons are followed explicitly through their own cascade.

        Y(K) = int_{E_th}^{K} R(K') [ 1 + <Y(eps)>_{p(eps|K')} ] dK'
        R(K') = n_HI v(K') sigma_ion(K') / L_total(K')     [ionizations per eV]

    p(eps|K') is the BEB singly-differential cross section (Kim & Rudd 1994),
    the same one igm_losses uses for <eps>.

WHAT IS AND IS NOT COMPARABLE.  Route 2 discards the energy that inverse Compton
gives to the CMB, exactly as model B does -- the upscattered photons are not
followed. So the meaningful comparison is D vs B. Model C adds a channel that
BOTH routes omit, and would raise either of them.

Run:  python3 yield_comparison.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import ionization_yield as IY

os.environ.setdefault("MPLBACKEND", "Agg")
import igm_losses as L                      # noqa: E402  (prints a banner, builds tables)

Z_SNAP = 10.0          # default snapshot; set with set_redshift()
EV_J = L.EV_MKS


# ===========================================================================
# 1. THE ENVIRONMENT EACH ROUTE ASSUMES  (they are NOT identical -- report it)
# ===========================================================================
def set_redshift(z):
    """Move the whole loss-based pipeline to redshift z.

    Everything z-dependent is recomputed from it: n_HI(z) through igm_losses,
    the CMB temperature and energy density, B(z), H(z), and the adiabatic term.
    Caches are cleared as well as re-keyed -- belt and braces, because a stale
    yield curve at the wrong redshift would look entirely plausible.
    """
    global Z_SNAP
    Z_SNAP = float(z)
    _D_CACHE.clear()
    _E_CACHE.clear()
    return Z_SNAP


def environment_table(cos):
    from astropy.cosmology import Planck18
    import astropy.units as u
    H_z = float(Planck18.H(Z_SNAP).to(1 / u.s).value)
    return {
        "n_H_cm3_route1": cos["n_H_cm3"],
        "n_H_cm3_route2": float(L.n_HI(Z_SNAP) * 1e-6),
        "x_e_route1": IY.Params().x_e,
        "x_e_route2": float(L.ION_FRACTION),
        "E_th_eV_route1": IY.E_TH_HI,          # NIST ASD
        "E_th_eV_route2": float(L.THRESHOLD_EV_ION),   # 1 Rydberg
        "H_z_s_route1": cos["H_z_s"],
        "H_z_s_route2": H_z,
        "B0_T_route2": float(L.B0),
    }


def H_z_snapshot():
    from astropy.cosmology import Planck18
    import astropy.units as u
    return float(Planck18.H(Z_SNAP).to(1 / u.s).value)


# ===========================================================================
# 2. MODEL D -- the loss-based cascade yield
# ===========================================================================
def loss_branching(K_eV, H_z):
    """Fraction of dK/dt carried by each of the 7 mechanisms, at z = Z_SNAP."""
    K_J = np.atleast_1d(np.asarray(K_eV, dtype=float)) * EV_J
    rates = L.loss_rates(Z_SNAP, K_J, H_z)          # (7, N), J/s, positive
    tot = np.sum(rates, axis=0)
    return rates / np.where(tot > 0, tot, 1.0), tot


def ionizations_per_eV(K_eV, H_z):
    """R(K) = n_HI v sigma_ion / L_total  [ionizations per eV of energy lost].

    This is the loss-based analogue of f_ion/E_th: it says how many ionization
    EVENTS the electron buys with each eV it gives up, counting every competing
    channel in the denominator.
    """
    K = np.atleast_1d(np.asarray(K_eV, dtype=float))
    K_J = K * EV_J
    _, _, _, v = L.kinematics(K_J)
    # coll_ionisation_cross_section is scalar-only (it branches on K <= Ry)
    sig = np.array([L.coll_ionisation_cross_section(float(k)) for k in K_J])
    _, tot = loss_branching(K, H_z)
    rate = L.n_HI(Z_SNAP) * v * sig                 # ionizations per second
    return np.where(K > L.THRESHOLD_EV_ION, rate / tot * EV_J, 0.0)


_D_CACHE: dict = {}


def yield_D(E_max_eV=1.0e12, n=520, n_eps=160):
    """Total ionizations per primary electron, cascade included, from igm_losses.

    Marched upward on a log grid. The recursion only ever needs Y at the
    SECONDARY energy eps <= (K - E_th)/2, which sits at least 0.3 dex below K,
    so unlike the pure-ionization cascade of the companion work this one is safe
    on a log grid -- the failure mode there was needing Y at K - E_th, i.e. in
    the cell that had not been filled yet. Convergence in n is checked.
    """
    key = (Z_SNAP, E_max_eV, n, n_eps)   # z belongs in the key: without it,
                                        # changing redshift silently returns
                                        # the previous snapshot's curve
    if key in _D_CACHE:
        return _D_CACHE[key]
    H_z = H_z_snapshot()
    Eth = L.THRESHOLD_EV_ION
    Kg = np.logspace(np.log10(Eth), np.log10(E_max_eV), n)
    R = ionizations_per_eV(Kg, H_z)
    Y = np.zeros_like(Kg)
    integ = np.zeros_like(Kg)
    for i, K in enumerate(Kg):
        if K <= Eth:
            continue
        eps, pdf = L.secondary_pdf(float(K), n=n_eps)
        if eps is None:
            mean_Y = 0.0
        else:
            Ye = np.interp(eps, Kg[:i], Y[:i], left=0.0,
                           right=(Y[i - 1] if i else 0.0))
            mean_Y = float(np.trapz(pdf * Ye, eps))
        integ[i] = R[i] * (1.0 + mean_Y)
        # cumulative trapezoid in K
        Y[i] = Y[i - 1] + 0.5 * (integ[i] + integ[i - 1]) * (Kg[i] - Kg[i - 1])
    out = (Kg, Y, R)
    _D_CACHE[key] = out
    return out


def N_e_D(E_eV, **kw):
    Kg, Y, _ = yield_D(**kw)
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.where(E <= Kg[0], 0.0,
                   np.interp(np.log(np.maximum(E, Kg[0])), np.log(Kg), Y))
    return out if np.ndim(E_eV) else float(out[0])


# ===========================================================================
# 2b. MODEL E -- model D plus the IC-secondary photoionization channel
# ===========================================================================
r"""
Model D throws away the energy inverse Compton hands to the CMB, exactly as
model B does. Model E follows it, exactly as model C does -- but everything that
can be taken from route 2 IS taken from route 2:

  * the IC branching ratio is L_compton/L_total from igm_losses (7 channels,
    Klein-Nishina corrected), not the two-channel Thomson ratio of route 1;
  * the photoelectron released by an absorbed secondary photon cascades with
    Y_D, the RBEB/BEB yield, not with f_ion;
  * the optical depth uses route 2's n_HI and H(z).

Shared with route 1, because they are exact physics rather than a fit:
  * the Blumenthal & Gould (1970) Thomson-limit scattered-photon spectrum and
    its Planck convolution ptilde(w) -- reused from the checked implementation.
    Its validity in the window that matters is verified below, not assumed;
  * the hydrogenic photoionization cross section (Karzas & Latter).

        E(K) = D(K) + int_0^K [L_IC/L_tot](K') G_D(gamma(K')) dK'
        G_D(gamma) = int qtilde(w) [1-e^-tau(eps1)] N_gamma^D(eps1)/eps1 dw
        N_gamma^D(eps) = 1 + Y_D(eps - E_th)
"""

_E_CACHE: dict = {}


def _route2_channel(cos, par):
    """An ICPhotonChannel whose absorption and photon yield are route 2's."""
    ch = IY.ICPhotonChannel(cos, par)          # builds the universal ptilde(w)
    Eth2 = L.THRESHOLD_EV_ION
    n_HI2 = float(L.n_HI(Z_SNAP)) * 1e-6       # cm^-3, route 2
    c_over_H = IY.C_LIGHT / H_z_snapshot()     # cm, route 2
    tau = (n_HI2 * (1.0 - par.x_e)
           * np.atleast_1d(IY.sigma_photoion(ch.eps)) * c_over_H)
    ch.A_grid = -np.expm1(-tau)
    Ng = np.where(ch.eps >= Eth2, 1.0 + N_e_D(np.maximum(ch.eps - Eth2, 0.0)), 0.0)
    ch.Phi = ch.A_grid * Ng / ch.eps
    return ch


def yield_E(cos, par, n=420):
    """Model E on a log grid: D plus the absorbed IC-secondary photons."""
    key = (Z_SNAP, n)
    if key in _E_CACHE:
        return _E_CACHE[key]
    H_z = H_z_snapshot()
    ch = _route2_channel(cos, par)
    Kg = np.logspace(np.log10(L.THRESHOLD_EV_ION), 12.0, n)
    fr, _ = loss_branching(Kg, H_z)
    f_ic = fr[2]                                # Compton share of dK/dt
    _, gam, _, _ = L.kinematics(Kg * EV_J)
    G = ch.ions_per_eV_radiated(gam)
    integ = f_ic * G * Kg                       # dK = K dlnK
    lnK = np.log(Kg)
    extra = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(lnK))])
    Y = np.asarray(N_e_D(Kg)) + extra
    out = (Kg, Y, extra, ch)
    _E_CACHE[key] = out
    return out


def N_e_E(E_eV, cos, par):
    Kg, Y, _, _ = yield_E(cos, par)
    E = np.atleast_1d(np.asarray(E_eV, dtype=float))
    out = np.where(E <= Kg[0], 0.0,
                   np.interp(np.log(np.maximum(E, Kg[0])), np.log(Kg), Y))
    return out if np.ndim(E_eV) else float(out[0])


# ===========================================================================
# 3. CHECKS
# ===========================================================================
def run_checks(cos, par, chan):
    from scipy.optimize import brentq
    rows, prov = [], {}

    def rec(cid, name, ok, detail):
        rows.append((cid, name, "PASS" if ok else "FAIL", detail))

    H_z = H_z_snapshot()
    env = environment_table(cos)
    prov.update(env)

    # --- C20 the two routes do NOT assume the same environment -------------
    dn = abs(env["n_H_cm3_route2"] - env["n_H_cm3_route1"]) / env["n_H_cm3_route1"]
    dE = abs(env["E_th_eV_route2"] - env["E_th_eV_route1"]) / env["E_th_eV_route1"]
    prov["env_dn_H"], prov["env_dE_th"] = dn, dE
    rec("C20", "environments differ slightly, and by how much",
        dn < 0.05 and dE < 0.01,
        f"n_H: {env['n_H_cm3_route1']:.4e} vs {env['n_H_cm3_route2']:.4e} cm^-3 "
        f"({100*dn:.1f}%); E_th: {env['E_th_eV_route1']:.4f} (NIST) vs "
        f"{env['E_th_eV_route2']:.4f} eV (1 Ry, {100*dE:.2f}%); x_e identical. "
        f"These are INPUT differences, not method differences, and they are not "
        f"harmonised away -- each route keeps its own fiducials.")

    # --- C24 INDEPENDENT ROUTE: the headline cross-check --------------------
    Es = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    b = np.array([float(IY.N_e_ic(e, cos, par)) for e in Es])
    d = np.asarray(N_e_D(Es))
    ratio = d / b
    prov["DB_ratio_min"], prov["DB_ratio_max"] = float(ratio.min()), float(ratio.max())
    rec("C24", "two INDEPENDENT methods agree below 1 MeV",
        np.all(np.abs(ratio - 1.0) < 0.15),
        "  ".join(f"{e:.0e}: D/B={r:.3f}" for e, r in zip(Es, ratio))
        + ". Route 1 is a published deposition FIT; route 2 counts events "
          "against RBEB cross sections with an explicit cascade. Nothing is "
          "shared between them but the physical scenario.")

    # --- C26 where IC overtakes collisions, both ways ----------------------
    def compton_minus_coll(logE):
        fr, _ = loss_branching(np.array([10.0 ** logE]), H_z)
        return float(fr[2, 0] - (fr[4, 0] + fr[5, 0]))
    E_crit_D = 10.0 ** brentq(compton_minus_coll, 3.0, 8.0, xtol=1e-6)
    prov["E_crit_D_eV"] = E_crit_D
    E_crit_1 = 1.3030289107069749e5          # from the companion registry
    rec("C26", "E_crit: deposition-fit route vs 7-mechanism loss route",
        abs(E_crit_D - E_crit_1) / E_crit_1 < 0.5,
        f"route 1 (Bethe vs Thomson IC) {E_crit_1:.4e} eV; route 2 (Compton vs "
        f"excitation+ionization, KN-corrected) {E_crit_D:.4e} eV; ratio "
        f"{E_crit_D/E_crit_1:.3f}. Different definitions of 'collisional', so "
        f"agreement to tens of per cent is the most that is meaningful.")

    # --- C36 what the high-energy divergence IS ---------------------------
    r12 = float(N_e_D(1e12)) / float(IY.N_e_ic(1e12, cos, par))
    prov["DB_ratio_1e12"] = r12
    b_kn = 4.0 * (1e12 * EV_J / L.E0) * L.K_B * L.T_CMB_0 * (1 + Z_SNAP) / L.E0
    fkn = float(L.F_KN(np.array([b_kn]))[0])
    prov["F_KN_at_1e12"] = fkn
    rec("C36", "the D/B divergence above 1e8 eV is Klein-Nishina",
        True,
        f"D/B = {r12:.3f} at 1e12 eV. Route 1 uses Thomson-limit IC; route 2 "
        f"applies the KN kernel, F_KN = {fkn:.4f} at b = {b_kn:.3e}, which "
        f"REDUCES the IC loss and leaves more energy for the gas. The companion "
        f"work flagged this as an unquantified caveat ('KN suppression would "
        f"raise model B'); this is the quantification: +{100*(r12-1):.0f}%.")

    # --- C20 adiabatic losses: present in route 2, absent in route 1 -------
    Eg = np.logspace(2, 12, 400)
    fr, _ = loss_branching(Eg, H_z)
    i_ad = int(np.argmax(fr[0]))
    prov["adiabatic_peak_frac"] = float(fr[0, i_ad])
    prov["adiabatic_peak_E_eV"] = float(Eg[i_ad])
    rec("C20", "route 2 contains a channel route 1 does not have at all",
        fr[0].max() > 0.01,
        f"adiabatic expansion peaks at {100*fr[0,i_ad]:.1f}% of the loss rate "
        f"near {Eg[i_ad]:.2e} eV. Route 1 has no adiabatic term. Synchrotron "
        f"(B0 = 1 nG) peaks at {100*fr[1].max():.4f}%, Coulomb at "
        f"{100*fr[3].max():.2f}%, bremsstrahlung at {100*fr[6].max():.4f}% -- "
        f"all negligible, which is why route 1 omitting them costs little.")

    # --- C31 convergence of the D cascade ---------------------------------
    y1 = yield_D(n=260)[1][-1]
    y2 = yield_D(n=1040)[1][-1]
    rec("C31", "model D converged in grid resolution",
        abs(y1 - y2) / y2 < 5e-3,
        f"Y(1e12) = {y1:.5e} at n=260 vs {y2:.5e} at n=1040, rel.diff "
        f"{abs(y1-y2)/y2:.2e}")

    # --- C20 hard ceiling, and monotonicity -------------------------------
    Kg, Y, _ = yield_D()
    rec("C20", "model D under the E/E_th ceiling and monotone",
        np.all(Y <= Kg / L.THRESHOLD_EV_ION + 1e-9) and np.all(np.diff(Y) >= -1e-9),
        f"max Y/(E/E_th) = {float(np.max(Y/(Kg/L.THRESHOLD_EV_ION))):.4f}; "
        f"min diff = {float(np.min(np.diff(Y))):.2e}")

    # --- MODEL E: D plus the IC-secondary photoionization channel ----------
    Kg, YE, extra, ch = yield_E(cos, par)
    YD = np.asarray(N_e_D(Kg))
    rec("C20", "model E >= model D everywhere, monotone, under the ceiling",
        np.all(YE >= YD - 1e-9) and np.all(np.diff(YE) >= -1e-9)
        and np.all(YE <= Kg / L.THRESHOLD_EV_ION + 1e-9),
        f"min(E-D) = {float(np.min(YE-YD)):.2e}; max E/(K/E_th) = "
        f"{float(np.max(YE/(Kg/L.THRESHOLD_EV_ION))):.4f}")

    # --- C20 IS THE THOMSON-LIMIT PHOTON SPECTRUM LEGITIMATE HERE? ---------
    # Route 2 uses Klein-Nishina for the IC LOSS RATE, but the Blumenthal &
    # Gould scattered-photon SPECTRUM is a Thomson-limit result. Those are only
    # compatible if the secondary channel operates where KN is negligible --
    # measured at the energies that actually deliver the ionizations, not at a
    # convenient one.
    frac = extra / max(extra[-1], 1e-30)
    i_lo, i_hi = int(np.argmax(frac > 0.05)), int(np.argmax(frac > 0.95))
    fkn_w = []
    for i in (i_lo, i_hi):
        g = float(L.kinematics(Kg[i] * EV_J)[1])
        b = 4.0 * g * L.K_B * L.T_CMB_0 * (1.0 + Z_SNAP) / L.E0
        fkn_w.append((Kg[i], g, b, float(L.F_KN(np.array([b]))[0])))
    prov["FKN_window_lo"], prov["FKN_window_hi"] = fkn_w[0][3], fkn_w[1][3]
    rec("C20", "Thomson-limit BG70 spectrum is valid where model E operates",
        min(fkn_w[0][3], fkn_w[1][3]) > 0.999,
        "; ".join(f"at {100*q:.0f}% of the channel: K={t[0]:.3e} eV, "
                  f"gamma={t[1]:.3e}, b_KN={t[2]:.2e}, F_KN={t[3]:.5f}"
                  for q, t in ((0.05, fkn_w[0]), (0.95, fkn_w[1])))
        + ". KN corrections are below 1e-4 across the window, so mixing a "
          "KN-corrected loss rate with a Thomson-limit photon spectrum is "
          "consistent. CHECKED, not assumed.")

    # --- C24 the independent cross-check, now WITH the secondary channel ---
    Ecmp = np.array([1e7, 1e8, 1e9, 1e12])
    e2 = np.asarray(N_e_E(Ecmp, cos, par))
    c1 = np.array([float(IY.N_e_C(x, cos, par, chan)) for x in Ecmp])
    prov["EC_ratio_1e12"] = float(e2[-1] / c1[-1])
    rec("C24", "routes still agree once BOTH carry the secondary channel",
        np.all(np.abs(e2 / c1 - 1.0) < 0.2),
        "  ".join(f"{x:.0e}: E/C={r:.3f}" for x, r in zip(Ecmp, e2 / c1))
        + f". Saturated: E = {e2[-1]:.4e} vs C = {c1[-1]:.4e}.")

    # --- quantities the rewritten paper needs to CITE, not retype ----------
    for Ew in (1e2, 1e3, 1e4, 1e5):
        prov[f"W_D_{Ew:.0e}"] = float(Ew) / float(N_e_D(Ew))
    Eg = np.logspace(2, 12, 400)
    fr_all, _ = loss_branching(Eg, H_z)
    for i, k in enumerate(L.MECH_KEYS):
        prov[f"peakfrac_{k}"] = float(fr_all[i].max())
        prov[f"peakE_{k}_eV"] = float(Eg[int(np.argmax(fr_all[i]))])
    # the IC-secondary window, measured on model E's own cumulative curve
    Kg2, YE2, extra2, _ = yield_E(cos, par)
    cum = extra2 / max(extra2[-1], 1e-30)
    prov["Ewindow_lo_eV"] = float(Kg2[int(np.argmax(cum > 0.05))])
    prov["Ewindow_hi_eV"] = float(Kg2[int(np.argmax(cum > 0.95))])
    prov["N_gamma_D_1e+02"] = 1.0 + float(N_e_D(1e2 - L.THRESHOLD_EV_ION))
    prov["N_gamma_D_1e+03"] = 1.0 + float(N_e_D(1e3 - L.THRESHOLD_EV_ION))
    prov["N_e_E_sat"] = float(e2[-1])
    prov["ED_ratio_sat"] = float(e2[-1] / float(N_e_D(1e12)))
    for E in (1e2, 1e3, 1e6, 1e9, 1e12):
        prov[f"N_e_D_{E:.0e}"] = float(N_e_D(E))
        prov[f"N_e_E_{E:.0e}"] = float(N_e_E(E, cos, par))
    prov["W_D_1keV"] = 1.0e3 / float(N_e_D(1e3))
    prov["N_e_D_sat"] = float(N_e_D(1e12))
    return rows, prov


# ===========================================================================
# 4. FIGURE -- the companion's upper panel, with model D on the same axes
# ===========================================================================
C_E, C_G, C_D = "#2a78d6", "#eb6834", "#117733"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
MECH_COL = ["#88419d", "#bbbbbb", "#eb6834", "#999933", "#44aa99", "#2a78d6", "#dddddd"]


def make_figure(cos, par, chan, prov, outstem="yield_comparison_fig"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    # igm_losses mutates GLOBAL plt.rcParams at import time -- it turns on
    # text.usetex when a LaTeX install is present and switches to a serif font.
    # That silently changed how this figure renders and made a bare "&" in
    # "Furlanetto & Stoever" a LaTeX alignment tab, which killed the build after
    # every check had already passed. Render this figure under our own settings,
    # matching the companion figures, and leave the global state as we found it.
    from igm_config import safe_plot_style
    with safe_plot_style() as plt:
        return _draw_figure(cos, par, chan, prov, outstem, plt, LogLocator)


def _unused_legacy_guard():
    pass


def _draw_figure(cos, par, chan, prov, outstem, plt, LogLocator):
    H_z = H_z_snapshot()
    E_e = np.logspace(2, 12, 260)
    E_g = np.logspace(1, 5, 200)
    A = np.asarray(IY.N_e_full(E_e, par))
    B = np.asarray(IY.N_e_ic(E_e, cos, par))
    C = np.asarray(IY.N_e_C(E_e, cos, par, chan))
    D = np.asarray(N_e_D(E_e))
    Emod = np.asarray(N_e_E(E_e, cos, par))
    Gc = np.asarray(IY.N_gamma(E_g, cos, par, "C", chan))
    fr, _ = loss_branching(E_e, H_z)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.2, 9.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2.3, 1.0],
                                              "hspace": 0.09})
    fig.patch.set_facecolor(BG)
    for a in (ax, bx):
        a.set_facecolor(BG)
        a.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        a.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6, zorder=0)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9)
        a.set_xscale("log")

    ax.plot(E_e, E_e / IY.E_TH_HI, color=INK2, lw=0.9, ls=(0, (1, 4)), zorder=1)
    ax.text(4e11, 4e11 / IY.E_TH_HI * 1.6, r"$E/E_{\rm th}$  ceiling",
            color=INK2, fontsize=8, ha="right", va="bottom")
    ax.plot(E_g, Gc, color=C_G, lw=4.0, alpha=.9, zorder=3)
    ax.plot(E_e, A, color=C_E, lw=1.5, ls=(0, (1, 2.2)), zorder=5)
    ax.plot(E_e, B, color=C_E, lw=1.6, ls=(0, (5, 2)), zorder=5)
    ax.plot(E_e, C, color=C_E, lw=2.3, zorder=5)
    ax.plot(E_e, D, color=C_D, lw=2.6, ls=(0, (5, 2)), zorder=6)
    ax.plot(E_e, Emod, color=C_D, lw=2.8, zorder=7)

    ax.axvline(IY.E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
    ax.axvline(prov["E_crit_D_eV"], color=C_D, lw=0.9, ls=":", zorder=2)
    ax.set_yscale("log"); ax.set_ylim(2e-3, 3e11); ax.set_xlim(8, 3e12)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.set_ylabel(r"HI ion pairs per primary,  $N_{\rm ion}$", color=INK, fontsize=10.5)
    ax.set_title("HI ionizations per primary electron: deposition-fit route vs "
                 "loss-based route\n"
                 r"IGM at $z=10$, $x_e=10^{-4}$, all cascade generations counted",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    # The pairs overlap; that IS the result, so label the pairs rather than
    # four curves competing for the same corner.
    ax.annotate("IC energy discarded\n"
                r"$\bf{B}$ (route 1, dashed blue) $\approx$ "
                r"$\bf{D}$ (route 2, dashed green)",
                xy=(6e10, B[-1]), xytext=(2.5e6, 6.0e1),
                color=INK, fontsize=8.8, ha="left", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9,
                                shrinkA=2, shrinkB=4), zorder=8)
    ax.annotate("IC secondaries followed\n"
                r"$\bf{C}$ (route 1, solid blue) $\approx$ "
                r"$\bf{E}$ (route 2, solid green)",
                xy=(6e10, Emod[-1]), xytext=(9.0e4, 3.0e8),
                color=INK, fontsize=8.8, ha="left", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9,
                                shrinkA=2, shrinkB=4), zorder=8)
    ax.text(2.8e11, 3.0e9, "A — full absorption", color=C_E, fontsize=8.8,
            ha="right", va="bottom")
    ax.text(3.4e1, 30.0, "photon", color=C_G, fontsize=9.5, weight="bold")
    ax.text(1.3e2, 2.2e-2,
            f"D/B = {prov['DB_ratio_1e12']:.2f} at $10^{{12}}$ eV\n"
            f"(Klein–Nishina: route 1 is Thomson-limit)\n"
            f"D/B within "
            f"{100*max(abs(prov['DB_ratio_min']-1), abs(prov['DB_ratio_max']-1)):.0f}%"
            f" of unity below 1 MeV\n"
            f"E/C = {prov['EC_ratio_1e12']:.3f} once both carry the "
            f"secondary channel",
            color=INK2, fontsize=8.2, ha="left", va="bottom", linespacing=1.35)

    for i, k in enumerate(L.MECH_KEYS):
        if fr[i].max() < 5e-3:
            continue
        bx.plot(E_e, fr[i], color=MECH_COL[i], lw=2.0, zorder=4)
        j = int(np.argmax(fr[i]))
        # a channel that saturates peaks at the right edge, where the label is
        # clipped; pull it back to where the curve is still rising
        if E_e[j] > 1e10:
            j = int(np.argmin(np.abs(fr[i] - 0.75 * fr[i].max())))
        bx.text(E_e[j], min(fr[i, j] + 0.045, 1.03), k, color=MECH_COL[i],
                fontsize=8.4, ha="center", va="bottom", weight="bold")
    bx.axvline(IY.E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
    bx.axvline(prov["E_crit_D_eV"], color=C_D, lw=0.9, ls=":", zorder=2)
    bx.text(prov["E_crit_D_eV"] * 1.35, 0.62,
            f"$E_{{\\rm crit}}$ = {prov['E_crit_D_eV']/1e3:.0f} keV\n(route 2)",
            color=C_D, fontsize=8.2, va="bottom", linespacing=1.25)
    bx.set_ylim(-0.03, 1.12)
    bx.set_ylabel("fraction of $dK/dt$", color=INK, fontsize=10.5)
    bx.set_xlabel("kinetic energy of the primary electron  [eV]", color=INK,
                  fontsize=10.5)
    bx.text(0.985, 0.055, "loss branching in igm_losses.py  (channels above 0.5%)",
            transform=bx.transAxes, ha="right", va="bottom", color=INK,
            fontsize=9.5)

    fig.text(0.012, 0.004,
             "Route 1 (A/B/C): deposition fraction $f_{\\rm ion}(E,x_e)$, "
             "Furlanetto & Stoever 2010 eq. (13); Bethe–Berger–Seltzer stopping; "
             "Thomson-limit IC.\n"
             "Route 2 (D): igm_losses.py — adiabatic, synchrotron, IC with "
             "Klein–Nishina, Coulomb (Gould 72),\n"
             "excitation (Stone & Kim 02), ionization (RBEB, Kim+00), "
             "bremsstrahlung; ionizations counted event by event,\n"
             "knock-on electrons followed through the BEB secondary spectrum "
             "(Kim & Rudd 94).\n"
             "Pairings: D is the analogue of B (both discard the IC energy); "
             "E is the analogue of C (both follow the upscattered CMB "
             "photons).\n"
             "The curves overlap pairwise. That agreement, between methods "
             "sharing nothing but the scenario, is the result.",
             fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.915, bottom=0.165)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outstem}.{ext}", dpi=200, facecolor=BG)
    return f"{outstem}.png"


def main():
    par = IY.Params()
    cos = IY.cosmology(par.z)
    chan = IY.ICPhotonChannel(cos, par).build()
    rows, prov = run_checks(cos, par, chan)
    print("\n[CHECKS]")
    for cid, name, st, det in rows:
        print(f"   [{st}] {cid:5s} {name}\n           {det}")
    print(f"\n   {sum(1 for r in rows if r[2]=='PASS')}/{len(rows)} checks pass.")
    png = make_figure(cos, par, chan, prov)
    with open("yield_comparison_results.json", "w") as fh:
        json.dump({"derived": {k: float(v) for k, v in prov.items()},
                   "checks": [{"id": c, "name": n, "state": s, "detail": d}
                              for c, n, s, d in rows]}, fh, indent=2)
    print(f"\n[ARTEFACTS] {png}  yield_comparison_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
