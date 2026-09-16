#!/usr/bin/env python3
r"""
ionization_yield_fig.png, reproduced with the igm_losses prescription.

Same two-panel layout as the original: yields on top, branching fractions below.
The physics underneath is entirely replaced --

  original : f_ion(E,x_e) deposition fit + Bethe stopping + Thomson-limit IC
  here     : igm_losses.py, 7 loss mechanisms, RBED event counting, BEB
             secondary cascade, Klein-Nishina IC   (the loss routes and E)

Written to a SEPARATE filename on purpose. ionization_yield.tex is a 10-page
paper whose 56 \src-tagged literals are tied to the original prescription's
registry; overwriting its figure would leave the text describing physics the
figure no longer shows. Migrating that paper is a separate job.
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import numpy as np
import ionization_yield as IY
import igm_losses as L
import yield_comparison as YC
from igm_config import IGMConfig, safe_plot_style, environment_report, fig_stem

C_E, C_G, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
MECH_COL = ["#88419d", "#bbbbbb", "#eb6834", "#999933", "#44aa99", "#2a78d6", "#dddddd"]


def main(cfg=IGMConfig()):
    # z has to reach ALL THREE places or the figure mixes redshifts: Params
    # (used by the loss+IC route's channel), cosmology, and the the loss route/E cache key.
    YC.set_redshift(cfg.z)
    par = IY.Params(z=cfg.z); cos = IY.cosmology(cfg.z)
    env = environment_report(cfg)
    Eth = L.THRESHOLD_EV_ION
    E_e = np.logspace(2, 12, 300)
    E_g = np.logspace(1, 5, 240)
    D = np.asarray(YC.N_e_loss(E_e))
    E = np.asarray(YC.N_e_loss_ic(E_e, cos, par))
    # photon yield on the SAME prescription: one ionization + the photoelectron's
    # IY.at_or_above, not a bare '>=': see the 2026-09-13 ULP fix. No grid
    # point here coincides with Eth today, but the pattern is the bug.
    Ng = np.where(IY.at_or_above(E_g, Eth), 1.0 + np.asarray(YC.N_e_loss_ic(np.maximum(E_g - Eth, 1e-9),
                                                        cos, par)), 0.0)
    fr, _ = YC.loss_branching(E_e, YC.H_z_snapshot())

    with safe_plot_style() as plt:
        from matplotlib.ticker import LogLocator
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.0, 8.8), sharex=True,
                                     gridspec_kw={"height_ratios": [2.3, 1.0],
                                                  "hspace": 0.09})
        fig.patch.set_facecolor(BG)
        for a in (ax, bx):
            a.set_facecolor(BG)
            a.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): a.spines[sp].set_color(GRID)
            a.tick_params(colors=INK2, labelsize=9); a.set_xscale("log")
        ax.plot(E_e, E_e / Eth, color=INK2, lw=0.9, ls=(0, (1, 4)), zorder=1)
        ax.text(4e11, 4e11 / Eth * 1.6, r"$E/E_{\rm th}$  ceiling", color=INK2,
                fontsize=8, ha="right", va="bottom")
        ax.plot(E_g, Ng, color=C_G, lw=4.0, alpha=.9, zorder=3)
        ax.plot(E_e, D, color=C_E, lw=1.8, ls=(0, (5, 2)), zorder=5)
        ax.plot(E_e, E, color=C_E, lw=2.6, zorder=6)
        ax.axvline(Eth, color=INK2, lw=0.9, ls=":", zorder=2)
        ax.set_yscale("log"); ax.set_ylim(2e-3, 3e11); ax.set_xlim(8, 3e12)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
        ax.set_ylabel(r"HI ion pairs per primary,  $N_{\rm ion}$", color=INK, fontsize=10.5)
        ax.set_title("Average HI ionizations per primary particle — igm_losses "
                     "prescription\n"
                     rf"IGM at $z={cfg.z:.0f}$, "
                     rf"$x_e=10^{{{np.log10(env['x_e']):.0f}}}$, "
                     "all cascade generations counted",
                     color=INK, fontsize=11.5, loc="left", pad=10)
        ax.text(2.8e11, E[-1] * 2.8, "E — IC secondaries followed",
                color=C_E, fontsize=9.2, ha="right", va="bottom", weight="bold")
        ax.text(2.8e11, D[-1] * 0.30, "D — IC energy discarded",
                color=C_E, fontsize=8.8, ha="right", va="top")
        ax.text(3.4e1, 30.0, "photon", color=C_G, fontsize=9.5, weight="bold")
        ax.text(1.3e2, 2.2e-2,
                f"saturated:  D = {D[-1]:.3e},  E = {E[-1]:.3e} ion pairs\n"
                f"the IC-secondary channel is worth a factor {E[-1]/D[-1]:.0f}",
                color=INK2, fontsize=8.4, ha="left", va="bottom", linespacing=1.35)
        for i, k in enumerate(L.MECH_KEYS):
            if fr[i].max() < 5e-3: continue
            bx.plot(E_e, fr[i], color=MECH_COL[i], lw=2.0, zorder=4)
            j = int(np.argmax(fr[i]))
            if E_e[j] > 1e10: j = int(np.argmin(np.abs(fr[i] - 0.75 * fr[i].max())))
            bx.text(E_e[j], min(fr[i, j] + 0.045, 1.03), k, color=MECH_COL[i],
                    fontsize=8.4, ha="center", va="bottom", weight="bold")
        bx.axvline(Eth, color=INK2, lw=0.9, ls=":", zorder=2)
        bx.set_ylim(-0.03, 1.12)
        bx.set_ylabel("fraction of $dK/dt$", color=INK, fontsize=10.5)
        bx.set_xlabel("kinetic energy of the primary particle  [eV]", color=INK, fontsize=10.5)
        bx.text(0.985, 0.055, "loss branching in igm_losses.py  (channels above 0.5%)",
                transform=bx.transAxes, ha="right", va="bottom", color=INK, fontsize=9.5)
        fig.text(0.012, 0.004,
                 "Electron propagation: igm_losses.py — adiabatic, synchrotron, IC "
                 "with Klein–Nishina, Coulomb (Gould 72),\n"
                 "excitation (Stone & Kim 02), ionization (RBED, Kim+00), "
                 "bremsstrahlung. Ionizations counted event by event;\n"
                 "knock-on electrons followed through the BED secondary spectrum "
                 "(Kim & Rudd 94). Photon yield on the same prescription.\n"
                 f"n_HI = {env['n_HI_cm3_at_z']:.4e} cm⁻³, B₀ = {env['B0_T']:.1e} T, "
                 f"H(z) = {env['H_z_s']:.4e} s⁻¹ — all adjustable via igm_config.",
                 fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
        fig.subplots_adjust(left=0.125, right=0.975, top=0.915, bottom=0.155)
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_stem('ionization_yield_fig_igm')}.{ext}",
                        dpi=200, facecolor=BG)
    print(f"[ARTEFACT] {fig_stem('ionization_yield_fig_igm')}.png")
    print(f"   D(1e12) = {D[-1]:.4e}   E(1e12) = {E[-1]:.4e}   E/D = {E[-1]/D[-1]:.1f}")
    print(f"   environment: {env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
