#!/usr/bin/env python3
r"""
Fig. 3 of photon_vs_electron.pdf, recomputed for two initial mass functions.

    Salpeter   : dN/dm ~ m^-2.35, 0.1-100 Msun, K_UV = 1.15e-28  [md14 eq.(10)]
    top-heavy  : dN/dm ~ m^-1.30, 1-100  Msun, K_UV = 1.15e-28/3.570
                 [jeong25 eq.(2); the K_UV ratio is their eta_UV pair]

DECISIONS ON RECORD (user, 2026-09-14):
  * Salpeter K_UV stays at OUR value 1.15e-28 and the top-heavy value is
    obtained by applying Jeong's RATIO, so the Salpeter curve is identical to
    the published Fig. 3. The provenance of both K_UV values, and of the 15%
    difference between them, is written into the manuscript.
  * Jeong's top-heavy eta_UV = 3.57e28 is borrowed from an EXTREME (Pop III
    like) IMF in Zackrisson+11, not from their own alpha = 1.3 law -- they say
    so, and the figure says so.
  * xi_ion is HELD FIXED: it is a measured line-to-continuum ratio and already
    carries whatever IMF nature used. zeta_gamma is therefore IMF-independent.
  * The figure shows the consistent comparison only, with the cancellation
    stated on it.
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import contextlib, json
import numpy as np
import ionization_yield as IY
import photon_vs_electron as P
from igm_config import safe_plot_style, fig_stem

ETA_UV_TOPHEAVY = 3.57e28      # erg/s/Hz per Msun/yr   [jeong25, sec. 5]
K_UV_JEONG_SALP = 1.00e-28     # their Salpeter baseline, read off md14 Fig. 3
K_RATIO = (1.0 / ETA_UV_TOPHEAVY) / K_UV_JEONG_SALP

IMFS = {
    "salpeter":  dict(slope=2.35, m_lo=0.1, m_hi=100.0, K=P.K_UV,
                      label=r"Salpeter $\alpha=2.35$, $0.1$--$100\,M_\odot$"),
    "topheavy":  dict(slope=1.30, m_lo=1.0, m_hi=100.0, K=P.K_UV * K_RATIO,
                      label=r"top-heavy $\alpha=1.30$, $1$--$100\,M_\odot$"),
}
C_G, C_E, INK, INK2, GRID, BG = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
C_TH = "#117733"


@contextlib.contextmanager
def imf(spec):
    """Override the IMF globals, restore them, and ASSERT the restoration."""
    old = (P.IMF_SLOPE, P.M_MIN, P.M_MAX, P.K_UV)
    P.IMF_SLOPE, P.M_MIN, P.M_MAX, P.K_UV = (spec["slope"], spec["m_lo"],
                                             spec["m_hi"], spec["K"])
    try:
        yield
    finally:
        P.IMF_SLOPE, P.M_MIN, P.M_MAX, P.K_UV = old
        assert (P.IMF_SLOPE, P.M_MIN, P.M_MAX, P.K_UV) == old


def main():
    par = IY.Params(z=10.0); cos = IY.cosmology(10.0)
    chan = IY.ICPhotonChannel(cos, par).build()
    Eg = np.logspace(np.log10(IY.E_TH_HI), np.log10(P.photon_band_max()), 260)
    Ee = np.logspace(np.log10(P.CR_E_MIN_EV), np.log10(P.CR_E_MAX_EV), 260)
    prov, D = {}, {}
    for key, spec in IMFS.items():
        with imf(spec):
            D[key] = dict(
                n_e=P.dndlog10E(Ee, "electron"),
                z_e=P.dzeta_dlog10E(Ee, "electron", cos, par, chan),
                zeta_e=P.zeta_total("electron", cos, par, chan),
                snM=P.sn_per_solar_mass(), K=P.K_UV,
                rho_sfr=P.rho_sfr(), ndot=P.ndot_electrons_comoving()[0])
        for k in ("zeta_e", "snM", "K", "rho_sfr", "ndot"):
            prov[f"{k}_{key}"] = float(D[key][k])
    # the photon channel does not move: xi_ion and rho_UV are observations
    n_g = P.dndlog10E(Eg, "photon")
    z_g = P.dzeta_dlog10E(Eg, "photon", cos, par, chan)
    zeta_g = P.zeta_total("photon", cos, par, chan)
    prov["zeta_gamma"] = float(zeta_g)
    for key in IMFS:
        prov[f"ratio_{key}"] = float(zeta_g / D[key]["zeta_e"])
        prov[f"eps_parity_{key}"] = float(P.EPS_CR_FE_FID * zeta_g / D[key]["zeta_e"])
    S, T = D["salpeter"], D["topheavy"]
    prov["f_snM"] = T["snM"] / S["snM"]
    prov["f_K"] = T["K"] / S["K"]
    prov["f_product"] = (T["snM"] / S["snM"]) * (T["K"] / S["K"])
    prov["f_zeta_e"] = T["zeta_e"] / S["zeta_e"]
    prov["f_ratio"] = prov["ratio_topheavy"] / prov["ratio_salpeter"]
    prov["K_ratio_jeong"] = K_RATIO
    prov["eta_UV_topheavy"] = ETA_UV_TOPHEAVY

    with safe_plot_style() as plt:
        fig = plt.figure(figsize=(13.2, 7.2)); fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], hspace=0.09, wspace=0.20)
        ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        cx = fig.add_subplot(gs[:, 1])
        for a in (ax, bx):
            a.set_facecolor(BG); a.set_xscale("log"); a.set_yscale("log")
            a.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): a.spines[sp].set_color(GRID)
            a.tick_params(colors=INK2, labelsize=9); a.set_xlim(8, 3e12)
        for a, yg, ye_key in ((ax, n_g, "n_e"), (bx, z_g, "z_e")):
            a.plot(Eg, yg, color=C_G, lw=2.6, zorder=5)
            a.plot(Ee, S[ye_key], color=C_E, lw=2.8, zorder=5)
            a.plot(Ee, T[ye_key], color=C_TH, lw=1.6, ls=(0, (5, 2)), zorder=6)
            a.axvline(IY.E_TH_HI, color=INK2, lw=0.9, ls=":", zorder=2)
        ax.set_ylabel(r"$d\dot n_{\rm ion}/d\log_{10}E$" "\n"
                      r"[s$^{-1}$cMpc$^{-3}$dex$^{-1}$]", color=INK, fontsize=9.5)
        bx.set_ylabel(r"$d\zeta/d\log_{10}E$" "\n"
                      r"[s$^{-1}$ per H atom dex$^{-1}$]", color=INK, fontsize=9.5)
        bx.set_xlabel("primary energy  $E$  [eV]", color=INK, fontsize=10)
        ax.set_xticklabels([])
        ax.text(0.985, 0.93, "(a)  differential ionizing emissivity",
                transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=11)
        bx.text(0.985, 0.93, "(b)  ionization rate per target atom",
                transform=bx.transAxes, ha="right", va="top", color=INK, fontsize=11)
        ax.text(0.20, 0.60, "stellar UV\n" r"identical for both IMFs:"
                "\n" r"$\xi_{\rm ion}$ and $\rho_{\rm UV}$ are measured",
                transform=ax.transAxes, color=C_G, fontsize=8.4, ha="left",
                va="top", weight="bold", linespacing=1.3)
        ax.text(0.985, 0.58, "CR electrons\nSalpeter (solid)\n"
                "top-heavy (dashed)",
                transform=ax.transAxes, color=C_E, fontsize=8.4, ha="right",
                va="top", weight="bold", linespacing=1.3)

        # ---- the map, with both parity loci ------------------------------
        fe = np.logspace(-4, 0, 240); ec = np.logspace(-6, 0, 240)
        F, Ec = np.meshgrid(fe, ec, indexing="ij")
        R = prov["ratio_salpeter"]*(F/P.F_ESC_FID)/(Ec/P.EPS_CR_FE_FID)
        im = cx.pcolormesh(fe, ec, np.log10(R).T, cmap="RdBu_r", shading="auto",
                           vmin=-4, vmax=4, zorder=1)
        for key, col, ls, lab in (("salpeter", "#0b0b0b", "-", "Salpeter"),
                                  ("topheavy", C_TH, (0, (6, 3)), "top-heavy")):
            slope = prov[f"ratio_{key}"]*P.EPS_CR_FE_FID/P.F_ESC_FID
            fl = np.array([fe[0], min(1.0/slope, fe[-1])])
            cx.plot(fl, slope*fl, color=col, lw=2.2, ls=ls, zorder=4,
                    label=f"{lab}: parity at $f_{{\\rm esc}}<{100/slope:.2f}\\%$")
        cx.legend(loc="upper left", fontsize=8.4, framealpha=0.92)
        cx.plot([P.F_ESC_FID], [P.EPS_CR_FE_FID], "o", ms=9, mfc="none",
                mec="#0b0b0b", mew=2.0, zorder=5)
        cx.set_xscale("log"); cx.set_yscale("log")
        cx.set_xlabel(r"escape fraction  $f_{\rm esc}$", color=INK, fontsize=10)
        cx.set_ylabel(r"$\epsilon_{\rm CR}\times f_e$", color=INK, fontsize=10)
        cx.tick_params(colors=INK2, labelsize=9)
        cx.set_title(r"(c)  competition map:  $\log_{10}(\zeta_\gamma/\zeta_e)$",
                     color=INK, fontsize=11, loc="left")
        cb = fig.colorbar(im, ax=cx, pad=0.02, extend="both")
        cb.set_label(r"$\log_{10}(\zeta_\gamma/\zeta_e)$", color=INK2, fontsize=8.5)
        cb.ax.tick_params(colors=INK2, labelsize=8)
        cx.text(0.97, 0.05,
                "the two loci are %.1f%% apart" % (100*abs(prov["f_ratio"]-1)),
                transform=cx.transAxes, ha="right", va="bottom",
                color="#ffffff", fontsize=9)

        # ---- the cancellation, stated ------------------------------------
        bx.text(0.018, 0.055,
                "Why the curves nearly coincide\n"
                r"SN per $M_\odot$:  $\times$%.3f" "\n"
                r"$K_{\rm UV}$:  $\times$%.4f" "\n"
                r"product entering $\zeta_e$:  $\times$%.3f" "\n"
                r"$\zeta_\gamma/\zeta_e$:  %.0f $\to$ %.0f"
                % (prov["f_snM"], prov["f_K"], prov["f_product"],
                   prov["ratio_salpeter"], prov["ratio_topheavy"]),
                transform=bx.transAxes, ha="left", va="bottom", fontsize=8.2,
                color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fcfcfbee", ec=GRID, lw=0.8))

        fig.suptitle("Stellar UV versus cosmic-ray electrons at $z=10$: "
                     "Salpeter against a top-heavy IMF",
                     color=INK, fontsize=13, x=0.012, ha="left", y=0.985)
        fig.text(0.012, 0.004,
                 "Top-heavy IMF: Jeong, Jeon, Song & Bromm (2025) eq. (2), "
                 r"$\xi(m)\propto m^{-1.3}$ on 1-100 $M_\odot$, stated for "
                 "Pop II at $z=10$.\n"
                 r"$K_{\rm UV}$: Salpeter $1.15\times10^{-28}$ "
                 "(Madau & Dickinson 2014 eq. 10); top-heavy obtained by "
                 r"applying their $\eta_{\rm UV}=3.57\times10^{28}$ ratio, "
                 r"i.e. $\times%.4f$." "\n"
                 "CAVEAT CARRIED FROM THE SOURCE: that "
                 r"$\eta_{\rm UV}$ is derived from an EXTREME "
                 "(Pop III-like) IMF in Zackrisson+11, not from the "
                 r"$\alpha=1.3$ law - Jeong et al. state this themselves." "\n"
                 r"$\xi_{\rm ion}$ and $\rho_{\rm UV}$ are held fixed: both are "
                 "measured, so the stellar channel is IMF-independent and "
                 "panels (a) and (b) show ONE orange curve."
                 % K_RATIO,
                 fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
        fig.subplots_adjust(left=0.085, right=0.965, top=0.905, bottom=0.205)
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_stem('imf_comparison_fig')}.{ext}", dpi=200, facecolor=BG)
        plt.close(fig)

    json.dump({"derived": prov}, open("imf_comparison_results.json", "w"),
              indent=2, sort_keys=True)
    print("[ARTEFACT] imf_comparison_fig.png  imf_comparison_results.json")
    for k in sorted(prov):
        print("   %-24s %.6g" % (k, prov[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
