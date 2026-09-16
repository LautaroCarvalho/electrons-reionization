"""
emis_figure.py -- the deliverable figures.

For each residual ionized fraction x_e in {1e-4, 1e-3, 1e-2, 1e-1} and each
branch (HYDROGEN, HELIUM) one three-panel figure is produced:

  (a) N_ion(E)          ionizations per PRIMARY particle, all the way down the
                        cascade.  Engine A (this project's transport/CSDA
                        cascade + Verner 1996 photoionization) as solid lines;
                        engine B (Shull & van Steenberg 1985, Valdes & Ferrara
                        2008, Valdes, Evoli & Ferrara 2010) as markers, each
                        drawn ONLY inside its own published validity range.

  (b) dndot_ion/dlog10(E)   differential ionizing emissivity
                        [ionizations s^-1 cMpc^-3 dex^-1].  Photons normalised
                        to the measured Gaikwad et al. (2023) emissivity and to
                        the JWST Giovinazzo et al. (2026) emissivity; cosmic-ray
                        electrons as a band spanning the Tueros et al. (2014)
                        and Gessey-Jones et al. (2023) injection spectra and the
                        10-50 per cent / per-cent-level efficiency ranges.

  (c) dzeta/dlog10(E)   the same divided by the comoving target density,
                        [s^-1 per target atom dex^-1], directly comparable with
                        the MEASURED Gamma_HI of Gaikwad et al. (2023).

Plus two summary figures:
  fig_emissivity_vs_z      integrated ionization emissivity and the CR/photon
                           ratio as functions of redshift.
  fig_engine_comparison    engine A / engine B, the systematic of the whole
                           calculation.

HELIUM IS FULLY NEUTRAL by instruction: n_HeI = n_He at every x_e.  This is an
upper bound on the helium target and is inconsistent with x_e > 0; it is
stated on every helium panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import emis_common as CM
import emis_engine_A as A
import emis_engine_A_xe as AX
import emis_engine_B as B
import emis_sources as S

OUT = Path(__file__).resolve().parent.parent / "figures_emissivity"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9.5,
    "legend.fontsize": 6.4, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "axes.titlesize": 9,
})

E_GRID = np.geomspace(CM.E_TH["HI"] * 1.001, 1.0e13, 700)
Z_SHOW = (6.0, 10.0)
C_PH = {6.0: "#b2182b", 10.0: "#ef8a62"}
C_CR = {6.0: "#2166ac", 10.0: "#67a9cf"}
LN10 = np.log(10.0)


def nion_primary(E, z, x_e, species, kind, flavour="csda"):
    """Ionizations of `species` per primary of energy E, engine A.

    flavour = "csda"      collisional ionizations only, any x_e
              "transport" + re-absorbed radiated photons, x_e = 1e-4 only
    """
    if kind == "electron":
        tot = (AX.nion_electron_transport(E, z) if flavour == "transport"
               else AX.nion_electron_total_xe(E, z, x_e))
        return tot * A.f_species(E, species)
    # photon: absorbed on H I or He I, then the photoelectron cascades
    out = np.zeros_like(E)
    for k in ("HI", "HeI"):
        Pk = A.p_absorb(E, k)
        Ke = np.maximum(E - CM.E_TH[k], 1e-3)
        tot = (AX.nion_electron_transport(Ke, z) if flavour == "transport"
               else AX.nion_electron_total_xe(Ke, z, x_e))
        casc = tot * A.f_species(Ke, species)
        casc = np.where(E > CM.E_TH[k], casc, 0.0)
        out = out + Pk * (casc + (1.0 if k == species else 0.0))
    return np.where(E >= CM.E_TH[species], out, np.nan)


SVS_EMAX_COMPUTED = 3.0e3      # SvS85 ran their Monte Carlo up to 3 keV


def panel_a(ax, x_e, species):
    for z in Z_SHOW:
        ax.plot(E_GRID, nion_primary(E_GRID, z, x_e, species, "photon"),
                "-", color=C_PH[z], lw=3.0, alpha=0.45, zorder=1,
                label=r"$\gamma$, $z=%.0f$" % z)
        ax.plot(E_GRID, nion_primary(E_GRID, z, x_e, species, "electron"),
                "-", color=C_CR[z], lw=1.5, label=r"$e^-$, $z=%.0f$" % z)
    if abs(np.log10(x_e) + 4.0) < 1e-6:
        for z in Z_SHOW:
            ax.plot(E_GRID, nion_primary(E_GRID, z, x_e, species, "electron",
                                         "transport"),
                    "-", color=C_CR[z], lw=1.0, alpha=0.55,
                    label=r"$e^-$ +radiated $\gamma$, $z=%.0f$" % z)
    # ---- engine B, each inside its own published range
    Eb1 = np.geomspace(B.SVS85_EMIN, SVS_EMAX_COMPUTED, 60)
    Eb2 = np.geomspace(SVS_EMAX_COMPUTED, 1.0e13, 120)
    ax.plot(Eb1, B.svs85_nion(Eb1, x_e, species), ":", color="0.1", lw=2.0,
            label="SvS85 (B), computed range")
    ax.plot(Eb2, B.svs85_nion(Eb2, x_e, species), ":", color="0.55", lw=1.2,
            label="SvS85 (B), limiting-value extrap.")
    if species == "HI":
        Ev = np.geomspace(B.VF08_EMIN, B.VF08_EMAX, 60)
        ax.plot(Ev, B.vf08_fion(Ev, x_e) * Ev / CM.E_TH["HI"], "--",
                color="#1b7837", lw=1.6, label="V&F08 (B), total ion.")
        Ew = np.geomspace(B.VEF_EMIN, B.VEF_EMAX, 90)
        ax.plot(Ew, B.vef2010_fion(Ew, x_e, 10.0) * Ew / CM.E_TH["HI"], "-.",
                color="#762a83", lw=1.6, label="VEF2010 (B), total, $z{=}10$")
    ax.set_xscale("log"); ax.set_yscale("log")
    # C44: limits are set to fixed physical values, identical across all eight
    # figures so the panels are comparable, and NOT tuned per panel.
    # x: threshold to the top of engine A's grid.  y: 1e-2 is below the
    # smallest non-zero yield (0.117 at 20 eV); 3e7 is above the largest
    # (4.5e6, A-transport at 1e10 eV).
    ax.set_xlim(CM.E_TH["HI"], 1e13); ax.set_ylim(1e-2, 3e7)
    ax.axvline(CM.VERNER_EMAX, color="0.6", lw=0.6, ls=":")
    ax.text(CM.VERNER_EMAX * 1.5, 1.5e-2, "Verner fit\nceiling", fontsize=5.4,
            color="0.45", rotation=90, va="bottom")
    ax.set_xlabel(r"primary energy $E$ [eV]")
    ax.set_ylabel(r"$N_{\rm ion}$ per primary particle")
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=1.9,
              columnspacing=0.7, labelspacing=0.22, fontsize=5.8)
    ax.set_title("(a) ionizations per primary, full cascade")


def _mask(y):
    """Zeros -> NaN, so fill_between and plot skip a model's dead range."""
    y = np.asarray(y, float)
    return np.where(np.isfinite(y) & (y > 0.0), y, np.nan)


def _emissivities(E, z, x_e, species):
    ph = {}
    for w, lab in (("gaikwad", "Gaikwad+23"), ("giovinazzo", "Giovinazzo+26")):
        if w == "gaikwad" and abs(z - 6.0) > 1e-6:
            continue
        spec = S.photon_spectrum(E, z, w)
        ph[lab] = _mask(LN10 * E * spec
                        * nion_primary(E, z, x_e, species, "photon"))
    cr = {}
    for m in S.CR_SPECTRA:
        y = {}
        for lv in ("lo", "mid", "hi"):
            spec = S.cr_spectrum(E, z, m, level=lv)
            y[lv] = _mask(LN10 * E * spec
                          * nion_primary(E, z, x_e, species, "electron"))
        cr[m] = y
    return ph, cr


CR_LS = {"Tueros+2014": "-", "GesseyJones+2023": (0, (4, 1.5))}


def panel_bc(axb, axc, x_e, species):
    n_t = CM.n_target_comoving(species) * CM.MPC ** 3      # per cMpc^3
    for z in Z_SHOW:
        ph, cr = _emissivities(E_GRID, z, x_e, species)
        ls = "-" if z == 6.0 else "--"
        for lab, y in ph.items():
            for ax, sc in ((axb, 1.0), (axc, 1.0 / n_t)):
                ax.plot(E_GRID, y * sc, ls, color=C_PH[z],
                        lw=1.5 if "Giov" in lab else 2.6,
                        alpha=1.0 if "Giov" in lab else 0.40, zorder=3,
                        label=r"$\gamma$ %s, $z=%.0f$" % (lab, z))
        for m in cr:
            for ax, sc in ((axb, 1.0), (axc, 1.0 / n_t)):
                ax.fill_between(E_GRID, cr[m]["lo"] * sc, cr[m]["hi"] * sc,
                                color=C_CR[z], alpha=0.22, lw=0, zorder=1)
                ax.plot(E_GRID, cr[m]["mid"] * sc, ls=CR_LS[m], color=C_CR[z],
                        lw=1.2, zorder=2,
                        label=(r"CR $e^-$ %s, $z=%.0f$"
                               % (m.replace("+", " "), z)) if z == 6.0
                        else None)
    for ax in (axb, axc):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(CM.E_TH["HI"], 1e13)
        ax.set_xlabel(r"primary energy $E$ [eV]")
    axb.set_ylabel(r"$\mathrm{d}\dot n_{\rm ion}/\mathrm{d}\log_{10}E$"
                   r"  [s$^{-1}$ cMpc$^{-3}$ dex$^{-1}$]")
    # C44: fixed across all eight figures.  Spans the photon peak
    # (~8e51 at z=6) down to five decades below the CR band, so nothing
    # plotted is clipped.
    axb.set_ylim(1e38, 1e53)
    axb.set_title("(b) differential ionizing emissivity")
    axb.legend(loc="lower left", frameon=False, ncol=1, handlelength=1.9,
               fontsize=5.8, labelspacing=0.25)
    axc.set_ylabel(r"$\mathrm{d}\zeta/\mathrm{d}\log_{10}E$"
                   r"  [s$^{-1}$ per atom dex$^{-1}$]")
    # C44: panel (b) limits divided by n_H(comoving) in cMpc^-3, so the two
    # panels are the same plot in different units; not independently tuned.
    axc.set_ylim(1e-28, 1e-11)
    if species == "HI":
        g = S.GAIKWAD_GAMMA_HI
        axc.axhline(g[0], color="0.2", lw=1.0, ls="-.")
        axc.axhspan(g[0] + g[2], g[0] + g[1], color="0.5", alpha=0.25, lw=0)
        axc.text(1.7e1, 1.3e-28,
                 r"$\Gamma_{\rm H\,I}(z{=}6)=1.45^{+1.57}_{-0.87}"
                 r"\times10^{-13}$ s$^{-1}$"
                 "\n" r"(Gaikwad+23) is per RESIDUAL NEUTRAL H inside"
                 "\n" r"ionized gas; $\zeta$ here is per TOTAL H, so the"
                 "\n" r"two differ by the neutral fraction $x_{\rm H\,I}$.",
                 fontsize=5.2, color="0.25", linespacing=1.4, va="bottom")
    axc.set_title("(c) ionization rate per target atom")


def make_figure(x_e, species):
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.5))
    panel_a(ax[0], x_e, species)
    panel_bc(ax[1], ax[2], x_e, species)
    name = {"HI": r"H\,\textsc{i}", "HeI": "He I"}
    sp = "HYDROGEN" if species == "HI" else "HELIUM"
    extra = ""
    if species == "HeI":
        extra = ("   helium taken FULLY NEUTRAL (instruction): "
                 r"$n_{\rm He\,I}=n_{\rm He}$, an upper bound, "
                 r"inconsistent with $x_e>0$")
    fig.suptitle(r"%s branch    $x_e = 10^{%d}$%s"
                 % (sp, int(round(np.log10(x_e))), extra), fontsize=9.5, y=1.02)
    fig.tight_layout()
    tag = "%s_xe%d" % ("H" if species == "HI" else "He",
                       int(round(-np.log10(x_e))))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("fig_ionizing_emissivity_%s.%s" % (tag, ext)),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    return tag


if __name__ == "__main__":
    for sp in ("HI", "HeI"):
        for xe in AX.X_E_GRID:
            t = make_figure(xe, sp)
            print("wrote fig_ionizing_emissivity_%s.pdf" % t)
