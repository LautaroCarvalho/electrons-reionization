"""
emis_summary_figure.py -- the two companion figures.

fig_emissivity_vs_z      integrated ionization emissivity of each population
                         and their ratio, as functions of redshift, for both
                         branches and every x_e.  This is the plot that
                         actually answers "how much do CR electrons matter
                         relative to stellar photons".
fig_engine_comparison    engine A against engine B over the whole energy axis,
                         i.e. the systematic uncertainty of the calculation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

import emis_common as CM
import emis_engine_A as A
import emis_engine_A_xe as AX
import emis_engine_B as B
import emis_figure as F
import emis_sources as S

OUT = Path(__file__).resolve().parent.parent / "figures_emissivity"
plt.rcParams.update(F.plt.rcParams)

E = np.geomspace(CM.E_TH["HI"] * 1.001, 1.0e13, 500)
LN10 = np.log(10.0)
Z = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])  # Giovinazzo+26 Table B.1 range
XE = AX.X_E_GRID


def integrated(z, x_e, species):
    """Total ionization emissivity [s^-1 cMpc^-3] of each population."""
    lE = np.log10(E)
    yg = LN10 * E * S.photon_spectrum(E, z, "giovinazzo") \
        * F.nion_primary(E, z, x_e, species, "photon")
    out = {"photons": trapezoid(np.nan_to_num(yg), lE)}
    for m in S.CR_SPECTRA:
        for lv in ("lo", "mid", "hi"):
            y = LN10 * E * S.cr_spectrum(E, z, m, level=lv) \
                * F.nion_primary(E, z, x_e, species, "electron")
            out["%s_%s" % (m, lv)] = trapezoid(np.nan_to_num(y), lE)
    return out


def fig_vs_z():
    fig, ax = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(XE)))
    for col, species in enumerate(("HI", "HeI")):
        a0, a1 = ax[0, col], ax[1, col]
        for k, xe in enumerate(XE):
            r = [integrated(z, xe, species) for z in Z]
            ph = np.array([q["photons"] for q in r])
            lo = np.array([min(q["%s_lo" % m] for m in S.CR_SPECTRA) for q in r])
            hi = np.array([max(q["%s_hi" % m] for m in S.CR_SPECTRA) for q in r])
            mid = np.sqrt(np.maximum(lo * hi, 1e-300))
            if k == 0:
                a0.plot(Z, ph, "-", color="#b2182b", lw=2.0,
                        label=r"$\gamma$ (Giovinazzo+26), all $x_e$"
                        if col == 0 else None)
            a0.fill_between(Z, lo, hi, color=cols[k], alpha=0.30, lw=0)
            a0.plot(Z, mid, "-", color=cols[k], lw=1.4,
                    label=r"CR $e^-$, $x_e=10^{%d}$" % round(np.log10(xe)))
            a1.fill_between(Z, lo / ph, hi / ph, color=cols[k], alpha=0.30, lw=0)
            a1.plot(Z, mid / ph, "-", color=cols[k], lw=1.4)
        a0.set_yscale("log")
        a0.set_title("%s branch" % ("HYDROGEN" if species == "HI" else "HELIUM"))
        a0.set_ylabel(r"$\dot n_{\rm ion}$ [s$^{-1}$ cMpc$^{-3}$]")
        a1.set_yscale("log")
        a1.set_xlabel(r"redshift $z$")
        a1.set_ylabel(r"CR $e^-$ / photons")
        a1.axhline(1.0, color="0.3", lw=0.8, ls=":")
        a1.axhline(0.01, color="0.6", lw=0.6, ls="--")
        a1.text(14.8, 0.012, "1 %", fontsize=6, color="0.45", ha="right")
        for a in (a0, a1):
            a.set_xlim(Z[0], Z[-1])
    ax[0, 0].legend(loc="lower left", frameon=False, fontsize=6.2)
    fig.suptitle("Integrated ionizing emissivity: stellar photons vs "
                 "cosmic-ray electrons\n"
                 "(photon curve is $x_e$-independent; helium taken fully "
                 "neutral)", fontsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("fig_emissivity_vs_z." + ext), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_emissivity_vs_z.pdf")


def fig_engines():
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    Eg = np.geomspace(1.0e2, 1.0e13, 400)
    for k, xe in enumerate(XE):
        c = plt.cm.viridis(k / (len(XE) - 1) * 0.85 + 0.05)
        aH = AX.nion_electron_total_xe(Eg, 10.0, xe) * A.f_species(Eg, "HI")
        bH = B.svs85_nion(Eg, xe, "HI")
        ax[0].plot(Eg, aH / bH, "-", color=c, lw=1.5,
                   label=r"$x_e=10^{%d}$" % round(np.log10(xe)))
        vv = B.vef2010_fion(Eg, xe, 10.0) * Eg / CM.E_TH["HI"]
        ax[1].plot(Eg, AX.nion_electron_transport(Eg, 10.0)
                   * A.f_species(Eg, "HI") / vv, "-", color=c, lw=1.5)
    for a, t, r in ((ax[0], "A-csda / SvS85", (B.SVS85_EMIN, 3e3)),
                    (ax[1], "A-transport / VEF2010", (B.VEF_EMIN, B.VEF_EMAX))):
        a.axhline(1.0, color="0.3", lw=0.8, ls=":")
        a.axvspan(r[0], r[1], color="0.85", alpha=0.6, lw=0, zorder=0)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(1e2, 1e13); a.set_ylim(3e-3, 3e2)
        a.set_xlabel(r"electron energy $K$ [eV]")
        a.set_title(t)
        a.text(0.03, 0.06, "shaded: engine B's\npublished range",
               transform=a.transAxes, fontsize=6, color="0.35")
    ax[0].set_ylabel("engine A / engine B")
    ax[0].legend(loc="upper right", frameon=False, fontsize=6.4)
    fig.suptitle("Engine A against engine B for H I, $z=10$ "
                 "-- the systematic of the calculation", fontsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("fig_engine_comparison." + ext), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_engine_comparison.pdf")


if __name__ == "__main__":
    fig_vs_z()
    fig_engines()
