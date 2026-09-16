"""
stage9_figure.py -- why UV wins: efficiency per erg times available energy.

Writes  manuscript/figures_redo/fig_reionization_budget.{pdf,png}

(a) Ionizations per erg of INJECTED energy, 1/W(E), for both primary species
    at z_i = 10.  This is the quantity that matters for reionization, and it
    is maximised at the hydrogen threshold, not at high energy.  Vertical
    bands mark the characteristic energies of the three source classes.
(b) Ionizations actually DELIVERED per solar mass of star formation, i.e.
    (a) multiplied by the energy budget of each channel (stage9_budget.py).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import stage9_budget as B

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "manuscript" / "figures_redo"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 7.4, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

EV_ERG = B.EV_ERG
fig, ax = plt.subplots(1, 2, figsize=(7.15, 3.2))

# ---------------------------------------------------------------- panel (a)
bands = [(13.6, 5.0e1, "#f6c667", "stellar\nUV"),
         (5.0e2, 8.0e3, "#8fbcd4", "HMXB\nX-rays"),
         (1.0e8, 1.0e10, "#c9a8d4", "CR $e^-$")]
for lo, hi, c, lbl in bands:
    ax[0].axvspan(lo, hi, color=c, alpha=0.45, lw=0, zorder=0)

styles = {20.0: ("#2166ac", "-"), 10.0: ("#b2182b", "--")}
for zi, (color, ls) in styles.items():
    Eg = np.geomspace(13.7, 1.0e13, 600)
    Ke = np.geomspace(B.K[0], B.K[-1], 600)
    ax[0].plot(Eg, B.yield_photon(Eg, zi) / (Eg * EV_ERG), ls, color=color,
               lw=1.5, dashes=(1.2, 1.2) if ls == "-" else (4, 1.4, 1, 1.4),
               label=r"$\gamma$, $z_i=%.0f$" % zi)
    ax[0].plot(Ke, B.yield_electron(Ke, zi) / (Ke * EV_ERG), ls, color=color,
               lw=1.5, label=r"$e^-$, $z_i=%.0f$" % zi)

ax[0].axhline(1.0 / (13.6057 * EV_ERG), color="0.35", lw=0.8, ls=":")
ax[0].text(1.0e7, 1.0 / (13.6057 * EV_ERG) / 1.9,
           r"one ionization per $B_{\rm H}$ = absolute maximum",
           fontsize=6.6, color="0.3", ha="center", va="top")
for (lo, hi, c, lbl), ha, xt in zip(bands, ("left", "center", "center"),
                                    (60.0, np.sqrt(5e2 * 8e3),
                                     np.sqrt(1e8 * 1e10))):
    ax[0].text(xt, 2.4e11, lbl, fontsize=6.8, color="0.25",
               ha=ha, va="top", linespacing=1.2)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlim(13.6, 1.0e13)
ax[0].set_ylim(1.0e5, 3.0e11)
ax[0].set_xlabel(r"primary energy $E$  [eV]")
ax[0].set_ylabel(r"ionizations per erg injected, $1/W$")
ax[0].legend(loc="lower left", frameon=False, ncol=2, handlelength=2.6)
ax[0].set_title("(a)  efficiency per unit energy", fontsize=9)

# ---------------------------------------------------------------- panel (b)
n_uv, n_x, n_x_hi, n_cre, n_crp = B.delivered()
labels = [r"stellar UV, $f_{\rm esc}=0.1$",
          r"HMXB X-rays, $L_X/{\rm SFR}=10^{40}$",
          r"HMXB X-rays, $L_X/{\rm SFR}=3\times10^{39}$",
          r"CR protons (scaled, indicative)",
          r"CR electrons, 1 GeV"]
vals = [n_uv, n_x_hi, n_x, n_crp, n_cre]
cols = ["#f0a63a", "#8fbcd4", "#b8d3e0", "#b9b9b9", "#c9a8d4"]
y = np.arange(len(vals))[::-1]
ax[1].barh(y, vals, color=cols, edgecolor="0.3", lw=0.6, height=0.62)
for yy, v, lb in zip(y, vals, labels):
    ax[1].text(1.4e54, yy + 0.36, lb, fontsize=6.8, va="bottom", ha="left")
    ax[1].text(v * 1.6, yy, r"$%.1e$" % v if v < 1e58 else r"$%.2e$" % v,
               fontsize=6.6, va="center", ha="left", color="0.25")
ax[1].set_xscale("log")
ax[1].set_xlim(1.0e54, 3.0e61)
ax[1].set_yticks([])
ax[1].set_ylim(-0.7, len(vals) - 0.25)
ax[1].set_xlabel(r"ionizations delivered per $M_\odot$ of star formation")
ax[1].set_title("(b)  efficiency $\\times$ available energy", fontsize=9)
ax[1].text(0.97, 0.04,
           "$z_i=10$, counted to $z=5.5$\nCR protons: energy scaled by\n"
           r"$1/K_{ep}$, no proton cascade computed",
           transform=ax[1].transAxes, fontsize=6.2, color="0.35",
           ha="right", va="bottom", linespacing=1.3)

fig.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / ("fig_reionization_budget." + ext), dpi=300)
print("wrote", OUT / "fig_reionization_budget.pdf")
