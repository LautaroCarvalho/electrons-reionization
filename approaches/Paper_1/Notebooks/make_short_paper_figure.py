"""Central result figure for the <=7-page manuscript."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "manuscript" / "figures"
d = np.load(HERE / "photon_cascade_table.npz")
K, z = d["K"], d["z"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 8.2, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

fig, ax = plt.subplots(1, 2, figsize=(7.15, 3.05))
styles = {20.0: ("#2166ac", "-"), 10.0: ("#b2182b", "--")}

for zi, (color, ls) in styles.items():
    iz = int(np.argmin(np.abs(z - zi)))
    Ye = np.maximum(d["Ye"][iz], 0.0)
    Yf = np.maximum(d["Yph_fixed"][iz], 0.0)
    Ym = np.maximum(d["Yph_mfp"][iz], 0.0)
    Tf, Tm = Ye + Yf, Ye + Ym
    m = Ye > 0

    ax[0].loglog(K[m], Ye[m], color=color, ls=":", lw=1.45,
                 label=rf"electron impact, $z_i={zi:.0f}$")
    ax[0].loglog(K[m], Tf[m], color=color, ls=ls, lw=2.05,
                 label=rf"total, $z_i={zi:.0f}$")
    ax[0].fill_between(K[m], Tf[m], Tm[m], color=color, alpha=0.17, lw=0)

    ax[1].loglog(K[m], K[m] / Ye[m], color=color, ls=":", lw=1.45)
    ax[1].loglog(K[m], K[m] / Tf[m], color=color, ls=ls, lw=2.05,
                 label=rf"$z_i={zi:.0f}$")
    ax[1].fill_between(K[m], K[m] / Tm[m], K[m] / Tf[m],
                       color=color, alpha=0.17, lw=0)

ax[0].set(xlabel=r"injection energy $K_{\rm ini}$ [eV]",
          ylabel=r"ionizations / injected $e^-$",
          xlim=(1e2, 1e13), ylim=(1, 2e7))
ax[1].set(xlabel=r"injection energy $K_{\rm ini}$ [eV]",
          ylabel=r"$W=K_{\rm ini}/Y$ [eV]",
          xlim=(1e2, 1e13), ylim=(1e1, 1e8))
ax[1].axhline(35.0, color="0.35", lw=1.0, ls="-.", label="35 eV neutral-gas limit")

for a, tag in zip(ax, ("(a)", "(b)")):
    a.axvline(13.6, color="0.5", lw=0.8)
    a.axvline(1e3, color="0.6", lw=0.7, ls=":")
    a.axvline(1e5, color="0.6", lw=0.7, ls=":")
    a.axvline(3e8, color="0.6", lw=0.7, ls=":")
    a.grid(which="major", alpha=0.22, lw=0.55)
    a.text(0.02, 0.04, tag, transform=a.transAxes, fontweight="bold")
ax[0].legend(loc="upper left", ncol=1, framealpha=0.92)
ax[1].legend(loc="upper left", framealpha=0.92)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "fig_photon_yield.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_photon_yield.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_photon_yield.pdf")
