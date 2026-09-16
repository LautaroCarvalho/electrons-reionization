"""
stage8_figure.py -- ionizations per primary particle: photons vs electrons
==========================================================================

Draws the deliverable figure

    fig_ionizations_per_primary.{pdf,png}     ->  manuscript/figures_redo/

Panel (a)  N_ion(E), the average TOTAL number of hydrogen ionizations produced
           by one primary particle of energy E injected at redshift z_i,
           counting every ionization made anywhere down the cascade until the
           end of the reionization window (z = Z_FINAL = 5.5).
Panel (b)  the same information as the energy cost per ionization,
           W = E / N_ion, which is the flat ~35 eV plateau for electrons and
           diverges wherever the primary escapes the window without depositing.

Two primary species, two injection redshifts:
    electrons  -- Y + Ype from redo_cascade_table.npz  (stage2_cascade.py)
    photons    -- Ne         from redo_photon_yield_table.npz (stage8)

Shaded band: above E ~ 3.7e7 eV the omitted channels (Bethe-Heitler pair
production on H nuclei for photons; the 3e4 eV photon-escape ceiling of the
electron table) make BOTH curves lower bounds.  See stage8_verify.py blocks
D, E, F for the numbers quoted in the annotations.

Nothing in Paper_draft/ is touched; output goes to manuscript/figures_redo/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "manuscript" / "figures_redo"
OUT.mkdir(parents=True, exist_ok=True)

ele = np.load(HERE / "redo_cascade_table.npz")
pho = np.load(HERE / "redo_photon_yield_table.npz")

n0 = int(ele["nlow"])
K, ze = ele["K"][n0:], ele["z"]
E, zp = pho["E"], pho["z"]

# Validity edges, all computed in stage8_verify.py
E_PAIR = 3.676e7        # sigma_pair = sigma_KN (complete-screening edge) [eV]
E_GG = 2.441e12         # gamma-gamma on the CMB sets in, z_i = 20 [eV]
E_PIC = 2.333e3         # sigma_pi = sigma_C [eV]
B_H = 13.6057           # [eV]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 7.4, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

styles = {20.0: ("#2166ac", "-"), 10.0: ("#b2182b", "--")}

fig, ax = plt.subplots(1, 2, figsize=(7.15, 3.15))

for zi, (color, ls) in styles.items():
    ie = int(np.argmin(np.abs(ze - zi)))
    ip = int(np.argmin(np.abs(zp - zi)))

    Ne = np.maximum(ele["Y"][ie, n0:] + ele["Ype"][ie, n0:], 0.0)
    Np = np.maximum(pho["Ne"][ip], 0.0)

    me, mp = Ne > 0.0, Np > 0.0
    ax[0].plot(K[me], Ne[me], ls, color=color, lw=1.5,
               label=r"$e^-$, $z_i=%.0f$" % zi)
    ax[0].plot(E[mp], Np[mp], ls, color=color, lw=1.5, alpha=0.95,
               marker="", dashes=(1.2, 1.2) if ls == "-" else (4, 1.4, 1, 1.4),
               label=r"$\gamma$, $z_i=%.0f$" % zi)

    ax[1].plot(K[me], K[me] / Ne[me], ls, color=color, lw=1.5)
    ax[1].plot(E[mp], E[mp] / Np[mp], ls, color=color, lw=1.5,
               dashes=(1.2, 1.2) if ls == "-" else (4, 1.4, 1, 1.4))

for a in ax:
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(B_H, 1.0e13)
    a.axvspan(E_PAIR, 1.0e13, color="0.87", alpha=0.6, lw=0, zorder=0)
    a.axvspan(E_GG, 1.0e13, color="0.72", alpha=0.6, lw=0, zorder=0)
    a.axvline(E_PIC, color="0.45", lw=0.7, ls=":", zorder=1)
    a.set_xlabel(r"primary energy $E$  [eV]")

ax[0].set_ylabel(r"ionizations per primary  $N_{\rm ion}$")
ax[0].set_ylim(1.0e-1, 1.0e7)
ax[0].axhline(1.0, color="0.6", lw=0.7, zorder=1)
ax[0].legend(loc="upper left", frameon=False, ncol=2, handlelength=2.6,
             columnspacing=1.1)
ax[0].text(5.5e7, 1.5e-1, "pair production\nomitted: $N$ is\na lower bound",
           fontsize=6.6, color="0.32", ha="left", va="bottom", linespacing=1.35)
ax[0].text(3.0e12, 3.0e5, r"$\gamma\gamma_{\rm CMB}$", fontsize=6.6,
           color="0.32", rotation=90, ha="left", va="center")
ax[0].text(E_PIC / 1.7, 1.7e-1, r"$\sigma_{\rm pi}=\sigma_{\rm C}$",
           fontsize=6.6, color="0.35", rotation=90, va="bottom", ha="right")
ax[0].set_title(r"(a)  all ionizations down the cascade, $z>%.1f$"
                % float(pho["z_final"]), fontsize=9)

ax[1].set_ylabel(r"$W = E/N_{\rm ion}$  [eV]")
ax[1].set_ylim(1.0e1, 1.0e9)
ax[1].axhline(35.0, color="0.5", lw=0.8, ls="-.", zorder=1)
ax[1].text(3.0e1, 15.5, r"$W=35$ eV (electron plateau)", fontsize=6.6,
           color="0.35")
ax[1].text(1.2e8, 3.0e8, "upper bound\non $W$", fontsize=6.6, color="0.32",
           ha="left", va="top")
ax[1].set_title(r"(b)  energy cost per ionization", fontsize=9)

fig.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / ("fig_ionizations_per_primary." + ext), dpi=300)
print("wrote", OUT / "fig_ionizations_per_primary.pdf")

# ---------------------------------------------------------------------------
# companion numbers printed for the write-up
# ---------------------------------------------------------------------------
print("\nN_ion per primary  (z_i = 20 / z_i = 10)")
print("  %-12s %-22s %s" % ("E [eV]", "electron", "photon"))
ie20, ie10 = int(np.argmin(abs(ze - 20))), int(np.argmin(abs(ze - 10)))
ip20, ip10 = int(np.argmin(abs(zp - 20))), int(np.argmin(abs(zp - 10)))
Ye20 = ele["Y"][ie20, n0:] + ele["Ype"][ie20, n0:]
Ye10 = ele["Y"][ie10, n0:] + ele["Ype"][ie10, n0:]
for Et in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12):
    je = int(np.argmin(abs(np.log(K) - np.log(Et))))
    jp = int(np.argmin(abs(np.log(E) - np.log(Et))))
    print("  %-12.3g %-10.4g %-11.4g %-10.4g %.4g"
          % (Et, Ye20[je], Ye10[je], pho["Ne"][ip20, jp], pho["Ne"][ip10, jp]))
