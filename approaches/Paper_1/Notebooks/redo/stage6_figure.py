"""Stage 6 figure: distance travelled from the source, vs energy and vs redshift.

Reads ``redo_range_table.npz`` written by ``stage6_range.py`` and writes
manuscript/figures_redo/fig_electron_range.{pdf,png}.

Panels
------
(a) Comoving ballistic range chi(K_ini) for a set of injection redshifts, with
    the comoving Hubble radius c/H(z)(1+z) as a reference band and the
    tangled-field random-walk estimate sqrt(lambda_c chi) for z_i = 20.
(b) The same quantity read the other way: chi(z_i) at fixed injection energy.
(c) The reason for the shape of (a): the time the electron takes to thermalize,
    as a fraction of the Hubble time at injection.  Open circles in (a) and (b)
    mark trajectories that had not thermalized when the integration window
    closed at z = 5.5, so their range is a lower bound.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import redo_common as R                                          # noqa: F401
import igm_losses as L

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "manuscript" / "figures_redo"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(HERE / "redo_range_table.npz")
K, z = d["K"], d["z"]
chi, ell = d["chi"], d["ell"]
tcool, cooled = d["tcool"], d["cooled"]
LAM = d["lambda_c"]
MPC = L.MEGAPARSEC_MKS

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 10.5,
    "legend.fontsize": 7.2, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})


def hubble_time(zz):
    return 1.0 / float(L.Planck18.H(zz).to(L.u.s ** -1).value)


def hubble_comoving_mpc(zz):
    return L.C_LIGHT * hubble_time(zz) * (1.0 + zz) / MPC


Z_SHOW = [20.0, 14.0, 10.0, 7.0]
CMAP = plt.cm.viridis(np.linspace(0.05, 0.80, len(Z_SHOW)))
K_SHOW = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e12]
CMAP_K = plt.cm.plasma(np.linspace(0.03, 0.80, len(K_SHOW)))

fig, ax = plt.subplots(1, 3, figsize=(10.9, 3.45))

# ----------------------------------------------------------------- panel (a)
# The comoving Hubble radius changes by only 1.6x over 7 <= z <= 20, so the
# four individual lines would be indistinguishable: draw the range they span.
hb = [hubble_comoving_mpc(v) for v in Z_SHOW]
ax[0].axhspan(min(hb), max(hb), color="#c6dbef", alpha=0.75, lw=0, zorder=0)

iz20 = int(np.argmin(np.abs(z - 20.0)))
lo = np.minimum(np.sqrt(LAM[0] * chi[iz20]), chi[iz20])
hi = np.minimum(np.sqrt(LAM[1] * chi[iz20]), chi[iz20])
ax[0].fill_between(K, lo, hi, color="0.70", alpha=0.5, lw=0, zorder=0)

for c, zi in zip(CMAP, Z_SHOW):
    iz = int(np.argmin(np.abs(z - zi)))
    ax[0].loglog(K, chi[iz], color=c, lw=1.6, zorder=3,
                 label=rf"$z_i={z[iz]:.0f}$")
    m = ~cooled[iz]
    if m.any():
        ax[0].plot(K[m], chi[iz][m], "o", mfc="none", mec=c, ms=4.2, mew=0.9,
                   zorder=4)

ax[0].set_xlabel(r"injection energy $K_{\mathrm{ini}}$ [eV]")
ax[0].set_ylabel(r"comoving range $\chi$ from source [cMpc]")
ax[0].set_xlim(K[0], K[-1])
ax[0].set_ylim(1e-6, 1e4)
ax[0].add_artist(ax[0].legend(loc="upper left", frameon=False,
                              handlelength=1.4, borderaxespad=0.4))
ax[0].legend(handles=[
    Patch(fc="#c6dbef", ec="none", label=r"comoving $c/H(z_i)$"),
    Patch(fc="0.70", alpha=0.5, ec="none",
          label=r"tangled field $\sqrt{\lambda_c\chi}$, $z_i{=}20$,"
                "\n" r"$\lambda_c = 1$ ckpc $-$ 1 cMpc"),
    Line2D([], [], ls="none", marker="o", mfc="none", mec="0.3", ms=4.2,
           label=r"not cooled by $z=5.5$"),
], loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.4)
ax[0].set_title("(a) range vs. energy", fontsize=9.5)

# ----------------------------------------------------------------- panel (b)
for c, K0 in zip(CMAP_K, K_SHOW):
    ik = int(np.argmin(np.abs(K - K0)))
    ax[1].semilogy(1.0 + z, chi[:, ik], color=c, lw=1.6, zorder=3)
    m = ~cooled[:, ik]
    if m.any():
        ax[1].plot((1.0 + z)[m], chi[m, ik], "o", mfc="none", mec=c, ms=4.2,
                   mew=0.9, zorder=4)
    dy = {6: -8.0, 12: 7.0}.get(int(round(np.log10(K[ik]))), -3.0)
    ax[1].annotate(rf"$10^{{{np.log10(K[ik]):.0f}}}$", (1.0 + z[-1], chi[-1, ik]),
                   color=c, fontsize=7.6, xytext=(4, dy),
                   textcoords="offset points", annotation_clip=False)

zc = np.linspace(z[0], z[-1], 120)
ax[1].semilogy(1.0 + zc, [hubble_comoving_mpc(v) for v in zc],
               color="#2171b5", lw=1.0, ls=":", zorder=2)
ax[1].annotate(r"comoving $c/H(z_i)$", (1.0 + zc[55], hubble_comoving_mpc(zc[55])),
               color="#2171b5", fontsize=7.2, xytext=(0, -12),
               textcoords="offset points", ha="center")
ax[1].set_xlabel(r"injection redshift $1+z_i$")
ax[1].set_ylabel(r"comoving range $\chi$ [cMpc]")
ax[1].set_xlim(1.0 + z[0], 1.0 + z[-1] + 2.2)
ax[1].set_ylim(1e-4, 1e4)
ax[1].text(0.03, 0.035, r"curve labels: $K_{\mathrm{ini}}$ [eV]",
           transform=ax[1].transAxes, va="bottom", fontsize=7.2, color="0.30")
ax[1].set_title("(b) range vs. redshift", fontsize=9.5)

# ----------------------------------------------------------------- panel (c)
for c, zi in zip(CMAP, Z_SHOW):
    iz = int(np.argmin(np.abs(z - zi)))
    ax[2].loglog(K, tcool[iz] / hubble_time(z[iz]), color=c, lw=1.6,
                 label=rf"$z_i={z[iz]:.0f}$")
ax[2].axhline(1.0, color="0.35", lw=0.8, ls="--")
ax[2].text(0.03, 0.915, r"$t_{\rm cool}=t_H(z_i)$", transform=ax[2].transAxes,
           fontsize=7.2, color="0.30", ha="left", va="top")
ax[2].set_xlabel(r"injection energy $K_{\mathrm{ini}}$ [eV]")
ax[2].set_ylabel(r"$t_{\mathrm{cool}}/t_H(z_i)$")
ax[2].set_xlim(K[0], K[-1])
ax[2].set_ylim(1e-9, 3.0)
ax[2].legend(loc="lower right", frameon=False, handlelength=1.4,
             borderaxespad=0.4)
ax[2].set_title("(c) cooling time", fontsize=9.5)

for a in ax:
    a.grid(True, which="major", ls=":", lw=0.4, alpha=0.45)

fig.tight_layout(pad=0.6, w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_electron_range.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "fig_electron_range.pdf")

# ------------------------------------------------------------------ summary
print()
hdr = " | ".join(f"chi(z={v:.0f})" for v in Z_SHOW)
print(f"{'K_ini [eV]':>12} | {hdr}      [cMpc]")
for K0 in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9, 1e11, 1e13):
    ik = int(np.argmin(np.abs(K - K0)))
    row = " | ".join(f"{chi[int(np.argmin(np.abs(z - v))), ik]:11.4e}" for v in Z_SHOW)
    print(f"{K[ik]:12.4g} | {row}")
print()
for zi in Z_SHOW:
    iz = int(np.argmin(np.abs(z - zi)))
    ikm = int(np.argmax(chi[iz]))
    print(f"z_i={z[iz]:5.1f}: plateau chi = {chi[iz][ikm]:9.4g} cMpc "
          f"(proper path {ell[iz][ikm]:8.4g} Mpc); comoving c/H = "
          f"{hubble_comoving_mpc(z[iz]):7.4g} cMpc; "
          f"t_cool/t_H = {tcool[iz][ikm]/hubble_time(z[iz]):.3f}; "
          f"{int((~cooled[iz]).sum())} nodes window-truncated")
