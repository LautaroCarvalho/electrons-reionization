"""Stage 7 figure: electron range in a COMPLETELY IONIZED medium.

Same three panels as ``stage6_figure.py``, computed from
``redo_range_table_ionized.npz`` (x_e = 1, T_gas = 1e4 K).  Writes
manuscript/figures_redo/fig_electron_range_ionized.{pdf,png}.

If the neutral table ``redo_range_table.npz`` is present it is overplotted as
thin dashed curves in panels (a) and (c), because the interesting statement is
not the ionized curve on its own but how much shorter it is: switching the
medium on removes the two collisional channels that stop the electron below
1 keV and replaces them with a Coulomb rate 1e4 times larger, which is a net
shortening at every energy.
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

d = np.load(HERE / "redo_range_table_ionized.npz")
K, z = d["K"], d["z"]
chi, ell = d["chi"], d["ell"]
tcool, cooled, below = d["tcool"], d["cooled"], d["below"]
LAM = d["lambda_c"]
K_TERM = float(d["k_term_eV"])
T_GAS = float(d["t_gas"])
MPC = L.MEGAPARSEC_MKS

NEU = HERE / "redo_range_table.npz"
dn = np.load(NEU) if NEU.exists() else None

ok = ~below                        # nodes injected above the thermal floor

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
hb = [hubble_comoving_mpc(v) for v in Z_SHOW]
ax[0].axhspan(min(hb), max(hb), color="#c6dbef", alpha=0.75, lw=0, zorder=0)

iz20 = int(np.argmin(np.abs(z - 20.0)))
lo = np.minimum(np.sqrt(LAM[0] * chi[iz20]), chi[iz20])
hi = np.minimum(np.sqrt(LAM[1] * chi[iz20]), chi[iz20])
ax[0].fill_between(K[ok], lo[ok], hi[ok], color="0.70", alpha=0.5, lw=0, zorder=0)

for c, zi in zip(CMAP, Z_SHOW):
    iz = int(np.argmin(np.abs(z - zi)))
    ax[0].loglog(K[ok], chi[iz][ok], color=c, lw=1.6, zorder=3,
                 label=rf"$z_i={z[iz]:.0f}$")
    if dn is not None:
        jz = int(np.argmin(np.abs(dn["z"] - zi)))
        ax[0].loglog(dn["K"], dn["chi"][jz], color=c, lw=0.8, ls="--",
                     alpha=0.55, zorder=2)
    m = (~cooled[iz]) & ok
    if m.any():
        ax[0].plot(K[m], chi[iz][m], "o", mfc="none", mec=c, ms=4.2, mew=0.9,
                   zorder=4)

ax[0].axvline(K_TERM, color="0.45", lw=0.8, ls="-.")
ax[0].text(K_TERM * 1.45, 2.0e-6, r"$K_{\rm term}=1.5\,k_BT_{\rm gas}$",
           fontsize=6.8, color="0.35", rotation=90, va="bottom", ha="left")
ax[0].set_xlabel(r"injection energy $K_{\mathrm{ini}}$ [eV]")
ax[0].set_ylabel(r"comoving range $\chi$ from source [cMpc]")
ax[0].set_xlim(K[0], K[-1])
ax[0].set_ylim(1e-9, 1e4)
ax[0].add_artist(ax[0].legend(loc="upper left", frameon=False,
                              handlelength=1.4, borderaxespad=0.4))
handles = [
    Patch(fc="#c6dbef", ec="none", label=r"comoving $c/H(z_i)$"),
    Patch(fc="0.70", alpha=0.5, ec="none",
          label=r"tangled field $\sqrt{\lambda_c\chi}$, $z_i{=}20$,"
                "\n" r"$\lambda_c = 1$ ckpc $-$ 1 cMpc"),
    Line2D([], [], ls="none", marker="o", mfc="none", mec="0.3", ms=4.2,
           label=r"not cooled by $z=5.5$"),
]
if dn is not None:
    handles.insert(0, Line2D([], [], color="0.3", lw=0.8, ls="--", alpha=0.7,
                             label=r"neutral IGM, $x_e=10^{-4}$"))
ax[0].legend(handles=handles, loc="lower right", frameon=False,
             handlelength=1.4, borderaxespad=0.4)
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
ax[1].set_ylim(1e-6, 1e4)
ax[1].text(0.03, 0.035, r"curve labels: $K_{\mathrm{ini}}$ [eV]",
           transform=ax[1].transAxes, va="bottom", fontsize=7.2, color="0.30")
ax[1].set_title("(b) range vs. redshift", fontsize=9.5)

# ----------------------------------------------------------------- panel (c)
for c, zi in zip(CMAP, Z_SHOW):
    iz = int(np.argmin(np.abs(z - zi)))
    ax[2].loglog(K[ok], (tcool[iz] / hubble_time(z[iz]))[ok], color=c, lw=1.6,
                 label=rf"$z_i={z[iz]:.0f}$")
    if dn is not None:
        jz = int(np.argmin(np.abs(dn["z"] - zi)))
        ax[2].loglog(dn["K"], dn["tcool"][jz] / hubble_time(dn["z"][jz]),
                     color=c, lw=0.8, ls="--", alpha=0.55)
ax[2].axhline(1.0, color="0.35", lw=0.8, ls="--")
ax[2].text(0.03, 0.915, r"$t_{\rm cool}=t_H(z_i)$", transform=ax[2].transAxes,
           fontsize=7.2, color="0.30", ha="left", va="top")
ax[2].set_xlabel(r"injection energy $K_{\mathrm{ini}}$ [eV]")
ax[2].set_ylabel(r"$t_{\mathrm{cool}}/t_H(z_i)$")
ax[2].set_xlim(K[0], K[-1])
ax[2].set_ylim(1e-12, 3.0)
ax[2].legend(loc="lower right", frameon=False, handlelength=1.4,
             borderaxespad=0.4)
ax[2].set_title("(c) cooling time", fontsize=9.5)

for a in ax:
    a.grid(True, which="major", ls=":", lw=0.4, alpha=0.45)

fig.suptitle(rf"fully ionized medium: $x_e=1$, $n_e=n_p=n_H(z)$, "
             rf"$T_{{\rm gas}}={T_GAS:.0f}$ K", fontsize=9.5, y=1.015)
fig.tight_layout(pad=0.6, w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_electron_range_ionized.{ext}", dpi=200,
                bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "fig_electron_range_ionized.pdf")

# ------------------------------------------------------------------ summary
print()
hdr = " | ".join(f"chi(z={v:.0f})" for v in Z_SHOW)
print(f"{'K_ini [eV]':>12} | {hdr}      [cMpc]" + ("   | ionized/neutral" if dn is not None else ""))
for K0 in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9, 1e11, 1e13):
    ik = int(np.argmin(np.abs(K - K0)))
    row = " | ".join(f"{chi[int(np.argmin(np.abs(z - v))), ik]:11.4e}" for v in Z_SHOW)
    rat = ""
    if dn is not None:
        jk = int(np.argmin(np.abs(dn["K"] - K0)))
        rat = "   | " + " ".join(
            f"{chi[int(np.argmin(np.abs(z-v))), ik]/dn['chi'][int(np.argmin(np.abs(dn['z']-v))), jk]:6.3f}"
            for v in Z_SHOW)
    print(f"{K[ik]:12.4g} | {row}{rat}")
print()
for zi in Z_SHOW:
    iz = int(np.argmin(np.abs(z - zi)))
    ikm = int(np.argmax(chi[iz]))
    extra = ""
    if dn is not None:
        jz = int(np.argmin(np.abs(dn["z"] - zi)))
        extra = f"; neutral plateau {dn['chi'][jz].max():8.4g} cMpc"
    print(f"z_i={z[iz]:5.1f}: plateau chi = {chi[iz][ikm]:9.4g} cMpc "
          f"(proper path {ell[iz][ikm]:8.4g} Mpc); comoving c/H = "
          f"{hubble_comoving_mpc(z[iz]):7.4g} cMpc; "
          f"t_cool/t_H = {tcool[iz][ikm]/hubble_time(z[iz]):.3f}"
          f"{extra}; {int(((~cooled[iz]) & ok).sum())} window-truncated")
