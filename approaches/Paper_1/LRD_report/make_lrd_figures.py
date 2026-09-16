#!/usr/bin/env python3
"""Figures for the LRD cosmic-ray electron report."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

c, e_es, me, sT = 2.99792458e10, 4.803204e-10, 9.1093837e-28, 6.6524587e-25
mec2, erg_eV = me*c**2, 6.241509e11

sites = {                                    # name: (B[G], U_B, U_rad, R[cm], colour)
    "jet funnel":   (0.815, 0.0264, 6.635, 1e16, "#c0392b"),
    "wind shock":   (2.58, 0.265, 26.54, 1e16, "#e67e22"),
    "disc corona":  (100.0, 397.9, 1.22e8, 4.67e12, "#8e44ad"),
    "host SNe":     (1e-4, 3.98e-10, 2.59e-9, 9.26e20, "#2980b9"),
}

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

# ---- panel (a): maximum electron energy, Hillas vs radiative burn-off -------
a = ax[0]
names, EH, EC, cols = [], [], [], []
for nm, (B, UB, Ur, R, col) in sites.items():
    names.append(nm); cols.append(col)
    beta = 1.0 if "SNe" not in nm else 0.033
    EH.append(e_es*B*R*beta*erg_eV)
    EC.append(mec2*np.sqrt(3*e_es*B/(4*sT*(UB+Ur)))*erg_eV)
x = np.arange(len(names))
a.bar(x-0.19, EH, 0.36, color=cols, alpha=0.45, label=r"Hillas  $eBR\beta$")
a.bar(x+0.19, EC, 0.36, color=cols, label=r"burn-off  $t_{\rm acc}=t_{\rm cool}$")
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(names, fontsize=9)
a.set_ylabel(r"$E_{\max}$  [eV]"); a.set_ylim(1e9, 1e20)
a.axhline(1.4e5, ls=":", c="k", lw=1)
a.text(3.42, 1.9e5, r"$K_2$", fontsize=8, ha="right")
a.axhline(4e8, ls="--", c="k", lw=1)
a.text(3.42, 5.5e8, r"$K_4$", fontsize=8, ha="right")
a.set_title("(a)  maximum CR-electron energy per site", fontsize=10)
h = [plt.Rectangle((0,0),1,1,fc="grey",alpha=0.45), plt.Rectangle((0,0),1,1,fc="grey")]
a.legend(h, ["Hillas confinement", "radiative burn-off"], fontsize=8, loc="upper left")

# ---- panel (b): spectrum-averaged energy per ionization ---------------------
b = ax[1]
d = np.load(os.path.join(ROOT, "cascade_traj_table.npz"))
Kt, zt, Yt = d["K"], d["z"], d["Y"]
Kf = np.geomspace(1e2, 1e12, 4001)
def Wbar(iz, p):
    lY = np.log10(np.where(Yt[iz] > 0, Yt[iz], 1e-300))
    YY = 10**np.interp(np.log10(Kf), np.log10(Kt), lY)
    return np.trapz(Kf**(1-p)*Kf, np.log(Kf))/np.trapz(YY*Kf**(-p)*Kf, np.log(Kf))
ps = np.linspace(1.7, 3.2, 60)
for iz, lab, st in ((7, r"$z_i=20$", "-"), (3, r"$z_i=10$", "--"), (1, r"$z_i=7$ (LRD epoch)", "-")):
    b.plot(ps, [Wbar(iz, p) for p in ps], st,
           lw=2.0 if iz == 1 else 1.3, color="#16a085" if iz == 1 else "0.35", label=lab)
for p, v in ((1.8, 888), (2.0, 107), (2.2, 46.3), (2.5, 37.8), (3.0, 37.4)):
    b.plot(p, v, "o", ms=5, mfc="none", mec="k", mew=1.2)
b.plot([], [], "o", ms=5, mfc="none", mec="k", mew=1.2, label="manuscript Table 3 ($z_i=20$)")
b.axvspan(2.0, 2.4, color="#f1c40f", alpha=0.18)
b.text(2.2, 400, "DSA range", ha="center", fontsize=8, color="#7f6000")
b.set_yscale("log"); b.set_xlabel(r"injection index $p$")
b.set_ylabel(r"$\bar W$  [eV per ionization]")
b.set_title("(b)  energy per ionization for a power-law injection", fontsize=10)
b.legend(fontsize=8)
for q in ax: q.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_lrd_cr.pdf"), dpi=200)
print("wrote fig_lrd_cr.pdf")
