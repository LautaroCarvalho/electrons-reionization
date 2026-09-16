"""Stage 4: regenerate Fig. 4 of manuscript_13page from the independent table.

Writes to manuscript/figures_redo/ so that nothing already in the project is
overwritten; the published figure can be compared side by side.

Two figures are produced.

1. ``fig_photon_yield_redo``  -- ionizations per injected electron and the
   W-value.  If the table carries the cosmological photon-transport columns
   (``Ype``/``Ypt``, written by the current stage2_cascade.py) they are drawn
   alongside the hard-band curves, so the size of the band-ceiling
   approximation is visible directly.

2. ``fig_heat_per_electron``  -- the heat deposited PER INJECTED PRIMARY
   ELECTRON.  This replaces the earlier heat-per-ionization figure.  The two
   are not the same statement: Q = H/Y answers "how much gas heating comes with
   each ionization", which is the quantity that enters the heat-ionization lock,
   whereas H(K_ini) answers "how much of the energy of one cosmic-ray electron
   actually ends up as heat", which is the quantity that a CR-electron energy
   budget needs.  Panel (b) gives the same thing as a fraction of the injected
   energy, which is the readable form since H itself spans thirteen decades.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "manuscript" / "figures_redo"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(HERE / "redo_cascade_table.npz")
old = np.load(HERE.parent / "photon_cascade_table.npz")
n0 = int(d["nlow"])
K, z = d["K"][n0:], d["z"]
HAS_TR = "Ype" in d.files              # transport columns present?

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "legend.fontsize": 8.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

styles = {20.0: ("#2166ac", "-"), 10.0: ("#b2182b", "--")}


def col(key, iz):
    return np.maximum(d[key][iz, n0:], 0.0)


# =====================================================================
# 1.  Ionization yield and W-value
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(7.15, 3.05))

for zi, (color, ls) in styles.items():
    iz = int(np.argmin(np.abs(z - zi)))
    Ye = col("Y", iz)
    Tf = Ye + col("Ypf", iz)
    Tm = Ye + col("Ypm", iz)
    m = Ye > 0
    ax[0].loglog(K[m], Ye[m], color=color, ls=":", lw=1.45,
                 label=rf"electron impact, $z_i={zi:.0f}$")
    ax[0].loglog(K[m], Tf[m], color=color, ls=ls, lw=2.05,
                 label=rf"total, hard band, $z_i={zi:.0f}$")
    ax[0].fill_between(K[m], Tf[m], Tm[m], color=color, alpha=0.17, lw=0)
    if HAS_TR:
        Te = Ye + col("Ype", iz)
        ax[0].loglog(K[m], Te[m], color=color, ls="-.", lw=1.55,
                     label=rf"total, transported, $z_i={zi:.0f}$")
    # published curve, for visual comparison
    op = np.maximum(old["Ye"][iz] + old["Yph_fixed"][iz], 1e-300)
    ax[0].loglog(K, op, color="0.45", lw=0.8, alpha=0.85, zorder=0)

    ax[1].loglog(K[m], K[m] / Ye[m], color=color, ls=":", lw=1.45)
    ax[1].loglog(K[m], K[m] / Tf[m], color=color, ls=ls, lw=2.05,
                 label=rf"$z_i={zi:.0f}$")
    ax[1].fill_between(K[m], K[m] / Tm[m], K[m] / Tf[m],
                       color=color, alpha=0.17, lw=0)
    if HAS_TR:
        ax[1].loglog(K[m], K[m] / (Ye + col("Ype", iz))[m],
                     color=color, ls="-.", lw=1.55)
    ax[1].loglog(K, K / op, color="0.45", lw=0.8, alpha=0.85, zorder=0)

ax[0].plot([], [], color="0.45", lw=0.8, label="published table")
ax[0].set(xlabel=r"injection energy $K_{\rm ini}$ [eV]",
          ylabel=r"ionizations / injected $e^-$", xlim=(1e2, 1e13), ylim=(1, 2e7))
ax[1].set(xlabel=r"injection energy $K_{\rm ini}$ [eV]",
          ylabel=r"$W=K_{\rm ini}/Y$ [eV]", xlim=(1e2, 1e13), ylim=(1e1, 1e8))
ax[1].axhline(35.0, color="0.35", lw=1.0, ls="-.", label="35 eV neutral-gas limit")
for a, tag in zip(ax, ("(a)", "(b)")):
    a.axvline(13.6, color="0.5", lw=0.8)
    for xv in (1e3, 1e5, 3e8):
        a.axvline(xv, color="0.6", lw=0.7, ls=":")
    a.grid(which="major", alpha=0.22, lw=0.55)
    a.text(0.02, 0.04, tag, transform=a.transAxes, fontweight="bold")
ax[0].legend(loc="upper left", framealpha=0.92, fontsize=6.8)
ax[1].legend(loc="upper left", framealpha=0.92)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "fig_photon_yield_redo.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_photon_yield_redo.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_photon_yield_redo.pdf")


# =====================================================================
# 2.  Heat deposited PER INJECTED PRIMARY ELECTRON
# =====================================================================
# Panel (a) is drawn on the FULL K grid, sub-threshold nodes included.  Those
# nine nodes below 13.6 eV are the whole point of the panel: an electron that
# cannot ionize and (below 10.2 eV) cannot even excite has nowhere to put its
# energy except Coulomb heat, so H must equal K_ini there exactly.  The table
# gives H/K_ini = 0.9753 at 0.3 eV rising to 0.998 at 8.4 eV, the deficit being
# precisely the 7.4 meV of residual kinetic energy at the thermal floor
# (1.5 k_B T_CMB at z = 20).  That analytic anchor is what makes the panel
# unambiguously a plot of ENERGY and not of a particle count, and it was
# cropped off by the old xlim.
#
# The second guard against the same confusion is the thin grey curve 13.6 Y:
# because Q = H/Y is nearly constant (5.9-6.6 eV over the plateau), log H(K)
# is just log Y(K) shifted by a constant, so on its own panel (a) looks like a
# rescaled copy of the yield figure.  Drawing the ionization-potential
# expenditure next to it separates the two by the factor 13.6/Q = 2.06 and
# makes the distinction visible.
KA = d["K"]                                  # full grid, K_LOW included


def cola(key, iz):
    return np.maximum(d[key][iz], 0.0)


fig2, bx = plt.subplots(1, 3, figsize=(10.4, 3.05))

for zi, (color, ls) in styles.items():
    iz = int(np.argmin(np.abs(z - zi)))
    Hq = cola("Hq", iz)                      # electron branch, all generations
    Ht = Hq + cola("Hpf", iz)                # + IC photoelectron cascades
    Yt = cola("Y", iz) + cola("Ypf", iz)     # for the 13.6 Y reference curve
    m = Hq > 0

    bx[0].loglog(KA[m], Hq[m], color=color, ls=":", lw=1.45,
                 label=rf"electron impact, $z_i={zi:.0f}$")
    bx[0].loglog(KA[m], Ht[m], color=color, ls=ls, lw=2.05,
                 label=rf"incl. IC photons, $z_i={zi:.0f}$")
    bx[1].loglog(KA[m], (Hq / KA)[m], color=color, ls=":", lw=1.45)
    bx[1].loglog(KA[m], (Ht / KA)[m], color=color, ls=ls, lw=2.05,
                 label=rf"$z_i={zi:.0f}$")
    if HAS_TR:
        He = Hq + cola("Hpe", iz)
        Hx = Hq + cola("Hpt", iz)
        bx[0].loglog(KA[m], He[m], color=color, ls="-.", lw=1.55,
                     label=rf"transported, $z_i={zi:.0f}$")
        bx[0].fill_between(KA[m], He[m], Hx[m], color=color, alpha=0.17, lw=0)
        bx[1].loglog(KA[m], (He / KA)[m], color=color, ls="-.", lw=1.55)
        bx[1].fill_between(KA[m], (He / KA)[m], (Hx / KA)[m],
                           color=color, alpha=0.17, lw=0)

    # the ionization-potential expenditure, for contrast: this is the curve
    # that WOULD be a rescaled ionization count, and H sits a factor ~2 below it
    mY = Yt > 0
    bx[0].loglog(KA[mY], 13.6057 * Yt[mY], color=color, lw=0.8, alpha=0.55,
                 zorder=0)

    bx[2].semilogx(KA[m], ((13.6057 * cola("Y", iz) + Hq + cola("Xq", iz)) / KA)[m],
                   color=color, ls=ls, lw=1.9, label=rf"$z_i={zi:.0f}$")

# the ceiling: every eV of the primary thermalized
bx[0].loglog(KA, KA, color="0.35", lw=0.9, ls="--", zorder=0,
             label=r"$H=K_{\rm ini}$ (all energy thermalized)")
bx[0].plot([], [], color="0.45", lw=0.8, alpha=0.7,
           label=r"$13.6\,Y$ (ionization potential)")

bx[0].set(xlabel=r"$K_{\rm ini}$ [eV]",
          ylabel=r"heat $H$ per injected $e^-$ [eV]",
          xlim=(0.25, 1e13), ylim=(0.1, 3e13))
bx[1].set(xlabel=r"$K_{\rm ini}$ [eV]",
          ylabel=r"heat fraction $f_{\rm heat}=H/K_{\rm ini}$",
          xlim=(0.25, 1e13), ylim=(1e-6, 2.0))
bx[1].axhline(1.0, color="0.4", lw=0.8, ls="--")
bx[2].set(xlabel=r"$K_{\rm ini}$ [eV]",
          ylabel=r"$(13.6\,Y+H+X)/K_{\rm ini}$", xlim=(0.25, 1e13), ylim=(0, 1.05))
bx[2].axhline(1.0, color="0.4", lw=0.8, ls="--")
for a, tag in zip(bx, ("(a)", "(b)", "(c)")):
    a.axvline(13.6057, color="0.5", lw=0.8)
    for xv in (1e3, 1e5, 3e8):
        a.axvline(xv, color="0.6", lw=0.7, ls=":")
    a.grid(which="major", alpha=0.22, lw=0.55)
    a.text(0.02, 0.045, tag, transform=a.transAxes, fontweight="bold")
bx[0].legend(loc="upper left", fontsize=6.2, framealpha=0.92)
bx[1].legend(loc="lower left", fontsize=7.5, framealpha=0.92)
bx[2].legend(loc="lower left", fontsize=7.5, framealpha=0.92)
fig2.tight_layout(w_pad=1.0)
fig2.savefig(OUT / "fig_heat_per_electron.pdf", bbox_inches="tight")
fig2.savefig(OUT / "fig_heat_per_electron.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_heat_per_electron.pdf")

# ---- a short table of the same numbers, for the text ----------------------
print("\nCoulomb heat per injected electron  H [eV]  and  f_heat = H/K_ini")
hdr = f"{'K_ini [eV]':>11}"
for zi in (20.0, 10.0):
    hdr += f" | {'H_e':>10} {'H_tot':>10} {'f_heat':>8}"
print(hdr + "     (z_i = 20 | z_i = 10)")
for K0 in (1.0, 8.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12):
    ik = int(np.argmin(np.abs(KA - K0)))
    row = f"{KA[ik]:11.3g}"
    for zi in (20.0, 10.0):
        iz = int(np.argmin(np.abs(z - zi)))
        He = cola("Hq", iz)[ik]
        Ht = He + cola("Hpf", iz)[ik]
        row += f" | {He:10.4g} {Ht:10.4g} {Ht/KA[ik]:8.4f}"
    print(row)


# =====================================================================
# 3.  IONIZATIONS PER INJECTED PRIMARY ELECTRON
# =====================================================================
# The exact counterpart of figure 2, panel for panel, with the ionization
# count in place of the Coulomb heat:
#
#   (a) the absolute number of ionizations per injected electron, against the
#       ceiling Y = K_ini/13.6057 that every eV of the primary would give if it
#       were all spent on binding energy.  The contrast curve here is N_1, the
#       primary's own ionizations: the gap between N_1 and Y_e is the
#       collisional shower, and the gap between Y_e and Y_tot is the IC-photon
#       branch, so the panel shows the three contributions at once.
#
#   (b) the fraction of the injected energy that ends as ionization potential,
#       f_ion = 13.6057 Y / K_ini.  This is the strict analogue of f_heat, and
#       its ceiling is 1 with the physically reachable maximum at
#       13.6057/W_min = 13.6057/35.38 = 0.385.
#
#   (c) the share of the LOCALLY DEPOSITED energy that goes into ionization,
#       13.6057 Y_e / (13.6057 Y_e + H_q + X_q), electron branch only, on the
#       same linear 0-1 axis as panel 2(c).  Panel 2(c) says how much of the
#       primary's energy is deposited at all; this one says how that deposit is
#       divided.  The two together are the whole story: the partition is
#       essentially universal (0.37-0.39 above 100 eV, at both redshifts) while
#       the deposited amount collapses by five decades.
fig3, cx = plt.subplots(1, 3, figsize=(10.4, 3.05))

W_MIN = 35.38                                   # eV, from the same table
F_ION_MAX = 13.6057 / W_MIN

for zi, (color, ls) in styles.items():
    iz = int(np.argmin(np.abs(z - zi)))
    N1 = cola("N1", iz)
    Ye = cola("Y", iz)
    Yt = Ye + cola("Ypf", iz)
    Xq = cola("Xq", iz)
    Hq = cola("Hq", iz)
    m = Yt > 0
    mn = N1 > 0

    cx[0].loglog(KA[mn], N1[mn], color=color, lw=0.8, alpha=0.55, zorder=0)
    cx[0].loglog(KA[m], Ye[m], color=color, ls=":", lw=1.45,
                 label=rf"electron impact, $z_i={zi:.0f}$")
    cx[0].loglog(KA[m], Yt[m], color=color, ls=ls, lw=2.05,
                 label=rf"incl. IC photons, $z_i={zi:.0f}$")

    cx[1].loglog(KA[m], (13.6057 * Ye / KA)[m], color=color, ls=":", lw=1.45)
    cx[1].loglog(KA[m], (13.6057 * Yt / KA)[m], color=color, ls=ls, lw=2.05,
                 label=rf"$z_i={zi:.0f}$")

    if HAS_TR:
        Te = Ye + cola("Ype", iz)
        Tx = Ye + cola("Ypt", iz)
        cx[0].loglog(KA[m], Te[m], color=color, ls="-.", lw=1.55,
                     label=rf"transported, $z_i={zi:.0f}$")
        cx[0].fill_between(KA[m], Te[m], Tx[m], color=color, alpha=0.17, lw=0)
        cx[1].loglog(KA[m], (13.6057 * Te / KA)[m], color=color, ls="-.", lw=1.55)
        cx[1].fill_between(KA[m], (13.6057 * Te / KA)[m],
                           (13.6057 * Tx / KA)[m], color=color, alpha=0.17, lw=0)

    dep = 13.6057 * Ye + Hq + Xq
    md = dep > 0
    cx[2].semilogx(KA[md], (13.6057 * Ye / dep)[md],
                   color=color, ls=ls, lw=1.9, label=rf"$z_i={zi:.0f}$")

# the ceiling: every eV of the primary spent on binding energy
cx[0].loglog(KA, KA / 13.6057, color="0.35", lw=0.9, ls="--", zorder=0,
             label=r"$Y=K_{\rm ini}/13.6$ eV (all energy into ionization)")
cx[0].plot([], [], color="0.45", lw=0.8, alpha=0.7,
           label=r"$N_1$ (primary alone, no shower)")

cx[0].set(xlabel=r"$K_{\rm ini}$ [eV]",
          ylabel=r"ionizations $Y$ per injected $e^-$",
          xlim=(0.25, 1e13), ylim=(1e-4, 3e12))
cx[1].set(xlabel=r"$K_{\rm ini}$ [eV]",
          ylabel=r"$f_{\rm ion}=13.6\,Y/K_{\rm ini}$",
          xlim=(0.25, 1e13), ylim=(1e-6, 2.0))
cx[1].axhline(1.0, color="0.4", lw=0.8, ls="--")
cx[1].axhline(F_ION_MAX, color="0.35", lw=1.0, ls="-.",
              label=rf"$13.6/W_{{\rm min}}={F_ION_MAX:.3f}$")
cx[2].set(xlabel=r"$K_{\rm ini}$ [eV]",
          ylabel=r"$13.6\,Y_{\rm e}/(13.6\,Y_{\rm e}+H+X)$",
          xlim=(0.25, 1e13), ylim=(0, 1.05))
cx[2].axhline(F_ION_MAX, color="0.4", lw=0.8, ls="--")
for a, tag in zip(cx, ("(a)", "(b)", "(c)")):
    a.axvline(13.6057, color="0.5", lw=0.8)
    for xv in (1e3, 1e5, 3e8):
        a.axvline(xv, color="0.6", lw=0.7, ls=":")
    a.grid(which="major", alpha=0.22, lw=0.55)
    a.text(0.02, 0.045, tag, transform=a.transAxes, fontweight="bold")
cx[0].legend(loc="upper left", fontsize=6.2, framealpha=0.92)
cx[1].legend(loc="lower left", fontsize=7.0, framealpha=0.92)
cx[2].legend(loc="lower right", fontsize=7.5, framealpha=0.92)
fig3.tight_layout(w_pad=1.0)
fig3.savefig(OUT / "fig_ionization_per_electron.pdf", bbox_inches="tight")
fig3.savefig(OUT / "fig_ionization_per_electron.png", dpi=220, bbox_inches="tight")
print("wrote", OUT / "fig_ionization_per_electron.pdf")

print("\nIonizations per injected electron, and the two fractions")
print(f"{'K_ini [eV]':>11} | {'N1':>9} {'Y_e':>10} {'Y_tot':>11} {'f_ion':>7} "
      f"{'ion share':>10} {'Y_e/N1':>7}   (z_i = 20)")
for K0 in (1.0, 20.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12):
    ik = int(np.argmin(np.abs(KA - K0)))
    iz = int(np.argmin(np.abs(z - 20.0)))
    N1 = cola("N1", iz)[ik]; Ye = cola("Y", iz)[ik]
    Yt = Ye + cola("Ypf", iz)[ik]
    dep = 13.6057 * Ye + cola("Hq", iz)[ik] + cola("Xq", iz)[ik]
    print(f"{KA[ik]:11.3g} | {N1:9.4g} {Ye:10.4g} {Yt:11.4g} "
          f"{13.6057*Yt/KA[ik]:7.4f} {13.6057*Ye/dep if dep>0 else 0:10.4f} "
          f"{Ye/max(N1,1e-30):7.3f}")
