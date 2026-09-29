"""Figures from the augmented ODE: accumulated energy lost per process (cell 37) and energy lost per fixed Δz bin (cell 39, E14).

Physics  integrate.evolve(augmented=True) integrates dE_i/dt = L_i for each process together with K.
cell 37  lost_i(z)/K_ini vs z, one panel per initial energy; x axis 10 → 7 as the notebook (C44: display choice kept).
cell 39  ΔE_i in bins of fixed Δz = 0.1 (E14): ΔE_i,k = lost_i(z_k+1) − lost_i(z_k), plotted at bin centres (does not
         depend on the time grid, unlike the notebook's np.diff).
Components whose maximum is below 1e-10 of K_ini (37) or 1e-10 eV (39) are not drawn (log axes).
"""

import numpy as np

import figlib as F
from figlib import L, plt, plotting
from igm_losses import losses

PB = "electron_losses_IGM/scripts/fig_accumulated.py::"
DRAW_MIN = 1e-10   # display threshold of the docstring: fraction of K_ini (37) or eV (39); not a physical input


def fig37(x_e):
    slug = "accumulated_losses"
    runs = F.run(slug, x_e, augmented=True)
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 16), sharex=True)
    for a, (K0, o) in zip(axes, runs):
        for p in losses.PROCESSES:
            y = o["lost_" + p] / K0
            if y.max() > DRAW_MIN:
                a.plot(o["z"], y, color=plotting.PROCESS_COLORS[p], label=plotting.LABELS["process"][p][plotting.LANG])
        a.set_yscale("log"); a.set_ylim(bottom=DRAW_MIN); a.set_ylabel(L("lost_fraction")); a.set_title(F.exp_label(K0), fontsize=11)
    # C44: 10 → 7 as the notebook; every trajectory of this figure reaches the floor at z ≥ 7.023 (numbers.json
    # z_therm_*, all x_e), so the curves are flat beyond z = 7 and nothing is cut.
    axes[-1].set_xlim(10.0, 7.0); axes[-1].set_xlabel(L("redshift")); axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, "accumulated energy lost to each process divided by K_ini vs z, for four initial energies",
           "augmented ODE", ["x axis 10 → 7 as the notebook (display choice)", "label corrected: fraction, not eV (E10)"],
           produced_by=PB + "fig37")


def fig39(x_e):
    slug = "step_losses"
    s = F.spec(slug)
    dz = s["delta_z"]
    runs = F.run(slug, x_e, augmented=True)
    zi, zf = F.z_range(s)
    edges = np.arange(zi, zf - 1e-12, -dz)
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 16), sharex=True)
    for a, (K0, o) in zip(axes, runs):
        zc = 0.5 * (edges[:-1] + edges[1:])
        for p in losses.PROCESSES:
            acc = np.interp(edges[::-1], o["z"][::-1], o["lost_" + p][::-1])[::-1]
            dE = np.diff(acc)
            if dE.max() > DRAW_MIN:
                a.step(zc, dE, where="mid", color=plotting.PROCESS_COLORS[p], label=plotting.LABELS["process"][p][plotting.LANG])
        a.set_yscale("log"); a.set_ylabel(L("lost_per_dz", dz=f"{dz:g}")); a.set_title(F.exp_label(K0), fontsize=11)
    axes[-1].set_xlim(10.0, 7.0)   # C44: same reason as in fig37 (all z_therm ≥ 7.023)
    axes[-1].set_xlabel(L("redshift")); axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, f"energy lost to each process in fixed Δz = {dz:g} bins vs z, for four initial energies",
           "differences of the accumulated losses at fixed z edges", ["fixed Δz bins instead of grid steps (E14)",
                                                                       "x axis 10 → 7 as the notebook"], produced_by=PB + "fig39",
           extra={"delta_z": dz})


def main():
    for f, slug in ((fig37, "accumulated_losses"), (fig39, "step_losses")):
        for x_e in F.xe_values(slug):
            f(x_e)


if __name__ == "__main__":
    main()
