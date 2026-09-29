"""Figure (objective 1b): fraction of K_ini that each process takes before the electron reaches K_floor, vs K_ini,
medium fixed at z_init (A19, A20; continuous slowing down), one figure per x_e of the A04 sweep.

Physics  em_cascades.fractions.integrated_fractions: f_i(K_ini) = (1/K_ini) ∫_{K_floor}^{K_ini} (L_i/L_tot) dK.
         The unlost part K_floor/K_ini is drawn too, so the curves add up to 1 at every K_ini.
Writes   figures/cas_loss_fraction_integrated[_xe*].{png,pdf} and provenance/figures/cas_loss_fraction_integrated[_xe*].json
Run      python3 EM_cascades/scripts/fig_loss_fraction_integrated.py   (from the project root)
"""

import figlib as F
from figlib import L, P, plt, plotting
from em_cascades import fractions as FR
from igm_losses import losses

SLUG = "loss_fraction_integrated"
PB = "EM_cascades/scripts/fig_loss_fraction_integrated.py::main"


def main():
    Kg = FR.energy_grid(F.spec(SLUG))
    for x_e in F.xe_values(SLUG):
        f = FR.integrated_fractions(Kg, x_e)
        fig, ax = plt.subplots(figsize=(9, 6))
        for p in losses.PROCESSES:
            ax.plot(Kg, f[p], color=plotting.PROCESS_COLORS[p], label=plotting.process_name(p))
        ax.plot(Kg, P["K_floor"] / Kg, "k:", lw=1.5, label=L("unlost"))
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)          # C44: a fraction lies in [0, 1]
        ax.set_xlim(Kg[0], Kg[-1])        # C44: the whole grid (K_floor … K_max of the spec)
        ax.set_xlabel(L("initial_energy_eV")); ax.set_ylabel(L("integrated_fraction")); ax.set_title(F.title(x_e), fontsize=11)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        F.save(fig, SLUG, x_e, "fraction of K_ini lost to each process down to K_floor vs K_ini, medium fixed at z_init",
               "em_cascades.fractions.integrated_fractions", ["energy integral at fixed z (checked against the time ODE)",
                                                              "unlost part K_floor/K_ini drawn", "z fixed (A19)"],
               PB, extra={"K_grid": F.spec(SLUG)["grid"]})


if __name__ == "__main__":
    main()
