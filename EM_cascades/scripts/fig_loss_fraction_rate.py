"""Figure (objective 1a): instantaneous fraction L_i/L_tot of the energy-loss rate of each process vs K, medium fixed at
z_init (A19), one figure per x_e of the A04 sweep.

Physics  em_cascades.fractions.rate_fractions (rates from igm_losses.losses, one implementation per process).
Writes   figures/cas_loss_fraction_rate[_xe*].{png,pdf} and provenance/figures/cas_loss_fraction_rate[_xe*].json
Run      python3 EM_cascades/scripts/fig_loss_fraction_rate.py   (from the project root)
"""

import numpy as np

import figlib as F
from figlib import L, plt, plotting
from em_cascades import fractions as FR
from igm_losses import losses

SLUG = "loss_fraction_rate"
PB = "EM_cascades/scripts/fig_loss_fraction_rate.py::main"


def main():
    Kg = FR.energy_grid(F.spec(SLUG))
    for x_e in F.xe_values(SLUG):
        phi = FR.rate_fractions(Kg, x_e)
        fig, ax = plt.subplots(figsize=(9, 6))
        for p in losses.PROCESSES:
            ax.plot(Kg, phi[p], color=plotting.PROCESS_COLORS[p], label=plotting.process_name(p))
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)          # C44: a fraction lies in [0, 1]
        ax.set_xlim(Kg[0], Kg[-1])        # C44: the whole grid (K_floor … K_max of the spec)
        ax.set_xlabel(L("kinetic_energy_eV")); ax.set_ylabel(L("rate_fraction")); ax.set_title(F.title(x_e), fontsize=11)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        F.save(fig, SLUG, x_e, "fraction L_i/L_tot of the energy-loss rate of each process vs kinetic energy, medium fixed at z_init",
               "em_cascades.fractions.rate_fractions", ["all 7 processes drawn (no threshold)", "linear y, log x", "z fixed (A19)"],
               PB, extra={"K_grid": F.spec(SLUG)["grid"]})


if __name__ == "__main__":
    main()
