"""Figure: K/K_ini for adiabatic (Hubble) losses only, from z = 30 to 5.5 (z_init = 30 as in the notebook cell 9) (notebook cell 9).

Physics  igm_losses.losses (processes: ["adiabatic"]), integrate.evolve (Radau, D05, D06).
Choices  z_init = 30 (figure parameter, cell 9); floor K_floor = 10.2 eV.
Writes   figures/igm_adiabatic_only[_xe*].png/.pdf and provenance/figures/igm_adiabatic_only[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_adiabatic_only.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "adiabatic_only"
SHOWS = "K/K_ini for adiabatic (Hubble) losses only, from z = 30 to 5.5 (z_init = 30 as in the notebook cell 9)"
CHOICES = ['z_init = 30 (figure parameter, cell 9)', 'floor K_floor = 10.2 eV']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["adiabatic"], floor_eV=None,
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["adiabatic"][plotting.LANG]))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
