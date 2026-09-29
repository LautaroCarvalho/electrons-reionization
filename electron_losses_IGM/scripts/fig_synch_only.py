"""Figure: K/K_ini vs cosmic time and vs z for synchrotron losses only, B = B0 (1+z)², isotropic pitch angles (notebook cell 8).

Physics  igm_losses.losses (processes: ["synchrotron"]), integrate.evolve (Radau, D05, D06).
Choices  isotropic pitch-angle average (A08); floor K_floor = 10.2 eV (A07, D09).
Writes   figures/igm_synch_only[_xe*].png/.pdf and provenance/figures/igm_synch_only[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_synch_only.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "synch_only"
SHOWS = "K/K_ini vs cosmic time and vs z for synchrotron losses only, B = B0 (1+z)², isotropic pitch angles"
CHOICES = ['isotropic pitch-angle average (A08)', 'floor K_floor = 10.2 eV (A07, D09)']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["synchrotron"], floor_eV=None,
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["synchrotron"][plotting.LANG]))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
