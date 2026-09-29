"""Figure: K/K_ini for Coulomb losses on free electrons (Gould 1972), cold plasma (notebook cell 14).

Physics  igm_losses.losses (processes: ["coulomb"]), integrate.evolve (Radau, D05, D06).
Choices  Gould Eq. 5.5 above β = α, Eq. 2.20 below (E18); δ = 1/2; floor K_floor = 10.2 eV.
Writes   figures/igm_coulomb_only[_xe*].png/.pdf and provenance/figures/igm_coulomb_only[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_coulomb_only.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "coulomb_only"
SHOWS = "K/K_ini for Coulomb losses on free electrons (Gould 1972), cold plasma"
CHOICES = ['Gould Eq. 5.5 above β = α, Eq. 2.20 below (E18)', 'δ = 1/2', 'floor K_floor = 10.2 eV']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["coulomb"], floor_eV=None,
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["coulomb"][plotting.LANG]))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
