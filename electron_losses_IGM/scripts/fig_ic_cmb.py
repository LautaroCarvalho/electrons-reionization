"""Figure: K/K_ini for inverse Compton on the CMB with the corrected Klein–Nishina factor (E05) (notebook cell 11).

Physics  igm_losses.losses (processes: ["ic"]), integrate.evolve (Radau, D05, D06).
Choices  F_KN with the factor q (E05); floor K_floor = 10.2 eV (the notebook stopped at k_B T_CMB).
Writes   figures/igm_ic_cmb[_xe*].png/.pdf and provenance/figures/igm_ic_cmb[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_ic_cmb.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "ic_cmb"
SHOWS = "K/K_ini for inverse Compton on the CMB with the corrected Klein–Nishina factor (E05)"
CHOICES = ['F_KN with the factor q (E05)', 'floor K_floor = 10.2 eV (the notebook stopped at k_B T_CMB)']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["ic"], floor_eV=None,
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["ic"][plotting.LANG]))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
