"""Figure: K/K_ini for collisional excitation of H(1s→np), n = 2…10; stops at E_2 = 10.204 eV (notebook cell 17).

Physics  igm_losses.losses (processes: ["excitation"]), integrate.evolve (Radau, D05, D06).
Choices  n = 2…10 (D01); Born+BE ≤ 3 keV, Eq. 5 3–10 keV, Inokuti relativistic > 10 keV (E12); stop at E_2 = 10.204 eV from Stone Table 1 (D09, E19).
Writes   figures/igm_excitation_only[_xe*].png/.pdf and provenance/figures/igm_excitation_only[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_excitation_only.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "excitation_only"
SHOWS = "K/K_ini for collisional excitation of H(1s→np), n = 2…10; stops at E_2 = 10.204 eV"
CHOICES = ['n = 2…10 (D01)', 'Born+BE ≤ 3 keV, Eq. 5 3–10 keV, Inokuti relativistic > 10 keV (E12)', 'stop at E_2 = 10.204 eV from Stone Table 1 (D09, E19)']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["excitation"], floor_eV=excitation.DATA["levels"][2]["E_eV"],
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["excitation"][plotting.LANG]))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
