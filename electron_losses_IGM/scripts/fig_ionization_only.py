"""Figure: K/K_ini for collisional ionization of H (RBED) with mean loss B + <ε> (notebook cell 19).

Physics  igm_losses.losses (processes: ["ionization"]), integrate.evolve (Radau, D05, D06).
Choices  RBED, B = 13.6057 eV (E16); Furlanetto & Stoever 2010 secondary spectrum (E17); stop at R_H = 13.598 eV (D09); between 13.598 and 13.6057 eV the ionization rate is zero.
Writes   figures/igm_ionization_only[_xe*].png/.pdf and provenance/figures/igm_ionization_only[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_ionization_only.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "ionization_only"
SHOWS = "K/K_ini for collisional ionization of H (RBED) with mean loss B + <ε>"
CHOICES = ['RBED, B = 13.6057 eV (E16)', 'Furlanetto & Stoever 2010 secondary spectrum (E17)', 'stop at R_H = 13.598 eV (D09, D21); between 13.598 and 13.6057 eV the ionization rate is zero, so K levels off at B and the event never fires (note on the figure)']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["ionization"], floor_eV=K.R_H_eV,
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["ionization"][plotting.LANG]))
        fig.text(0.5, 0.005, L("plateau_B", B=f"{F.P['B_ion_RBED']:g}"), ha="center", va="bottom", fontsize=9)
        fig.subplots_adjust(bottom=0.08)
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
