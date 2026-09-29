"""Figure: K/K_ini for bremsstrahlung on neutral H (BREMS CS_int) and on protons (Karzas & Latter + B&G) (notebook cell 21).

Physics  igm_losses.losses (processes: ["bremsstrahlung"]), integrate.evolve (Radau, D05, D06).
Choices  C¹ splices (D19); e–e neglected (A15); tail k/T > 0.95 ignored; floor K_floor = 10.2 eV.
Writes   figures/igm_bremsstrahlung_only[_xe*].png/.pdf and provenance/figures/igm_bremsstrahlung_only[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_bremsstrahlung_only.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "bremsstrahlung_only"
SHOWS = "K/K_ini for bremsstrahlung on neutral H (BREMS CS_int) and on protons (Karzas & Latter + B&G)"
CHOICES = ['C¹ splices (D19)', 'e–e neglected (A15)', 'tail k/T > 0.95 ignored', 'floor K_floor = 10.2 eV']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=["bremsstrahlung"], floor_eV=None,
                                  title=L("title_process_only", proc=plotting.LABELS["process"]["bremsstrahlung"][plotting.LANG]))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
