"""Figure: K/K_ini with all seven processes (adiabatic, synchrotron, IC, Coulomb, excitation, ionization, bremsstrahlung) (notebook cell 23).

Physics  igm_losses.losses (processes: all), integrate.evolve (Radau, D05, D06).
Choices  bremsstrahlung included in the total (D14); floor K_floor = 10.2 eV (A07).
Writes   figures/igm_total_cooling[_xe*].png/.pdf and provenance/figures/igm_total_cooling[_xe*].json (x_e sweep if listed).
Run      python3 electron_losses_IGM/scripts/fig_total_cooling.py   (from the project root)
"""

import figlib as F
from figlib import L, plotting, K
from igm_losses import excitation  # noqa: F401  (threshold data for the excitation figure)

SLUG = "total_cooling"
SHOWS = "K/K_ini with all seven processes (adiabatic, synchrotron, IC, Coulomb, excitation, ionization, bremsstrahlung)"
CHOICES = ['bremsstrahlung included in the total (D14)', 'floor K_floor = 10.2 eV (A07)']


def main():
    for x_e in F.xe_values(SLUG):
        fig, _ = F.k_ratio_figure(SLUG, x_e, include=None, floor_eV=None,
                                  title=L("title_all"))
        F.save(fig, SLUG, x_e, SHOWS, "trajectory K(t) from the rate registry; all rates re-implemented (see igm_losses)",
               CHOICES)


if __name__ == "__main__":
    main()
