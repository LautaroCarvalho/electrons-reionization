"""Figure (objective 2): number of H ionizations produced by an electron and its whole cascade down to K_floor, vs
K_ini, medium fixed at z_init (A19), one figure per x_e of the A04 sweep.

Curves  collisional C(K_ini) (electron-impact ionizations not below an IC photon), secondary photoionization Ph(K_ini)
        (IC photons in the window [R_H, E_max] of A22 plus everything their photoelectrons produce, A23), total C + Ph,
        and the ceiling K_ini/R_H (each ionization costs at least R_H).
Physics em_cascades.yields.cascade with the window of em_cascades.photoionization.window (Verner et al. 1996, A24).
Writes  figures/cas_ionization_yield[_xe*].{png,pdf} and provenance/figures/cas_ionization_yield[_xe*].json
Run     python3 EM_cascades/scripts/fig_ionization_yield.py   (from the project root; ~9 min per x_e)
"""

import numpy as np

import figlib as F
from figlib import L, P, plt
from em_cascades import fractions as FR, photoionization as PI, yields as Y
from igm_losses import constants as K

SLUG = "ionization_yield"
PB = "EM_cascades/scripts/fig_ionization_yield.py::main"
Y_FLOOR = 1.0e-2          # C44: lower y limit; below 1e-2 ionizations per primary nothing is plotted on the log axis


def main():
    Kg = FR.energy_grid(F.spec(SLUG))
    for x_e in F.xe_values(SLUG):
        lo, hi = PI.window(x_e)
        C, Ph = Y.cascade(Kg, x_e, hi, photons=True)
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(Kg, C, color="tab:brown", label=L("curve_collisional"))
        ax.plot(Kg, Ph, color="tab:green", label=L("curve_photo"))
        ax.plot(Kg, C + Ph, color="k", lw=2.4, label=L("curve_total"))
        ax.plot(Kg, Kg / K.R_H_eV, "k:", lw=1.2, label=L("ceiling"))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(Kg[0], Kg[-1])                       # C44: the whole grid (K_floor … K_max of the spec)
        ax.set_ylim(Y_FLOOR, 3 * (Kg[-1] / K.R_H_eV))    # C44: up to the ceiling at K_max
        ax.set_xlabel(L("initial_energy_eV")); ax.set_ylabel(L("n_ion")); ax.set_title(F.title(x_e), fontsize=11)
        ax.legend(fontsize=10, loc="upper left")
        fig.tight_layout()
        F.save(fig, SLUG, x_e, "H ionizations per primary electron down to K_floor vs K_ini: collisional, secondary "
               "photoionization by IC photons (with the photoelectrons' cascades) and total; ceiling K_ini/R_H",
               "em_cascades.yields.cascade + em_cascades.photoionization.window",
               ["z fixed (A19)", f"photoionization window [R_H, {hi:.4g} eV] (A22)", "photoelectron trees counted in the photo curve (A23)",
                "continuous slowing down for every particle (A20)"], PB,
               extra={"K_grid": F.spec(SLUG)["grid"], "window_eV": [lo, hi]})


if __name__ == "__main__":
    main()
