"""Figure: gyroradius r_g(t) of electrons cooling by synchrotron only, proper and comoving (notebook cell 2).

Physics  r_g = p sinθ/(e B(z)), B = B0 (1+z)² (A01); K(t) from synchrotron with fixed sinθ = 1 (D11, E06).
         Comoving: r_g (1+z) (A09).
Writes   figures/igm_gyroradius_synch.{png,pdf}, provenance/figures/igm_gyroradius_synch.json
"""

import numpy as np

import figlib as F
from figlib import L, K, plt
from igm_losses import kinematics
from igm_losses.medium import Medium

SLUG = "gyroradius_synch"


def main():
    s = F.spec(SLUG)
    sin_t = s["sin_theta"]
    x_e = F.P["x_e"]
    runs = F.run(SLUG, x_e, include=["synchrotron"], sin_theta=sin_t)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for K0, out in runs:
        p = np.sqrt(kinematics.p2c2(out["K"] * K.e)) / K.c
        B = np.array([Medium(z, x_e).B for z in out["z"]])
        rg = p * sin_t / (K.e * B) / K.parsec
        axes[0].plot(F.time_axis_yr(out), rg, label=F.exp_label(K0))
        axes[1].plot(F.time_axis_yr(out), rg * (1 + out["z"]))
    for a, key in zip(axes, ("panel_proper", "panel_comoving")):
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylabel(L("gyroradius_pc")); a.set_title(L(key), fontsize=11)
    axes[1].set_xlabel(L("cosmic_time_yr"))
    axes[1].set_xlim(F.time_axis_yr(runs[0][1])[0], F.time_axis_yr(runs[0][1])[-1])
    axes[0].legend(fontsize=10)
    fig.suptitle(L("pitch_fixed", s=f"{sin_t:g}"), fontsize=11)
    fig.tight_layout()
    F.save(fig, SLUG, x_e, "gyroradius r_g = p sinθ/(eB) vs cosmic time, proper (top) and comoving (1+z) r_g (bottom), synchrotron cooling only",
           "r_g from the K(t) trajectory", ["sin θ = 1 (D11, E06)", "comoving = (1+z) × proper (A09)"], extra={"sin_theta": sin_t})


if __name__ == "__main__":
    main()
