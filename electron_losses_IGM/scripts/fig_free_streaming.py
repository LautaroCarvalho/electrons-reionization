"""Figure: free streaming (no losses): maximum distance reached and time to travel R_target (notebook cell 6).

Physics  K constant; proper distance D(t) = v (t − t_init); comoving χ(t) = ∫ v (1+z) dt (A09).
         Maximum over [t_init, t_final]; time to reach R_target = 10 Mpc (parameter R_target); if never reached,
         the point is not drawn (the notebook drew nothing there either).
Writes   figures/igm_free_streaming.{png,pdf} and record.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

import figlib as F
from figlib import L, K, plt
from igm_losses import integrate, kinematics

SLUG = "free_streaming"


def main():
    s = F.spec(SLUG)
    zi, zf = F.z_range(s)
    t = integrate.time_grid(zi, zf)
    z = np.array([float(F.cosmology.redshift(ti)) for ti in t])
    R = F.P["R_target"] * K.parsec
    Kini = np.array(F.energies(s))
    res = {"proper": ([], []), "comoving": ([], [])}
    for K0 in Kini:
        v = float(kinematics.speed(K0 * K.e))
        for key, fac in (("proper", np.ones_like(z)), ("comoving", 1 + z)):
            D = cumulative_trapezoid(v * fac, t, initial=0.0)
            res[key][0].append(D[-1] / K.parsec)
            res[key][1].append(np.interp(R, D, t - t[0]) / K.year if D[-1] >= R else np.nan)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for key, st, col in (("proper", "-", "darkblue"), ("comoving", "--", "darkred")):
        a1.plot(Kini, res[key][0], st, marker="o", ms=3, color=col, label=L("panel_" + key))
        a2.plot(Kini, res[key][1], st, marker="o", ms=3, color=col, label=L("panel_" + key))
    for a in (a1, a2):
        a.set_xscale("log"); a.set_yscale("log"); a.legend(fontsize=10)
    a1.set_ylabel(L("max_distance_pc"))
    a2.set_ylabel(L("time_to_reach_yr", R=f"{F.P['R_target'] / 1e6:g}"))
    a2.set_xlabel(L("initial_energy_eV"))
    fig.tight_layout()
    F.save(fig, SLUG, F.P["x_e"], "maximum distance reached between z_init and z_final (top) and time to travel R_target (bottom) vs initial energy, without losses; proper and comoving",
           "distances integrated on the cosmic-time grid", ["no energy losses (as the notebook)", "comoving distance added (A09)"],
           extra={"R_target_pc": F.P["R_target"]})


if __name__ == "__main__":
    main()
