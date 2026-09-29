"""Figure: inverse Compton cooling with z held at z_init ("UTOPIA") vs dynamical z (notebook cell 12).

Physics  IC only (corrected F_KN, E05); fixed_z → U_CMB, T_CMB frozen at z_init. Vertical line at the redshift reached
         Δt = delta_t_marker after t_init (30 Myr; E09: the notebook label said 3.2 Myr).
Writes   figures/igm_ic_cmb_fixed_vs_dynamic.{png,pdf} and record.
"""

import figlib as F
from figlib import L, K, plt
from igm_losses import cosmology

SLUG = "ic_cmb_fixed_vs_dynamic"


def main():
    x_e = F.P["x_e"]
    zi, zf = F.z_range(F.spec(SLUG))
    z_mark = float(cosmology.redshift(cosmology.age(zi) + F.P["delta_t_marker"] * K.year))
    fig, axes = plt.subplots(2, 1, figsize=(8, 10.5), sharex=True, sharey=True)
    for a, fixed, title in ((axes[0], True, L("utopia_fixed", z=f"{zi:g}")), (axes[1], False, L("utopia_dynamic"))):
        for K0, out in F.run(SLUG, x_e, include=["ic"], fixed_z=fixed):
            a.plot(out["z"] if not fixed else F.cosmology.redshift(out["t"]), out["K"] / K0, label=F.exp_label(K0))
        F.ratio_lines(a)
        a.axvline(z_mark, color="g", ls="--", lw=1.5, label=L("dt_marker", Myr=f"{F.P['delta_t_marker'] / 1e6:g}"))
        a.set_yscale("log"); a.set_ylabel(L("K_over_Kini")); a.set_title(title, fontsize=11)
    axes[1].set_xlim(zi, zf); axes[1].set_xlabel(L("redshift"))
    axes[0].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    F.save(fig, SLUG, x_e, "K/K_ini vs z for IC on the CMB with the CMB frozen at z_init (top) and evolving (bottom); the fixed-z curves are plotted against the redshift of the elapsed cosmic time",
           "trajectories from integrate.evolve(fixed_z=True/False)", ["x axis of the fixed-z panel = z(t) as time proxy (as the notebook)",
                                                                       "marker Δt = 30 Myr (E09)"], extra={"z_marker": z_mark})


if __name__ == "__main__":
    main()
