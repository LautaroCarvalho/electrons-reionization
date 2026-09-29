"""Figures comparing the full cooling with the cooling without one process (notebook cells 25, 27, 40, 42).

cell 25  ratio_without_synch_ad     K_tot(t)/K_no_sync(t) and K_tot(t)/K_no_ad(t) vs cosmic time
cell 27  loss_rate_fractions        (L_tot − L_sync)/L_tot and (L_tot − L_ad)/L_tot along the full trajectory, vs z
cell 40  component_ratios           (K_ini − K_tot)/(K_ini − K_no_X) for X = sync, ad, vs z (failed in the notebook, E11)
cell 42  with_vs_without_adiabatic  K_with_ad/K_no_ad vs cosmic time and vs z (never run in the notebook)
"all" processes include bremsstrahlung (D14). Each figure is repeated for the x_e sweep values (A04).
"""

import numpy as np

import figlib as F
from figlib import L, K, plt
from igm_losses import losses

PB = "electron_losses_IGM/scripts/fig_comparisons.py::"


def fig25(x_e):
    slug = "ratio_without_synch_ad"
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    for a, rem in zip(axes, ("synchrotron", "adiabatic")):
        tot, wo = F.runs_without(slug, x_e, rem)
        for (K0, o1), (_, o2) in zip(tot, wo):
            a.plot(F.time_axis_yr(o1), o1["K"] / o2["K"], label=F.exp_label(K0))
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylabel(L("ratio_K_without", proc=F.short(rem)))
    axes[1].set_xlabel(L("cosmic_time_yr")); axes[0].legend(ncol=2, fontsize=9)
    axes[0].set_title(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, "ratio of K with all processes to K without synchrotron (top) / without adiabatic losses (bottom) vs cosmic time",
           "pairs of trajectories", ["spikes where one trajectory reaches the 10.2 eV floor before the other (as in the notebook)"],
           produced_by=PB + "fig25")


def fig27(x_e):
    slug = "loss_rate_fractions"
    tot = F.run(slug, x_e)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    for K0, o in tot:
        m = F.mask_after_floor(o)
        frac = {p: [] for p in ("synchrotron", "adiabatic")}
        for Ke, z in zip(o["K"][m], o["z"][m]):
            r = losses.rates(Ke * K.e, z, x_e)
            Lt = sum(r.values())
            for p in frac:
                frac[p].append((Lt - r[p]) / Lt)
        for a, p in zip(axes, frac):
            a.plot(o["z"][m], frac[p], label=F.exp_label(K0))
    for a, p in zip(axes, ("synchrotron", "adiabatic")):
        a.axhline(1.0, color="k", ls="--", lw=1.0); a.set_ylabel(L("rate_fraction", proc=F.short(p)))
    axes[1].set_xlim(*F.z_range(F.spec(slug))); axes[1].set_xlabel(L("redshift")); axes[0].legend(ncol=2, fontsize=9)
    axes[0].set_title(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, "fraction of the total loss rate not due to synchrotron (top) / adiabatic (bottom) along the full trajectory, vs z",
           "rates from losses.rates on the K(t) trajectory", ["curves stop at the 10.2 eV floor", "linear axes (as the notebook)"],
           produced_by=PB + "fig27")


def fig40(x_e):
    slug = "component_ratios"
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    for a, rem in zip(axes, ("synchrotron", "adiabatic")):
        tot, wo = F.runs_without(slug, x_e, rem)
        for (K0, o1), (_, o2) in zip(tot, wo):
            with np.errstate(divide="ignore", invalid="ignore"):
                a.plot(o1["z"], (K0 - o1["K"]) / (K0 - o2["K"]), label=F.exp_label(K0))
        a.set_yscale("log"); a.set_ylabel(L("lost_ratio", proc=F.short(rem)))
    axes[1].set_xlim(*F.z_range(F.spec(slug))); axes[1].set_xlabel(L("redshift")); axes[0].legend(ncol=2, fontsize=9)
    axes[0].set_title(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, "energy lost with all processes over energy lost without synchrotron (top) / without adiabatic (bottom), vs z",
           "pairs of trajectories", ["rebuilt from scratch (the notebook cell failed, E11)", "bremsstrahlung included (D14)"],
           produced_by=PB + "fig40")


def fig42(x_e):
    slug = "with_vs_without_adiabatic"
    tot, wo = F.runs_without(slug, x_e, "adiabatic")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 8.5))
    for (K0, o1), (_, o2) in zip(tot, wo):
        a1.plot(F.time_axis_yr(o1), o1["K"] / o2["K"], label=F.exp_label(K0))
        a2.plot(o1["z"], o1["K"] / o2["K"])
    for a in (a1, a2):
        a.set_yscale("log"); a.axhline(1.0, color="k", lw=1.0); a.set_ylabel(L("K_with_without_ad"))
    a1.set_xscale("log"); a1.set_xlabel(L("cosmic_time_yr")); a2.set_xlim(*F.z_range(F.spec(slug))); a2.set_xlabel(L("redshift"))
    a1.legend(ncol=2, fontsize=9); a1.set_title(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, "ratio of K with and without adiabatic losses vs cosmic time and vs z",
           "pairs of trajectories", ["KN IC (D10; the notebook cell used Thomson)", "synchrotron included (the notebook cell omitted it)",
                                     "ionization with the eV→J factor (E08)"], produced_by=PB + "fig42")


def main():
    for f, slug in ((fig25, "ratio_without_synch_ad"), (fig27, "loss_rate_fractions"),
                    (fig40, "component_ratios"), (fig42, "with_vs_without_adiabatic")):
        for x_e in F.xe_values(slug):
            f(x_e)


if __name__ == "__main__":
    main()
