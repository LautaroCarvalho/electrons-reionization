"""Figures: distance travelled with synchrotron cooling and gyration geometry (notebook cells 4 and 5).

Physics  r_g = p ⟨sinα⟩/(eB), drift y = ∫ v ⟨|cosα|⟩ dt; d_max = √((r_g + r_g,0)² + y²), d_min = √((r_g − r_g,0)² + y²)
         (the notebook's heuristic bounds, kept per A09). Cell 4: v ⟂ B (sinα = 1, cosα = 0, fixed-pitch synchrotron).
         Cell 5: isotropic averages ⟨sinα⟩ = π/4, ⟨|cosα|⟩ = 1/2 (exact over the sphere) with isotropic synchrotron.
         Comoving panel: gyration lengths × (1+z) and drift ∫ v⟨|cosα|⟩(1+z) dt (A09; my choice for the gyration part).
Writes   figures/igm_distance_synch_fixed.*, figures/igm_distance_synch_isotropic.* and their records.
"""

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid

import figlib as F
from figlib import L, K, plt
from igm_losses import kinematics
from igm_losses.medium import Medium


def make(slug, sin_mean, cos_mean, sin_theta_rate):
    x_e = F.P["x_e"]
    runs = F.run(slug, x_e, include=["synchrotron"], sin_theta=sin_theta_rate)
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for i, (K0, out) in enumerate(runs):
        c = plt.cm.tab10.colors[i % 10]
        Kj = out["K"] * K.e
        p = np.sqrt(kinematics.p2c2(Kj)) / K.c
        v = kinematics.speed(Kj)
        B = np.array([Medium(z, x_e).B for z in out["z"]])
        rg = p * sin_mean / (K.e * B)
        for a, fac in ((axes[0], np.ones_like(out["z"])), (axes[1], 1 + out["z"])):
            y = cumulative_trapezoid(v * cos_mean * fac, out["t"], initial=0.0)
            r, r0 = rg * fac, rg[0] * fac[0]
            dmax = np.sqrt((r + r0) ** 2 + y ** 2) / K.parsec
            dmin = np.maximum(np.sqrt((r - r0) ** 2 + y ** 2) / K.parsec, 1e-12)
            a.plot(F.time_axis_yr(out), dmax, color=c, label=F.exp_label(K0) if a is axes[0] else None)
            a.fill_between(F.time_axis_yr(out), dmin, dmax, color=c, alpha=0.15)
    for a, key in zip(axes, ("panel_proper", "panel_comoving")):
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylabel(L("distance_pc")); a.set_title(L(key), fontsize=11)
    axes[1].set_xlabel(L("cosmic_time_yr"))
    axes[1].set_xlim(F.time_axis_yr(runs[0][1])[0], F.time_axis_yr(runs[0][1])[-1])
    axes[0].legend(ncol=2, fontsize=10)
    fig.suptitle(L("pitch_fixed", s="1") if sin_theta_rate else L("pitch_iso"), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e,
           "distance travelled (d_max line, band d_min–d_max) vs cosmic time with synchrotron cooling, proper and comoving",
           "gyroradius and drift integrated from the K(t) trajectory",
           ["notebook heuristic bounds d_min/d_max kept (A09)", "comoving: gyration × (1+z), drift ∫v(1+z)dt (my choice)",
            f"⟨sinα⟩ = {sin_mean:.6g}, ⟨|cosα|⟩ = {cos_mean:g}"],
           extra={"sin_mean": sin_mean, "cos_mean": cos_mean}, produced_by="electron_losses_IGM/scripts/fig_distance_gyration.py::make")


def main():
    make("distance_synch_fixed", 1.0, 0.0, 1.0)
    make("distance_synch_isotropic", math.pi / 4, 0.5, None)


if __name__ == "__main__":
    main()
