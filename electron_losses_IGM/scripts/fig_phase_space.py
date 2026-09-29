"""Phase-space figures (notebook cells 29, 30, 32, 34, 35, 41).

cell 29  phase_space_map             dominant process on a (z, K) grid + trajectories + up-scatter threshold curves
cell 30  phase_space_colored         trajectories coloured by the dominant process along the track + thresholds
cell 32  phase_space_without_ad      coloured trajectories: all processes (top) vs without adiabatic (bottom)
cell 34  distance_travelled          K vs distance travelled, all vs without synchrotron (E13), proper and comoving (A09);
                                     x limits = distance at speed c over [t_init, t_final] (D17)
cell 35  phase_space_map_without_ad  dominant-process map and trajectories without adiabatic losses
cell 41  adiabatic_fraction_map      Γ_ad = L_ad/L_tot on z ∈ [5.5, 15] (D16), with the Γ_ad = 0.5 contour
The notebook's hand-placed text boxes of cell 41 ("Radiative Dominated", …) are not reproduced: they were not derived
from the data. Grids and energies from parameters.yaml → figures.
"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.integrate import cumulative_trapezoid

import figlib as F
from figlib import L, K, plt, plotting
from igm_losses import losses, phase_space, thresholds, kinematics, cosmology

PB = "electron_losses_IGM/scripts/fig_phase_space.py::"
NAMES = lambda: plotting.LABELS["process"]


def _pname(p):
    return NAMES()[p][plotting.LANG]


def _dominant_along(out, x_e, include):
    idx = []
    for Ke, z in zip(out["K"], out["z"]):
        r = losses.rates(Ke * K.e, z, x_e, include)
        idx.append(max(r, key=r.get))
    return idx


def _colored(ax, x, y, dom):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, colors=[plotting.PROCESS_COLORS[d] for d in dom[:-1]], linewidths=2.0)
    ax.add_collection(lc)


def _thresholds(ax, z, x=None):
    Kc, Kh = thresholds.threshold_curves(z)
    xx = z if x is None else x
    ax.plot(xx, Kc, "k--", lw=1.4, label=L("threshold_curve"))
    ax.plot(xx, Kh, "k--", lw=1.4)


def _legend(ax, procs, extra=()):
    h = [mpatches.Patch(color=plotting.PROCESS_COLORS[p], label=_pname(p)) for p in procs]
    ax.legend(handles=h + list(extra), ncol=2, fontsize=9, loc="upper right")


def _map(ax, slug, x_e, include):
    s = F.spec(slug)
    zi, zf = F.z_range(s)
    zg = np.linspace(zf, zi, s["map"]["z_points"])
    a, b, n = s["map"]["K_eV"]["logspace"]
    Kg = np.logspace(a, b, n)
    names, idx = phase_space.dominant_process(zg, Kg, x_e, include)
    cmap = ListedColormap([plotting.PROCESS_COLORS[p] for p in names])
    ax.pcolormesh(zg, Kg, idx, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, alpha=0.35, shading="auto")   # C44: one colour per process index
    return sorted(set(np.unique(idx)))


def map_figure(slug, x_e, include, pb):
    runs = F.run(slug, x_e, include=include)
    fig, ax = plt.subplots(figsize=(10, 7))
    present = _map(ax, slug, x_e, include)
    cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(runs)))
    traj = []
    for (K0, o), c in zip(runs, cols):
        tr = F.track(o)
        traj += ax.plot(tr["z"], tr["K"], "--", lw=1.6, color=c, label=F.exp_label(K0))
    zi, zf = F.z_range(F.spec(slug))
    _thresholds(ax, np.linspace(zf, zi, 200))
    # C44: y from 1 eV (below the 10.2 eV floor) to 1e14 eV (top of the map grid, above the highest K_ini = 1e13 eV);
    # x starts 0.1 before z_init to leave the injection point visible
    ax.set_yscale("log"); ax.set_xlim(zi + 0.1, zf); ax.set_ylim(1.0, 1e14)
    ax.set_xlabel(L("redshift")); ax.set_ylabel(L("kinetic_energy_eV"))
    _legend(ax, [include[i] for i in present], traj + ax.get_legend_handles_labels()[0][-1:])
    ax.set_title(F.xe_text(x_e), fontsize=11)
    fig.tight_layout()
    F.save(fig, slug, x_e, "dominant energy-loss process on the (z, K) plane (background), trajectories (dashed) and IC up-scatter thresholds",
           "rate maps from losses.rates; trajectories from integrate.evolve", ["background opacity 0.35", "y axis 1 eV – 1e14 eV (all trajectories visible)",
                                                                                    "trajectories end at the floor event point", "one colour per initial energy (viridis)"],
           produced_by=PB + pb)


def colored_panel(ax, slug, x_e, include, title):
    for K0, o in F.run(slug, x_e, include=include):
        tr = F.track(o)
        _colored(ax, tr["z"], tr["K"], _dominant_along(tr, x_e, include))
        ax.text(o["z"][0] + 0.05, K0, f"$10^{{{np.log10(K0):.0f}}}$ eV", fontsize=9, ha="right", va="center")
    zi, zf = F.z_range(F.spec(slug))
    _thresholds(ax, np.linspace(zf, zi, 200))
    # C44: y as in map_figure, up to 3e14 eV to fit the 1e13 eV label; x starts 0.5 before z_init for the K_ini labels
    ax.set_yscale("log"); ax.set_xlim(zi + 0.5, zf); ax.set_ylim(1.0, 3e14)
    ax.set_ylabel(L("kinetic_energy_eV")); ax.set_title(title, fontsize=11)
    _legend(ax, include, ax.get_legend_handles_labels()[0][:1])


def fig30(x_e):
    slug = "phase_space_colored"
    fig, ax = plt.subplots(figsize=(10, 7))
    colored_panel(ax, slug, x_e, list(losses.PROCESSES), L("all_active") + ", " + F.xe_text(x_e))
    ax.set_xlabel(L("redshift")); fig.tight_layout()
    F.save(fig, slug, x_e, "trajectories K(z) coloured by the dominant loss process at each point, with the IC up-scatter thresholds",
           "dominant process from losses.rates along each trajectory", ["curves stop at the 10.2 eV floor"], produced_by=PB + "fig30")


def fig32(x_e):
    slug = "phase_space_without_ad"
    fig, axes = plt.subplots(2, 1, figsize=(10, 14), sharex=True)
    inc = list(losses.PROCESSES)
    colored_panel(axes[0], slug, x_e, inc, L("all_active") + ", " + F.xe_text(x_e))
    colored_panel(axes[1], slug, x_e, [p for p in inc if p != "adiabatic"], L("without", proc=_pname("adiabatic")))
    axes[1].set_xlabel(L("redshift")); fig.tight_layout()
    F.save(fig, slug, x_e, "coloured trajectories with all processes (top) and without adiabatic losses (bottom)",
           "dominant process along each trajectory", ["file name without spaces or accents (the notebook's had both)"], produced_by=PB + "fig32")


def fig34(x_e):
    slug = "distance_travelled"
    s = F.spec(slug)
    zi, zf = F.z_range(s)
    inc = list(losses.PROCESSES)
    variants = [(inc, L("all_active")), ([p for p in inc if p != "synchrotron"], L("without", proc=_pname("synchrotron")))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    t = integrate_time = F.integrate.time_grid(zi, zf)
    zt = np.array([float(cosmology.redshift(ti)) for ti in t])
    Mpc = 1e6 * K.parsec
    dmax = {"proper": K.c * (t[-1] - t[0]) / Mpc, "comoving": np.trapz(K.c * (1 + zt), t) / Mpc}
    for row, (include, title) in enumerate(variants):
        runs = F.run(slug, x_e, include=include, augmented=True)
        for col, key in enumerate(("proper", "comoving")):
            a = axes[row, col]
            for K0, o in runs:
                dk = "D_proper" if key == "proper" else "D_comoving"
                tr = F.track(o, ("z", "K", dk))
                _colored(a, tr[dk] / Mpc, tr["K"], _dominant_along(tr, x_e, include))
            Dthr = (K.c * (t - t[0]) if key == "proper" else cumulative_trapezoid(K.c * (1 + zt), t, initial=0.0)) / Mpc
            _thresholds(a, zt, x=Dthr)
            # C44: x up to the distance at speed c over [t_init, t_final] (D17), y as in colored_panel
            a.set_xlim(-0.01 * dmax[key], dmax[key]); a.set_yscale("log"); a.set_ylim(1.0, 3e14)
            a.set_title(f"{title} — {L('panel_' + key)}", fontsize=11)
            a.set_xlabel(L("distance_Mpc"))
            if col == 0:
                a.set_ylabel(L("kinetic_energy_eV"))
    _legend(axes[0, 0], inc, axes[0, 0].get_legend_handles_labels()[0][:1])
    fig.suptitle(F.xe_text(x_e), fontsize=11); fig.tight_layout()
    F.save(fig, slug, x_e, "K vs distance travelled (proper left, comoving right) with all processes (top) and without synchrotron (bottom), coloured by dominant process",
           "augmented ODE (distances) + dominant process along the track", ["comparison without synchrotron (E13)", "x limits = distance at speed c (D17)",
                                                                            "threshold curves placed at the distance a photon-speed particle covers by z"],
           produced_by=PB + "fig34", extra={"xmax_Mpc": dmax})


def fig41(x_e):
    slug = "adiabatic_fraction_map"
    s = F.spec(slug)
    z0, z1 = s["map"]["z_range"]
    zg = np.linspace(z0, z1, s["map"]["z_points"])
    a, b, n = s["map"]["K_eV"]["logspace"]
    Kg = np.logspace(a, b, n)
    G = phase_space.adiabatic_fraction(zg, Kg, x_e)
    fig, ax = plt.subplots(figsize=(9, 7))
    pc = ax.pcolormesh(zg, Kg, G, cmap="coolwarm_r", vmin=0, vmax=1, shading="gouraud")   # C44: Γ_ad is a fraction, [0, 1]
    cs = ax.contour(zg, Kg, G, levels=[0.5], colors="k", linewidths=1.5)
    ax.clabel(cs, fmt={0.5: L("dominance_horizon")}, fontsize=10)
    fig.colorbar(pc, ax=ax, label=L("gamma_ad"))
    ax.set_yscale("log"); ax.set_xlim(z1, z0); ax.set_xlabel(L("redshift")); ax.set_ylabel(L("kinetic_energy_eV"))
    ax.set_title(F.xe_text(x_e), fontsize=11); fig.tight_layout()
    F.save(fig, slug, x_e, "fraction of the total loss rate due to adiabatic expansion on the (z, K) plane, with the Γ_ad = 0.5 contour",
           "rate maps from losses.rates", ["z range 5.5–15 (D16)", "hand-placed text boxes of the notebook not reproduced",
                                           "ionization with the eV→J factor (E08); bremsstrahlung from BREMS CS_int + Karzas & Latter + B&G (D18, D19)",
                                           "the Γ_ad = 0.5 contour is drawn only where Γ_ad reaches 0.5"],
           produced_by=PB + "fig41")


def main():
    inc = list(losses.PROCESSES)
    for x_e in F.xe_values("phase_space_map"):
        map_figure("phase_space_map", x_e, inc, "map_figure")
    for x_e in F.xe_values("phase_space_colored"):
        fig30(x_e)
    for x_e in F.xe_values("phase_space_without_ad"):
        fig32(x_e)
    for x_e in F.xe_values("phase_space_map_without_ad"):
        map_figure("phase_space_map_without_ad", x_e, [p for p in inc if p != "adiabatic"], "map_figure")
    for x_e in F.xe_values("distance_travelled"):
        fig34(x_e)
    for x_e in F.xe_values("adiabatic_fraction_map"):
        fig41(x_e)


if __name__ == "__main__":
    main()
