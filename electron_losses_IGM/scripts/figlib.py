"""Shared helpers for the figure scripts fig_*.py (F5).

Every figure script: reads its specification from provenance/parameters.yaml → figures.<slug> (single place for
the energies and options), computes with igm_losses only, saves figures/igm_<slug>[_xe<value>].{png,pdf} and writes
its record provenance/figures/<slug>[_xe<value>].json (file, produced_by, shows, from_scratch, from_library,
choices, supports, parameters). scripts/build_claims.py (single writer) assembles provenance/claims.yaml from them.
"""

import json
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "electron_losses_IGM" / "src"))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

from igm_losses import constants as K, cosmology, integrate, losses, plotting  # noqa: E402,F401

PARAMS = yaml.safe_load((ROOT / "provenance" / "parameters.yaml").read_text(encoding="utf-8"))
P = {k: v["value"] for k, v in PARAMS["parameters"].items()}
FIG_DIR = ROOT / "figures"
REC_DIR = ROOT / "provenance" / "figures"
LIB = "numpy, scipy (solve_ivp Radau, quad, brentq, CubicSpline), astropy (Planck18), mpmath, matplotlib"
plotting.apply_style()
L = plotting.label


def spec(slug):
    return PARAMS["figures"][slug]


def energies(s):
    e = s["energies_eV"]
    if isinstance(e, dict):
        a, b, n = e["logspace"]
        return list(np.logspace(a, b, n))
    return list(e)


def processes(s):
    p = s["processes"]
    if p == "all":
        return list(losses.PROCESSES)
    if p == "all_but_adiabatic":
        return [q for q in losses.PROCESSES if q != "adiabatic"]
    return list(p)


# [A04] x_e constant in z; swept over parameters.yaml → x_e.sweep
def xe_values(slug):
    sw = PARAMS["parameters"]["x_e"]["sweep"]
    return sw["values"] if slug in sw["figures"] else [P["x_e"]]


def suffix(slug, x_e):
    return "" if len(xe_values(slug)) == 1 or x_e == P["x_e"] else f"_xe{x_e:g}"


def z_range(s):
    return s.get("z_init", P["z_init"]), P["z_final"]


def exp_label(K0):
    e = np.log10(K0)
    return L("K_ini_eV", exp=f"{e:.0f}" if abs(e - round(e)) < 1e-9 else f"{e:.1f}")


def xe_text(x_e):
    return L("xe_value", xe=f"{x_e:g}")


def save(fig, slug, x_e, shows, from_scratch, choices, supports=(), extra=None, produced_by=None):
    """Save PNG+PDF and the record. `produced_by` defaults to the calling script's main()."""
    sfx = suffix(slug, x_e)
    FIG_DIR.mkdir(exist_ok=True)
    REC_DIR.mkdir(parents=True, exist_ok=True)
    name = f"igm_{slug}{sfx}"
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    caller = produced_by or f"electron_losses_IGM/scripts/fig_{slug}.py::main"
    rec = {"file": f"figures/{name}.png", "file_pdf": f"figures/{name}.pdf", "produced_by": caller,
           "shows": shows, "from_scratch": from_scratch, "from_library": LIB, "choices": list(choices),
           "supports": list(supports), "notebook_cell": spec(slug)["cell"],
           "parameters": {"x_e": x_e, "z_init": z_range(spec(slug))[0], "z_final": P["z_final"],
                          "language": P["language"], **(extra or {})}}
    (REC_DIR / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote figures/{name}.png/.pdf and provenance/figures/{name}.json")


def time_axis_yr(out):
    return out["t"] / K.year


def ratio_lines(ax):
    ax.axhline(0.9, color="k", ls="--", lw=1.2, label=L("loss_10"))
    ax.axhline(0.5, color="r", ls="--", lw=1.2, label=L("loss_50"))


def run(slug, x_e, include=None, floor_eV=None, fixed_z=False, augmented=False, K0s=None, z_init=None, **opts):
    """Trajectories for every initial energy of the figure. Returns list of (K0, out)."""
    s = spec(slug)
    zi, zf = z_range(s)
    zi = z_init if z_init is not None else zi
    inc = processes(s) if include is None else include
    return [(K0, integrate.evolve(K0, zi, zf, x_e, include=inc, floor_eV=floor_eV, fixed_z=fixed_z,
                                   augmented=augmented, **opts)) for K0 in (energies(s) if K0s is None else K0s)]


def k_ratio_figure(slug, x_e, include=None, floor_eV=None, ylim=None, title=None, **opts):
    """Two panels K/K_ini vs cosmic time (log–log) and vs z (cells 8, 9, 11, 14, 17, 19, 21, 23)."""
    runs = run(slug, x_e, include, floor_eV, **opts)
    s = spec(slug)
    zi, zf = z_range(s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 8.5))
    for K0, out in runs:
        a1.plot(time_axis_yr(out), out["K"] / K0, label=exp_label(K0))
        a2.plot(out["z"], out["K"] / K0)
    for a in (a1, a2):
        a.set_yscale("log")
        ratio_lines(a)
        a.set_ylabel(L("K_over_Kini"))
        if ylim:
            a.set_ylim(*ylim)
    a1.set_xscale("log")
    a1.set_xlim(time_axis_yr(runs[0][1])[0], time_axis_yr(runs[0][1])[-1])
    a1.set_xlabel(L("cosmic_time_yr"))
    a2.set_xlim(zi, zf)
    a2.set_xlabel(L("redshift"))
    a1.legend(ncol=2, fontsize=10)
    t = (title + ", " if title else "") + xe_text(x_e)
    a1.set_title(t, fontsize=11)
    fig.tight_layout()
    return fig, runs


def short(proc):
    return plotting.LABELS["short"][proc][plotting.LANG]


def runs_without(slug, x_e, removed):
    """(runs with all processes of the figure, runs without `removed`)."""
    inc = processes(spec(slug))
    return run(slug, x_e, include=inc), run(slug, x_e, include=[p for p in inc if p != removed])


def mask_after_floor(out):
    """Boolean mask of the output points before the floor event (the electron is still followed)."""
    return out["t"] <= out["t_floor"] if out["t_floor"] is not None else np.ones_like(out["t"], bool)


def track(out, keys=("z", "K")):
    """Output arrays up to the floor event, with the event point appended (the grid point before the event can be
    well above the floor). After the event K and the augmented components are held at their event values; z is
    replaced by z(t_floor)."""
    m = mask_after_floor(out)
    res = {k: out[k][m] for k in keys}
    if out["t_floor"] is not None and not m.all():
        j = int(np.argmin(m))                         # first grid point after the event (holds the event state)
        for k in keys:
            v = float(cosmology.redshift(out["t_floor"])) if k == "z" else out[k][j]
            res[k] = np.append(res[k], v)
    return res
