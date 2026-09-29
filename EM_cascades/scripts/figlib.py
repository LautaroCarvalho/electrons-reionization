"""Shared helpers for the EM_cascades figure scripts (F5).

Each script reads its specification from provenance/parameters.yaml → figures.<slug> (single place), computes with
em_cascades / igm_losses only, saves figures/cas_<slug>[_xe<value>].{png,pdf} and writes its record
provenance/figures/cas_<slug>[_xe<value>].json (same record format as electron_losses_IGM; the gate checks it).
The x_e values come from parameters.yaml → x_e.sweep (A04); the default x_e has no suffix.
"""

import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "EM_cascades" / "src"))
import em_cascades._paths  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
import yaml  # noqa: E402

from em_cascades import plotting  # noqa: E402

PARAMS = yaml.safe_load((ROOT / "provenance" / "parameters.yaml").read_text(encoding="utf-8"))
P = {k: v["value"] for k, v in PARAMS["parameters"].items()}
FIG_DIR = ROOT / "figures"
REC_DIR = ROOT / "provenance" / "figures"
LIB = "numpy, scipy (quad, hyp2f1), mpmath, matplotlib; rates from igm_losses (electron_losses_IGM)"
plotting.apply_style()
L = plotting.label


def spec(slug):
    return PARAMS["figures"][slug]


def xe_values(slug):
    sw = PARAMS["parameters"]["x_e"]["sweep"]
    return sw["values"] if slug in sw["figures"] else [P["x_e"]]


def suffix(x_e):
    return "" if x_e == P["x_e"] else f"_xe{x_e:g}"


def title(x_e):
    return L("fixed_z_title", z=f"{P['z_init']:g}", xe=f"{x_e:g}")


def save(fig, slug, x_e, shows, from_scratch, choices, produced_by, supports=(), extra=None):
    name = f"cas_{slug}{suffix(x_e)}"
    FIG_DIR.mkdir(exist_ok=True)
    REC_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    rec = {"file": f"figures/{name}.png", "file_pdf": f"figures/{name}.pdf", "produced_by": produced_by,
           "shows": shows, "from_scratch": from_scratch, "from_library": LIB, "choices": list(choices),
           "supports": list(supports), "task": spec(slug)["task"],
           "parameters": {"x_e": x_e, "z": P["z_init"], "language": P["language"], **(extra or {})}}
    (REC_DIR / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote figures/{name}.png/.pdf and provenance/figures/{name}.json")
