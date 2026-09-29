"""Single figure style and bilingual labels (language from parameters.yaml, es|en; decision 2026-09-27).

Style: the notebook's cell-0 rcParams (usetex, Computer Modern, ticks in, minor ticks, tab10 colours), defined once.
Labels: electron_losses_IGM/labels.yaml, key → {es, en}. Every axis label, legend entry and title of the new figures
comes from there, so a label cannot disagree with the plotted quantity silently (E10).
"""

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import yaml

from . import constants as K

ROOT = pathlib.Path(__file__).resolve().parents[3]
LABELS = yaml.safe_load((ROOT / "electron_losses_IGM" / "labels.yaml").read_text(encoding="utf-8"))
LANG = K._P["language"]
PROCESS_COLORS = dict(zip(("adiabatic", "synchrotron", "ic", "coulomb", "excitation", "ionization", "bremsstrahlung"),
                          plt.cm.tab10.colors))


def apply_style():
    matplotlib.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14, "font.size": 12, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "-",
        "axes.prop_cycle": plt.cycler("color", plt.cm.tab10.colors),
        "savefig.dpi": 300, "savefig.bbox": "tight"})


def label(key, **fmt):
    """Text for `key` in the configured language, formatted with **fmt."""
    return LABELS[key][LANG].format(**fmt)
