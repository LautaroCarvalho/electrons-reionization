"""Figure style and labels for EM_cascades: the igm_losses style (one style for the whole project) and labels from
EM_cascades/labels.yaml first, then electron_losses_IGM/labels.yaml (process names, energy axes), never duplicated."""

import pathlib

import yaml

from . import _paths  # noqa: F401
from igm_losses import plotting as IP

ROOT = pathlib.Path(__file__).resolve().parents[3]
OWN = yaml.safe_load((ROOT / "EM_cascades" / "labels.yaml").read_text(encoding="utf-8")) or {}
LANG = IP.LANG
PROCESS_COLORS = IP.PROCESS_COLORS
apply_style = IP.apply_style


def label(key, **fmt):
    d = OWN.get(key) or IP.LABELS[key]
    return d[LANG].format(**fmt)


def process_name(p):
    return IP.LABELS["process"][p][LANG]
