#!/usr/bin/env python3
r"""
The single source of truth: loads ../parameters.yaml.

WHY THIS EXISTS.  Until 2026-09-15 scenario values were declared in whichever
module first needed them: Z_SNAP lived in three modules, QUAD_TOL in two, and
the photon band top had three different values across the project -- which had
the project reporting BOTH 2.280x and 2.073x for the same physical statement.
Everything a scenario depends on now lives in parameters.yaml, and modules read
it from here.

DELIBERATELY DEPENDENCY-LIGHT.  This module imports yaml, hashlib and pathlib
and nothing else.  make_inputs_table.py can therefore generate the LaTeX table
without importing the physics stack -- which matters, because importing
ionization_yield + igm_losses + source_map together OOM-kills this machine.

Usage:
    import parameters as PR
    PR.val("cr_channel.index")          # 2.2
    PR.entry("cr_channel.index")        # the whole dict, with units/source/note
    PR.Z, PR.X_E, PR.ZETA_N             # the hot ones, as module constants
    PR.scenario_hash()                  # stamps artefacts, so scenarios cannot mix

Check it:
    python3 python/parameters.py --check
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import yaml

YAML_PATH = Path(__file__).resolve().parent.parent / "parameters.yaml"

_RAW = YAML_PATH.read_text()
_P = yaml.safe_load(_RAW)

VALID_STATUS = set("VCDSUX")


# ---------------------------------------------------------------- accessors
def entry(path: str) -> dict:
    """The full record at a dotted path, e.g. 'cr_channel.index'."""
    node = _P
    for part in path.split("."):
        if part not in node:
            raise KeyError(f"parameters.yaml has no entry '{path}' "
                           f"(failed at '{part}')")
        node = node[part]
    return node


def val(path: str):
    """Just the value at a dotted path.  Ranges come back as a list."""
    e = entry(path)
    if isinstance(e, dict) and "value" in e:
        return e["value"]
    return e


def units(path: str) -> str:
    return entry(path).get("units", "--")


def source(path: str) -> str:
    return entry(path).get("source", "")


def status(path: str) -> str:
    return entry(path).get("status", "")


def note(path: str) -> str:
    return " ".join(entry(path).get("note", "").split())


def scenario_hash(n: int = 12) -> str:
    """Short hash of the whole file.

    Stamped into every registry, figure and generated .tex so that a figure
    made under one scenario cannot be silently combined with a number from
    another.  Any edit to parameters.yaml changes it.
    """
    return hashlib.sha256(_RAW.encode()).hexdigest()[:n]


def stamp() -> dict:
    """The scenario stamp written into every registry.

    Two artefacts built under different parameters.yaml must never be combined
    -- a z = 10 figure beside a z = 20 number is the classic way a result goes
    quietly wrong. Every registry carries this, and check_provenance refuses a
    set whose stamps disagree.
    """
    return {"scenario": _P["meta"]["scenario_name"],
            "scenario_hash": scenario_hash()}


def photon_band(emit_max_eV: float, igm_cutoff_eV: float) -> tuple:
    """The canonical photon band: (E_th, min(source cutoff, IGM cutoff)).

    Settled 2026-09-15.  4 Ryd is a property of STELLAR spectra -- the He II
    edge, above which BPASS spectra drop -- while tau_IGM = 1 is a property of
    the IGM.  A source is therefore integrated over its own emission range,
    capped where the IGM stops absorbing.  Both numbers are passed in rather
    than computed here, so this module stays free of the physics stack.
    """
    return (E_TH_EV, min(float(emit_max_eV), float(igm_cutoff_eV)))


# ------------------------------------------------- hot constants, by name
Z = float(val("scenario.z"))
X_E = float(val("scenario.x_e"))
Z_TARGET = float(val("scenario.Z_target"))
B0_TESLA = float(val("scenario.B0_tesla"))

E_TH_EV = float(val("photon_band.E_min_eV"))
STELLAR_CUTOFF_RYD = float(val("photon_band.stellar_cutoff_ryd"))

SED_ALPHA = float(val("stellar_source.sed_alpha"))
SED_MODEL = str(val("stellar_source.sed_model"))
CR_INDEX = float(val("cr_channel.index"))
CR_E_MIN_EV = float(val("cr_channel.E_min_eV"))
CR_E_MAX_EV = float(val("cr_channel.E_max_eV"))
F_DEP = float(val("cr_channel.f_dep"))
E_SN_ERG = float(val("cr_channel.E_SN_erg"))

F_ESC_FID = float(val("scanned.f_esc"))
EPS_CR_FE_FID = float(val("cr_channel.eps_CR_f_e"))
JET_E_FRAC = tuple(val("scanned.jet_e_frac"))

ZETA_N = int(val("numerics.zeta_n"))
MASTER_GRID_N = int(val("numerics.master_grid_n"))
QUAD_TOL = float(val("numerics.quad_tol"))
E_TOP_MASTER_EV = float(val("numerics.e_top_master_eV"))


# --------------------------------------------------------------- validation
def check() -> int:
    """Every entry must carry units and a valid status.  A table whose status
    column is optional is a table whose provenance quietly rots."""
    bad, n = [], 0
    def walk(node, path):
        nonlocal n
        if isinstance(node, dict) and "value" in node:
            n += 1
            if "units" not in node:
                bad.append(f"{path}: no units")
            st = node.get("status")
            if st not in VALID_STATUS:
                bad.append(f"{path}: status {st!r} not in {sorted(VALID_STATUS)}")
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
    for k, v in _P.items():
        if k != "meta":
            walk(v, k)

    print(f"parameters.yaml  --  {n} entries, scenario '{_P['meta']['scenario_name']}'")
    print(f"  scenario hash: {scenario_hash()}")
    counts = {}
    def tally(node):
        if isinstance(node, dict) and "value" in node:
            counts[node.get("status", "?")] = counts.get(node.get("status", "?"), 0) + 1
            return
        if isinstance(node, dict):
            for v in node.values():
                tally(v)
    for k, v in _P.items():
        if k != "meta":
            tally(v)
    print("  status: " + "  ".join(f"{k}={counts[k]}" for k in sorted(counts)))
    if counts.get("X"):
        print("  UNSOURCED (X):")
        def show(node, path):
            if isinstance(node, dict) and "value" in node:
                if node.get("status") == "X":
                    print(f"     {path:34s} {node['value']}   {note(path)[:70]}")
                return
            if isinstance(node, dict):
                for k2, v2 in node.items():
                    show(v2, f"{path}.{k2}" if path else k2)
        for k, v in _P.items():
            if k != "meta":
                show(v, k)
    if bad:
        print(f"\n  {len(bad)} PROBLEM(S):")
        for b in bad:
            print("     " + b)
        return 1
    print("\n  all entries carry units and a valid status.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check() if "--check" in sys.argv or len(sys.argv) == 1 else 0)
