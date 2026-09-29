#!/usr/bin/env python3
"""Single writer of provenance/numbers.json for the whole project (C2, C4, C17, C26, C56; user decision 2026-09-28).

Collects the entries of every subproject and writes them once:
    electron_losses_IGM/scripts/compute_numbers.py::compute_all   (trajectories, dominance, Γ_ad, Bethe, aggregates)
    EM_cascades/scripts/numbers.py::compute_all                    (loss fractions and ionization yield at fixed z)
Each subproject function returns {key: entry}; a key present in two subprojects is an error (no silent overwrite).
The entries keep their own produced_by (the function that computes them); this script only serializes.
Imports never write bytecode (the task of EM_cascades forbids creating files inside electron_losses_IGM/).

Run   python3 scripts/compute_numbers.py      (from the project root; ~15 min)
"""

import importlib.util
import json
import pathlib
import sys
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "provenance" / "numbers.json"
SOURCES = [("electron_losses_IGM", ROOT / "electron_losses_IGM" / "scripts" / "compute_numbers.py"),
           ("EM_cascades", ROOT / "EM_cascades" / "scripts" / "numbers.py")]


def load(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"numbers_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    allnum, origin = {}, {}
    for name, path in SOURCES:
        if not path.exists():
            print(f"{name}: {path.relative_to(ROOT)} not present yet, skipped")
            continue
        entries = load(name, path).compute_all()
        clash = sorted(set(entries) & set(allnum))
        if clash:
            raise KeyError(f"keys produced by both {origin[clash[0]]} and {name}: {clash[:5]}")
        for k in entries:
            origin[k] = name
        allnum.update(entries)
    meta = {"produced_by": "scripts/compute_numbers.py::main", "date": time.strftime("%Y-%m-%d"),
            "rng": "none (no stochastic method)", "n_entries": len(allnum),
            "per_subproject": {n: sum(1 for v in origin.values() if v == n) for n, _ in SOURCES}}
    OUT.write_text(json.dumps({"_meta": meta, **allnum}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(allnum)} entries {meta['per_subproject']} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
