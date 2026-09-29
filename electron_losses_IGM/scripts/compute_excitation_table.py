"""Tabulate the BE-scaled plane-wave Born excitation cross sections of H(1s → np), n = 2…10, for T ≤ 3 keV.

Single writer of provenance/data/excitation_table.json. Physics: igm_losses.excitation (Stone, Kim & Desclaux 2002
Eqs. 1–2 with the exact hydrogen GOS). Grid per level: T − E_n log-spaced from 1e-3 eV to 3 keV − E_n,
12 points per decade.
Run:  python3 electron_losses_IGM/scripts/compute_excitation_table.py      (from the project root; a few minutes)
"""

import json
import pathlib
import sys
import time
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "electron_losses_IGM" / "src"))
from igm_losses import excitation as X   # noqa: E402

PER_DECADE = 12
DT_MIN_EV = 1.0e-3


def main():
    warnings.filterwarnings("ignore", category=IntegrationWarning)   # roundoff notices of the GOS radial quad (tested)
    out, t0 = {}, time.time()
    for n in X.LEVELS:
        E = X.DATA["levels"][n]["E_eV"]
        dT_max = X.T_BORN_MAX_EV - E
        m = int(np.ceil(np.log10(dT_max / DT_MIN_EV) * PER_DECADE)) + 1
        dT = np.logspace(np.log10(DT_MIN_EV), np.log10(dT_max), m)
        s = [X.sigma_born_be(n, E + d) for d in dT]
        out[str(n)] = {"E_eV": E, "dT_eV": dT.tolist(), "sigma_m2": s}
        print(f"n = {n}: {m} points  [{time.time() - t0:.0f} s]", flush=True)
    X.TABLE_FILE.write_text(json.dumps({
        "description": "BE-scaled plane-wave Born cross sections for H(1s -> np), T <= 3 keV",
        "produced_by": "electron_losses_IGM/scripts/compute_excitation_table.py::main",
        "physics": "Stone, Kim & Desclaux 2002 Eqs. 1-2, exact hydrogen GOS; data provenance/data/stone2002_H_1s_np.yaml",
        "grid": {"dT_min_eV": DT_MIN_EV, "points_per_decade": PER_DECADE, "T_max_eV": X.T_BORN_MAX_EV},
        "levels": out}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {X.TABLE_FILE.relative_to(ROOT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
