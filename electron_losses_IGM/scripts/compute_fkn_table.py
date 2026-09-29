"""Tabulate the Klein–Nishina suppression F_KN(b) (single writer of provenance/data/F_KN_table.json).

Physics and definitions: igm_losses.inverse_compton (B&G 1970 Eqs. 2.48, 2.56 with the factor q, E05).
Grid: 200 points log-spaced in b ∈ [1e-4, 1e4] (D15).
Run:  python3 electron_losses_IGM/scripts/compute_fkn_table.py      (from the project root)
"""

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "electron_losses_IGM" / "src"))
from igm_losses.inverse_compton import compute_F_kn, B_MIN, B_MAX, TABLE_FILE   # noqa: E402

N = 200


def main():
    b = np.logspace(np.log10(B_MIN), np.log10(B_MAX), N)
    t0 = time.time()
    F = [compute_F_kn(bi) for bi in b]
    TABLE_FILE.write_text(json.dumps({
        "description": "Klein–Nishina suppression of the IC energy loss on a blackbody, F_KN(b), b = 4 gamma k T / m c^2",
        "produced_by": "electron_losses_IGM/scripts/compute_fkn_table.py::main",
        "physics": "B&G 1970 Eqs. 2.48 and 2.56 with the factor q (E05); normalization 135/pi^4",
        "b": b.tolist(), "F": F}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TABLE_FILE.relative_to(ROOT)}: {N} points in {time.time() - t0:.0f} s; F(1e-4) = {F[0]:.8f}, F(1) = {F[N // 2]:.6f}")


if __name__ == "__main__":
    main()
