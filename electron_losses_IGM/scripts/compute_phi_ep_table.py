"""Tabulate the non-relativistic e–p radiative energy-loss function φ_rad^KL(T) from the exact Gaunt factor.

Purpose
    The exact free-free Gaunt factor (igm_losses.gaunt_ff, Karzas & Latter 1961) costs seconds per energy,
    too slow to evaluate inside the ODE. This script is the SINGLE WRITER of the table the code interpolates.

Physics
    φ_rad(T) ≡ (∫ k dσ/dk dk) / (α r_e² Z² (T + mc²)) = (8π/(3√3)) ⟨g⟩(T/(Z²Ry∞)) · mc²/(T + mc²),
    derived from [KarzasLatter1961 Eq. 22]; ⟨g⟩ from igm_losses.gaunt_ff.mean_gff. Z = 1.
    Ry∞ = α² m_e c²/2 because K&L define η = Z e²/(ħv) with the electron mass (Eq. 8).

Reads      provenance/parameters.yaml (through igm_losses.constants)
Writes     provenance/data/phi_ep_KarzasLatter.json  (T grid [eV], φ_rad, ⟨g⟩, grid spec, code version info)

Grid       log-spaced from 10 eV to 1 MeV, 8 points per decade (41 points). The upper end is beyond the
           validity of K&L (β ≪ 1): it is only used to find and smooth the splice with B&G 1970 Eq. (3.53).

Run        python3 electron_losses_IGM/scripts/compute_phi_ep_table.py      (from the project root; ~5 min)
"""

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "electron_losses_IGM" / "src"))
from igm_losses import constants as K          # noqa: E402
from igm_losses.gaunt_ff import mean_gff       # noqa: E402

OUT = ROOT / "provenance" / "data" / "phi_ep_KarzasLatter.json"
T_MIN_EV, T_MAX_EV, PER_DECADE = 10.0, 1.0e6, 8


def main():
    n = int(round(np.log10(T_MAX_EV / T_MIN_EV) * PER_DECADE)) + 1
    T = np.logspace(np.log10(T_MIN_EV), np.log10(T_MAX_EV), n)
    g, phi = [], []
    t0 = time.time()
    for Ti in T:
        gi = float(mean_gff(Ti / K.Ry_inf_eV))
        g.append(gi)
        phi.append(8 * np.pi / (3 * np.sqrt(3)) * gi * K.m_e_c2_eV / (Ti + K.m_e_c2_eV))
        print(f"T = {Ti:10.4g} eV   <g> = {gi:.8f}   phi = {phi[-1]:.8f}   [{time.time() - t0:5.0f} s]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "description": "Non-relativistic e–p (Z = 1) radiative energy-loss function from the exact K&L 1961 Gaunt factor",
        "produced_by": "electron_losses_IGM/scripts/compute_phi_ep_table.py::main",
        "physics": "phi = (8 pi/(3 sqrt 3)) <g>(T/Ry_inf) mc^2/(T+mc^2); <g> = igm_losses.gaunt_ff.mean_gff",
        "grid": {"T_min_eV": T_MIN_EV, "T_max_eV": T_MAX_EV, "points_per_decade": PER_DECADE},
        "Ry_inf_eV": K.Ry_inf_eV, "m_e_c2_eV": K.m_e_c2_eV,
        "T_eV": T.tolist(), "mean_g": g, "phi_rad": phi,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({n} points, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
