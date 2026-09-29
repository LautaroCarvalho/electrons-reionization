"""Number functions of EM_cascades for provenance/numbers.json (library; the single writer is scripts/compute_numbers.py).

Produces (every key starts with cas_; entries have the 12 fields of the registry)
    cas_E_max_xe<x>                    upper edge of the photoionization window [eV] (A22, A24)
    cas_frac_<process>_K<e>_xe<x>      fraction of K_ini lost to each process down to K_floor, fixed z (objective 1b)
    cas_N_coll_K<e>_xe<x>              collisional ionizations of the cascade (objective 2)
    cas_N_photo_K<e>_xe<x>             ionizations due to IC photons and their photoelectrons (objective 2)
    cas_N_tot_K<e>_xe<x>               total
    cas_W_3keV_xe<x>                   mean energy per collisional ionization of a 3 keV electron [eV]
    cas_phi_HI_3keV_xe<x>              fraction of a 3 keV electron's energy that ends in H ionizations (vs SvS 1985)
x_e: parameters.yaml → x_e.sweep (A04); z fixed at z_init (A19); grid: figures.ionization_yield.grid (D22).
Run   python3 scripts/compute_numbers.py   (the root writer calls compute_all(); ~45 min for the cascades)
"""

import importlib.metadata as md
import pathlib
import sys
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "EM_cascades" / "src"))
import em_cascades._paths  # noqa: E402,F401

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from em_cascades import fractions as FR, photoionization as PI, yields as Y  # noqa: E402
from igm_losses import constants as K, losses  # noqa: E402

PARAMS = yaml.safe_load((ROOT / "provenance" / "parameters.yaml").read_text(encoding="utf-8"))
P = K._P
SWEEP = PARAMS["parameters"]["x_e"]["sweep"]["values"]
SPEC = PARAMS["figures"]["ionization_yield"]
SVS = yaml.safe_load((ROOT / "provenance" / "data" / "shull1985_table1.yaml").read_text(encoding="utf-8"))
ME = "EM_cascades/scripts/numbers.py::"
TC = "EM_cascades/tests/"
E_REPORT = [1.0e4, 1.0e6, 1.0e8, 1.0e10, 1.0e13]
A_CASC = ["A03", "A04", "A07", "A11", "A13", "A14", "A17", "A19", "A20", "A21", "A22", "A23", "A24"]
LIB = "scipy=={s}, scipy.integrate.quad, scipy.optimize.brentq; mpmath=={m}".format(
    s=md.version("scipy"), m=md.version("mpmath"))
NUMBERS = {}


def put(key, value, unit, statement, fn, checked_by, sigma=None, kind="numerical", assumptions=A_CASC,
        inputs=("K_floor", "z_init", "B0", "Y_p", "T_CMB0"), choices=()):
    NUMBERS[key] = {"value": float(value), "unit": unit, "uncertainty": {"sigma": sigma, "kind": kind}, "tag": "derived",
                    "statement": statement, "produced_by": ME + fn, "from_scratch": ME + fn, "from_library": LIB,
                    "choices": list(choices), "assumptions": list(assumptions), "inputs": list(inputs),
                    "checked_by": [TC + c for c in checked_by]}


def k_label(e):
    return f"1e{np.log10(e):.0f}"


def window_numbers():
    for x_e in SWEEP:
        put(f"cas_E_max_xe{x_e:g}", PI.window(x_e)[1], "eV",
            f"Borde superior de la ventana de fotoionización (n_HI σ_pi c = H) en z = {P['z_init']:g}, x_e = {x_e:g}",
            "window_numbers", ["test_photoionization.py::test_window_definition", "test_photoionization.py::test_verner_vs_exact_hydrogenic"],
            sigma=None, kind="root to 1e-12 (brentq)", assumptions=["A03", "A04", "A19", "A22", "A24"])


def fraction_numbers():
    grid = FR.energy_grid({"grid": {"K_max_eV": max(E_REPORT), "points_per_decade": SPEC["grid"]["points_per_decade"]}})
    for x_e in SWEEP:
        f = FR.integrated_fractions(grid, x_e)
        for e in E_REPORT:
            i = int(np.argmin(np.abs(np.log(grid / e))))
            for p in losses.PROCESSES:
                put(f"cas_frac_{p}_K{k_label(e)}_xe{x_e:g}", f[p][i], "",
                    f"Fracción de K_ini = {grid[i]:.3g} eV perdida por {p} hasta K_floor, medio fijo en z = {P['z_init']:g}, x_e = {x_e:g}",
                    "fraction_numbers", ["test_fractions.py::test_integrated_vs_time_integration", "test_fractions.py::test_grid_convergence",
                                         "test_fractions.py::test_integrated_budget_closes"],
                    sigma=1e-3, kind="grid error bound (test_grid_convergence)", assumptions=["A07", "A11", "A19", "A20"])


def cascade_numbers():
    grid = FR.energy_grid(SPEC)
    E0 = SVS["E0_eV"]
    for x_e in SWEEP:
        t0 = time.time()
        C, Ph = Y.cascade(grid, x_e, PI.window(x_e)[1], photons=True)
        print(f"  cascade x_e = {x_e:g} [{time.time() - t0:.0f} s]", flush=True)
        for e in E_REPORT:
            i = int(np.argmin(np.abs(np.log(grid / e))))
            lab = f"K{k_label(e)}_xe{x_e:g}"
            chk = ["test_cascade.py::test_ceiling_and_no_photons", "test_cascade.py::test_photon_branch_bounded_by_ic_energy",
                   "test_cascade.py::test_grid_convergence_collisional"]
            put(f"cas_N_coll_{lab}", C[i], "", f"Ionizaciones colisionales de la cascada de un electrón de {grid[i]:.3g} eV, x_e = {x_e:g}",
                "cascade_numbers", chk)
            put(f"cas_N_photo_{lab}", Ph[i], "", f"Ionizaciones por fotones de IC y sus fotoelectrones, electrón de {grid[i]:.3g} eV, x_e = {x_e:g}",
                "cascade_numbers", chk + ["test_cascade.py::test_photon_grid_convergence"])
            put(f"cas_N_tot_{lab}", C[i] + Ph[i], "", f"Ionizaciones totales de la cascada de un electrón de {grid[i]:.3g} eV, x_e = {x_e:g}",
                "cascade_numbers", chk + ["test_cascade.py::test_photon_grid_convergence"])
        c3 = float(np.interp(np.log(E0), np.log(grid), C))
        put(f"cas_W_3keV_xe{x_e:g}", E0 / c3 if c3 > 0 else float("inf"), "eV",
            f"Energía media por ionización colisional de un electrón de 3 keV, x_e = {x_e:g}", "cascade_numbers",
            ["test_cascade.py::test_collisional_yield_vs_shull_van_steenberg"])
        put(f"cas_phi_HI_3keV_xe{x_e:g}", c3 * SVS["I_eV"] / E0, "",
            f"Fracción de la energía de un electrón de 3 keV que termina en ionizaciones de H (a comparar con Shull & van Steenberg 1985), x_e = {x_e:g}",
            "cascade_numbers", ["test_cascade.py::test_collisional_yield_vs_shull_van_steenberg"])


def compute_all():
    """Fill NUMBERS with every EM_cascades entry and return it (no file is written here)."""
    t0 = time.time()
    window_numbers()
    fraction_numbers()
    cascade_numbers()
    print(f"EM_cascades numbers: {len(NUMBERS)} entries [{time.time() - t0:.0f} s]", flush=True)
    return NUMBERS


if __name__ == "__main__":
    print("This module only computes; the single writer of provenance/numbers.json is scripts/compute_numbers.py")
