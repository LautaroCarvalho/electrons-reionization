"""Number functions of electron_losses_IGM for provenance/numbers.json (library; it does NOT write the file).

Since 2026-09-28 the single writer of provenance/numbers.json is scripts/compute_numbers.py at the project root
(user decision when EM_cascades was added): it calls compute_all() below and the EM_cascades number functions.

Produces (keys of numbers.json; every entry has value, unit, uncertainty, tag, statement, produced_by, from_scratch,
from_library, choices, assumptions, inputs, checked_by)
    n_H_zinit                       n_H at z_init [m^-3] (A02)
    dt_zinit_zfinal_Myr             cosmic time between z_init and z_final
    dz_marker                       Δz covered in delta_t_marker (30 Myr) after z_init (cell 12, E09)
    z_therm_K<e>_xe<x>              redshift at which K reaches K_floor (A07), energies of figures.total_cooling
    t_therm_K<e>_xe<x>_Myr          time from injection to K_floor
    frac_<process>_K<e>_xe<x>       energy lost to each process / K_ini up to the floor (figures.accumulated_losses)
    D_proper_K<e>_xe<x>_Mpc         proper path length up to the floor (A09, figures.distance_travelled)
    D_comoving_K<e>_xe<x>_Mpc       comoving distance up to the floor
    K_dom_<a>_to_<b>_z<z>_xe<x>     kinetic energy where the dominant loss process changes from a (below) to b (above)
    gamma_ad_max_xe<x>              maximum of Γ_ad = L_ad/L_tot on the map of figures.adiabatic_fraction_map (cell 41)
    bethe_ratio_T<e>                (excitation + ionization) / Bethe stopping (Inokuti 1971 Eq. 4.65), E21
    frac_max_<process>              max of frac_<process>_* over the energies of accumulated_losses and every x_e
    z_therm_spread_Kge1e9_xe<x>     max − min of z_therm over K_ini ≥ 1e9 eV (loss of memory of the initial energy)
x_e takes every value of parameters.yaml → x_e.sweep (A04).

Numerical uncertainty
    ODE results: |value(rtol) − value(rtol/100)|, re-running every trajectory with rtol/100 (D05).
    Fractions: also the energy-bookkeeping residual |Σ lost + K_end − K_ini| / K_ini.
    Crossings: brentq xtol; Γ_ad max: grid value (the grid spacing is stated in `choices`).
Reads      provenance/parameters.yaml (through igm_losses), provenance/data/* tables (through igm_losses)
Writes     nothing (fills NUMBERS; the root writer serializes it)
Checked by tests/test_numbers.py (independent re-derivations, see each entry's checked_by)
Run        python3 scripts/compute_numbers.py      (the root writer; from the project root)
"""

import contextlib
import importlib.metadata as md
import json
import math
import pathlib
import sys
import time

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "electron_losses_IGM" / "src"))
import yaml  # noqa: E402
from igm_losses import constants as K, cosmology, excitation, integrate, ionization, losses, phase_space  # noqa: E402
from igm_losses.medium import Medium  # noqa: E402

PARAMS = yaml.safe_load((ROOT / "provenance" / "parameters.yaml").read_text(encoding="utf-8"))
P = K._P
FIG = PARAMS["figures"]
SWEEP = PARAMS["parameters"]["x_e"]["sweep"]["values"]
ME = "electron_losses_IGM/scripts/compute_numbers.py::"
TN = "electron_losses_IGM/tests/test_numbers.py::"
MPC = 1e6 * K.parsec
MYR = 1e6 * K.year
ALL_A = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17"]


def lib(*calls):
    v = {p: md.version(p) for p in ("numpy", "scipy", "astropy", "mpmath")}
    return "; ".join(c.format(**v) for c in calls)


LIB_ODE = lib("scipy=={scipy}, scipy.integrate.solve_ivp, method=Radau rtol=1e-8 atol=1e-25 J (atol_distance 1 m)",
              "astropy=={astropy}, astropy.cosmology.Planck18 (splined in igm_losses.cosmology)")
NUMBERS = {}


def put(key, value, unit, statement, produced_by, sigma=None, kind="numerical", tag="derived", choices=(),
        assumptions=(), inputs=(), checked_by=(), from_library=LIB_ODE):
    NUMBERS[key] = {"value": value, "unit": unit, "uncertainty": {"sigma": sigma, "kind": kind}, "tag": tag,
                    "statement": statement, "produced_by": ME + produced_by, "from_scratch": ME + produced_by,
                    "from_library": from_library, "choices": list(choices), "assumptions": list(assumptions),
                    "inputs": list(inputs), "checked_by": [c if "::" in c else TN + c for c in checked_by]}


def k_label(e):
    return f"1e{math.log10(e):.0f}"


@contextlib.contextmanager
def rtol_scaled(f):
    old = P["rtol"]
    P["rtol"] = old * f
    try:
        yield
    finally:
        P["rtol"] = old


# ------------------------------------------------------------------ medium and cosmology
def medium_numbers():
    zi, zf = P["z_init"], P["z_final"]
    n = Medium(zi, 0.0).n_H
    put("n_H_zinit", n, "m-3", f"Densidad de hidrógeno total n_H en z = {zi:g} (1 − Y_p) Ω_b ρ_c (1+z)³/m_p (A02)",
        "medium_numbers", sigma=None, kind="exact (formula of the inputs)", assumptions=["A02", "A10"],
        inputs=["Y_p", "m_p"], checked_by=["test_n_H_vs_astropy"],
        from_library=lib("astropy=={astropy}, Planck18.Ob0, Planck18.critical_density0"))
    dt = (cosmology.age(zf) - cosmology.age(zi)) / MYR
    put("dt_zinit_zfinal_Myr", float(dt), "Myr", f"Tiempo cósmico entre z = {zi:g} y z = {zf:g}", "medium_numbers",
        sigma=1e-8 * float(dt), assumptions=["A10"], inputs=["z_init", "z_final", "year"],
        choices=["error relativo < 1e-8 de los splines de cosmología (D12)"], checked_by=["test_dt_and_dz_vs_astropy"])
    t1 = cosmology.age(zi) + P["delta_t_marker"] * K.year
    dz = zi - float(cosmology.redshift(t1))
    put("dz_marker", dz, "", f"Δz recorrido en Δt = {P['delta_t_marker'] / 1e6:g} Myr después de z = {zi:g} (celda 12; E09)",
        "medium_numbers", sigma=1e-8 * (1 + zi), assumptions=["A10"], inputs=["z_init", "delta_t_marker", "year"],
        choices=["error relativo < 1e-8 en (1+z) de los splines (D12)"], checked_by=["test_dt_and_dz_vs_astropy"])


# ------------------------------------------------------------------ trajectories
def _summary(K0, x_e):
    o = integrate.evolve(K0, P["z_init"], P["z_final"], x_e, include=losses.PROCESSES, augmented=True)
    s = {"z_therm": float(cosmology.redshift(o["t_floor"])) if o["t_floor"] is not None else None,
         "t_therm": (o["t_floor"] - o["t"][0]) / MYR if o["t_floor"] is not None else None,
         "D_proper": o["D_proper"][-1] / MPC, "D_comoving": o["D_comoving"][-1] / MPC, "K_end": o["K"][-1]}
    for p in losses.PROCESSES:
        s["frac_" + p] = o["lost_" + p][-1] / K0
    s["closure"] = abs(sum(s["frac_" + p] for p in losses.PROCESSES) + s["K_end"] / K0 - 1)
    return s


def trajectory_numbers():
    E_th = FIG["total_cooling"]["energies_eV"]
    E_fr = FIG["accumulated_losses"]["energies_eV"]
    E_d = FIG["distance_travelled"]["energies_eV"]
    energies = sorted(set(E_th) | set(E_fr) | set(E_d))
    for x_e in SWEEP:
        for K0 in energies:
            t0 = time.time()
            a = _summary(K0, x_e)
            with rtol_scaled(1e-2):
                b = _summary(K0, x_e)
            print(f"  K0 = {K0:.0e} eV, x_e = {x_e:g}: z_therm = {a['z_therm']}  [{time.time() - t0:.0f} s]", flush=True)
            tag = f"K{k_label(K0)}_xe{x_e:g}"
            d = lambda q: abs(a[q] - b[q]) if a[q] is not None and b[q] is not None else None
            common = dict(assumptions=ALL_A, inputs=["z_init", "z_final", "K_floor", "B0", "Y_p", "T_CMB0", "rtol", "atol"],
                          choices=["incertidumbre = |valor(rtol) − valor(rtol/100)| (D05)",
                                   "todos los procesos (A13–A17); piso K_floor (A07)"])
            if K0 in E_th:
                if a["z_therm"] is None:
                    stm = f"El electrón con K_ini = {K0:.0e} eV (x_e = {x_e:g}) no llega a K_floor antes de z_final"
                else:
                    stm = f"Redshift en que K llega a K_floor = {P['K_floor']:g} eV, K_ini = {K0:.0e} eV en z = {P['z_init']:g}, x_e = {x_e:g}"
                put(f"z_therm_{tag}", a["z_therm"], "", stm, "trajectory_numbers", sigma=d("z_therm"),
                    checked_by=["test_z_therm_independent_driver", "test_trajectory_entries_consistent"], **common)
                put(f"t_therm_{tag}_Myr", a["t_therm"], "Myr", stm.replace("Redshift en que", "Tiempo desde la inyección hasta que"),
                    "trajectory_numbers", sigma=d("t_therm"), checked_by=["test_trajectory_entries_consistent"], **common)
            if K0 in E_fr:
                for p in losses.PROCESSES:
                    put(f"frac_{p}_{tag}", a["frac_" + p], "",
                        f"Fracción de K_ini perdida por {p} hasta el piso (o z_final), K_ini = {K0:.0e} eV, x_e = {x_e:g}",
                        "trajectory_numbers", sigma=max(d("frac_" + p), a["closure"]),
                        checked_by=["test_fractions_close_the_budget"], **common)
            if K0 in E_d:
                for q in ("D_proper", "D_comoving"):
                    put(f"{q}_{tag}_Mpc", a[q], "Mpc",
                        f"Distancia {'propia' if q == 'D_proper' else 'comóvil'} recorrida hasta el piso, K_ini = {K0:.0e} eV, x_e = {x_e:g} (A09)",
                        "trajectory_numbers", sigma=d(q), checked_by=["test_distance_bounds"], **common)


# ------------------------------------------------------------------ dominant-process transitions
def dominance_numbers():
    lo = math.log10(P["K_floor"] * 1.0001)
    grid = np.logspace(lo, 14.0, 700)
    for x_e in SWEEP:
        for z in (P["z_init"], P["z_final"]):
            dom = []
            for Ke in grid:
                r = losses.rates(Ke * K.e, z, x_e)
                dom.append(max(r, key=r.get))
            for i in range(len(grid) - 1):
                a, b = dom[i], dom[i + 1]
                if a == b:
                    continue
                def f(u, a=a, b=b):   # normalized difference: defined where one rate is 0 (ionization below B)
                    r = losses.rates(math.exp(u) * K.e, z, x_e, [a, b])
                    return (r[a] - r[b]) / (r[a] + r[b])
                u = brentq(f, math.log(grid[i]), math.log(grid[i + 1]), xtol=1e-12)
                Kx = math.exp(u)
                put(f"K_dom_{a}_to_{b}_z{z:g}_xe{x_e:g}", Kx, "eV",
                    f"Energía cinética en que el proceso de pérdida dominante pasa de {a} (debajo) a {b} (arriba), z = {z:g}, x_e = {x_e:g}",
                    "dominance_numbers", sigma=Kx * 1e-12, assumptions=ALL_A,
                    inputs=["B0", "Y_p", "T_CMB0", "K_floor"],
                    choices=["barrido log de 700 puntos entre K_floor y 1e14 eV y refinamiento con brentq sobre (L_a − L_b)/(L_a + L_b)"],
                    checked_by=["test_dominance_crossings"],
                    from_library=lib("scipy=={scipy}, scipy.optimize.brentq, xtol=1e-12"))


def gamma_ad_numbers():
    s = FIG["adiabatic_fraction_map"]["map"]
    zg = np.linspace(*s["z_range"], s["z_points"])
    a, b, n = s["K_eV"]["logspace"]
    Kg = np.logspace(a, b, n)
    for x_e in SWEEP:
        G = phase_space.adiabatic_fraction(zg, Kg, x_e)
        i, j = np.unravel_index(np.argmax(G), G.shape)
        put(f"gamma_ad_max_xe{x_e:g}", float(G[i, j]), "",
            f"Máximo de Γ_ad = L_ad/L_tot en z ∈ [{s['z_range'][0]:g}, {s['z_range'][1]:g}], K ∈ [1e{a}, 1e{b}] eV, x_e = {x_e:g}; "
            f"alcanzado en z = {zg[j]:.3f}, K = {Kg[i]:.4g} eV",
            "gamma_ad_numbers", sigma=None, kind="grid maximum (resolution in choices)", assumptions=ALL_A,
            inputs=["B0", "Y_p", "T_CMB0"],
            choices=[f"grilla de la figura 41: {s['z_points']} valores de z y {n} de K (log); el máximo verdadero puede ser "
                     "algo mayor que el de la grilla", f"argmax en z = {zg[j]!r}, K_eV = {Kg[i]!r}"],
            checked_by=["test_gamma_ad_max"], from_library="none")


# ------------------------------------------------------------------ Bethe comparison (E21)
def hydrogen_I0_eV():
    """I0 = R exp L(0), L(0) = Σ f ln(E/R) over the exact H(1s) oscillator strengths [Inokuti 1971 Eq. 4.62]."""
    with mp.workdps(30):
        f_n = lambda n: mp.mpf(2) ** 8 * n ** 5 * (n - 1) ** (2 * n - 4) / (3 * (n + 1) ** (2 * n + 4))
        k = lambda E: mp.sqrt(E - 1)
        dfdE = lambda E: mp.mpf(2) ** 7 * mp.exp(-4 / k(E) * mp.atan(k(E))) / (3 * E ** 4 * (1 - mp.exp(-2 * mp.pi / k(E))))
        pts = [1, 2, 10, mp.inf]
        L0 = mp.nsum(lambda n: f_n(n) * mp.log(1 - 1 / mp.mpf(n) ** 2), [2, mp.inf]) + mp.quad(lambda E: dfdE(E) * mp.log(E), pts)
        return float(mp.exp(L0)) * K.Ry_inf_eV


def bethe_numbers():
    I0 = hydrogen_I0_eV() * K.e
    put("I0_hydrogen", I0 / K.e, "eV", "Energía media de excitación del H(1s), I0 = R exp L(0) (Inokuti 1971 ec. 4.62)",
        "hydrogen_I0_eV", sigma=1e-10, kind="numerical (mpmath 30 digits)", inputs=["c", "h", "e", "m_e", "mu0"],
        choices=["intensidades de oscilador exactas del H (recordadas, Bethe & Salpeter §69–71), validadas con TRK = 1"],
        checked_by=["test_bethe_ratios"], from_library=lib("mpmath=={mpmath}, mp.nsum, mp.quad, dps=30"))
    m = Medium(P["z_init"], 0.0)
    for T in (1e4, 1e5, 1e6, 1e8, 1e10):
        TJ = T * K.e
        b2 = 1 - 1 / (1 + TJ / K.m_e_c2) ** 2
        mv2 = K.m_e * K.c ** 2 * b2
        s = math.sqrt(1 - b2)
        br = math.log(mv2 * TJ / (I0 ** 2 * (1 - b2))) - (2 * s + b2) * math.log(2) + 1 - b2 + (1 - s) ** 2 / 8
        bethe = m.n_HI * K.c * math.sqrt(b2) * K.Ry_inf * 8 * math.pi * K.a0 ** 2 * (K.Ry_inf / mv2) * br
        r = (excitation.loss_rate(TJ, m) + ionization.loss_rate(TJ, m)) / bethe
        put(f"bethe_ratio_T{k_label(T)}", r, "",
            f"(excitación + ionización)/Bethe relativista con intercambio (Inokuti 1971 ec. 4.65) a T = {T:.0e} eV (E21)",
            "bethe_numbers", sigma=None, kind="model comparison", assumptions=["A03", "A13", "A14"],
            inputs=["B_ion_RBED", "secondary_E_bar", "secondary_exponent"],
            choices=["el test de Bethe (5 %) cubre sólo 1e4–1e6 eV; los valores a 1e8 y 1e10 eV no tienen check y son la errata E21"],
            checked_by=["test_bethe_ratios"] + ([f"electron_losses_IGM/tests/test_collisions.py::test_collisional_stopping_vs_bethe[{T:.1f}]"] if T <= 1e6 else []), from_library="none")


def aggregate_numbers():
    """Aggregates of entries already in NUMBERS (no new physics): maxima of fractions, spread of z_therm."""
    for p in ("adiabatic", "synchrotron", "bremsstrahlung", "coulomb"):
        keys = [k for k in NUMBERS if k.startswith(f"frac_{p}_K")]
        best = max(keys, key=lambda k: NUMBERS[k]["value"])
        put(f"frac_max_{p}", NUMBERS[best]["value"], "",
            f"Máximo de la fracción de K_ini perdida por {p} sobre las energías de la figura 37 y todos los x_e (alcanzado en {best})",
            "aggregate_numbers", sigma=NUMBERS[best]["uncertainty"]["sigma"], assumptions=ALL_A, inputs=["K_floor"],
            choices=[f"máximo sobre {len(keys)} entradas frac_{p}_*"], checked_by=["test_aggregates"], from_library="none")
    for x_e in SWEEP:
        keys = [k for k in NUMBERS if k.startswith("z_therm_K") and k.endswith(f"_xe{x_e:g}")
                and float(k.split("_K")[1].split("_")[0]) >= 1e9 and NUMBERS[k]["value"] is not None]
        v = [NUMBERS[k]["value"] for k in keys]
        put(f"z_therm_spread_Kge1e9_xe{x_e:g}", max(v) - min(v), "",
            f"Dispersión (máx − mín) del redshift de termalización para K_ini ≥ 1e9 eV, x_e = {x_e:g}",
            "aggregate_numbers", sigma=max(NUMBERS[k]["uncertainty"]["sigma"] for k in keys), assumptions=ALL_A,
            inputs=["K_floor"], choices=[f"sobre {len(keys)} entradas z_therm_*"], checked_by=["test_aggregates"],
            from_library="none")


def compute_all():
    """Fill NUMBERS with every entry of this subproject and return it (no file is written here)."""
    t0 = time.time()
    medium_numbers()
    bethe_numbers()
    print("medium, Bethe done", flush=True)
    dominance_numbers()
    print(f"dominance done [{time.time() - t0:.0f} s]", flush=True)
    gamma_ad_numbers()
    print(f"gamma_ad done [{time.time() - t0:.0f} s]", flush=True)
    trajectory_numbers()
    aggregate_numbers()
    print(f"electron_losses_IGM numbers: {len(NUMBERS)} entries [{time.time() - t0:.0f} s]", flush=True)
    return NUMBERS


if __name__ == "__main__":
    print("This module only computes; the single writer of provenance/numbers.json is scripts/compute_numbers.py")
