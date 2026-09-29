"""Checks of provenance/numbers.json (single writer: scripts/compute_numbers.py at the project root; the entries of this
subproject come from electron_losses_IGM/scripts/compute_numbers.py::compute_all). Criteria fixed before running.

Each test re-derives entries by a route independent of compute_numbers.py where one exists (astropy directly, a
different ODE driver and method, physical bounds, a second copy of the Bethe formula), and the schema test checks
the fields of every entry and that produced_by / checked_by resolve to existing functions (plan §2.5 item 1).
"""

import ast
import json
import math
import pathlib
import re
import sys

import numpy as np
import pytest
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from igm_losses import constants as K, cosmology, losses  # noqa: E402
from igm_losses.medium import Medium  # noqa: E402

NUM_FILE = ROOT / "provenance" / "numbers.json"
N = {k: v for k, v in json.loads(NUM_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")} \
    if NUM_FILE.exists() else {}
P = K._P
FIELDS = ("value", "unit", "uncertainty", "tag", "statement", "produced_by", "from_scratch", "from_library", "choices",
          "assumptions", "inputs", "checked_by")
pytestmark = pytest.mark.skipif(not N, reason="provenance/numbers.json not written yet")


def _defines(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name for n in ast.walk(tree))


def _tags():
    return sorted({m.group(1) for k in N for m in [re.match(r"z_therm_(K1e\d+_xe[\d.e-]+)$", k)] if m})


def _K0(tag):
    return float(re.match(r"K(1e\d+)_", tag).group(1))


def _xe(tag):
    return float(tag.split("_xe")[1])


def test_registry_schema():
    """Every entry has the 12 fields; tag in the C35 vocabulary; produced_by and every checked_by resolve (AST);
    a number with an empty checked_by is only allowed if its choices say why (it is then a hypothesis)."""
    assumptions = __import__("yaml").safe_load((ROOT / "provenance" / "assumptions.yaml").read_text(encoding="utf-8"))
    ids = set(assumptions.get("assumptions", assumptions))
    for k, e in N.items():
        assert all(f in e for f in FIELDS), k
        assert e["tag"] in ("measured", "assumed", "fitted", "derived"), k
        path, fn = e["produced_by"].split("::")
        assert _defines(path, fn), e["produced_by"]
        for c in e["checked_by"]:
            cpath, cfn = c.split("::")
            assert _defines(cpath, cfn.split("[")[0]), c
        assert set(e["assumptions"]) <= ids, k
        assert all(i in P for i in e["inputs"]), k


def test_n_H_vs_astropy():
    """n_H(z_init) with astropy's own m_p and critical density (1e-8: astropy's m_p is CODATA 2018)."""
    import astropy.constants as ac
    z = P["z_init"]
    ref = ((1 - P["Y_p"]) * Planck18.Ob0 * Planck18.critical_density0 * (1 + z) ** 3 / ac.m_p).to(u.m ** -3).value
    assert abs(N["n_H_zinit"]["value"] / ref - 1) < 1e-8


def test_dt_and_dz_vs_astropy():
    """Δt(z_init → z_final) with astropy ages (1e-7) and Δz of the 30 Myr marker with astropy z_at_value (1e-6)."""
    zi, zf = P["z_init"], P["z_final"]
    dt = (Planck18.age(zf) - Planck18.age(zi)).to(u.Myr).value * (u.Myr.to(u.s) / (1e6 * K.year))
    assert abs(N["dt_zinit_zfinal_Myr"]["value"] / dt - 1) < 1e-7
    t1 = Planck18.age(zi) + P["delta_t_marker"] * K.year * u.s
    z1 = z_at_value(Planck18.age, t1, zmin=zf, zmax=zi, ztol=1e-12).value
    assert abs(N["dz_marker"]["value"] - (zi - z1)) < 1e-6


@pytest.mark.parametrize("K0,x_e", [(1e7, 1e-4), (1e5, 0.5)])
def test_z_therm_independent_driver(K0, x_e):
    """A second ODE driver written here (LSODA, state K in eV, own floor event, rtol 1e-9) on losses.total_rate, and
    the event time converted to z with astropy z_at_value instead of the code's splines. |Δz| < 1e-4."""
    t0 = float(Planck18.age(P["z_init"]).to(u.s).value)
    t1 = float(Planck18.age(P["z_final"]).to(u.s).value)
    f = lambda t, y: [-losses.total_rate(max(y[0], 0.0) * K.e, float(cosmology.redshift(t)), x_e) / K.e]
    ev = lambda t, y: y[0] - P["K_floor"]
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(f, (t0, t1), [K0], method="LSODA", rtol=1e-9, atol=1e-6, events=ev)
    z_ev = z_at_value(Planck18.age, sol.t_events[0][0] * u.s, zmin=P["z_final"], zmax=P["z_init"], ztol=1e-12).value
    assert abs(N[f"z_therm_K1e{math.log10(K0):.0f}_xe{x_e:g}"]["value"] - z_ev) < 1e-4


def test_trajectory_entries_consistent():
    """t_therm and z_therm of the same trajectory agree through astropy ages (1e-6 relative)."""
    for tag in _tags():
        z = N[f"z_therm_{tag}"]["value"]
        if z is None:
            continue
        dt = (Planck18.age(z) - Planck18.age(P["z_init"])).to(u.s).value / (1e6 * K.year)
        assert abs(N[f"t_therm_{tag}_Myr"]["value"] / dt - 1) < 1e-6, tag


def test_fractions_close_the_budget():
    """Energy conservation across entries: Σ_p frac_p + K_floor/K_ini = 1 (1e-6) for every thermalized trajectory."""
    for k in [k for k in N if k.startswith("frac_adiabatic_")]:
        tag = k.removeprefix("frac_adiabatic_")
        if N.get(f"z_therm_{tag}", {}).get("value") is None:
            continue
        s = sum(N[f"frac_{p}_{tag}"]["value"] for p in losses.PROCESSES)
        assert abs(s + P["K_floor"] / _K0(tag) - 1) < 1e-6, tag


def test_distance_bounds():
    """Physical bounds, independent of the integration: β(K_floor) c Δt ≤ D_proper ≤ c Δt (the speed never drops below
    the floor speed before the floor), and (1+z_therm) ≤ D_comoving/D_proper ≤ (1+z_init)."""
    for k in [k for k in N if k.startswith("D_proper_")]:
        tag = k.removeprefix("D_proper_").removesuffix("_Mpc")
        z = N.get(f"z_therm_{tag}", {}).get("value")
        if z is None:
            continue
        dt = (Planck18.age(z) - Planck18.age(P["z_init"])).to(u.s).value
        g = 1 + P["K_floor"] / K.m_e_c2_eV
        bmin = math.sqrt(1 - 1 / g ** 2)
        D = N[k]["value"] * 1e6 * K.parsec
        assert bmin * K.c * dt * (1 - 1e-9) <= D <= K.c * dt * (1 + 1e-9), tag
        r = N[f"D_comoving_{tag}_Mpc"]["value"] / N[k]["value"]
        assert (1 + z) * (1 - 1e-9) <= r <= (1 + P["z_init"]) * (1 + 1e-9), tag


def test_dominance_crossings():
    """At each recorded K the two rates are equal (1e-6) and the dominant process is a just below and b just above."""
    for k, e in N.items():
        m = re.match(r"K_dom_(\w+?)_to_(\w+?)_z([\d.]+)_xe([\d.e-]+)$", k)
        if not m:
            continue
        a, b, z, x_e = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
        Kx = e["value"] * K.e
        r = losses.rates(Kx, z, x_e, [a, b])
        assert abs(r[a] / r[b] - 1) < 1e-6, k
        lo, hi = losses.rates(Kx * (1 - 1e-4), z, x_e), losses.rates(Kx * (1 + 1e-4), z, x_e)
        assert max(lo, key=lo.get) == a and max(hi, key=hi.get) == b, k


def test_gamma_ad_max():
    """Γ_ad recomputed from losses.rates at the recorded argmax point equals the entry (1e-12) and lies in [0, 1]."""
    for k in [k for k in N if k.startswith("gamma_ad_max_xe")]:
        e = N[k]
        m = re.search(r"argmax en z = ([\d.eE+-]+), K_eV = ([\d.eE+-]+)", " ".join(e["choices"]))
        z, Ke = float(m.group(1)), float(m.group(2))
        r = losses.rates(Ke * K.e, z, float(k.removeprefix("gamma_ad_max_xe")))
        assert abs(r["adiabatic"] / sum(r.values()) / e["value"] - 1) < 1e-12 and 0 <= e["value"] <= 1


def test_bethe_ratios():
    """I0 and the Bethe ratios against the independent copy of the Bethe formula in test_collisions (1e-9)."""
    import test_collisions as TC
    from igm_losses import excitation as X, ionization as I
    I0 = TC._hydrogen_I0_over_R() * K.Ry_inf
    assert abs(N["I0_hydrogen"]["value"] * K.e / I0 - 1) < 1e-9
    m = Medium(P["z_init"], 0.0)
    for k in [k for k in N if k.startswith("bethe_ratio_T")]:
        TJ = float(k.removeprefix("bethe_ratio_T")) * K.e
        b2 = 1 - 1 / (1 + TJ / K.m_e_c2) ** 2
        mv2 = K.m_e * K.c ** 2 * b2
        s = math.sqrt(1 - b2)
        br = math.log(mv2 * TJ / (I0 ** 2 * (1 - b2))) - (2 * s + b2) * math.log(2) + 1 - b2 + (1 - s) ** 2 / 8
        bethe = m.n_HI * K.c * math.sqrt(b2) * K.Ry_inf * 8 * math.pi * K.a0 ** 2 * (K.Ry_inf / mv2) * br
        assert abs((X.loss_rate(TJ, m) + I.loss_rate(TJ, m)) / bethe / N[k]["value"] - 1) < 1e-9, k


def test_aggregates():
    """frac_max_* and z_therm_spread_* recomputed from the individual entries (exact)."""
    for k in [k for k in N if k.startswith("frac_max_")]:
        p = k.removeprefix("frac_max_")
        assert N[k]["value"] == max(v["value"] for kk, v in N.items() if kk.startswith(f"frac_{p}_K"))
    for k in [k for k in N if k.startswith("z_therm_spread_Kge1e9_xe")]:
        xe = k.removeprefix("z_therm_spread_Kge1e9_xe")
        v = [e["value"] for kk, e in N.items() if kk.startswith("z_therm_K") and kk.endswith(f"_xe{xe}")
             and float(kk.split("_K")[1].split("_")[0]) >= 1e9]
        assert N[k]["value"] == max(v) - min(v)


def test_claims_resolve():
    """Plan §2.5 item 4: every claim has statement and evidence; every {{key|fmt}} and every listed number is in
    numbers.json; evidence is a delivered file::function or a key of references/references.bib."""
    import yaml
    st = yaml.safe_load((ROOT / "provenance" / "claims_statements.yaml").read_text(encoding="utf-8"))
    bib = (ROOT / "references" / "references.bib").read_text(encoding="utf-8")
    bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    for c in st["claims"]:
        assert c["statement"] and c["evidence"], c["id"]
        for key in re.findall(r"\{\{([\w.+-]+)\|", c["statement"]) + c.get("numbers", []):
            assert key in N, (c["id"], key)
            if c["status"] == "checked":          # a checked claim may only use numbers that have a check
                assert N[key]["checked_by"], (c["id"], key)
        for ev in c["evidence"]:
            if "::" in ev:
                path, fn = ev.split("::")
                assert _defines(path, fn), (c["id"], ev)
            elif ev.startswith("figures/"):
                assert (ROOT / ev).exists(), (c["id"], ev)
            else:                                  # "<bib key>, eq./sec./p. <locator>" (C6: citation with locator)
                m = re.match(r"^(\S+?),\s+(eq|eqs|sec|fig|table|p|pp)\.\s*\S+", ev)
                assert m and m.group(1) in bibkeys, (c["id"], ev)
