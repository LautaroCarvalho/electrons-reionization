#!/usr/bin/env python3
r"""
Merge the z = 20 result registries into one file with every key suffixed _z20.

WHY. The provenance gate refuses (check C3b) two registries in which the same
key carries different values -- which is exactly what "the same calculation at
a second redshift" produces. Suffixing keeps both redshifts auditable in one
paper without the gate having to guess which one a tag meant.

Input : photon_vs_electron_results_z20.json, yield_comparison_results_z20.json
Output: provenance/registry_z20.json   (a "derived" section, keys ..._z20)
"""
import project_paths  # noqa: F401  -- anchors CWD to the project root
import json, os

SRC = ["photon_vs_electron_results_z20.json", "yield_comparison_results_z20.json"]
SECTIONS = ("derived", "cosmology", "scenario", "inputs")

out, seen = {}, {}
for path in SRC:
    doc = json.load(open(path))
    for sec in SECTIONS:
        for k, v in (doc.get(sec) or {}).items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            key = f"{k}_z20"
            if key in out and abs(float(out[key]) - float(v)) > 1e-12 * max(
                    1.0, abs(float(v))):
                raise SystemExit(
                    f"CLASH: {key} = {out[key]} in {seen[key]} but {v} in {path}")
            out[key], seen[key] = float(v), path

# --- cross-redshift quantities, COMPUTED here from the two registries ------
# The paper quotes changes between the redshifts ("falls by 36.5%"). Those are
# numerals like any other and need a source, so they are derived here rather
# than typed into the prose.
z10 = {}
for path in ["photon_vs_electron_results.json", "yield_comparison_results.json"]:
    doc = json.load(open(path))
    for sec in SECTIONS:
        for k, v in (doc.get(sec) or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                z10[k] = float(v)

F_ESC_FID, EPS_CR_FE_FID = 0.10, 1.0e-3        # photon_vs_electron.py


def drop(key):
    """fractional fall from z=10 to z=20 (positive means it fell)."""
    return 1.0 - out[key + "_z20"] / z10[key]


out["loss_sat_drop_frac_z20"] = drop("N_e_loss_sat")
out["Esat_drop_frac_z20"] = drop("N_e_loss_ic_sat")
out["zeta_e_drop_frac_z20"] = drop("zeta_e_s")
# how much the competition ratio itself moves between the two redshifts
out["zeta_ratio_rise_z20_pct"] = 100.0 * (
    out["zeta_ratio_z20"] / z10["zeta_ratio"] - 1.0)
out["n_H_ratio_z20_over_z10"] = out["n_H_cm3_route1_z20"] / z10["n_H_cm3_route1"]
out["H_ratio_z20_over_z10"] = out["H_z_s_route2_z20"] / z10["H_z_s_route2"]
out["U_CMB_ratio_z20_over_z10"] = (21.0 / 11.0) ** 4
# parity escape fraction: both channels are linear in their own normalisation,
# so zeta_gamma = zeta_e at f_esc = F_ESC_FID / (ratio * EPS_CR_FE_FID).
out["fesc_parity_pct_z10"] = 100.0 * F_ESC_FID / (z10["zeta_ratio"] * EPS_CR_FE_FID)
out["fesc_parity_pct_z20"] = 100.0 * F_ESC_FID / (out["zeta_ratio_z20"] * EPS_CR_FE_FID)

# lookback between the last row of Donnan+24 Table 3 and z = 20.
# The table ends at z = 14.5, NOT 12.5 -- corrected 2026-09-12 by reading the
# table rather than trusting the earlier note.
try:
    from astropy.cosmology import Planck18
    out["dt_z14p5_to_z20_Myr"] = float(
        (Planck18.age(14.5) - Planck18.age(20.0)).to("Myr").value)
    out["dt_z12p5_to_z20_Myr"] = float(
        (Planck18.age(12.5) - Planck18.age(20.0)).to("Myr").value)
except Exception as exc:                      # never invent it if astropy is absent
    print(f"   [skip] lookback keys: {exc}")

os.makedirs("provenance", exist_ok=True)
dst = "provenance/registry_z20.json"
json.dump({"derived": out}, open(dst, "w"), indent=2, sort_keys=True)
print(f"[ARTEFACT] {dst}  --  {len(out)} keys from {len(SRC)} registries")
