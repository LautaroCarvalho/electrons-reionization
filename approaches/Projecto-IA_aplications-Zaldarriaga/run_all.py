#!/usr/bin/env python3
r"""
Run the whole project, in dependency order, one step at a time.

    python3 run_all.py --dry-run      # print the order and what each step touches
    python3 run_all.py --stale        # report which steps are out of date, run nothing
    python3 run_all.py                # run everything that is stale
    python3 run_all.py --force        # run everything regardless
    python3 run_all.py --from source_map
    python3 run_all.py --only xcomp

WHY THIS EXISTS.  Nineteen scripts with an undocumented order is a project only
its author can rebuild.  Worse, nothing detected staleness: after the 2026-09-15
folder reorganisation photon_vs_electron.py silently wrote its registry into
Images/ while the provenance gate kept validating the stale copy at the root.
It passed -- for the wrong reason.  A step here is out of date when any of its
inputs is newer than any of its outputs, and --stale says so without running
anything.

STRICTLY SEQUENTIAL, deliberately.  This machine OOM-kills two concurrent
imports of the physics stack.  Never add parallelism here without checking that.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "python"
PARAMS = ["parameters.yaml"]        # every step depends on the source of truth


def S(name, cmd, inputs, outputs, note=""):
    return dict(name=name, cmd=cmd, inputs=PARAMS + inputs,
                outputs=outputs, note=note)


# Registries the gate must be given.  Kept here so there is ONE list: passing an
# incomplete set makes C3 report "unresolved key", which looks like drift but is
# only a short command line.  The ledger step consumes the same list.
REGISTRY_FILES = [
    "photon_vs_electron_results.json", "yield_comparison_results.json",
    "imf_comparison_results.json",
    "provenance/registry_z20.json", "provenance/reionization_budget.json",
    "provenance/figure_claims.json", "provenance/audit_registry.json",
    "provenance/source_map.json", "provenance/xcomp.json",
]


# Order matters: a step may only depend on artefacts produced above it.
STEPS = [
    S("parameters", ["python3", "python/parameters.py", "--check"], [], [],
      "validates the source of truth; produces nothing"),
    S("yield_comparison", ["python3", "python/yield_comparison.py"], [],
      ["yield_comparison_results.json", "Images/yield_comparison_fig.png"]),
    # ionization_yield.tex SHOWS this figure, but nothing rebuilt it: the file on
    # disk predated the pipeline and no step owned it.  Found by gate C7b.
    S("ionization_yield_igm", ["python3", "python/ionization_yield_igm.py"], [],
      ["Images/ionization_yield_fig_igm.png", "Images/ionization_yield_fig_igm.pdf"],
      "models D and E on the igm_losses prescription; figure only"),
    S("photon_vs_electron", ["python3", "python/photon_vs_electron.py"], [],
      ["photon_vs_electron_results.json",
       "provenance/photon_vs_electron_md5.txt",
       "Images/photon_vs_electron_fig1.png", "Images/photon_vs_electron_fig3.png"]),
    S("reionization_budget", ["python3", "python/reionization_budget.py"], [],
      ["provenance/reionization_budget.json"]),
    S("verify_kim_and_z", ["python3", "python/verify_kim_and_z.py"], [],
      ["provenance/audit_registry.json"]),
    S("imf_comparison", ["python3", "python/imf_comparison.py"], [],
      ["imf_comparison_results.json", "Images/imf_comparison_fig.png"]),
    S("source_map", ["python3", "python/source_map.py"], [],
      ["provenance/source_map.json", "Images/source_budget_plane.png",
       "Images/source_budget_objects.png", "Images/source_spectral_index.png"]),
    S("xcomp", ["python3", "python/make_xcomp_definitions.py",
                "--check", "--fig", "--tex"],
      ["provenance/source_map.json"],
      ["provenance/xcomp.json", "Images/xcomp_map.png",
       "Text_files/xcomp_definitions.tex"],
      "imports source_map's master grids"),
    S("figure_claims", ["python3", "python/figure_claims.py"], [],
      ["provenance/figure_claims.json"]),
    S("redshift_suite", ["python3", "python/run_redshift_suite.py"],
      ["photon_vs_electron_results.json", "yield_comparison_results.json"],
      ["photon_vs_electron_results_z20.json", "yield_comparison_results_z20.json",
       "redshift_suite_results.json"],
      "re-runs the z=20 snapshot; slow"),
    S("z20_registry", ["python3", "python/make_z20_registry.py"],
      ["photon_vs_electron_results_z20.json", "yield_comparison_results_z20.json"],
      ["provenance/registry_z20.json"]),
    S("inputs_table", ["python3", "python/make_inputs_table.py"], [],
      ["Text_files/inputs_table.tex"]),
    # Last, deliberately: the ledger reads every registry and both manuscripts,
    # so it can only be honest once everything above it has been rebuilt.
    S("ledger", ["python3", "python/make_ledger.py"],
      REGISTRY_FILES + ["papers/references.bib",
                        "Text_files/photon_vs_electron.tex",
                        "Text_files/ionization_yield.tex"],
      ["provenance/ledger.json", "Text_files/ledger.html"],
      "reads every registry + references.bib; computes nothing itself"),
]

REGISTRIES = ",".join(REGISTRY_FILES)

MANUSCRIPTS = ["photon_vs_electron", "ionization_yield", "inputs_table",
               "xcomp_definitions"]


def mtime(rel):
    p = ROOT / rel
    return p.stat().st_mtime if p.exists() else None


def staleness(step):
    """(is_stale, reason).  Missing output beats out-of-date output."""
    if not step["outputs"]:
        return True, "no outputs (always runs)"
    missing = [o for o in step["outputs"] if mtime(o) is None]
    if missing:
        return True, f"missing {missing[0]}"
    oldest_out = min(mtime(o) for o in step["outputs"])
    newer = [i for i in step["inputs"]
             if mtime(i) is not None and mtime(i) > oldest_out]
    if newer:
        return True, f"{newer[0]} is newer than its outputs"
    return False, "up to date"


def run(step, dry):
    stale, why = staleness(step)
    mark = "STALE" if stale else "ok   "
    print(f"\n[{mark}] {step['name']:20s} {why}")
    if step["note"]:
        print(f"         note: {step['note']}")
    print(f"         consumes: {', '.join(step['inputs'])}")
    print(f"         produces: {', '.join(step['outputs']) or '(nothing)'}")
    if dry:
        return True
    t0 = time.time()
    r = subprocess.run(step["cmd"], cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"         FAILED after {dt:.0f}s (exit {r.returncode})")
        print("         " + "\n         ".join(r.stdout.strip().splitlines()[-6:]))
        print("         " + "\n         ".join(r.stderr.strip().splitlines()[-6:]))
        return False
    print(f"         done in {dt:.0f}s")
    n = stamp_outputs(step)
    if n:
        print(f"         stamped {n} registry file(s) with the scenario hash")
    return True


def stamp_outputs(step):
    """Write the scenario stamp into every JSON registry this step produced.

    Done here rather than in each producer so that no physics module has to know
    about scenario bookkeeping, and so a step added later is stamped for free.
    """
    import json
    sys.path.insert(0, str(PY))
    import parameters as PR
    n = 0
    for o in step["outputs"]:
        if not o.endswith(".json"):
            continue
        f = ROOT / o
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        d["_scenario"] = PR.stamp()
        f.write_text(json.dumps(d, indent=2, sort_keys=True))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stale", action="store_true", help="report only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-tex", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(PY))
    import parameters as PR
    print("=" * 74)
    print(f"run_all  --  scenario '{PR._P['meta']['scenario_name']}'  "
          f"hash {PR.scenario_hash()}")
    print("=" * 74)

    steps = STEPS
    if a.only:
        steps = [s for s in steps if s["name"] == a.only] or steps
    elif a.start:
        idx = [i for i, s in enumerate(steps) if s["name"] == a.start]
        steps = steps[idx[0]:] if idx else steps

    ran = skipped = 0
    for s in steps:
        stale, _ = staleness(s)
        if a.stale:
            run(s, dry=True)
            continue
        if not stale and not a.force:
            print(f"\n[ok   ] {s['name']:20s} up to date, skipping")
            skipped += 1
            continue
        if not run(s, dry=a.dry_run):
            print("\nSTOPPING: a step failed; later steps would build on it.")
            return 1
        ran += 1

    if a.dry_run or a.stale:
        print(f"\n(dry run: {len(steps)} steps listed, nothing executed)")
        return 0

    if not a.skip_tex:
        print("\n--- LaTeX ---")
        for m in MANUSCRIPTS:
            src = ROOT / "Text_files" / f"{m}.tex"
            if not src.exists():
                continue
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", m],
                               cwd=ROOT / "Text_files", capture_output=True)
            print(f"  built {m}.pdf")

        print("\n--- provenance gate ---")
        for m in ("photon_vs_electron", "ionization_yield"):
            r = subprocess.run(
                ["python3", "python/check_provenance.py",
                 f"Text_files/{m}.tex", REGISTRIES],
                cwd=ROOT, capture_output=True, text=True)
            npass = r.stdout.count("[PASS]")
            nfail = r.stdout.count("[FAIL]")
            print(f"  {m:24s} {npass} pass, {nfail} fail")

    print(f"\nran {ran}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
