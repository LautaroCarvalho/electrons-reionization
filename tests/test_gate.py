"""Can the provenance gate fail? (scripts/check_provenance.py; C18/C23 for the gate itself.)

Moved here from electron_losses_IGM/tests/test_gate.py on 2026-09-28 (user decision): the gate covers the whole
project, so its tests live at the root; the fixture now also copies EM_cascades/, and the EM_cascades cases (one
figure-record writer per subproject, prefix ↔ produced_by, C8 on its README) are included.

Each test builds a minimal copy of the project in a temporary directory (registries, code, READMEs, the .bib with empty
placeholder PDFs, empty placeholder figures, and the originals listed in MANIFEST.txt), plants ONE error, runs the
gate with --root on the copy and requires that exactly the targeted check FAILs and the exit status is 1. The clean
copy must PASS with exit status 0. The real project is never modified.
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

PROJ = pathlib.Path(__file__).resolve().parents[1]
GATE = PROJ / "scripts" / "check_provenance.py"


def make_fixture(tmp):
    root = tmp / "proj"
    root.mkdir()
    shutil.copy(PROJ / "CLAUDE.md", root)
    shutil.copytree(PROJ / "provenance", root / "provenance")
    shutil.copytree(PROJ / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJ / "tests", root / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    for sub in ("electron_losses_IGM", "EM_cascades"):
        for d in ("src", "scripts", "tests"):
            if (PROJ / sub / d).is_dir():
                shutil.copytree(PROJ / sub / d, root / sub / d, ignore=shutil.ignore_patterns("__pycache__", "last_run.txt"))
    shutil.copy(PROJ / "EM_cascades" / "README.md", root / "EM_cascades")
    for f in ("README.md", "labels.yaml"):
        shutil.copy(PROJ / "electron_losses_IGM" / f, root / "electron_losses_IGM")
    (root / "plantilla_proyecto_final").mkdir()
    (root / "plantilla_proyecto_final" / "PLAN_TRABAJO_FINAL.md").touch()
    (root / "electron_losses_IGM" / "reference_figures").mkdir()
    for p in (PROJ / "electron_losses_IGM" / "reference_figures").iterdir():
        (root / "electron_losses_IGM" / "reference_figures" / p.name).touch()
    (root / "figures").mkdir()
    for p in (PROJ / "figures").iterdir():
        (root / "figures" / p.name).touch()
    (root / "references").mkdir()
    bib = (PROJ / "references" / "references.bib").read_text(encoding="utf-8")
    (root / "references" / "references.bib").write_text(bib, encoding="utf-8")
    for f in re.findall(r"\bfile\s*=\s*[{\"]([^}\"]+)[}\"]", bib):
        (root / "references" / f).touch()
    for line in (PROJ / "provenance" / "MANIFEST.txt").read_text(encoding="utf-8").splitlines():
        if "<-" in line and not line.lstrip().startswith("#"):
            orig = line.split("<-", 1)[1].strip()
            (root / orig).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(PROJ / orig, root / orig)
    return root


def run_gate(root):
    r = subprocess.run([sys.executable, str(root / "scripts" / "check_provenance.py"), "--root", str(root)],
                       capture_output=True, text=True, timeout=300)
    status = {m.group(1): m.group(2) for m in re.finditer(r"^(\S+)\s+(PASS|FAIL|SKIPPED)\b", r.stdout, re.M)
              if m.group(1) != "GATE"}
    return r.returncode, status, r.stdout


@pytest.fixture
def root(tmp_path):
    return make_fixture(tmp_path)


def test_clean_copy_passes(root):
    code, st, out = run_gate(root)
    assert code == 0, out
    assert all(v in ("PASS", "SKIPPED") for v in st.values()), out


def _edit(path, old, new):
    s = path.read_text(encoding="utf-8")
    assert s.count(old) >= 1, old
    path.write_text(s.replace(old, new, 1), encoding="utf-8")


def _numbers(root, fn):
    p = root / "provenance" / "numbers.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    fn(d)
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


PLANTS = {
    "C80": lambda r: _edit(r / "provenance" / "parameters.yaml", "value: 0.2454", 'value: "0.2454"'),
    "NUMBERS": lambda r: _numbers(r, lambda d: d["n_H_zinit"].pop("checked_by")),
    "C4": lambda r: (r / "electron_losses_IGM" / "scripts" / "rogue.py").write_text(
        'import pathlib\nROOT = pathlib.Path(".")\nOUT = ROOT / "provenance" / "numbers.json"\nOUT.write_text("{}")\n'),
    "C5/C6/C37": lambda r: _edit(r / "provenance" / "claims.yaml", "- electron_losses_IGM/scripts/compute_numbers.py::dominance_numbers",
                                 "- well known"),
    "C2-claims": lambda r: _numbers(r, lambda d: d["K_dom_ionization_to_ic_z10_xe0.0001"].update(
        value=2 * d["K_dom_ionization_to_ic_z10_xe0.0001"]["value"])),
    "C7": lambda r: (r / "figures" / "igm_orphan.png").touch(),
    "C8": lambda r: _edit(r / "electron_losses_IGM" / "README.md", "## Estado", "## Estado\n\nVer `scripts/nonexistent_helper.py`.\n"),
    "C9-bib": lambda r: (r / "references" / "Kim (2000).pdf").unlink(),
    "MR7": lambda r: _edit(r / "provenance" / "data" / "CS_int_1.txt", "\n", "\n0\n"),
}


@pytest.mark.parametrize("check", list(PLANTS))
def test_planted_error_is_caught(root, check):
    PLANTS[check](root)
    code, st, out = run_gate(root)
    assert code == 1 and st.get(check) == "FAIL", out
    others = [k for k, v in st.items() if v == "FAIL" and k != check]
    assert not others, f"planted {check}, but also failed {others}\n{out}"


def _paper(root, body):
    (root / "paper").mkdir(exist_ok=True)
    (root / "paper" / "main.tex").write_text(
        "\\documentclass{article}\n\\newcommand{\\dataref}[2]{#2}\n\\begin{document}\n" + body + "\n\\end{document}\n",
        encoding="utf-8")


def test_document_checks_run_and_can_fail(root):
    """With paper/main.tex present, C1/C2/C3/C9-doc stop being SKIPPED: a correct tagged value passes; a wrong literal
    fails C2; a bare numeral fails C1; an unknown key fails C3; an unknown citation fails C9-doc."""
    N = json.loads((root / "provenance" / "numbers.json").read_text(encoding="utf-8"))
    v = N["n_H_zinit"]["value"]
    _paper(root, f"n_H vale \\dataref{{n_H_zinit}}{{{v:.1f}}} m$^{{-3}}$ \\cite{{Planck2018VI}}.")
    code, st, out = run_gate(root)
    assert code == 0 and all(st[c] == "PASS" for c in ("C1", "C2", "C3", "C9-doc")), out
    for body, check in ((f"\\dataref{{n_H_zinit}}{{{v * 1.1:.1f}}}", "C2"), ("un valor de 4.2 sin etiqueta", "C1"),
                        ("\\dataref{no_such_key}{1.0}", "C3"), ("\\cite{NoSuchPaper2099}", "C9-doc")):
        _paper(root, body)
        code, st, out = run_gate(root)
        assert code == 1 and st[check] == "FAIL", (check, out)


# ------------------------------------------------------------------ EM_cascades cases
def test_second_record_writer_in_em_cascades_is_caught(root):
    (root / "EM_cascades" / "scripts" / "rogue_fig.py").write_text(
        'import pathlib\nROOT = pathlib.Path(".")\nREC = ROOT / "provenance" / "figures"\n(REC / "x.json").write_text("{}")\n')
    code, st, out = run_gate(root)
    assert code == 1 and st["C4"] == "FAIL", out


def test_record_prefix_produced_by_mismatch_is_caught(root):
    rec = next((root / "provenance" / "figures").glob("cas_*.json"))
    d = json.loads(rec.read_text(encoding="utf-8"))
    d["produced_by"] = "electron_losses_IGM/scripts/fig_ic_cmb.py::main"      # exists, but belongs to the other subproject
    rec.write_text(json.dumps(d), encoding="utf-8")
    code, st, out = run_gate(root)
    assert code == 1 and st["C4"] == "FAIL", out


def test_broken_path_in_em_cascades_readme_is_caught(root):
    p = root / "EM_cascades" / "README.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nVer `EM_cascades/scripts/no_such_script.py`.\n", encoding="utf-8")
    code, st, out = run_gate(root)
    assert code == 1 and st["C8"] == "FAIL", out
