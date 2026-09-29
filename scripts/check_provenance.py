#!/usr/bin/env python3
"""Provenance gate of the project (plan §2.5; declared in CLAUDE.md: `gate: python3 scripts/check_provenance.py`).

Covers C1, C2, C3, C4, C5 (+C6, C37), C7, C8, C9, C80, the numbers.json schema (plan §2.5 item 1), staleness of
claims.yaml, and the MANIFEST copies (MR7). Paths come from the "rigorous-physics declarations" block of CLAUDE.md.
Each check prints one line: PASS, FAIL or SKIPPED (a check that cannot run is never reported as PASS).
Exit status 1 if any check FAILs, 0 otherwise.

Run   python3 scripts/check_provenance.py [--root <project dir>]      (from the project root; ~5 s)
Tests tests/test_gate.py (root) plants one error per check in a copy and requires a FAIL.
"""

import ast
import json
import math
import pathlib
import re
import sys

import yaml

ARGS = sys.argv[1:]
ROOT = pathlib.Path(ARGS[ARGS.index("--root") + 1] if "--root" in ARGS else pathlib.Path(__file__).resolve().parents[1])
RESULTS = []
TAGS = ("measured", "assumed", "fitted", "derived")
A_STATUS = ("proposed", "adopted", "tested", "flagged-outside-regime")
C_STATUS = ("hypothesis", "checked", "reproduced", "reviewed")
N_FIELDS = ("value", "unit", "uncertainty", "tag", "statement", "produced_by", "from_scratch", "from_library", "choices",
            "assumptions", "inputs", "checked_by")
SUBPROJECTS = ("electron_losses_IGM", "EM_cascades")
CODE_DIRS = tuple(f"{sp}/{d}" for sp in SUBPROJECTS for d in ("src", "scripts", "tests")) + ("scripts", "tests")
# Figure records in provenance/figures/: one writer per subproject, bound to the file-name prefix of its figures.
FIG_PREFIX = {"igm_": "electron_losses_IGM", "cas_": "EM_cascades"}


def report(check, ok, detail, skipped=False):
    status = "SKIPPED" if skipped else ("PASS" if ok else "FAIL")
    RESULTS.append((check, status))
    print(f"{check:10s} {status:8s} {detail}")


# ------------------------------------------------------------------ declarations
def declarations():
    txt = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    block = txt.split("## rigorous-physics declarations", 1)[1]
    d = {}
    for line in block.splitlines():
        m = re.match(r"^- ([\w ]+):\s*(.*?)\s*(?:#.*)?$", line)
        if m:
            d[m.group(1).strip()] = m.group(2).strip()
    return d


D = declarations()
first = lambda s: s.split(";")[0].strip()
REG = ROOT / first(D["registries"])
PARAMS = ROOT / D["parameters"]
ASSUMP = ROOT / D["assumptions"]
CLAIMS = ROOT / first(D["claims"])
FIGDIR = ROOT / first(D["figures"]).split("(")[0].strip()
DOC = ROOT / first(D["documents"])
BIB = ROOT / D["bibliography"]
NUMREF = D["numref"].split()[0]
STATEMENTS = ROOT / "provenance" / "claims_statements.yaml"
RECDIR = ROOT / "provenance" / "figures"


# ------------------------------------------------------------------ helpers
def load_yaml(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def defines(path, symbol):
    """path::symbol names a function/class defined in path (AST for Python; regex otherwise)."""
    p = ROOT / path
    if not p.is_file():
        return False
    name = symbol.split("[")[0].split(".")[-1]
    if p.suffix == ".py":
        return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
                   for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))))
    return re.search(rf"(\bdef|\bfunction|\bsubroutine|\bfn)\s+{re.escape(name)}\b", p.read_text(errors="replace")) is not None


def resolves(ref):
    m = re.fullmatch(r"([\w/.-]+\.\w+)::([\w.\[\]-]+)", str(ref))
    return bool(m) and defines(m.group(1), m.group(2))


def bib_entries():
    txt = BIB.read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"@\w+\{\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)", txt, re.S)}


def code_files():
    for d in CODE_DIRS:
        yield from (p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)


def is_numeric_text(v):
    if not isinstance(v, str):
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ C80
def c80():
    import astropy.units as u
    bib = set(bib_entries())
    probs = []
    P = load_yaml(PARAMS)["parameters"]
    for k, e in P.items():
        e = e or {}
        miss = [f for f in ("value", "unit", "uncertainty", "tag", "source") if f not in e]
        if miss:
            probs.append(f"{k}: missing {miss}")
            continue
        if is_numeric_text(e["value"]):
            probs.append(f"{k}: number stored as text {e['value']!r}")
        if e["tag"] not in TAGS:
            probs.append(f"{k}: tag {e['tag']!r}")
        if e["unit"]:
            try:
                u.Unit(e["unit"], parse_strict="raise")
            except ValueError:
                probs.append(f"{k}: unit {e['unit']!r} does not parse")
        s = e["source"] or {}
        if "bib" in s:
            if s["bib"] not in bib:
                probs.append(f"{k}: source bib {s['bib']} not in {BIB.name}")
            if not s.get("loc"):
                probs.append(f"{k}: source bib without loc")
        elif "produced_by" in s:
            if not resolves(s["produced_by"]):
                probs.append(f"{k}: produced_by {s['produced_by']} does not resolve")
        elif "user" not in s:
            probs.append(f"{k}: source is none of bib+loc / user / produced_by")
        if e["tag"] == "derived" and "produced_by" not in s:
            probs.append(f"{k}: tag derived without produced_by")
    A = load_yaml(ASSUMP)["assumptions"]
    for k, e in A.items():
        miss = [f for f in ("statement", "regime", "source", "status") if f not in e]
        if miss:
            probs.append(f"{k}: missing {miss}")
        elif e["status"] not in A_STATUS:
            probs.append(f"{k}: status {e['status']!r}")
        elif e["status"] in ("adopted", "tested") and not (e.get("approved") or {}).get("date"):
            probs.append(f"{k}: {e['status']} without approved.date")
    used = {}
    for p in code_files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = re.search(r"#(.*)$", line)
            if m:
                for a in re.findall(r"\[(A\d+)\]", m.group(1)):
                    used.setdefault(a, []).append(f"{p.relative_to(ROOT)}:{i}")
    if REG.exists():
        for k, e in json.loads(REG.read_text(encoding="utf-8")).items():
            for a in (e.get("assumptions", []) if isinstance(e, dict) else []):
                used.setdefault(a, []).append(f"{REG.name}:{k}")
    for a, where in used.items():
        if a not in A:
            probs.append(f"{a} used ({where[0]}) but not declared")
        elif A[a]["status"] not in ("adopted", "tested"):
            probs.append(f"{a} used ({where[0]}) but status {A[a]['status']}")
    report("C80", not probs, f"{len(P)} parameters, {len(A)} assumptions, {len(used)} ids used; problems: {probs[:8]}")


# ------------------------------------------------------------------ numbers.json schema (plan §2.5 item 1)
def numbers_schema():
    if not REG.exists():
        return report("NUMBERS", False, f"{REG.relative_to(ROOT)} missing")
    N = {k: v for k, v in json.loads(REG.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    P = load_yaml(PARAMS)["parameters"]
    probs = []
    for k, e in N.items():
        miss = [f for f in N_FIELDS if f not in e]
        if miss:
            probs.append(f"{k}: missing {miss}")
            continue
        if e["tag"] not in TAGS:
            probs.append(f"{k}: tag {e['tag']!r}")
        v = e["value"]
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
            probs.append(f"{k}: value {v!r} is not a finite number or null")
        if not resolves(e["produced_by"]):
            probs.append(f"{k}: produced_by {e['produced_by']} does not resolve")
        bad = [c for c in e["checked_by"] if not resolves(c)]
        if bad:
            probs.append(f"{k}: checked_by does not resolve {bad}")
        hypothesis_said = any(w in e["statement"].lower() for w in ("hipótesis", "hypothesis"))
        if not e["checked_by"] and not hypothesis_said:
            probs.append(f"{k}: no checked_by and the statement does not call it a hypothesis")
        bad_in = [i for i in e["inputs"] if i not in P]
        if bad_in:
            probs.append(f"{k}: inputs not in parameters {bad_in}")
    report("NUMBERS", not probs, f"{len(N)} entries; problems: {probs[:8]}")
    return N


# ------------------------------------------------------------------ C4 single writer
WRITE = re.compile(r"\.write_text\(|\.write_bytes\(|json\.dump\(|yaml\.(safe_)?dump\(|open\([^)]*['\"][wa]b?['\"]|np\.save|\.to_(json|csv)\(")


def c4():
    registries = {REG: "numbers", CLAIMS: "claims", RECDIR: "figure records"}
    registries.update({p: p.name for p in (ROOT / "provenance" / "data").glob("*.json")})
    binds = {}                    # (module, name) -> registry
    mods = {p: p.read_text(encoding="utf-8") for p in code_files()}

    def parts(reg):
        return reg.relative_to(ROOT).parts

    for p, src in mods.items():
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                consts = [n for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                lits = [n.value for n in sorted(consts, key=lambda n: (n.lineno, n.col_offset))]   # source order (walk is BFS)
                joined = "/".join(lits)
                for reg in registries:
                    if "/".join(parts(reg)) in joined or (lits and lits[-1] == parts(reg)[-1] and parts(reg)[-2] in lits):
                        binds[(p, node.targets[0].id)] = reg
    writers = {reg: set() for reg in registries}
    for p, src in mods.items():
        local = {name: reg for (m, name), reg in binds.items() if m == p}
        for node in ast.walk(ast.parse(src)):                       # names imported from another module
            if isinstance(node, ast.ImportFrom):
                for al in node.names:
                    for (m, name), reg in binds.items():
                        if name == al.name and m.stem == (node.module or "").split(".")[-1]:
                            local[al.asname or al.name] = reg
        modalias = {}                                               # alias -> module stem ("X" -> "excitation")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom):
                for al in node.names:
                    modalias[al.asname or al.name] = al.name
            elif isinstance(node, ast.Import):
                for al in node.names:
                    modalias[al.asname or al.name.split(".")[-1]] = al.name.split(".")[-1]
        for line in src.splitlines():
            if not WRITE.search(line) or line.lstrip().startswith("#"):
                continue
            for alias, name in re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", line):   # module.NAME
                for (m, bname), reg in binds.items():
                    if bname == name and modalias.get(alias) == m.stem:
                        writers[reg].add(str(p.relative_to(ROOT)))
            for name in set(re.findall(r"\b([A-Za-z_]\w*)\b", line)):
                if name in local:
                    writers[local[name]].add(str(p.relative_to(ROOT)))
            for reg in registries:
                if "/".join(parts(reg)) in line:
                    writers[reg].add(str(p.relative_to(ROOT)))
    mk = ROOT / "Makefile"
    if mk.exists():
        for reg in registries:
            if re.search(r">\s*" + re.escape(str(reg.relative_to(ROOT))), mk.read_text()):
                writers[reg].add("Makefile (shell redirect)")
    # provenance/figures/: exactly one writer inside each subproject that owns a prefix, and every record's
    # produced_by belongs to the subproject of its file-name prefix
    fw = writers.pop(RECDIR)
    per_sub = {sp: sorted(w for w in fw if w.startswith(sp + "/")) for sp in FIG_PREFIX.values()}
    stray = sorted(w for w in fw if not any(w.startswith(sp + "/") for sp in FIG_PREFIX.values()))
    mismatch = []
    for rp in RECDIR.glob("*.json"):
        pre = next((k for k in FIG_PREFIX if rp.name.startswith(k)), None)
        pb = json.loads(rp.read_text(encoding="utf-8")).get("produced_by", "")
        if pre is None or not pb.startswith(FIG_PREFIX[pre] + "/"):
            mismatch.append(rp.name)
    bad = {str(r.relative_to(ROOT)): sorted(w) for r, w in writers.items() if len(w) != 1}
    for sp, w in per_sub.items():
        if len(w) != 1:
            bad[f"provenance/figures ({sp})"] = w
    if stray:
        bad["provenance/figures (outside the subprojects)"] = stray
    if mismatch:
        bad["provenance/figures (prefix ↔ produced_by)"] = mismatch[:5]
    writers[RECDIR] = set(fw)
    who = {str(r.relative_to(ROOT)): sorted(w)[0].split("/")[-1] for r, w in writers.items() if len(w) == 1}
    who["provenance/figures"] = {sp: w[0] for sp, w in per_sub.items() if len(w) == 1}
    report("C4", not bad, f"{len(registries)} registries, one writer each: {who}" if not bad else f"not exactly one writer: {bad}")


# ------------------------------------------------------------------ C5 / C6 / C37 + staleness of claims.yaml
PLACEHOLDER = re.compile(r"\{\{([\w.+-]+)\|([^}]*)\}\}")
CITE = re.compile(r"^(\S+?),\s+(eq|eqs|sec|section|fig|figure|table|tab|p|pp|page)\.?\s*\S+", re.I)


def c5(N):
    if not CLAIMS.exists():
        return report("C5", False, f"{CLAIMS.relative_to(ROOT)} missing")
    doc = load_yaml(CLAIMS)
    claims = doc.get("claims") or []
    bib = set(bib_entries())
    A = load_yaml(ASSUMP)["assumptions"]
    ids = {c["id"] for c in claims}
    fails = []
    for c in claims:
        cid, st, ev = c.get("id", "?"), str(c.get("statement") or "").strip(), c.get("evidence") or []
        if not st:
            fails.append((cid, "C5 no statement"))
        if c.get("status") not in C_STATUS:
            fails.append((cid, f"status {c.get('status')!r}"))
        if not ev and "hypothes" not in st.lower() and "hipótesis" not in st.lower():
            fails.append((cid, "C37 no evidence and not marked hypothesis"))
        for e in ev:
            e = str(e)
            if "::" in e:
                if not resolves(e):
                    fails.append((cid, f"C6 code evidence does not resolve: {e}"))
            elif e.startswith("figures/"):
                if not (ROOT / e).exists():
                    fails.append((cid, f"C6 figure evidence missing: {e}"))
            else:
                m = CITE.match(e)
                if not m:
                    fails.append((cid, f"C6 neither path::symbol, figures/<file> nor citation+locator: {e}"))
                elif m.group(1) not in bib:
                    fails.append((cid, f"C6 citation key not in bib: {e}"))
        for k in c.get("numbers", []):
            if k not in N:
                fails.append((cid, f"number {k} not in numbers.json"))
            elif c.get("status") == "checked" and not N[k]["checked_by"]:
                fails.append((cid, f"status checked but {k} has no checked_by"))
        for d in c.get("depends_on", []):
            if d not in ids:
                fails.append((cid, f"depends_on {d} unknown"))
        for a in c.get("assumptions", []):
            if a not in A or A[a]["status"] not in ("adopted", "tested"):
                fails.append((cid, f"assumption {a} not adopted"))
    report("C5/C6/C37", not fails, f"{len(claims)} claims; {fails[:6]}")
    # staleness: statements and figures section must equal what build_claims would write now
    stale = []
    if STATEMENTS.exists():
        tmpl = {c["id"]: c["statement"] for c in (load_yaml(STATEMENTS).get("claims") or [])}
        for c in claims:
            t = tmpl.get(c["id"])
            if t is None:
                stale.append(f"{c['id']} not in {STATEMENTS.name}")
                continue
            try:
                now = PLACEHOLDER.sub(lambda m: format(N[m.group(1)]["value"], m.group(2)), t)
            except KeyError as ex:
                stale.append(f"{c['id']}: key {ex} not in numbers.json")
                continue
            if now != c["statement"]:
                stale.append(f"{c['id']} statement differs from its template rendered with the current registry")
        if set(tmpl) != ids:
            stale.append(f"claim ids differ: {sorted(set(tmpl) ^ ids)}")
    recs = sorted(json.loads(p.read_text(encoding="utf-8"))["file"] for p in RECDIR.glob("*.json"))
    if sorted(f["file"] for f in doc.get("figures") or []) != recs:
        stale.append("figures section differs from provenance/figures/*.json")
    report("C2-claims", not stale, "claims.yaml up to date with numbers.json and the figure records"
           if not stale else f"stale claims.yaml (run build_claims.py): {stale[:4]}")


# ------------------------------------------------------------------ C7
def c7():
    exts = tuple("." + e for e in re.search(r"\(([^)]*)\)", D["figures"]).group(1).split(","))
    figs = sorted(p for p in FIGDIR.glob("*") if p.suffix in exts)
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in RECDIR.glob("*.json")]
    covered = {r["file"] for r in recs} | {r.get("file_pdf") for r in recs}
    missing = [str(p.relative_to(ROOT)) for p in figs if str(p.relative_to(ROOT)) not in covered]
    bad = [r["produced_by"] for r in recs if not resolves(r["produced_by"])]
    orphan = [r["file"] for r in recs if not (ROOT / r["file"]).exists()]
    ok = not (missing or bad or orphan)
    report("C7", ok, f"{len(figs) - len(missing)}/{len(figs)} figure files recorded; no record: {missing[:5]}; "
                     f"broken produced_by: {bad[:3]}; records without file: {orphan[:3]}")


# ------------------------------------------------------------------ C8
C8_EXTS = "py json yaml yml csv npz npy h5 hdf5 txt md tex bib pdf png svg ipynb sh lock".split()


def c8():
    """Path-like tokens in the READMEs and the declared document must resolve. Not failures, reported separately:
    patterns (the whitespace-delimited word contains <…>, * or {…}), and the declared document if not created yet.
    A token written after "~" is looked up in the home directory."""
    docs = [p for p in (ROOT / "README.md", *(ROOT / sp / "README.md" for sp in SUBPROJECTS), DOC) if p.exists()]
    ext = r"(?:" + "|".join(C8_EXTS) + r")"
    missing, patterns, planned, total = [], 0, set(), 0
    for d in docs:
        txt = re.sub(r"https?://\S+", " ", d.read_text(errors="replace"))
        roots = [ROOT, d.parent, ROOT / "provenance", ROOT / "provenance" / "data"] + \
                [d.parent / s for s in ("src", "scripts", "tests", "reference_figures")]
        for m in re.finditer(r"(?<![\w/.-])([\w./-]+\." + ext + r")\b", txt):
            t = m.group(1)
            lo, hi = txt.rfind(" ", 0, m.start()) + 1, m.end()
            while hi < len(txt) and not txt[hi].isspace():
                hi += 1
            word = txt[lo:hi]
            if any(c in word for c in "<*{"):
                patterns += 1
                continue
            total += 1
            if m.start() > 0 and txt[m.start() - 1] == "~":
                ok = (pathlib.Path.home() / t.lstrip("/")).exists()
            else:
                ok = any((r / t).exists() for r in roots)
            if not ok and (ROOT / t) == DOC:
                planned.add(t)
            elif not ok:
                missing.append(f"{d.relative_to(ROOT)}: {t}")
    report("C8", not missing, f"{total - len(missing)}/{total} path tokens resolve in {[str(d.relative_to(ROOT)) for d in docs]}; "
                              f"missing: {missing}; {patterns} patterns not checked; declared but not created: {sorted(planned)}")


# ------------------------------------------------------------------ C9
def c9():
    entries = bib_entries()
    refdir = BIB.parent
    no_file = []
    for k, body in entries.items():
        f = re.search(r"\bfile\s*=\s*[{\"]([^}\"]+)[}\"]", body, re.I)
        if not f or not any((refdir / x.strip()).is_file() or (ROOT / x.strip()).is_file() for x in f.group(1).split(";")):
            no_file.append(k)
    report("C9-bib", not no_file, f"{len(entries) - len(no_file)}/{len(entries)} bib entries have their file in references/; "
                                  f"missing: {no_file}")
    if not DOC.exists():
        return report("C9-doc", True, f"{DOC.relative_to(ROOT)} does not exist: no citations to resolve", skipped=True)
    body = re.sub(r"(?<!\\)%.*", "", DOC.read_text(errors="replace"))
    cites = {k.strip() for m in re.finditer(r"\\[A-Za-z]*[Cc]ite[A-Za-z]*\*?(?:\s*\[[^\]]*\])*((?:\s*\{[^}]*\})+)", body)
             for g in re.findall(r"\{([^}]*)\}", m.group(1)) for k in g.split(",") if k.strip() and k.strip() != "*"}
    miss = sorted(cites - set(entries))
    report("C9-doc", not miss, f"{len(cites) - len(miss)}/{len(cites)} cited keys in {BIB.name}; missing: {miss}")


# ------------------------------------------------------------------ C1, C2, C3 on the declared document
def _macro_calls(text, name, nargs):
    """[(start, end, [arg1, …])] for every \\name{…}{…} with balanced braces (nested braces allowed)."""
    out, i = [], 0
    while True:
        i = text.find("\\" + name + "{", i)
        if i < 0:
            return out
        j, args = i + len(name) + 1, []
        for _ in range(nargs):
            if j >= len(text) or text[j] != "{":
                break
            depth, k = 0, j
            while k < len(text):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    break
                k += 1
            args.append(text[j + 1:k])
            j = k + 1
        if len(args) == nargs:
            out.append((i, j, args))
        i = j


def _registry_values(N):
    """Keys a document may cite: numbers.json keys (outputs) and parameters.yaml (inputs): a parameter name gives
    its value, and dotted paths reach nested values and list items (e.g. x_e.sweep.values.2, figures.<slug>.…)."""
    vals = {k: e["value"] for k, e in N.items()}
    P = load_yaml(PARAMS)

    def flat(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flat(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                flat(v, f"{path}.{i}")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            vals.setdefault(path, obj)
    for name, e in P["parameters"].items():
        if isinstance((e or {}).get("value"), (int, float)):
            vals.setdefault(name, e["value"])
        flat(e, name)
    flat(P.get("figures", {}), "figures")
    return vals


def c1_c2_c3(N):
    if not DOC.exists():
        for c in ("C1", "C2", "C3"):
            report(c, True, f"{DOC.relative_to(ROOT)} does not exist (no document yet)", skipped=True)
        return
    tex = re.sub(r"(?<!\\)%.*", "", DOC.read_text(errors="replace"))
    body = tex.split(r"\begin{document}", 1)[-1]
    body = re.split(r"\\bibliography\{|\\printbibliography", body, maxsplit=1)[0]
    tagged = [tuple(a) for _, _, a in _macro_calls(body, NUMREF, 2)]
    exempt = [tuple(a) for _, _, a in _macro_calls(body, "exempt", 2)]
    # display equations are formulas (their integers and exponents are structure, not results): excluded from C1 and
    # counted; inline math is still scanned, so a result written in the text must be tagged
    eqs = re.findall(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", body, re.S)
    stripped = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
    for name, n in ((NUMREF, 2), ("exempt", 2)):
        for i0, i1, _ in sorted(_macro_calls(stripped, name, n), reverse=True):
            stripped = stripped[:i0] + " " + stripped[i1:]
    stripped = re.sub(r"\\(ref|eqref|label|[A-Za-z]*cite[A-Za-z]*|url|href|includegraphics|begin|end|usepackage|"
                      r"vspace|hspace|input|bibliographystyle)\*?(\[[^\]]*\])*\{[^}]*\}", " ", stripped)
    stripped = re.sub(r"\[[^\]]*(width|height|scale)[^\]]*\]", " ", stripped)
    bare = [m.group() for m in re.finditer(r"(?<![\w.{\\+-])[-+]?\d+(?:\.\d+)?", stripped)
            if not re.fullmatch(r"(19|20)\d\d", m.group())]
    report("C1", not bare, f"{len(tagged)} tagged numbers, {len(eqs)} display equations excluded, {len(exempt)} exempt with reason "
                           f"{[e[0] for e in exempt][:8]}; untagged numerals: {bare[:20]}")
    vals = _registry_values(N)
    unresolved = sorted({k for k, _ in tagged if k not in vals})
    report("C3", not unresolved, f"{len({k for k, _ in tagged})} keys (numbers.json + parameters.yaml); unresolved: {unresolved}")
    bad = []
    for k, lit in tagged:
        if k not in vals or vals[k] is None:
            continue
        m = re.fullmatch(r"\$?\s*([-+]?[\d.]+)(?:\s*(?:\\times\s*10\^\{?|[eE])([-+]?\d+)\}?)?\s*\$?", lit.strip())
        if not m:
            bad.append((k, lit, "unparsed"))
            continue
        mant, exp = m.group(1), int(m.group(2) or 0)
        dec = len(mant.split(".")[1]) if "." in mant else 0
        if round(vals[k] / 10 ** exp, dec) != round(float(mant), dec):
            bad.append((k, lit, vals[k]))
    report("C2", not bad, f"{len(tagged)} tagged literals vs registries at displayed precision; mismatches/unparsed: {bad[:6]}")


# ------------------------------------------------------------------ MR7 MANIFEST copies
def manifest():
    mf = ROOT / "provenance" / "MANIFEST.txt"
    if not mf.exists():
        return report("MR7", False, "provenance/MANIFEST.txt missing")
    bad, n = [], 0
    for line in mf.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "<-" not in line:
            continue
        copy, orig = (s.strip() for s in line.split("<-", 1))
        n += 1
        a, b = ROOT / copy, ROOT / orig
        if not a.is_file() or not b.is_file() or a.read_bytes() != b.read_bytes():
            bad.append(copy)
    report("MR7", not bad, f"{n - len(bad)}/{n} MANIFEST copies identical to their originals; stale or missing: {bad}")


def main():
    print(f"provenance gate — {ROOT}")
    c80()
    N = numbers_schema() or {}
    c4()
    c5(N)
    c7()
    c8()
    c9()
    c1_c2_c3(N)
    manifest()
    fails = [c for c, s in RESULTS if s == "FAIL"]
    skipped = [c for c, s in RESULTS if s == "SKIPPED"]
    print(f"GATE {'FAIL' if fails else 'PASS'}: {len(RESULTS) - len(fails) - len(skipped)} pass, {len(fails)} fail {fails}, "
          f"{len(skipped)} skipped {skipped}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
