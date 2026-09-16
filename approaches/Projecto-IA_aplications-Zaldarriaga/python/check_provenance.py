#!/usr/bin/env python3
"""Provenance gate for ionization_yield.tex  (CHECKS_AUTO C1 C2 C3 C7 C8 C12).

Refuses the build unless the paper, the code and the registry agree.
Run AFTER python3 ionization_yield.py.  Exit 0 = all checks pass.
"""
import project_paths  # noqa: F401  -- anchors CWD to the project root
import json, os, re, sys

# Usage: check_provenance.py [paper.tex] [registry.json] [build.log]
# Defaults to the ionization-yield paper so existing invocations are unchanged.
TEX = sys.argv[1] if len(sys.argv) > 1 else "Text_files/ionization_yield.tex"
# Comma-separated: one consolidated paper may legitimately cite numbers from
# several runs. Keys must still be unique across them, and a collision is
# reported rather than silently resolved by load order.
REG = sys.argv[2] if len(sys.argv) > 2 else "results.json"
LOGF = sys.argv[3] if len(sys.argv) > 3 else TEX.replace(".tex", ".log")
rows = []
def rec(cid, name, ok, detail, skipped=False):
    rows.append((cid, name, "SKIP" if skipped else ("PASS" if ok else "FAIL"), detail))

reg, _seen, _clash = {}, {}, []
for _f in REG.split(","):
    _a = json.load(open(_f.strip()))
    for _sec in ("derived", "cosmology", "scenario", "inputs"):
        for _k, _v in _a.get(_sec, {}).items():
            if _k in _seen and _seen[_k] != _f.strip():
                prev = reg.get(_k)
                if not (isinstance(prev, (int, float)) and isinstance(_v, (int, float))
                        and prev != 0 and abs(prev - _v) <= abs(prev) * 1e-12):
                    _clash.append((_k, _seen[_k], _f.strip()))
            reg[_k] = _v
            _seen[_k] = _f.strip()
raw = open(TEX).read()
# strip comment lines: a % not preceded by a backslash starts a comment
body = "\n".join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in raw.splitlines())

rec("C3b", "no key means different things in different registries",
    not _clash,
    f"{len(REG.split(','))} registry file(s); colliding keys with DIFFERENT "
    f"values: {_clash or 'none'}")

keys = [k.replace("\\_", "_") for k in re.findall(r"\\src\{([^}]*)\}", body)]
missing = sorted({k for k in keys if k not in reg})
rec("C3", "every \\src key resolves in the registry", not missing,
    f"{len(keys)} uses, {len(set(keys))} unique; unresolved: {missing or 'none'}")

used, unused = set(keys), sorted(set(reg) - set(keys))
rec("C3", "registry keys that the paper never cites (orphans)", True,
    f"{len(unused)} unused: {unused}  [informational: the registry may legitimately "
    f"hold intermediate values]")

# --- C2 displayed-precision agreement -------------------------------------
# A trailing "$" or "\times" between the literal and its \src tag used to make
# the tag INVISIBLE here: 9 of 41 tags in one paper were silently unchecked, and
# a stale 10.63 rode through a "0 mismatches" report. TAIL absorbs the closers,
# and C2b now reports whatever still cannot be parsed instead of skipping it.
TAIL = r"(?:\\times)?\$?"
NUM = (r"(?:\\SI\{([-\d.eE+]+)\}\{[^}]*\}"
       r"|\$?([\d.]+)\\times10\^\{(-?\d+)\}\$?"
       r"|\$?10\^\{(-?\d+)\}\$?"
       r"|\$?([\d.]+)\$?)") + TAIL
bad = []
n_parsed = 0
for m in re.finditer(NUM + r"\s*\\src\{([^}]*)\}", body):
    si, mant, expo, pow10, plain, key = m.groups()
    key = key.replace("\\_", "_")
    if key not in reg:
        continue
    n_parsed += 1
    # significant figures come from the MANTISSA only, never the exponent
    if pow10 is not None:          # a bare 10^{b}: one significant figure
        lit = 10.0 ** int(pow10)
        n_parsed += 1
        if abs(lit - float("%.0e" % reg[key])) > abs(lit) * 1e-12:
            bad.append((key, lit, reg[key], 1))
        continue
    mantissa_str = (si.split("e")[0].split("E")[0] if si else (mant or plain))
    lit = float(si) if si else (float(mant) * 10 ** int(expo) if mant else float(plain))
    sig = len(mantissa_str.replace("-", "").replace(".", "").lstrip("0")) or 1
    val = reg[key]
    # the literal must equal the registry value rounded to `sig` significant figures
    rounded = float(f"%.{sig-1}e" % val)
    if abs(lit - rounded) > abs(rounded) * 1e-12:
        bad.append((key, lit, val, sig))
rec("C2", "displayed literal == registry value at displayed precision", not bad,
    f"{len(bad)} mismatch(es) over {n_parsed} parsed literal(s): {bad or 'none'}")

# --- C2b a tag whose literal cannot be parsed is NOT a passing check --------
all_src = [m.start() for m in re.finditer(r"\\src\{", body)]
ends = {m.end() for m in re.finditer(NUM + r"(?=\s*\\src\{)", body)}
unparsed = []
for pos in all_src:
    if not any(abs(pos - e) <= 2 for e in ends):
        k = re.match(r"\\src\{([^}]*)\}", body[pos:]).group(1).replace("\\_", "_")
        unparsed.append(k)
rec("C2b", "every \\src tag has a literal the gate can actually read",
    not unparsed,
    f"{len(all_src)} tag(s), {len(all_src)-len(unparsed)} readable; "
    f"UNREADABLE (silently unchecked): {sorted(set(unparsed)) or 'none'}")

# --- C8 paths named in the paper exist ------------------------------------
paths = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body))
paths |= {p.replace("\\_", "_") for p in re.findall(r"\\texttt\{([\w\\_./-]+\.(?:py|json|tex|pdf|md))\}", body)}
# Since the 2026-09-15 reorganisation a path named in the paper may live in
# Images/, python/ or Text_files/ rather than beside it, and the paper quite
# reasonably names it bare ("photon_vs_electron_fig1.pdf", "igm_losses.py").
# Resolve against the search path rather than weakening the check to a no-op.
SEARCH = ("", "Images/", "python/", "Text_files/", "papers/")
def _resolves(q):
    return any(os.path.exists(d + q) or os.path.exists(d + q + ".pdf")
               for d in SEARCH)
absent = sorted(p for p in paths if not _resolves(p))
rec("C8", "every file path named in the paper exists", not absent,
    f"checked {len(paths)} path(s) against {list(SEARCH)}; "
    f"absent: {absent or 'none'}")

# --- C7 every figure has a producer ---------------------------------------
cf = TEX.replace(".tex", "_claims.yaml")
cf = cf if os.path.exists(cf) else "provenance/claims.yaml"
claims = open(cf).read()
rec("C7", "every figure has a produced_by entry",
    claims.count("- file:") == claims.count("produced_by:"),
    f"{claims.count('- file:')} figure(s), {claims.count('produced_by:')} producer(s)")

# --- C7b every figure the PAPER shows is one the claims file knows about ---
# C7 on its own is vacuous: a claims file listing three of ten figures still
# has as many produced_by lines as file lines. This is the check that bites.
_shown = {os.path.splitext(os.path.basename(x))[0]
          for x in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body)}
_claimed = {os.path.splitext(os.path.basename(x))[0]
            for x in re.findall(r"^\s*-\s*file:\s*(\S+)", claims, re.M)}
_unclaimed = sorted(_shown - _claimed)
rec("C7b", "every figure the paper shows has a claims entry", not _unclaimed,
    f"{len(_shown)} shown, {len(_claimed)} claimed; "
    f"shown but unclaimed: {_unclaimed or 'none'}")

# --- C12 skipped means skipped --------------------------------------------
# This used to test ONLY for "Undefined control sequence", and therefore called
# a build clean while pdflatex was emitting "! Missing $ inserted" on every line
# where \src appeared inside an align. A check that cannot see the error class
# you actually have is not a check. It now reads every line pdflatex prefixes
# with "! ", which is how pdflatex marks an error, and reports them.
LOG = LOGF
if os.path.exists(LOG):
    log = open(LOG, errors="replace").read()
    errs = sorted({ln.strip() for ln in log.splitlines() if ln.startswith("! ")})
    rec("C12", "LaTeX build raises no errors at all", not errs,
        f"{len(errs)} distinct error line(s): {errs or 'none'}")
else:
    rec("C12", "LaTeX build raises no errors at all", False, "no log", skipped=True)

for cid, name, state, detail in rows:
    print(f"[{state}] {cid:4s} {name}\n        {detail}")
n_run = sum(1 for r in rows if r[2] != "SKIP")
n_ok = sum(1 for r in rows if r[2] == "PASS")
print(f"\nall checks pass ({n_ok} of {n_run} run)" if n_ok == n_run
      else f"\nGATE FAILED: {n_run - n_ok} of {n_run} checks failed")
sys.exit(0 if n_ok == n_run else 1)
