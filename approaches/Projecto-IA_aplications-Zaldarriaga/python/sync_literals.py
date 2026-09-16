#!/usr/bin/env python3
"""Rewrite every \src-tagged literal in a paper to the registry value, keeping
the precision the author already chose.

The provenance gate REFUSES a drifted literal; this is its companion, which
fixes one. Doing it by hand across 35 tagged numbers is how a paper and its
registry silently diverge.

Usage: sync_literals.py paper.tex registry.json [--dry-run]
"""
import project_paths  # noqa: F401  -- anchors CWD to the project root
import json
import re
import sys

tex_path, reg_path = sys.argv[1], sys.argv[2]
dry = "--dry-run" in sys.argv

reg = {}
for sec in ("derived", "cosmology", "scenario", "inputs"):
    reg.update(json.load(open(reg_path)).get(sec, {}))

NUM = (r"(?:\\SI\{([-\d.eE+]+)\}\{[^}]*\}"
       r"|\$?([\d.]+)\\times10\^\{(-?\d+)\}\$?"
       r"|\$?10\^\{(-?\d+)\}\$?"
       r"|\$?([\d.]+)\$?)")
pat = re.compile(NUM + r"(?=\s*\\src\{([^}]*)\})")
changed = []


def sigfigs(mantissa: str) -> int:
    return len(mantissa.replace("-", "").replace(".", "").lstrip("0")) or 1


def repl(m):
    si, mant, expo, pow10, plain, key = m.groups()
    k = key.replace("\\_", "_")
    val = reg.get(k)
    if not isinstance(val, (int, float)):
        return m.group(0)
    val = float(val)

    if pow10 is not None:
        return m.group(0)          # a bare 10^{b} carries no digits to resync
    if si is not None:
        sig = sigfigs(si.split("e")[0].split("E")[0])
        old = float(si)
        new_s = "%.{}e".format(sig - 1) % val
        a, b = new_s.split("e")
        lit = a if int(b) == 0 else "{}e{}".format(a, int(b))
        if abs(old - float(lit)) <= abs(float(lit)) * 1e-12:
            return m.group(0)
        changed.append((k, old, float(lit)))
        return m.group(0).replace(si, lit, 1)

    if mant is not None:
        sig = sigfigs(mant)
        old = float(mant) * 10 ** int(expo)
        new_s = "%.{}e".format(sig - 1) % val
        a, b = new_s.split("e")
        if abs(old - float(new_s)) <= abs(float(new_s)) * 1e-12:
            return m.group(0)
        changed.append((k, old, float(new_s)))
        return m.group(0).replace(mant, a, 1).replace(
            "10^{%s}" % expo, "10^{%d}" % int(b), 1)

    sig = sigfigs(plain)
    old = float(plain)
    rounded = float("%.{}e".format(sig - 1) % val)
    if abs(old - rounded) <= abs(rounded) * 1e-12:
        return m.group(0)
    lit = "%.{}g".format(sig) % rounded
    changed.append((k, old, rounded))
    return m.group(0).replace(plain, lit, 1)


raw = open(tex_path).read()
out = pat.sub(repl, raw)
if changed:
    print("%d literal(s) resynced:" % len(changed))
    for k, o, n in changed:
        print("   %-34s %r -> %r" % (k, o, n))
else:
    print("nothing to resync")
if not dry:
    open(tex_path, "w").write(out)
