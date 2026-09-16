#!/usr/bin/env python3
r"""
Build the provenance ledger: every number in the manuscripts, traced to source.

For each \src-tagged literal this collects
    the key -> the value in the registry -> which registry -> which producer
    -> which paper in references.bib -> its status code

so a reader can audit any figure in the manuscripts without running anything.
The gate (check_provenance.py) already proves these links hold; this makes them
browsable rather than a pass/fail line.

Dependency-light on purpose: json, re, pathlib, yaml. No physics imports, so it
runs in a second and cannot OOM this machine.

    python3 python/make_ledger.py            # write Text_files/ledger.html
"""
from __future__ import annotations
import project_paths  # noqa: F401
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NUM = r"(-?\d[\d,]*\.?\d*(?:\s*\\times\s*10\^\{-?\d+\})?|-?\d*\.\d+[eE][-+]?\d+)"


def registry_files():
    """The one registry list, taken from run_all.REGISTRY_FILES.

    Parsed rather than imported so this script stays dependency-light and does
    not execute the runner just to read a constant.
    """
    src = (ROOT / "run_all.py").read_text()
    m = re.search(r"REGISTRY_FILES = \[(.*?)\]", src, re.S)
    if not m:
        raise SystemExit("make_ledger: cannot find REGISTRY_FILES in run_all.py")
    return re.findall(r'"([^"]+\.json)"', m.group(1))


def load_registries():
    """key -> (value, registry file), over EXACTLY the files the gate is given.

    It used to glob every *.json under the root and provenance/.  That is wrong,
    and quietly so: the z=20 registries (yield_comparison_results_z20.json,
    photon_vs_electron_results_z20.json) reuse the SAME UNSUFFIXED key names for
    a different scenario, and "later files win" put them last by sort order.  76
    keys were served with their z=20 value under the z=10 name -- N_e_loss_ic_sat
    as 4.12e6 instead of 4.91e6, C_IGM as 1.256 instead of 1.838.  The gate never
    saw this because it is handed an explicit list; the ledger is supposed to
    MIRROR the gate, so it now reads that same list from run_all.py rather than
    guessing from the filesystem.
    """
    reg, where = {}, {}
    files = [ROOT / f for f in registry_files()]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("  WARNING: registry file(s) absent:",
              ", ".join(str(m.relative_to(ROOT)) for m in missing))
    for f in files:
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for sec in ("derived", "cosmology", "scenario", "inputs"):
            for k, v in (d.get(sec) or {}).items():
                reg[k] = v
                where[k] = f.name
    return reg, where


def load_bib():
    """bibkey -> {title, note, url} from the canonical bibliography.

    The entry regex is bounded by the closing brace so a field is never taken
    from a LATER entry -- an unbounded one silently hands each key the next
    entry's url, which is how a ledger ends up linking the wrong paper.
    """
    txt = (ROOT / "papers" / "references.bib").read_text()
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", txt, re.S):
        key, body = m.group(1).strip(), m.group(2)
        clean = lambda s: " ".join(re.sub(r"[{}\\]", "", s).split())
        # The trailing newline must be OPTIONAL: the entry regex stops before
        # "\n}", so the LAST field in an entry has no newline after its closing
        # brace. Requiring one silently dropped every entry whose url came last
        # -- NIST_ASD among them.
        fld = lambda name: (lambda mm: clean(mm.group(1)) if mm else "")(
            re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*(?:\n|$)", body, re.S))
        t, note = fld("title"), fld("note")
        # Link preference: an explicit url, else the DOI, else the arXiv id.
        url = fld("url")
        if not url and fld("doi"):
            url = "https://doi.org/" + fld("doi")
        if not url and fld("eprint"):
            e = fld("eprint")
            url = "https://arxiv.org/abs/" + e
        out[key] = dict(title=t or key, note=note[:400], url=url)
    return out


def producers():
    """figure stem -> the function that makes it, from the claims files."""
    out = {}
    for f in (ROOT / "Text_files").glob("*claims.yaml"):
        try:
            d = yaml.safe_load(f.read_text())
        except Exception:
            continue
        for fig in d.get("figures", []):
            out[Path(fig["file"]).stem] = fig.get("produced_by", "")
    return out


def scan_manuscript(tex_path, reg, where):
    """Every \\src tag in one manuscript, with the literal printed beside it."""
    body = tex_path.read_text()
    body = re.sub(r"(?<!\\)%.*", "", body)          # strip comments
    rows = []
    for m in re.finditer(NUM + r"\s*\$?\s*\\src\{([^}]*)\}", body):
        lit, key = m.group(1), m.group(2).replace("\\_", "_")
        rows.append(dict(key=key, literal=" ".join(lit.split()),
                         value=reg.get(key), registry=where.get(key, ""),
                         doc=tex_path.stem))
    # tags whose literal the regex could not pair (the gate's C2b territory)
    tagged = {r["key"] for r in rows}
    for m in re.finditer(r"\\src\{([^}]*)\}", body):
        k = m.group(1).replace("\\_", "_")
        if k not in tagged:
            rows.append(dict(key=k, literal="", value=reg.get(k),
                             registry=where.get(k, ""), doc=tex_path.stem))
            tagged.add(k)
    return rows


def parameter_rows():
    """Every entry of parameters.yaml, which is where provenance now starts."""
    d = yaml.safe_load((ROOT / "parameters.yaml").read_text())
    out = []
    def walk(node, path):
        if isinstance(node, dict) and "value" in node:
            out.append(dict(path=path, value=node["value"],
                            units=node.get("units", "--"),
                            source=node.get("source", ""),
                            status=node.get("status", ""),
                            note=" ".join(node.get("note", "").split())))
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
    for k, v in d.items():
        if k != "meta":
            walk(v, k)
    return out, d["meta"]


def build():
    reg, where = load_registries()
    bib = load_bib()
    prod = producers()
    params, meta = parameter_rows()
    numbers = []
    for tex in sorted((ROOT / "Text_files").glob("*.tex")):
        if tex.stem in ("photon_vs_electron", "ionization_yield"):
            numbers += scan_manuscript(tex, reg, where)
    return dict(meta=meta, numbers=numbers, params=params,
                bib=bib,
                producers=prod,
                registries=sorted({r["registry"] for r in numbers if r["registry"]}))




HTML_HEAD = """<title>Reionization Provenance Ledger</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  /* Palette taken from the project's own matplotlib constants: --ground is
     photon_vs_electron.BG, and the photon/electron hues are the labels on the
     xcomp_map colourbar. */
  --ground:#fcfcfb; --raised:#ffffff; --ink:#17191c; --muted:#6f6a62;
  --rule:#e4e0d8; --photon:#8b1a1a; --electron:#1a4f8b; --verified:#1b7837;
  --warn:#9a6b12; --unsourced:#b02a1e;
  --shadow:0 1px 2px rgba(23,25,28,.05);
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --ground:#15171a; --raised:#1d2024; --ink:#e9e6e0; --muted:#9a948a;
  --rule:#2d3136; --photon:#e08a7e; --electron:#84b3e6; --verified:#6fbf8a;
  --warn:#d7a848; --unsourced:#eb8578; --shadow:0 1px 2px rgba(0,0,0,.4);
}}
:root[data-theme="dark"]{
  --ground:#15171a; --raised:#1d2024; --ink:#e9e6e0; --muted:#9a948a;
  --rule:#2d3136; --photon:#e08a7e; --electron:#84b3e6; --verified:#6fbf8a;
  --warn:#d7a848; --unsourced:#eb8578; --shadow:0 1px 2px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
     font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5}
.wrap{max-width:1120px;margin:0 auto;padding-inline:20px;padding-block:32px 64px}
h1{font-family:Spectral,Georgia,serif;font-weight:600;font-size:clamp(1.7rem,4vw,2.4rem);
   margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--muted);max-width:66ch;margin:0 0 6px}
.ident{font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--muted);
       display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;
       padding-top:10px;border-top:1px solid var(--rule)}
.ident b{color:var(--ink);font-weight:500}
.tallies{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 8px}
.tally{background:var(--raised);border:1px solid var(--rule);border-radius:6px;
       padding:9px 13px;box-shadow:var(--shadow);min-width:88px}
.tally .n{font-family:"IBM Plex Mono",monospace;font-size:1.3rem;font-weight:500;
          font-variant-numeric:tabular-nums;display:block;line-height:1.15}
.tally .k{font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.legend{margin:20px 0 6px;font-size:.83rem;color:var(--muted)}
.legend dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:8px 0 0}
.legend dt{font-family:"IBM Plex Mono",monospace;font-weight:500;color:var(--ink)}
.legend dd{margin:0}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;
          margin:26px 0 12px;position:sticky;top:env(safe-area-inset-top,0px);
          background:var(--ground);padding-block:10px;z-index:5;
          border-bottom:1px solid var(--rule)}
input[type=search]{flex:1 1 240px;min-width:0;padding:8px 11px;border-radius:6px;
  border:1px solid var(--rule);background:var(--raised);color:var(--ink);
  font-family:"IBM Plex Mono",monospace;font-size:.85rem}
input[type=search]:focus-visible{outline:2px solid var(--electron);outline-offset:1px}
.chip{border:1px solid var(--rule);background:var(--raised);color:var(--muted);
  border-radius:999px;padding:5px 12px;font-size:.78rem;cursor:pointer;
  font-family:inherit}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--ground);border-color:var(--ink)}
.chip:focus-visible{outline:2px solid var(--electron);outline-offset:2px}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;
           background:var(--raised);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.84rem}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule);
      vertical-align:top}
th{font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
   font-weight:600;position:sticky;top:0;background:var(--raised);z-index:1}
tr:last-child td{border-bottom:0}
td.k,td.v,td.u{font-family:"IBM Plex Mono",monospace}
td.v{font-variant-numeric:tabular-nums;white-space:nowrap}
td.k{color:var(--electron);word-break:break-word}
.pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.7rem;
      font-weight:500;border-radius:4px;padding:1px 6px;border:1px solid currentColor}
.V{color:var(--verified)} .C{color:var(--warn)} .D{color:var(--electron)}
.S{color:var(--muted)} .U{color:var(--muted)} .X{color:var(--unsourced)}
.note{color:var(--muted);font-size:.78rem;max-width:62ch}
a.src{color:var(--electron);text-decoration:none;
      border-bottom:1px solid color-mix(in srgb,var(--electron) 35%,transparent)}
a.src:hover{border-bottom-color:var(--electron)}
a.src:focus-visible{outline:2px solid var(--electron);outline-offset:2px}
a.src::after{content:"\2197";font-size:.72em;vertical-align:super;
             margin-left:2px;opacity:.6}
.nolink{color:var(--ink)}
h2{font-family:Spectral,Georgia,serif;font-weight:600;font-size:1.25rem;
   margin:44px 0 2px}
.count{color:var(--muted);font-size:.8rem;margin:0 0 10px}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--rule);
       color:var(--muted);font-size:.78rem;max-width:72ch}
@media (max-width:560px){ th:nth-child(4),td:nth-child(4){display:none} }
</style>
"""


def emit_html(d, out_path):
    payload = json.dumps(d, separators=(",", ":"))
    st = {}
    for p in d["params"]:
        st[p["status"]] = st.get(p["status"], 0) + 1
    tallies = "".join(
        f'<div class="tally"><span class="n">{v}</span>'
        f'<span class="k">{k}</span></div>'
        for k, v in [("numbers", len(d["numbers"])),
                     ("parameters", len(d["params"])),
                     ("papers", len(d["bib"])),
                     ("figures", len(d["producers"])),
                     ("registries", len(d["registries"]))])
    html = HTML_HEAD + f"""
<div class="wrap">
<h1>Reionization Provenance Ledger</h1>
<p class="sub">Every number printed in the two manuscripts, traced to the registry
that holds it, the code that computed it, and the paper it came from. The
provenance gate proves these links hold on each run; this page makes them
browsable without running anything.</p>
<div class="ident">
  <span>scenario <b>{d['meta']['scenario_name']}</b></span>
  <span>hash <b>{d['meta'].get('scenario_hash','see parameters.yaml')}</b></span>
  <span>source of truth <b>parameters.yaml</b></span>
</div>

<div class="tallies">{tallies}</div>

<div class="legend"><strong>Status codes</strong>, as this project defines them —
the column exists to separate four very different kinds of number that a bare
citation makes look alike.
<dl>
<dt class="V">V</dt><dd>verified numerically against a source that is in <code>papers/</code></dd>
<dt class="C">C</dt><dd>cited, but the implementation has never been checked against the source's equations</dd>
<dt class="D">D</dt><dd>derived in this project from other entries</dd>
<dt class="S">S</dt><dd>a scanned parameter — a range, not a measurement</dd>
<dt class="U">U</dt><dd>the author's choice; no external source claimed</dd>
<dt class="X">X</dt><dd>conventional in the field, but <em>this project has established no source</em></dd>
</dl></div>

<h2>Parameters</h2>
<p class="count" id="pcount"></p>
<div class="controls">
  <input type="search" id="q" placeholder="filter by name, source, units or note…"
         aria-label="Filter parameters">
  <button class="chip" id="all" aria-pressed="true">all</button>
  <button class="chip V" data-s="V" aria-pressed="false">V</button>
  <button class="chip C" data-s="C" aria-pressed="false">C</button>
  <button class="chip D" data-s="D" aria-pressed="false">D</button>
  <button class="chip S" data-s="S" aria-pressed="false">S</button>
  <button class="chip U" data-s="U" aria-pressed="false">U</button>
  <button class="chip X" data-s="X" aria-pressed="false">X</button>
</div>
<div class="tablewrap"><table id="ptab"><thead><tr>
<th>parameter</th><th>value</th><th>units</th><th>source</th><th>st</th><th>note</th>
</tr></thead><tbody></tbody></table></div>

<h2>Manuscript numbers</h2>
<p class="count" id="ncount"></p>
<div class="tablewrap"><table id="ntab"><thead><tr>
<th>key</th><th>printed</th><th>registry value</th><th>registry</th><th>document</th>
</tr></thead><tbody></tbody></table></div>

<footer>Generated by <code>python/make_ledger.py</code> from
<code>parameters.yaml</code>, the provenance registries, the claims files and
<code>papers/references.bib</code>. It reads those artefacts and computes nothing,
so it cannot disagree with them — but it is only as current as the last run of
<code>run_all.py</code>.</footer>
</div>
<script>
const D = {payload};
const esc = s => String(s).replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
const fmt = v => v === null || v === undefined ? "—"
  : (typeof v === "number"
      ? (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)
          ? v.toExponential(4) : String(Number(v.toFixed(6))))
      : Array.isArray(v) ? v.join(" – ") : String(v));
let active = null, query = "";

// The source column links straight to the paper. Preference in load_bib is
// url -> DOI -> arXiv; the four entries with none of those (books and reports)
// render as plain text rather than an invented link.
function srcLink(key){{
  if(!key) return '<span class="note">—</span>';
  const b = D.bib[key];
  if(!b) return esc(key);
  const t = b.title ? esc(b.title) : esc(key);
  if(!b.url) return `<span class="nolink" title="${{t}} — no online link in references.bib">${{esc(key)}}</span>`;
  return `<a class="src" href="${{esc(b.url)}}" target="_blank" rel="noopener noreferrer" title="${{t}}">${{esc(key)}}</a>`;
}}

function drawParams(){{
  const rows = D.params.filter(p =>
    (!active || p.status === active) &&
    (!query || (p.path+" "+p.source+" "+p.units+" "+p.note).toLowerCase().includes(query)));
  document.querySelector("#ptab tbody").innerHTML = rows.map(p => `
    <tr><td class="k">${{esc(p.path)}}</td>
        <td class="v">${{esc(fmt(p.value))}}</td>
        <td class="u">${{esc(p.units)}}</td>
        <td>${{srcLink(p.source)}}</td>
        <td><span class="pill ${{p.status}}">${{p.status}}</span></td>
        <td class="note">${{esc(p.note)}}</td></tr>`).join("");
  document.getElementById("pcount").textContent =
    rows.length + " of " + D.params.length + " shown";
}}
function drawNumbers(){{
  const rows = D.numbers.filter(n =>
    !query || (n.key+" "+n.registry+" "+n.doc).toLowerCase().includes(query));
  document.querySelector("#ntab tbody").innerHTML = rows.slice(0,600).map(n => `
    <tr><td class="k">${{esc(n.key)}}</td>
        <td class="v">${{esc(n.literal || "—")}}</td>
        <td class="v">${{esc(fmt(n.value))}}</td>
        <td class="u">${{esc(n.registry)}}</td>
        <td>${{esc(n.doc)}}</td></tr>`).join("");
  document.getElementById("ncount").textContent =
    rows.length + " of " + D.numbers.length + " tagged literals";
}}
document.getElementById("q").addEventListener("input", e => {{
  query = e.target.value.toLowerCase().trim(); drawParams(); drawNumbers();
}});
document.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => {{
  active = b.id === "all" ? null : b.dataset.s;
  document.querySelectorAll(".chip").forEach(o =>
    o.setAttribute("aria-pressed", String(o === b)));
  drawParams();
}}));
drawParams(); drawNumbers();
</script>
"""
    out_path.write_text(html)
    return out_path

if __name__ == "__main__":
    d = build()
    print(f"  tagged numbers : {len(d['numbers'])}")
    print(f"  parameters     : {len(d['params'])}")
    print(f"  bib entries    : {len(d['bib'])}")
    print(f"  figures        : {len(d['producers'])}")
    print(f"  registries used: {len(d['registries'])}")
    unresolved = [n for n in d["numbers"] if n["value"] is None]
    print(f"  unresolved keys: {len(unresolved)}"
          + (f"  e.g. {unresolved[0]['key']}" if unresolved else ""))
    (ROOT / "provenance" / "ledger.json").write_text(json.dumps(d, indent=1, sort_keys=True))
    print("  -> provenance/ledger.json")
    import parameters as PR
    d["meta"]["scenario_hash"] = PR.scenario_hash()
    out = emit_html(d, ROOT / "Text_files" / "ledger.html")
    print(f"  -> {out.relative_to(ROOT)}  ({out.stat().st_size/1024:.0f} KB)")
