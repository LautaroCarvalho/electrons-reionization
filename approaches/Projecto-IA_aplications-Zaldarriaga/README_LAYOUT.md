# Folder layout

Reorganised 2026-09-15. Four top-level folders plus the pre-existing `papers/`
and `provenance/`.

| folder | holds |
|---|---|
| `python/` | all 18 `.py` sources, plus `project_paths.py` (new) and the `.bak` |
| `Images/` | all 34 figure outputs — `.png` and `.pdf`, one pair per figure |
| `Text_files/` | the three LaTeX manuscripts (`.tex`), their compiled `.pdf`, their build artefacts (`.aux/.log/.out/.bbl/.blg`), the claims `.yaml`, and the `.tex.bak` |
| `papers/` | **unchanged** — the literature PDFs, `references.bib`, `MANIFEST.md` |
| `provenance/` | **unchanged** — the generated registries the gate reads |
| root | the `.json` result registries, the `.md` notes, the instruction files |

Three PDFs are *not* in `Images/`: `inputs_table.pdf`, `ionization_yield.pdf` and
`photon_vs_electron.pdf` are compiled manuscripts, so they live beside the `.tex`
that produces them. `papers/` was left alone for the same reason — its PDFs are
the bibliography, not figures.

## How paths survived the move

`python/project_paths.py` resolves the project root from its own location and
`chdir`s there on import; every script imports it. So the ~40 bare relative
paths written when the project was flat (`results.json`, `provenance/...`,
`papers/references.bib`) still mean exactly what they meant, and scripts run
correctly from anywhere:

    python3 python/source_map.py          # from the root
    cd python && python3 source_map.py    # from inside python/

Only the paths that genuinely had to change were changed at source:

- `igm_config.fig_stem()` now returns an `Images/`-prefixed stem, so every
  figure producer that already used it needed no edit of its own. The four that
  bypassed it (`imf_comparison`, `imf_schaerer`, `ionization_yield`,
  `yield_comparison`) were routed through it.
- `make_inputs_table.py` writes to `Text_files/inputs_table.tex` and emits
  `\bibliography{../papers/references}`.
- `check_provenance.py` defaults to `Text_files/ionization_yield.tex`, and its
  C8 existence check now resolves a bare filename against
  `["", "Images/", "python/", "Text_files/", "papers/"]` rather than only the
  CWD — so it still bites instead of quietly passing.
- Both manuscripts gained
  `\graphicspath{{../Images/}{Images/}{./}}`, so they build from `Text_files/`
  or from the root.

## Building the manuscripts

    cd Text_files
    pdflatex photon_vs_electron && bibtex photon_vs_electron && pdflatex photon_vs_electron && pdflatex photon_vs_electron

## Running the provenance gate

The gate takes the paper and a comma-separated list of registries. It needs
*all* of them: a missing registry shows up as C3 "unresolved key", which looks
like drift but is only an incomplete invocation.

    REGS="photon_vs_electron_results.json,yield_comparison_results.json,\
    imf_comparison_results.json,\
    provenance/registry_z20.json,provenance/reionization_budget.json,\
    provenance/figure_claims.json,provenance/audit_registry.json,\
    provenance/source_map.json,provenance/xcomp.json"

(The authoritative list is `REGISTRY_FILES` in `run_all.py`.)

    python3 python/check_provenance.py Text_files/photon_vs_electron.tex "$REGS"

Verified 9/9 after the reorganisation, including C8 ("every file path named in
the paper exists") over 40 paths — the check that would have caught a botched
move.

## Known pre-existing issue, not caused by the move

Running the gate with its bare defaults targets `ionization_yield.tex` against
`results.json` alone and fails C3. That paper cites keys from registries the
default invocation does not load; `results.json` also dates from 2026-09-07 and
its model-C keys no longer match the model-D/E keys the paper cites. Re-running
`python3 python/ionization_yield.py` would regenerate it. Unrelated to the
folder reorganisation.
