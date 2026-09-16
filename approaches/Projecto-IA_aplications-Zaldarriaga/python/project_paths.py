#!/usr/bin/env python3
r"""
Project paths, anchored to the repository root rather than the shell's CWD.

WHY THIS EXISTS.  The scripts in this project were written when everything --
code, figures, LaTeX, papers/ and provenance/ -- sat in one flat directory, so
they address files by bare relative paths ("results.json", "provenance/...",
"papers/references.bib").  The 2026-09-15 reorganisation moved the code into
python/, the figures into Images/ and the manuscripts into Text_files/.

Rather than rewrite ~40 scattered path literals (and risk missing one silently,
which is exactly how a provenance gate starts passing for the wrong reason),
this module resolves the root from its OWN location and chdir's there on
import.  Every existing relative path therefore keeps its original meaning, and
the scripts run correctly from any working directory:

    python3 python/source_map.py          # from the project root
    cd python && python3 source_map.py    # from inside python/

Only paths that must now point somewhere NEW -- figure output, and the
generated inputs_table.tex -- are changed at their source, via IMAGES and TEXT
below and via igm_config.fig_stem().
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PYTHON = ROOT / "python"
IMAGES = ROOT / "Images"
TEXT = ROOT / "Text_files"
PAPERS = ROOT / "papers"
PROVENANCE = ROOT / "provenance"

# Anchor the process to the root so bare relative paths mean what they always
# meant.  Idempotent: importing this module twice is a no-op the second time.
os.chdir(ROOT)

for _d in (IMAGES, TEXT, PROVENANCE):
    _d.mkdir(exist_ok=True)


def image(stem: str) -> str:
    """Path for a figure output, relative to the root (so it prints tidily)."""
    return f"Images/{stem}"


def text(name: str) -> str:
    """Path for a LaTeX source or build product, relative to the root."""
    return f"Text_files/{name}"
