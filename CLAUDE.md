# Proyecto final — Aplicaciones de agentes de IA en astrofísica (M. Zaldarriaga)

Este proyecto sigue el plan de trabajo de `plantilla_proyecto_final/PLAN_TRABAJO_FINAL.md`. Leelo antes de
empezar cualquier tarea; la estructura de carpetas, los esquemas de los registros y el flujo por fases están ahí.
Las Master Rules globales (`~/.claude/CLAUDE.md`) aplican sin cambios; este archivo sólo dice dónde está cada cosa.

## Reglas del proyecto

- `Contenido_del_curso-Aplicaciones_de_IA_en_astrofisca/` es material de referencia: se lee, **nunca se modifica**.
- Si existen `OBJECTIVE.md` y `wiki/index.md`, leerlos al empezar y decir qué páginas se usaron.
- Todo número de salida va a `provenance/numbers.json` a través de un único script (`scripts/compute_numbers.py`);
  todo valor de entrada, a `provenance/parameters.yaml`; toda hipótesis, a `provenance/assumptions.yaml`,
  aprobada por el usuario antes de usarla (MR3).
- Los registros de `provenance/` son originales: no se copian ni llevan línea en `MANIFEST.txt` (MR7).
- No hay hooks registrados en este proyecto todavía: nada se hace cumplir automáticamente.

## rigorous-physics declarations
- numref: dataref                               # \dataref{key}{value} de paper/paperclaims.sty (C1, C2)
- registries: provenance/numbers.json           # único escritor: scripts/compute_numbers.py (C2, C4, C17, C26, C56)
- inputs: provenance/parameters.yaml            # (C26)
- parameters: provenance/parameters.yaml        # (C80)
- assumptions: provenance/assumptions.yaml      # (C80, MR3)
- claims: provenance/claims.yaml; fields: id,statement,evidence   # (C5)
- figures: figures/ (png,pdf,svg); record: provenance/claims.yaml  # sección figures: (C7)
- documents: paper/main.tex; format: latex; language: es           # (C1, C2, C8, C9, C47)
- bibliography: references/references.bib       # (C9, A12)
- code: py                                      # (código en src/, scripts/, derivations/, tests/)
- suite: pytest -q                              # (C23)
- pipeline: make reproduce; clean: make clean   # (C56, C58)
- outputs: figures/, site/, paper/main.pdf, provenance/numbers.json   # (C56, C57, C58, C60)
- build manifest: Makefile, README.md           # (C57)
- lock: conda-lock                              # (C50) — a confirmar al crear el entorno
- gate: python3 scripts/check_provenance.py; covers: C1,C2,C3,C4,C5,C7,C8,C9,C80   # creado 2026-09-28; C1/C2/C3/C9-doc SKIPPED mientras paper/main.tex no exista; tests: electron_losses_IGM/tests/test_gate.py
