# EM_cascades

Cascadas de electrones en el IGM: fracción de energía que se lleva cada proceso de enfriamiento y número de
ionizaciones (colisionales, por fotoionización secundaria de fotones de IC, y totales) que produce un electrón hasta
termalizar. Instrucciones: `contenido_inicial_generado_por_Carvalho/Instrucciones_codigo_cascadas_de_electrones`.
Plan: `~/.claude/plans/lee-el-archivo-instrucciones-codigo-casc-luminous-sonnet.md` (fases F0–F7).

Usa `igm_losses` (`electron_losses_IGM/src`) **sin modificarlo** y los registros únicos del proyecto en
`provenance/` (parámetros, hipótesis, decisiones, números, afirmaciones, registros de figuras). Las figuras van a
`figures/` de la raíz con prefijo `cas_`. Nada se escribe dentro de `electron_losses_IGM/` ni de
`contenido_inicial_generado_por_Carvalho/`: los scripts y tests importan primero `em_cascades._paths`, que desactiva
la escritura de bytecode.

## Estado

| Fase | Estado |
|---|---|
| F0 andamiaje | hecho (2026-09-28) |
| F1 registros | hecho (2026-09-28): A19–A24 adoptadas y A17 reescrita (absorbe la A25 propuesta) en `provenance/assumptions.yaml`; specs de 3 figuras en `parameters.yaml → figures`; D22 (grilla) |
| F2 referencias | hecho (2026-09-28): Verner et al. 1996 (`provenance/data/verner1996_HI.yaml`, Ec. 1 y Tabla 1) y Shull & van Steenberg 1985 (`provenance/data/shull1985_table1.yaml`, Tabla 1), ambos en `references/references.bib` |
| F3–F4 código y tests | hecho: `src/em_cascades/{fractions, ic_spectrum, secondary, yields, photoionization, plotting}.py`; 33 tests (Verner vs hidrogénico exacto, espectro de IC vs pérdida de igm_losses, fracciones vs ODE en el tiempo, cascada vs Shull & van Steenberg, convergencia); mutaciones 8/8 (`tests/mutants/`) |
| F5 figuras | hecho: `figures/cas_loss_fraction_rate*`, `cas_loss_fraction_integrated*`, `cas_ionization_yield*` (5 x_e cada una) |
| F6 números y afirmaciones | en curso: `scripts/numbers.py` (llamado por el escritor único de la raíz), afirmaciones C9–C12 en `provenance/claims_statements.yaml` |
| F7 integración | hecho: gate (C4 por subproyecto, C8 del README, tests del gate en `tests/test_gate.py` de la raíz), Makefile (`cas-*`, `numbers` en la raíz, `test` con las tres carpetas); falta la reproducción desde cero y el cierre |

## Figuras

| Clave (`parameters.yaml → figures`) | Archivo | Qué muestra |
|---|---|---|
| `loss_fraction_rate` | `figures/cas_loss_fraction_rate[_xe*].png` | L_i/L_tot de cada proceso en función de K, medio fijo en z_init |
| `loss_fraction_integrated` | `figures/cas_loss_fraction_integrated[_xe*].png` | fracción de K_ini que se lleva cada proceso hasta K_floor, en función de K_ini |
| `ionization_yield` | `figures/cas_ionization_yield[_xe*].png` | ionizaciones colisionales, por fotoionización secundaria y totales hasta K_floor, con el techo K_ini/R_H |

Tests: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider EM_cascades/tests` (desde la raíz).
