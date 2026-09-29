# electron_losses_IGM

Reescritura desde cero del notebook `contenido_inicial_generado_por_Carvalho/Perdidas_de_energia-Z_10.ipynb`:
pérdidas de energía de un electrón que atraviesa el medio intergaláctico desde `z_init` hasta `z_final`.
Plan de acción: `~/.claude/plans/radiant-humming-marble.md` (fases F0–F7). Convenciones del proyecto:
`plantilla_proyecto_final/PLAN_TRABAJO_FINAL.md` y `CLAUDE.md` en la raíz.

## Estado

| Fase | Estado |
|---|---|
| F0 andamiaje | hecho: árbol de carpetas, copia del notebook en `provenance/original/` (MANIFEST), 23 figuras originales en `reference_figures/` |
| F1 registros YAML | hecho y revisado con el usuario (2026-09-27): 17 hipótesis adoptadas, 17 decisiones tomadas, 10 erratas confirmadas y 4 pendientes del PDF (E01, E05, E07, E12). **Estado al 2026-09-28:** 21 decisiones (D01–D21) y 21 erratas (E01–E21; E20 confirmada con evidencia, E21 sospechada) |
| F2 referencias | hecho (2026-09-27): `references/references.bib` con 12 entradas, todas con su PDF en `references/`; el año juliano queda como assumed sin cita. **Estado al 2026-09-28:** 18 entradas |
| F3 verificación | terminada (los pendientes de esta fila se resolvieron en F3–F4: bremsstrahlung D18–D19, excitación relativista E12); texto original (2026-09-27): constantes verificadas contra SI 2019, CODATA 2022, PDG 2024, Planck 2018 VI y Fixsen 2009; fórmulas contra B&G 1970, Gould 1972, Kim 2000, Kim & Rudd 1994, Stone et al. 2002, R&L y Furlanetto & Stoever 2010; 19 erratas confirmadas y resueltas. Pendiente: bremsstrahlung no relativista contra H neutro y la fuente de la forma relativista de la excitación |
| F4 implementación | hecha (2026-09-28), 71 tests pasan en ese momento (~8 min; hoy son 110, ver F5–F6). Módulos en `src/igm_losses/`: `constants`, `cosmology`, `medium`, `kinematics`, `inverse_compton`, `excitation`, `ionization`, `coulomb`, `gaunt_ff`, `bremsstrahlung`, `losses` (registro único de procesos), `integrate` (driver ODE único), `thresholds`, `phase_space`, `plotting` (+ `labels.yaml`). Tablas en `provenance/data/` con un único script escritor cada una (`scripts/compute_*_table.py`) |
| F5 figuras | rehecha (2026-09-28) tras el cambio de A04: x_e = 1e-4, 1e-2, 1e-1, 0.5, 0.99 en toda figura con procesos que dependen de x_e (n_e o n_HI; 17 figuras). 93 PNG+PDF en `figures/`, cada una con su registro (C7 pasa); `claims.yaml` con la sección figures; comparaciones en `comparisons/`. Las figuras de la versión anterior (barrido [1e-4, 1e-2, 0.5] sólo en las que usan n_e) se borraron antes de regenerar. E20: la imagen guardada de la celda 34 no corresponde a su código |
| F6 números y afirmaciones | hecha (2026-09-28): `provenance/numbers.json` (desde 2026-09-28 el único escritor es `scripts/compute_numbers.py` de la raíz del proyecto, que llama a `compute_all()` de `electron_losses_IGM/scripts/compute_numbers.py` y a los números de EM_cascades; ~10 min; incertidumbre numérica = |valor(rtol) − valor(rtol/100)|), verificado por `tests/test_numbers.py` (rederivaciones independientes: astropy, otro integrador LSODA, cotas físicas); `provenance/claims_statements.yaml` (tesis + 8 afirmaciones con plantillas `{{clave|formato}}`) → `claims.yaml` vía `build_claims.py`. La celda 20 (tiempos con rayos cósmicos) no se reproduce: sus hipótesis (A18) quedaron fuera en F1 |
| F7 cierre | hecho (2026-09-28): checks §11 (C59, C44, C42, C80, C17 → corregido E20, MANIFEST sync, C7), 110 tests y 14/14 mutantes. Pendiente: Makefile (`make reproduce`, C57), gate `scripts/check_provenance.py`, `paper/main.tex`, errata E21 |

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Parámetros de entrada (único lugar con números) | `provenance/parameters.yaml` |
| Hipótesis físicas (MR3, todas `proposed`) | `provenance/assumptions.yaml` |
| Variantes duplicadas a elegir | `provenance/decisions.yaml` |
| Errores sospechados del original | `provenance/errata.yaml` |
| Copia del notebook original | `provenance/original/` (listada en `provenance/MANIFEST.txt`) |
| Figuras originales extraídas | `reference_figures/cell<NN>_0.png` + `index.json` |
| Figuras nuevas (F5) | `figures/igm_<slug>.{png,pdf}` en la raíz |

## Mapa: celda original → figura nueva

La clave es el nombre en `parameters.yaml` → `figures:`.

| Celda | Clave | Qué muestra |
|---|---|---|
| 2 | `gyroradius_synch` | giro-radio vs tiempo, sólo sincrotrón, pitch fijo |
| 4 | `distance_synch_fixed` | distancia recorrida, sincrotrón, v ⟂ B |
| 5 | `distance_synch_isotropic` | ídem con promedios isotrópicos |
| 6 | `free_streaming` | distancia máxima y tiempo para llegar a R_target, sin pérdidas |
| 8 | `synch_only` | K/K_ini, sincrotrón |
| 9 | `adiabatic_only` | K/K_ini, adiabático, z = 30 → 5.5 |
| 11 | `ic_cmb` | K/K_ini, IC sobre CMB (Klein–Nishina) |
| 12 | `ic_cmb_fixed_vs_dynamic` | IC con z fijo vs z dinámico |
| 14 | `coulomb_only` | K/K_ini, Coulomb |
| 17 | `excitation_only` | K/K_ini, excitación colisional del H |
| 19 | `ionization_only` | K/K_ini, ionización colisional del H |
| 21 | `bremsstrahlung_only` | K/K_ini, bremsstrahlung |
| 23 | `total_cooling` | K/K_ini, todos los procesos |
| 25 | `ratio_without_synch_ad` | K_tot/K_sin-sincrotrón y K_tot/K_sin-adiabático |
| 27 | `loss_rate_fractions` | (L_tot − L_x)/L_tot para sincrotrón y adiabático |
| 29 | `phase_space_map` | mapa del proceso dominante en (z, K) + trayectorias + umbrales |
| 30 | `phase_space_colored` | trayectorias coloreadas por proceso dominante |
| 32 | `phase_space_without_ad` | todos vs sin adiabático |
| 34 | `distance_travelled` | K vs distancia recorrida |
| 35 | `phase_space_map_without_ad` | mapa sin adiabático |
| 37 | `accumulated_losses` | pérdida acumulada por proceso / K_ini |
| 39 | `step_losses` | pérdida por proceso a lo largo de z (ver E14) |
| 40 | `component_ratios` | razones de pérdida con/sin sincrotrón y adiabático (falló en el original) |
| 41 | `adiabatic_fraction_map` | mapa de la fracción adiabática Γ_ad (nunca se ejecutó) |
| 42 | `with_vs_without_adiabatic` | K con/sin adiabático (nunca se ejecutó) |

## Tablas y cómo regenerarlas

| Tabla (`provenance/data/`) | Único escritor | Tiempo |
|---|---|---|
| `phi_ep_KarzasLatter.json` | `scripts/compute_phi_ep_table.py` | ~4 min |
| `F_KN_table.json` | `scripts/compute_fkn_table.py` | ~1 min |
| `excitation_table.json` | `scripts/compute_excitation_table.py` | ~60 min |
| `CS_int_1.txt` | copia de `references/Brems/CS_int/CS_int_1.txt` (MANIFEST) | — |
| `stone2002_H_1s_np.yaml`, `kim_rudd1994_H_1s.yaml` | transcritos a mano de las tablas de los papers | — |

Tests: `python3 -m pytest -q electron_losses_IGM/tests` (desde la raíz, ~10 min).
Mutaciones (C23): `python3 electron_losses_IGM/tests/mutants/run_mutants.py [filtro]` (~30 min): cada mutante planta un error realista y debe ser atrapado.

## Reproducir

Desde la raíz del proyecto (Makefile en la raíz; `pipeline: make reproduce` en `CLAUDE.md`):

```bash
make -j2 reproduce   # figuras + numbers.json + claims.yaml + comparaciones, y al final el gate (~1 h con -j2)
make reproduce-all   # además regenera las tablas de provenance/data (~65 min más)
make check           # sólo el gate: python3 scripts/check_provenance.py
make test            # suite pytest (~10 min);  make mutants: arnés de mutación (~40 min)
make clean           # borra las salidas de reproduce (no las tablas, ni reference_figures, ni las entradas)
```
Cada target y su script están listados en la cabecera del `Makefile`.
