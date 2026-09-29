# Estructura de trabajo para el proyecto final

## Contexto

Pedido en `../Instrucciones_generales_para_planear_el_trabajo_final`: (1) analizar el material del curso
(`../Contenido_del_curso-Aplicaciones_de_IA_en_astrofisca/`, abreviado `curso/` abajo) para entender
*provenance* y reproducibilidad, y (2) proponer una estructura de trabajo para el proyecto final que
sirva **con cualquier tema**. Todas las rutas citadas son relativas a `curso/`.

Fuentes principales leídas: `README.md`, `final-project.html`, `homework.html`, `using-the-tools.html`,
`day1..5/README.md`, las cinco `day*/session-ai*.html`, `day*/prompts.txt`, `day3/objective.txt`,
`day1/vault/` (AGENTS.md, plantillas), `day5/exercise/` (hook, skill, formatos), `day5/skills/`,
`day5/exercise-notes/wiki/` (conventions, reproducibility, environment, bibliography, log),
`how-a-page-is-built/`.

Lo que pide el curso (`final-project.html`): un problema propio con muchas componentes (derivación,
código, figuras, datos), hecho con agentes, con la provenance escrita *mientras se trabaja* y con cada
resultado **verificado** ("una figura con origen registrado y sin check sigue siendo una afirmación con un
dibujo al lado"). Entrega: un repo de GitHub con **una página HTML y un PDF** (para personas) y el resto
del repo (para máquinas). Cualquiera que lo clone tiene que poder reproducir todo con un agente:
instrucciones, entorno, datos y checks legibles por algo que nunca habló con vos.

Definición del curso: *la PROVENANCE de un resultado es qué archivo y qué función lo produjeron, qué
entró, qué se escribió desde cero frente a qué se llamó de una biblioteca, y cada elección que tenía una
alternativa defendible.*

## Parte 1 — Qué enseña el material (síntesis)

Qué agrega cada día sobre cómo trabajar (sesiones `day*/session-ai*.html`, `homework.html`, `using-the-tools.html`):

| Día | Lección de práctica | Falla documentada que la motiva |
|---|---|---|
| 1 | Pedir **con contrato**: "dejame un script, una figura y la lista de cada claim verificado y cómo; decí qué derivaste y qué recordaste". Registro por escrito (el agente arranca en frío cada sesión). Números estampados: `results.json` + `results_provenance.json` {file, func, line}, el `.tex` cita `\src{key}` y nunca dígitos | "23/23 verde" sin ningún assert sobre las 3 figuras con datos (todas mal: sin whitening, signo invertido, magnitud equivocada). Un check que ajusta 30 M☉ puestos a mano contra sí mismo |
| 2 | Checks con valor esperado **antes** de mirar el resultado (varianza 1, ⟨ρ²⟩=2 en ruido, recuperar inyección). Prompt congelado byte a byte; correr headless; juzgar el artefacto, no el resumen | Un modelo **ajustó** la normalización para que ⟨ρ²⟩=2 (el check no podía fallar). Sonnet headless: "success", 109 turns, ningún archivo. Bajar el esfuerzo apagó la verificación sin cambiar el número |
| 3 | `objective.txt` + `/goal`: entregables ordenados (reporte primero), qué es un reporte honesto, restricciones, reloj de pared. Correr dos veces. Revisión ciega por dos sesiones frescas (Claude y Codex). Tests de aceptación bit-idénticos (sha256) | Revisores coincidieron sólo en 2 de 5; bugs de herramientas que dan respuestas plausibles, no crashes; reportes "silenciosamente viejos" |
| 4 | Equipos con roles; **el verificador es de otra familia de modelos**; plantilla de prompt GOAL/SOURCES/FENCE/DISCIPLINE/DONE/DELIVERABLE; "decí qué NO chequeaste"; `checks.py` debe incluir un caso que falle a propósito; `report.html` (humanos) + `agent.html` (cada número → código) | Un claim refutado se publicó igual; un límite medido que era el percentil 95 del prior; figura de un paper sin abrir durante 6 rondas |
| 5 | La convención en `.md` una vez; skill que apunta; hook que verifica. Paper publicado como repo legible por máquina | Celda sin nada: el número muere en el chat. Con skill+hook o sólo hook: número en archivo con función, biblioteca y elecciones |

Reglas de prompt reutilizables: "Every number … comes from code in this job, or is quoted from a named paper
with its equation or page. Say which, every time"; "a few per cent is usually a modelling choice; an order of
magnitude is usually a bug — stop on the second kind"; "pick cases that would expose an error, not cases that
flatter it"; "a short list of the choices that could reasonably have gone another way"; registrar tokens y
reloj de pared ("durables; los precios no").

Modelo de referencia (día 5, release del paper PTA de Zaldarriaga & Sato-Polito, y `day5/exercise/`):

| Pieza en el paper publicado | Pieza a escala laptop (`day5/exercise/b`) | Rol |
|---|---|---|
| `structure/claims.yaml` (tesis + 11 claims: statement, sección, numbers, evidence, depends_on, status) | `provenance/claims.yaml` | el esqueleto del argumento: qué se afirma y qué lo respalda |
| `data/paper_numbers.json` (106 claves, **un solo escritor** `compute_paper_numbers.py`) | `provenance/numbers.json` (value, statement, produced_by, from_scratch, from_library, choices) | registro de números |
| `structure/figures.yaml` + `\genby{}` | `figures:` en claims.yaml | figura → script::función → inputs → claims |
| `paperclaims.sty` (`\dataref{key}{value}`, `\genby`, `\provnote`; `[draft]`/`[final]`) | `SKILL.md` + `.claude/provenance/*.md` | la convención, escrita **una sola vez**, cargada donde se escribe |
| `check_provenance.py` (10 checks; los no aplicables se reportan *skipped*, no *pass*) | `hooks/provenance_gate.py` (hook `Stop`) | la compuerta: no está terminado hasta que pasa |
| README §Reproducibility (bitwise / raster-idéntico / tolerancia medida 1e-12 / tolerancia MC / **no afirmado**) | — | honestidad sobre qué se reproduce y cómo |
| wiki (`index`, `conventions`, `log`, `todo`, `sources/`, `concepts/`, `code/`, `results/`, `figures/`) + `reproducibility.md` | `day1/vault/` + `AGENTS.md` | memoria entre sesiones; derivada, "la fuente gana" |

Reglas recurrentes (textuales del material): "prefer a check to a claim"; "a result with no assertion is a
hypothesis"; decir qué se derivó y qué se recordó; "never claim to have run something you did not run";
"state what you did *not* check"; "never invent a number → todo.md"; "anything untracked needs a
regeneration command"; "link, never copy"; "a convention written twice drifts"; verificar cada bib contra
arXiv **y** ADS (incidente "Lin & Loeb"); "la página no puede derivar del registro que reporta" (el
builder de las páginas lee los números de archivos y falla si no coinciden).

## Parte 2 — Estructura propuesta (plantilla independiente del tema)

Principio organizador (del hook del día 5): **la convención vive una vez en archivos `.md`; un skill /
CLAUDE.md apunta a ella; un script-compuerta verifica el resultado y, si falla, sirve el `.md` que aplica.**
Todo lo demás es consecuencia.

### 2.1 Árbol del repositorio

```
<proyecto>/
├── README.md              para humanos Y agentes: qué es, cómo reproducir (make reproduce), §Reproducibility honesta
├── CLAUDE.md              bloque `rigorous-physics declarations`: dice a las reglas globales dónde está cada registro
├── OBJECTIVE.md           el pedido escrito antes de empezar: entregables, formato, checks que debe pasar, qué NO hacer
├── environment.yml + lock entorno fijado (conda-lock o pixi.lock); versiones exactas registradas
├── Makefile / reproduce.sh etapas: env → data → derivations → numbers → figures → paper → site → check
│
├── references/            PDFs citados + references.bib + README.md (qué se usa de cada paper, ec./tabla/§)
├── provenance/            REGISTRO LEGIBLE POR MÁQUINA (única fuente de verdad)
│   ├── assumptions.yaml   hipótesis físicas / aproximaciones (id, régimen, fuente, estado, quién la aprobó)
│   ├── parameters.yaml    valores numéricos de ENTRADA con unidad, incertidumbre, fuente
│   ├── numbers.json       valores de SALIDA (un solo escritor: scripts/compute_numbers.py)
│   ├── claims.yaml        tesis + claims + figures (esquema día 5 extendido)
│   ├── scripts.yaml       productores: comando, inputs, outputs, semilla, tiempo
│   ├── run_metadata.md    receipts de corridas (C71): modelo, esfuerzo, tokens, reloj, turns, fecha, comando, log
│   └── MANIFEST.txt       sólo copias de archivos que viven FUERA de provenance/ (`copy <- original`, MR7);
│                          los registros de arriba son originales y no se copian
├── data/raw/              inmutable, con DATA.md (origen, versión, URL/DOI, fecha de descarga, sha256)
├── src/<paquete>/         módulos importables, documentados (ver 2.4)
├── derivations/           derivaciones analíticas en sympy (marimo .py → HTML); cada una termina en asserts
├── scripts/               run_*.py, fig_*.py, compute_numbers.py, check_provenance.py, build_site.py
├── tests/                 pytest: identidades sympy, límites, reproducción de números publicados, dimensiones
├── figures/               GENERADO. Nunca editado a mano
├── paper/                 main.tex + paperclaims.sty ([draft]/[final]) → PDF entregable
├── site/                  index.html GENERADO por build_site.py desde provenance/ (falla si hay drift)
├── reviews/               informes de revisores independientes (agente fresco, otro modelo, humanos) + respuesta
├── manuscripts/           (regla global MR2) resúmenes LaTeX/PDF de respuestas del agente; no es el paper
└── .claude/  (+ .codex/)  settings.json (hooks), skills/provenance-record/, provenance/*.md (formatos)
```

`references/`, `provenance/` y `manuscripts/` coinciden con las carpetas de las Master Rules globales;
`provenance/` también es donde el hook del día 5 espera `numbers.json` y `claims.yaml`, así que se reutiliza.

### 2.2 Esquemas YAML/JSON (extensión del día 5 con unidades, incertidumbre y fuentes)

`provenance/parameters.yaml` — **ningún número entra al código si no está acá**:
```yaml
parameters:
  G:
    value: 6.67430e-11
    unit: m^3 kg^-1 s^-2          # parseable por astropy.units (el gate lo verifica)
    uncertainty: {sigma: 1.5e-15, type: "1-sigma"}   # o null + motivo
    source: {bib: <clave_bib>, loc: "<tabla o ecuación>"}  # clave que existe en references.bib
    tag: measured | assumed | fitted | derived   # vocabulario de C35
    # source es UNA de: {bib, loc} (cita con localizador) | {user: "<fecha y cita textual>"} |
    #                   {produced_by: "<archivo>::<función>"} (obligatorio si tag: derived)
    note: ""
```
`provenance/assumptions.yaml`:
```yaml
assumptions:
  A01:
    statement: "Aproximación cuadrupolar, órbita kepleriana (0PN)"
    regime: "v/c << 1; separación >> radio de Schwarzschild"
    source: {bib: Peters1964, loc: "<sección o ecuación>"}
    status: proposed | adopted | tested | flagged-outside-regime
    approved: {by: "<usuario>", date: 2026-..}   # regla MR3: el agente no adopta hipótesis solo
    tested_by: [tests/test_limits.py::test_circular_limit]
    used_by: [src/pkg/orbit.py::merger_time]
```
`provenance/numbers.json` — los 6 campos del día 5 **+** `unit`, `uncertainty`, `tag` (C35), `assumptions`, `inputs`,
`checked_by` (sin `checked_by` el número se marca *hipótesis*). `from_scratch` = `<archivo>::<función>` o `none`;
`from_library` = `<paquete>==<versión>, <llamada>, <argumentos no default>` o `none` (formato de C34):
```json
{"enh_e09": {"value": 293.5112, "unit": "", "uncertainty": {"sigma": 1e-4, "kind": "numerical"},
  "statement": "Tc/T(e0=0.9) a semieje fijo", "produced_by": "src/pkg/orbit.py::merger_time_ratio", "tag": "derived",
  "from_scratch": "src/pkg/orbit.py::merger_time_ratio",
  "from_library": "scipy==<versión>, scipy.integrate.quad, epsabs=1e-14 epsrel=1e-12 (defaults: 1.49e-8)",
  "choices": ["epsrel=1e-12 porque ..."],
  "assumptions": ["A01"], "inputs": ["G","c"], "checked_by": ["tests/test_orbit.py::test_ode_crosscheck"]}}
```
`provenance/claims.yaml` — el formato del día 5 + `depends_on`, `assumptions`, `status`
(`hypothesis | checked | reproduced | reviewed`) y `figures:` (file, produced_by, shows, from_scratch,
from_library, choices, supports). Regla del release: *cada evidence apunta a código que se entrega o a una
clave del .bib; nada más es evidencia válida.*

Los formatos se escriben una vez en `.claude/provenance/{parameters,assumptions,numbers,claims,figures}.md`
(copiando/extendiendo los tres de `day5/exercise/b/.claude/provenance/`); skill y hook apuntan a
ellos, no los repiten.

### 2.3 Referencias: ninguna afirmación sin paper

- `references/` contiene el PDF de todo lo citado; `references.bib` con eprint siempre .
- Alta de una referencia = verificación contra arXiv **y** ADS, diff mostrado antes de editar el .bib (plantilla `day1/vault/templates/
  paper.md`: claims | su evidencia | la nuestra | qué no reproducimos).
- Toda afirmación física en código/paper/página cita `bib:loc` (ecuación, tabla, sección).

### 2.4 Convención de código (legible por humanos)

- Cabecera de cada módulo/script: propósito · qué produce (archivos, claves de numbers.json) · de qué lee ·
  ecuaciones implementadas `[Peters1964 Eq. 5.14]` · hipótesis `[A01]` · convención de unidades · cómo correrlo.
- Docstrings estilo numpy con **unidades en cada parámetro y retorno**; cantidades con `astropy.units`
  en la frontera (entrada/salida), SI adimensionalizado adentro si hace falta velocidad, documentado.
- Comentarios en los pasos no obvios con la etiqueta de la hipótesis o ecuación; sin números mágicos
  (todo viene de `parameters.yaml` vía un loader único `src/<pkg>/params.py`).
- Semillas explícitas: `np.random.default_rng(seed)` con el seed en `scripts.yaml` y en numbers.json.
- Separar "escrito desde cero" de "llamado de biblioteca" (alimenta `from_scratch`/`from_library`).
- Cada función que produce un número registrado tiene un test.

### 2.5 La compuerta: `scripts/check_provenance.py` (+ hook `Stop`)

Extiende `day5/exercise/b/.claude/hooks/provenance_gate.py` y la lista del `check_provenance.py` del paper:
1. numbers.json: campos obligatorios; `produced_by` resuelve a archivo **y** función existentes (AST).
2. parameters.yaml: toda `unit` parsea con astropy; toda `source.bib` existe en el .bib; `tag: derived` tiene productor (= check global C80).
3. assumptions.yaml: toda hipótesis usada está `adopted/tested` y aprobada; ids referenciados existen (= C80).
4. claims.yaml: statement + evidence; evidence = archivo que se entrega o clave bib; numbers ∈ numbers.json.
5. Toda figura en `figures/` tiene entrada; todo `\genby`/`\dataref` resuelve.
6. `\dataref{key}{valor}` coincide con el registro a la precisión mostrada (check 7 del paper).
7. Cada clave bib tiene PDF en `references/`; MANIFEST: copias = original (verificación de copias de MR7).
8. Análisis dimensional de las fórmulas clave (tests con astropy/pint) y pytest verde.
9. Los números de `site/index.html` y del PDF salen del registro (build_site.py falla si difieren).
10. Checks no aplicables → **SKIPPED** explícito, nunca pass silencioso. Resumen final "N de M corridos".

Wiring: `.claude/settings.json` (SessionStart `--start`, Stop) y `.codex/hooks.json`, igual que el día 5,
+ skill `provenance-record` que apunta a los formatos. Hook con límite de intentos (MAX_ATTEMPTS).

### 2.6 Verificación (lo que el curso más enfatiza)

Cada resultado necesita al menos una de estas, registrada en `checked_by`:
- **sympy**: la diferencia es cero (derivaciones); límites conocidos; orden de magnitud.
- **Reproducir un número publicado** de un paper en `references/` (assert).
- **Segunda implementación independiente** (p.ej. quad vs. ODE, como en el día 5).
- **Piso de ruido**: correr la misma tarea dos veces igual; diferencias menores al piso no son resultados.
- **Revisor fresco**: sesión nueva, sin contexto, sólo el output (y otro modelo/Codex); primero escribir la
  propia lista de defectos; guardar en `reviews/` el informe, qué era real y qué se corrigió.
- **Lectura ciega del PDF** (como en la wiki PTA): agente sin herramientas sólo con el texto.
- **Test de clon limpio, headless**: un agente sin historia clona el repo y corre `make reproduce`; se
  compara lo que dejó **en disco**, no su resumen (caso real: 109 turns, "éxito", ningún archivo).
- **Revisión humana**: lista de checks manuales en `reviews/HUMAN_CHECKLIST.md`.

### 2.7 Salidas para personas

- `paper/main.tex` con `paperclaims.sty`: `[draft]` muestra las anotaciones de provenance en color para
  revisar; `[final]` las oculta. El texto nunca nombra scripts fuera de las macros.
- `site/index.html` autocontenido (abre con `file://`), generado por `scripts/build_site.py` desde
  `provenance/` y `figures/` (estilo `how-a-page-is-built/build_show.py`: figura embebida desde el archivo
  renderizado, cada número con `[src:key]`, build con `--check` para detectar drift).
- README §Reproducibility con las categorías del paper PTA: bitwise demostrado · raster-idéntico
  (`SOURCE_DATE_EPOCH`) · tolerancia medida · error Monte Carlo · **qué no se afirma**.

### 2.8 Memoria y registro de sesiones

- `wiki/` (index, conventions con cada error ya cometido,
  log append-only, todo); instalar también el skill `project-notes` para que se lea.
- `provenance/run_metadata.md` = run receipts (check C71: modelo, esfuerzo, tokens, reloj de pared, máquina,
  turns, fecha, comando, ruta del log; lo no disponible se escribe `—`). Sin receipt, una página registra un
  plan, no un resultado.

## Parte 3 — Flujo de trabajo por fases

0. **Elegir el problema** (propio es mejor: se puede juzgar la respuesta) con derivación + código + figura
   + datos. Escribir `OBJECTIVE.md` (entregables, formato, checks) — el ejercicio 1 del curso muestra que es
   el cambio más barato.
1. **Andamiaje**: crear el árbol, AGENTS.md, formatos, hook, entorno fijado, `git init`.
2. **Referencias**: ingestar papers (PDF + bib verificado + `wiki/sources/`).
3. **Hipótesis y parámetros** antes de cualquier código: `assumptions.yaml` (aprobadas por vos),
   `parameters.yaml`, `wiki/conventions.md` (notación, signos, unidades).
4. **Derivaciones** en `derivations/` con sympy y asserts; marcar derivado vs. recordado.
5. **Código + figuras + números**: cada producto registrado; `compute_numbers.py` único escritor.
6. **Verificación** (2.6) y compuerta verde.
7. **Salidas**: PDF `[final]`, página HTML generada, README.
8. **Release**: test de clon limpio headless por un agente fresco; recién entonces push a GitHub.

## Parte 4 — Próximos pasos: implementar la plantilla (todavía no hecho)
- el árbol de 2.1 vacío con `README.md` de cada carpeta;
- `CLAUDE.md`, `OBJECTIVE.md` (plantilla con huecos);
- `.claude/provenance/*.md` (5 formatos), `.claude/skills/provenance-record/SKILL.md`,
  `.claude/settings.json`, `.codex/hooks.json` — reutilizando `day5/exercise/b/.claude/*` como base;
- `scripts/check_provenance.py` (checks 1–10, extensión de `provenance_gate.py`) y el hook que lo llama;
- `src/<pkg>/params.py` (loader de parameters.yaml → astropy Quantity);
- `paper/paperclaims.sty` mínimo (`\dataref`, `\genby`, `\provnote`, draft/final) y `main.tex` esqueleto;
- `scripts/build_site.py` mínimo; `Makefile` con `reproduce` y `check`;
- `wiki/{index,conventions,log,todo}.md`; `reviews/HUMAN_CHECKLIST.md`.

Opcional: un ejemplo mínimo de punta a punta (Peters 1964, factor a e=0.9) para probar que la compuerta,
el sitio y el PDF funcionan juntos. El valor 293.5112 que aparece en este documento está tomado de las
corridas del curso (`day5/session-ai5.html`, celdas b/ y c/). **No lo recalculé**, así que acá cuenta
como un valor de referencia a reproducir, no como un resultado.

## Cómo verificar la plantilla (cuando se implemente)

- `python scripts/check_provenance.py` en el árbol vacío → sólo SKIPPED, exit 0.
- Con el ejemplo Peters: registrar a mano un número sin `produced_by` → la compuerta falla y cita el `.md`
  correcto; completarlo → pasa. Figura sin entrada → falla.
- `\dataref` con un dígito cambiado → falla el check 6.
- Sesión headless (`claude -p`) en una copia limpia con `OBJECTIVE.md` del ejemplo → verificar en disco
  que dejó `provenance/` completo y que el hook disparó o no.
- `make reproduce` desde un clon limpio en entorno nuevo.

