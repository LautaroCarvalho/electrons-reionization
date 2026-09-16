# CHECKS_AUTO — run these myself, before answering

**What this is.** The checks I can execute alone, with tools I have, without asking anything.
Run them *before* handing back an answer, not after being challenged.
Companion: `CHECKS_MANUAL.md` (needs the user or a second run). Source: `CHECKS_CATALOGUE.md`.

**The one rule.** *A result with no assertion is a hypothesis — say so in those words.*
**If I could not check it, the answer says so. Silence is the failure, not the gap.**

---

## §0 · ALWAYS — the pre-flight, every answer with any number or claim in it

Run top to bottom. Each line is pass/fail on its own.

```
[ ] A1  No invented numbers.       Every numeral traces to: code I ran | a citation with
                                    eq/page | the user's own words. Otherwise DELETE it.   MR4
[ ] A2  Units on every quantity.   No bare numbers. Units carried through every step, not
                                    reattached at the end.                                 MR6
[ ] A3  Dimensional check.         Verify homogeneity BEFORE quoting any final numeric.    MR6·C29
[ ] A4  Sig figs + uncertainty.    State both. Propagate input errors. Round ONLY the
                                    final reported value; never an intermediate.           MR6·C59
[ ] A5  Derived vs recalled.       Label every step. Unprompted, every time.               MR4·C33
[ ] A6  Fact / model / hypothesis. Three separate registers, visibly separated.            MR4
[ ] A7  Regime of validity.        State it. Flag if the analysis pushes outside it.       MR4
[ ] A8  Cross-check or confess.    sympy, limit case, or order-of-magnitude — or an
                                    explicit line saying NO CROSS-CHECK WAS PERFORMED.     MR5·MR8
[ ] A9  "I ran it" ≠ "it produces". Paste actual output. Never claim an unrun run.         C38
[ ] A10 Name the uncovered half.   "N/N passing" alone is misleading. Say what is NOT
                                    checked.                                               C14
[ ] A11 Confidence flags.          Mark every low-confidence step. "I cannot verify this
                                    from what I have" beats a guess.                       MR8
[ ] A12 Running reference list.    Every source used, accumulated, auditable later.        MR7
```

**Stop conditions — halt and report instead of choosing:**
- Two verification methods disagree → **report the discrepancy, do not silently pick one.** `MR5`
- A disagreement with a published value is an **order of magnitude** → that is usually a bug.
  Stop. (A few per cent is usually a modelling choice.) `C41`
- I need a physical/astrophysical hypothesis that was not given → **ask** (`CHECKS_MANUAL §0`).

---

## §1 · Output contract (Master Rule 2)

Every substantive answer, in this order:

```
Assumptions          → each stated AT THE POINT IT IS INTRODUCED, not in a preamble   MR3
Derivation
Verification / cross-check
Final result         → with units and uncertainty                                     MR6
```
Then: **write the LaTeX file, in the same folder as the summary file.** `MR2`
Also ship runnable code beside every numeric result — never the number alone. `MR7`

---

## §2 · IF I produced a DERIVATION

| ID | Action | Pass criterion |
|---|---|---|
| **C25** | `sympy`: `simplify(derived - printed)` | `== 0` exactly. A machine check, not a re-reading by the mind that made the error. |
| **C26** | Evaluate the symbolic form numerically against the pipeline number | relative error reported, not asserted-as-small |
| **C28** | Reduce to every known special case | each limit reproduces the known result |
| **C30** | Independent order-of-magnitude estimate | **write the expected magnitude BEFORE running** |
| **C27** | Two independent literature routes to the same quantity | both computed, both reported |
| **C40** | If reproducing a figure/equation: derive it from scratch, never copy | if the data was unobtainable, say **"not reproduced"** |

```python
# C25 — the canonical form. Not "looks right"; zero.
import sympy as sp
resid = sp.simplify(derived_expr - paper_expr)
assert resid == 0, f"residual {resid}"
print("C25 PASS  residual is identically zero")
```

---

## §3 · IF I produced a NUMBER

| ID | Action |
|---|---|
| **C34** | Record `from_scratch` vs `from_library` (what I called, and from where) |
| **C35** | Tag it: **measured** / **assumed** / **fitted** |
| **C36** | List every choice that had a defensible alternative, **with its reason**. Empty list written `[]` explicitly, never omitted |
| **C52** | Report N beside every mean, and say whether samples are independent |
| **C49** | Record the RNG **API call**, not just the seed (`Halton(seed=)` ≠ `Halton(rng=)`) |
| **C37** | If evidence list is empty → render the word **hypothesis** |
| **C17** | Grep my own output for the same quantity reported twice with different values |

```bash
# C17 — the twelve-lines-apart failure. Two values for one quantity in one file.
grep -nE "chirp|SNR|<quantity>" run.log results.json | sort -k2
```

---

## §4 · IF I wrote a CHECK — can it fail?

**A check that cannot fail is not a check.** Run all four before trusting any suite.

| ID | Detector | Grep / test |
|---|---|---|
| **C19** | **Fitted constant** — a constant solved for so the check passes | `grep -nE "sqrt\(\s*[0-9.]+\s*/\s*(np\.)?mean" -r .` |
| **C20** | **Self-inverse** — test input produced by the inverse of the function under test | trace: does ground truth enter from **data**, or from the same module? |
| **C16** | **Tolerance audit** — tolerance tighter than ~1e-6 on a *data-derived* quantity | `grep -nE "< *1e-(7|8|9|1[0-9])" -r .` → each needs a justification |
| **C22** | **Independence** — does the check import the thing it checks? | list imports; a verifier sharing implementation is not a verifier |
| **C18** | **Ship a deliberate-disagreement case** — make the two methods disagree on purpose | the comparison is shown *capable* of failing |
| **C23** | **Mutation test** — flip a sign / perturb a constant | the suite must go **RED**. If nothing fails, it tests nothing |
| **C19b** | **Narrow detector** — the check greps for ONE error string while the tool emits several classes | read the raw log/output once by hand and compare with what the check would have caught |

> **C19b, seen live (2026-09-07).** A LaTeX gate tested only for `Undefined control
> sequence`. pdflatex was emitting `! Missing $ inserted` on every line where a macro
> was used inside an `align`; the gate reported the build clean for as long as it
> existed. **Fix the class, not the instance:** greping for `^! ` catches every error
> pdflatex can raise. Before shipping any log-scraping check, ask *what else can this
> tool print when it is unhappy?*

```bash
# C16 — suspiciously exact tolerances on measured quantities
grep -rnE "abs\(.*\) *< *1e-(0?[6-9]|1[0-9])" --include=*.py .
```

---

## §5 · IF I produced CODE or a PIPELINE

| ID | Action | Note |
|---|---|---|
| **C55** | **Normalisation test** — statistic follows its predicted distribution | on real *and* synthetic data |
| **C53** | **Null test** — run where the answer is known to be nothing | assert nothing comes back |
| **C54** | **Injection–recovery** — inject known amplitude, recover it | *the one that catches the worst bugs* |
| **C39** | **Sample vs population** — test the bound against a **fresh draw the construction never saw** | a guarantee over drawn candidates is not a guarantee over the continuum |
| **C31** | Recompute one key quantity with a different library/algorithm | |
| **C58** | Assert nothing is read from cache in place of being computed | *caching is a different program, not a speed-up* |
| **C43** | **Docstring ↔ implementation** — does the body do what the docstring says? | *the answer can still land in range* |
| **C50** | Emit a **lockfile**, not a package list | the environment is a file, not a description |
| **C56** | Print the hash of the registry you just wrote. **Do not assert byte-identity from one run** — it is a claim about two, so it belongs in `CHECKS_MANUAL` §1. Publishing the hash makes the comparison one command for anyone |

> The triple that transfers to almost any measurement pipeline: **normalisation (C55), null (C53),
> injection–recovery (C54).** Run all three before believing any headline number.

---

## §6 · IF I produced a FIGURE

| ID | Action |
|---|---|
| **C42** | Axis label must name the quantity the code actually returns (energy ≠ amplitude) |
| **C44** | Flag every hand-set `vmax` / `ylim` / `clip` and give it a recorded reason |
| **C7** | Entry with `produced_by` = `file::function`, named inputs, `shows`, `choices`, `supports` |
| **C46** | Every number in the caption resolves to the registry |
| **C47** | Identify by stable label key (`fig:xxx`), never by figure number |
| **C48** | Regenerate; assert zero differing pixels at fixed dpi; set `SOURCE_DATE_EPOCH` |

```bash
# C44 — tuned-to-appearance limits
grep -rnE "vmax *=|vmin *=|set_ylim|set_xlim|clip\(" --include=*.py .
```

---

## §7 · IF a REPO / PAPER BUILD exists — the gate

Wire to a `Stop` hook, not just a build script. Non-zero exit reopens the turn.
Loop guard: `MAX_ATTEMPTS = 3`, then print what is missing and exit 0.

| ID | Check |
|---|---|
| **C1** | Every number in prose carries a registry key — no bare numeral outside `\dataref` |
| **C2** | **Displayed-precision agreement**: literal == registry value rounded to the literal's precision |
| **C3** | Every named key exists; every registry key is used (orphans both ways) |
| **C4** | **Single-writer rule** — exactly one script writes the registry |
| **C5** | Every claim has a statement and ≥1 evidence; every `numbers:` slug resolves |
| **C6** | Evidence is shipping code, or a citation with eq/section/figure number. Nothing else |
| **C8** | Every path-like token in every shipped text file resolves |
| **C9** | Citations parse; every cited work is present; **concept DOIs, never record numbers** |
| **C10** | Session-start timestamp — handed-in files are not counted as produced |
| **C11** | Self-containment: no CDN, no external stylesheet, no remote font, no `fetch` |
| **C12** | A check that cannot run reports **skipped**, never passes silently |
| **C13** | Print **per-claim coverage numbers, not verdicts** |
| **C15** | **Data-touching audit** — every data-derived claim has a check that reads the data |
| **C57** | Every untracked file has an entry giving the literal command that rebuilds it |
| **C68** | Every supplied source was actually opened — flag any never read |
| **C69** | Assert nothing under a fenced path was read |

```bash
# C11 — self-containment, the check that is always skipped
grep -nEi "https?://|cdn\.|<link[^>]+href=|fetch\(|@import" out/*.html && echo "C11 FAIL"
```

---

## §8 · IF a run was UNATTENDED

| ID | Action |
|---|---|
| **C60** | **Check the artefact, never the summary.** Files exist and parse — *before* reading what the run says about itself. Exit 0 and `"success"` mean nothing |
| **C62** | Count verdicts from the raw transcripts myself; compare with the summary state |
| **C63** | Every refutation ruled on in writing by the round that follows |
| **C71** | Record: model, effort, tokens in/out, wall clock, machine, turns, date, command, receipt. Absent → print `—`, which is the honest answer |

---

## §9 · Self-interrogation — reasoning checks, no tool required

Ask these of my own output. They are cheap and they caught what nothing else did.

```
[ ] C21  Is any residual I quote as AGREEMENT bounded by one of my own assumptions?
         If yes → it cannot be large, so it is NOT EVIDENCE. Say "anchored", not "recovered".

[ ] C74  What could a WRONG answer still reproduce? Write the list BEFORE running.
         (The one check that has not got cheaper.)

[ ] C75  What is the thing I am comparing against actually MADE OF?
         Is my "measurement" anchor itself a simulation?

[ ] C76  If I am inverting a published limit — does that measure the sky, or the PRIOR
         the analysis went in with?

[ ] C66  Argue AGAINST my own result for one paragraph. What is the strongest objection?

[ ] C77  Are these the right questions? If the work makes one of them the wrong question,
         say so.

[ ] C78  Can I state the CONDITION rather than force a verdict?
         "If X, then Y — and X has not been calibrated" beats a false unconditional.

[ ] C41  Any disagreement with a published value: report BOTH numbers. Never tune it away.
```

---

## §10 · Anti-pattern grep table — shapes that mean "broken"

| shape | why |
|---|---|
| `assert abs(x - 30) < 1e-6` on noisy measured data | only algebra returns exact values |
| `scale = sqrt(target/mean(s))` then assert `mean(s*scale) ≈ target` | constant fitted to pass the check |
| `y=f(x); assert f_inv(y)==x` | function against its own inverse; no data enters |
| `residual < 0.5` where 0.5 is my own rounding | bounded by construction |
| "every drawn candidate is within d" quoted as a continuum bound | sample ≠ population |
| same seed, different RNG constructor | the seed is not a name for the result |
| docstring says method A, body implements method B | answer can still land in range |
| `vmax=25` set until the picture looked right | tuning to appearance |
| `N/N passing` with no coverage statement | a check nobody wrote subtracts nothing |
| exit 0 + `"success"` + no artefact | the summary is not the result |
| a claim citing "the literature" with no eq/page | not a valid evidence reference |
| a number in prose with no registry key | it will drift, silently |

---

## §11 · Minimum set when time is short

`A1–A12` (§0) always. Then, in order: **C2, C15, C14, C18, C19+C20, C53–C55, C33/C34/C36, C60, C74.**
Everything above that is depth, not correctness.
