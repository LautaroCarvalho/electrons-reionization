# CHECKS_MANUAL — needs the user, a second run, or resources I do not have

**What this is.** Checks I **cannot** complete alone. Each one names what I must ask for, the
exact wording to ask it in, and what it buys — so I can raise it in one line instead of
improvising.
Companion: `CHECKS_AUTO.md` (self-executing).
**These two files are the record.** There is no separate catalogue behind them. Every
check ID C1–C79 lives in one of the two, with its operational content, and — since
2026-09-10 — with the documented failure it is traced to, in the appendix below.
The split originally dropped 34 of those 40 traces; they were recovered from the
catalogue before it was deleted again. **The migration is now complete: 40 of 40.**

**Six checks are split across both files, deliberately — they are not duplicates.**
`C23` `C27` `C48` `C56` `C62` `C71`. In each case the half I can run alone lives in
`CHECKS_AUTO.md` and the half that needs a resource, a second run, or your terminal lives
in `CHECKS_MANUAL.md`. Read both entries before deciding a check is done.
Coverage across the pair: **C1–C79, none missing.**

**The rule for using this file.** Do not block on these. Do everything in `CHECKS_AUTO.md`
first, deliver the work, and then **name which of these are outstanding and what each would
buy.** The only genuine blockers are in §0.

---

## §0 · MUST ASK BEFORE STARTING — the two real blockers

| ID | When | Ask |
|---|---|---|
| **MR1** | **Every new request** | *"Follow the Master Rules rigorously for this, or relax them?"* — unless the user has already scoped an answer forward (e.g. *"relaxed until I describe the problem, rigorous after"*). **If scoped, do not re-ask.** |
| **MR3** | Before adopting **any** physical or astrophysical hypothesis not given | *"This needs assumption X. Adopt it, or do you want a different one?"* Never assume silently. |

Everything else in this file: **proceed under a stated assumption, and flag it.**

---

## §1 · Requires a SEPARATE RUN (I cannot launch it; the user must, or must authorise it)

| ID | Check | What to ask for | What it buys |
|---|---|---|---|
| **C24** | **Blind second implementation** | *"Open a fresh session, give it the spec and NOT my code, and have it write the analysis again."* | The strongest evidence available. Five independent programs agreed to 0.00014 M☉. **Now costs one prompt — it used to be the check you saved for a decade-defence result.** |
| **C65** | **Fresh-session blind review** | *"Give a new session the OUTPUT ONLY — no transcript, no reasoning, no context — and ask what is wrong with it."* First write your own list, then compare. | Two blind reviewers agreed on the winner and on **2 of 5** places. *You only learn the criterion was doing the work when there are two of them.* |
| **C61** | **Verifier on a different model family** | *"Run the verifier on the other CLI/model family from the workers."* | Otherwise it is the same model marking its own homework. |
| **C67** | **Noise-floor measurement** | *"Run the identical task twice — same model, settings, words — so we know how far the answer moves when nothing moves."* | **Nobody in the course has measured this.** Without it, no later comparison is interpretable. Do it once, early. |
| **C72** | **Resume test** | *"Give the finished run three more rounds and see what it reopens."* | Every job given more rounds used them on something it had already decided to leave undone. **A stopped job can always be resumed, so no round is ever the last one.** |
| **C73** | **Clean-clone agent test** | *"Hand the repo to an agent that has never spoken to us and ask it to reproduce the headline result."* | This *is* the final project's acceptance criterion. |
| **C23†** | Mutation test at scale | authorise the rerun cost | cheap locally (in `CHECKS_AUTO`), listed here only when the suite is expensive |
| **C56** | **Determinism** | run the pipeline twice and diff the registry hashes — the script now prints `md5(results.json)` and writes `provenance/results_md5.txt` | A single run **cannot** establish this, and writing "regenerates byte-identically" in a paper after one run is an unbacked claim. It sat in `CHECKS_AUTO` mislabelled until 2026-09-07. |

---

## §2 · Requires RESOURCES or DATA I do not have

| ID | Check | What to request |
|---|---|---|
| **C51** | **Measure the tolerance, do not choose it** | A second BLAS/OpenBLAS build with every Python package version held identical. Report the **worst disagreement over every reference array**. (Reference result: 1.1e-14 against a 1e-12 tolerance — *one hundred times inside*.) |
| **C32** | **External anchor** | The published table / figure / paper the number can be checked against. *Prefer an anchor the pipeline could not have produced over any internal consistency.* |
| **C45** | **Overlay against the source figure** | The original figure, digitised or as a PDF, so the comparison can be a difference panel rather than two pictures side by side. |
| **C27** | Second literature route | The second paper, if not already in `papers/`. |
| **C09** | Citations | The actual PDFs. **Never cite a paper that is not in the tree.** Concept DOIs, never record numbers. |
| **C48** | Raster comparison | Confirmation of the matplotlib version to pin, so PDFs can be made byte-identical via `SOURCE_DATE_EPOCH`. |

---

## §3 · Requires the USER'S OWN TERMINAL

I cannot run these; give the user the exact line to paste.

| ID | Need | Line to hand over |
|---|---|---|
| **C70** | context occupancy | `/context` (Claude Code) · `/status` (Codex) — run at the start of a run and again when it finishes |
| **C71** | tokens, wall clock, cost | the session log / the JSON from `claude -p … --output-format json` |
| — | trust a hook (Codex) | `/hooks` — **Codex will not run a hook it has not trusted. A cell whose hook silently never fires is a different experiment, and you would not know.** |
| **C64** | steering-arrival | `job say "<direction>"`, then verify it was read — *three roles ran after one direction landed on disk and the cursor never moved* |
| **C62** | verdict ledger | the raw `transcript/` directory, if outside my reach |
| — | interactive login | `claude` → `/login` · `codex` → `/login` |

> Prefix any of these with `!` in the prompt to run it in-session so the output lands here.

---

## §4 · Requires the USER'S JUDGEMENT — decisions that are not mine

Raise these; do not resolve them.

| ID | The question | Why it is theirs |
|---|---|---|
| **C79** | **How much evidence is enough?** | *"Not enough to look thorough, enough that you would stand behind the result. That judgement is yours and there is no rule for it, which is the point."* |
| **C77** | **Are these the right questions?** | I can *raise* that a question looks wrong; replacing it is a research decision. Keep the escape hatch open in every brief. |
| — | **Which criterion ranks the work?** | Two blind reviewers inverted a ranking because one asked *is the analysis good* and the other *is this folder safe to hand over.* **Neither is wrong; the brief did not say which mattered.** Ask which. |
| — | **Acceptance contract** | **Bit-identical** (a machine settles it, no tolerance) or a **science gate** (conclusions must survive; arithmetic is open)? These buy different things — bit-identical made the day-3 program *wider*, not cheaper; the science gate made it genuinely cheaper. |
| — | **Permission mode before walking away** | *An agent that stops to ask is not running while you are in the corridor; an agent that never asks is one you must be able to let loose in that directory.* Pick on purpose, not at the first prompt. |
| — | **Round / cost budget** | The budget is an input the lead plans against, not just a cost. |
| **C41** | Which disagreement to pursue | I report both numbers and stop; which to chase is theirs. |

---

## §5 · Escalation rule — ask, or proceed with a stated assumption?

```
Would proceeding under ANY assumption be unsafe, or make the work useless if wrong?
   YES → ask, and do the independent parts meanwhile
   NO  → proceed, state the assumption at the point it is introduced, flag it, and
         list it under "outstanding" at the end
```

Never: block with nothing delivered while waiting on something that is not §0.

---

## §6 · Copy-paste prompt templates

### Blind second implementation (C24)
```
Here is a specification and the input data. Write the analysis from scratch.
Do NOT look for an existing implementation in this tree or online.
Report: the headline number, its uncertainty, and every choice that could
reasonably have gone another way, with your reason for each.
```

### Blind review (C65)
```
Here is the output of a piece of work — the files, and nothing else.
No transcript, no reasoning, no context about how it was produced.
Tell me what is wrong with it: bugs, hidden assumptions, anything that only
works by luck. What "wrong" means here is your call, and I am as interested
in the criteria you chose as in the findings.
```

### Noise floor (C67)
```
Run this identical task twice, in two directories, changing nothing —
same model, same settings, same words. Then put the two answers side by side.
```

### Verifier discipline (C61)
```
Lead and verifier on <tool A>; workers on <tool B>.
The verifier must work from an implementation IT writes, not the worker's.
File verified / refuted / unclear on every claim, and name the lines.
```

---

## §7 · Standing wording for the "outstanding" section of any deliverable

End every substantive answer with this block, filled in:

```
NOT CHECKED
  - <what has no assertion behind it>
CHECKED BUT NOT INDEPENDENTLY
  - <what only has an internal consistency check>
OUTSTANDING — needs you
  - C__  <one line>   → buys: <one line>
ASSUMPTIONS IN FORCE
  - <each, with where it was introduced>
```

*A summary line that reports only what passed is the failure this file exists to prevent.*

---

## Appendix · Why these checks exist — the failure each one is traced to

Recovered 2 traces from the deleted catalogue on 2026-09-10, after a consistency check
found that the operational content had migrated but the provenance of the checks had not.
**A check you cannot justify is a check you will skip.** These are the documented failures
that motivated each entry; the quotations are from the course material they came from.

| ID | The failure it exists to prevent |
|---|---|
| **C45** | Both caught failures were caught by visual comparison; automate the comparison. |
| **C70** | Named failure mode: *"drifting under a full context — fine early, degraded late, no announcement."* And the cost argument points the same way: *something wrong in context is re-sent and re-read until you clear it.* |
