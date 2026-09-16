# GW-AI-course — notes on the AI-applications content

Working notes from a full read of `GW-AI-course-main/`. Physics of gravitational waves is
deliberately omitted except where it is the *vehicle* for a method. Everything here is about
how to do research with agents, and about **provenance**, which is the spine of the final
project.

Sources are cited as `path:section` so every claim below can be re-checked.

---

## 0. The one-paragraph summary

The course teaches that with agents, **generation is cheap and review is the bottleneck**.
Every technique in it is an answer to the same question: *how do I know this result is right,
and how does somebody who did not watch me produce it know?* The answer the course converges
on is **provenance** — a machine-readable record, written as the work happens, mapping every
number, figure and claim to the code, data, library call and decision that produced it — plus
**checks** that can actually fail, and **enforcement** (a hook) that does not depend on the
agent remembering.

The course's own materials obey the rule they teach: the session pages fail to build if a
number on them does not resolve to a file (`how-a-page-is-built/build_show.py`, rule 1c).

---

## 1. Vocabulary — the harness model

From `using-the-tools.html`, which is the week's single reference page.

| term | definition |
|---|---|
| **Chatbot** | text in, text out. Cannot see files, run anything, or check itself. Whatever it says about your data is a *guess* about your data. |
| **Agent** | the same model **in a loop with tools**: read a file, write a file, run a command, look at the output, decide what next. *The loop is the entire difference.* |
| **Harness** | everything around the agent — permissions, project instructions, skills, hooks, subagents, jobs. **The part you configure, and where all the leverage is.** |
| **Turn** | one step of the loop: act → see result → decide. The cheapest measure of how much work a run did. |
| **Context** | one flat, finite block of text: instructions + files read + commands run + their output. |
| **Headless** | fire at a bounded deliverable and walk away (`claude -p`, `codex exec`). |

Two modes you will use both of: **interactive** (watch, steer, interrupt) and
**headless/jobs** (bounded deliverable, nobody watching).

### 1.1 Context economics — five consequences of "every turn re-sends everything"

The model holds nothing between turns, so the whole conversation is re-sent each turn.

1. **A long session pays for its own history on every turn.** Tokens grow with the *square*
   of session length, not linearly.
2. **A long output early is paid for on every subsequent turn.** → tell the agent to read a
   *range* of a file, not the file; trim commands that print thousands of lines. Not
   tidiness — you buy the difference once per turn for the rest of the session.
3. **Changing subject without `/clear` means paying for the subject you abandoned.**
   `/clear` costs you the setup again and nothing else.
4. **Caching makes the re-send much cheaper but not free, and not *smaller*.** A cached turn
   is a cheaper turn, not a shorter one; it fills context at the same rate. *Cheap and full
   are separate problems and caching only touches the first.*
5. **Something wrong in context is re-sent and re-read until you clear it.** A file you
   corrected on disk is still in front of the agent in its original form. This is the cost
   argument for bounding a task, and it points the same way as the quality argument.

**Measured shape of a real run** (`day1/session-ai.html`, `using-the-tools.html`; measured
Aug 2026, Opus 5 at xhigh):

| | day 2 run | day 3 run |
|---|---|---|
| turns | 132 | 242 |
| tokens in | 19,475,555 | 58,906,227 |
| tokens out | 169,476 | 235,493 |
| **in per token out** | **115** | **250** |
| **cached share of input** | **98.7 %** | **99.4 %** |

Read the last two rows: a small amount written, an enormous amount re-read. *The two levers
that matter are how much you put in front of it and how often you start again.*
Token counts are quoted rather than prices deliberately — **a price is a fact about a price
list on the day; the ratios are the durable part.**

Inspection commands: Claude Code `/context`, `/compact`, `/clear`; Codex `/status`,
`/compact`, `/new`.

> The practical skill is **not writing clever prompts. It is deciding what deserves to be in
> context**: bounded tasks, fresh starts, and writing to disk rather than carrying things in
> the conversation.

### 1.2 Cold start

Close the terminal and everything the agent knew is gone. **There is exactly one fix: write
it down, in the project, in a form the agent will read next time.**

- **Project instructions**: `CLAUDE.md` (Claude Code) / `AGENTS.md` (Codex) at the root, read
  automatically at *every* start. Keep one real file plus a one-line pointer to avoid
  maintaining two.
- **A record of the work** — the *run receipt*: date, command, environment, actual output,
  checks passed. **An exercise without one records a plan, not a result.**
- Useful test: *if you were hit by a bus, could a stranger — or a fresh agent — pick this up
  from what is on disk?*

---

## 2. The two dials, and how to tell whether they moved anything

- **Model** — bigger is more expensive per token and usually better at holding a long argument
  together. **Not uniformly better.**
- **Reasoning effort** — how much thinking before acting, independent of model. *For a
  derivation this matters more than the model.*

Always record which model and which effort a run used. Flags rot every few months
(`claude --help`, `codex --help`); the fact that the two dials exist does not.

### 2.1 Day 2's grid — one prompt, eight cells

Design (`day2/prompts.txt`): **one prompt string, frozen before the first cell ran, byte for
byte identical in all eight cells.** Rewording for one cell would have ended the comparison.
Each cell ran headless in a fresh directory outside the course repo (so it could not find the
instructor's own answer), one at a time (wall time is a column).

Four deliberate design choices in the prompt, each a column in the scoreboard:

1. The **waveform generator is off the shelf and the filter is not** — every cell has the same
   easy half and the same hard half. That boundary is what makes the task discriminating.
2. It asks for the histogram of ρ² and its mean **and does not say the mean should be 2**.
   *Whether a run knows the expected value is one of the things being measured; putting the
   answer in the prompt would have destroyed it.*
3. It asks for a **self-contained HTML page and a `results.json`** — format is one of the
   varied things.
4. The last clause asks for **"the choices you made that could reasonably have gone another
   way."** This is the course's first rule turned into a prompt.

### 2.2 The four failure modes the grid exposed

These are the most transferable findings in the course.

**(a) Turning the dial down degrades *checking*, not only answers.**
Opus 5 at low effort: measured mean ρ² = 2.257 (13 % high → ρ 6 % high) and *explained it* as
a heavy-tailed detector-noise effect. The explanation is available and wrong; the same model
at xhigh returned 2.012 from the same data. It reported peak SNR 20.87 where others reported
19.7/19.9. **Its own check said so, on its own page, in a paragraph explaining why it did not
matter.** Separately, gpt-5.6-sol at low effort returned the *identical* number to four
figures and simply stopped verifying — never searched L1 at all. *The second failure is more
dangerous because nothing about the output says so.*

**(b) A check turned into a fit cannot fail.**
gpt-5.4 derived the normalisation correctly, then multiplied by
`np.sqrt(2.0 / np.mean(raw_means))` — a constant *defined as whatever makes the mean of ρ²
come out at 2*. It reported 1.959 on held-out windows and said plainly what it did. The answer
was right (SNR 19.53). But **the one check the prompt insisted on had been turned into a fit**;
held-out validation measures whether two stretches of noise look alike, not the normalisation,
because the normalisation was chosen to make it come out. → *the scoreboard needs a column for
how a number was arrived at, not only for its value.*

**(c) Without the expected value, a check becomes a ruler.**
Haiku 4.5 (21 ¢, 3 min, 17 turns) reported peak SNR **5,236**, chirp mass 36.3, at a GPS time
1000 s before the event at the edge where its filter wraps. Its own ρ² check came out at
**11,448** (ρ inflated ×76). It is the only cell that never names 2. *Without the expected
value the histogram stops being a check: "the signal is much bigger than the noise" comes out
the same whatever the normalisation is.* Its page also loaded a charting library from a CDN —
the only cell whose page reached outside itself; offline it opened to empty headings.
**Ask for a self-contained file and check that it is one.**

**(d) A sample count is not an uncertainty, but it is what a reader needs to compute one.**
1.884 from 960 independent samples: χ²₂ has variance 4, so σ_mean = 0.065 and 1.884 is 1.8σ
from 2 — consistent. You can only know that *because the count is there*. Contrast 2.012 from
7.9 M *correlated* samples, where the count does not convert into an uncertainty — that run
answered it a different way, by pushing synthetic Gaussian noise (where the answer is 2 by
construction) through the same filter and getting 2.007. **Same statistic, two ways of showing
it means something; neither page is wrong, one of them lets you check.**

**(e) Headless removes the thing you were relying on without noticing: you.**
Sonnet 5 wrote the whole pipeline, ran the search, then ended its turn with *"I'll now wait
quietly for the background notification rather than continue polling."* In a conversation that
is sensible; **headless there is no next turn.** It left no page and no `results.json`, and
left **exit code 0, `"subtype": "success"`, after 109 turns and 31 minutes.**

> **If you walk away from a run, the thing you check when you come back is the artefact. Not
> the exit code, and not the summary it wrote about itself.**

### 2.3 The three checks (day 2's card)

A generalisable pattern — checks that any pipeline of this kind must pass:

1. Is your whitened data's variance 1 per sample?
2. Run your filter on signal-free noise. Is the mean of ρ² equal to 2?
3. Inject a signal of known amplitude. Do you get the SNR back?

*A group that can answer all three has a working pipeline whether or not it found the event. A
group reporting SNR 20 that can answer none of them has found nothing.* Note the structure:
**a normalisation check, a null check, and an injection/recovery check.** That triple
generalises to almost any measurement pipeline.

---

## 3. Day 1 — what an agent is, and the seven failure modes

### 3.1 The catalogue of failure modes (`day1/session-ai.html`)

| failure mode | what it looks like | what defends against it |
|---|---|---|
| **Recall dressed as derivation** | the right final answer, with a derivation that does not produce it | ask which steps were *derived* and which *recalled* |
| **Confident interpolation** | a plausible coefficient where the real one is unusual | check against a printed number |
| **Silent unit errors** | everything consistent, one constant in the wrong system | dimensional analysis; a second implementation |
| **Claiming without running** | "the fit converges" — no log, no output | ask for the receipt, always |
| **Drifting under a full context** | fine early, degraded late, no announcement | bound the task; start fresh |
| **Agreeing with you** | you suggest something wrong and it goes along | ask it to argue against you |
| **Uncovered by construction** | a green summary line over work no assertion touches | make it state what it did **not** check |

> **None of these are fixed by a better model. They are fixed by checks.**

### 3.2 The central cautionary result: "23/23 reproduced" over untested work

Two papers were reproduced. **The algebra came back right; every figure made from data came
back wrong**, each differently, and nothing in either repository noticed.

- **Failure 1** — followed the paper's stated band-pass literally, but the pass band is not
  flat (noise varies ×few-hundred inside it), so the low edge dominates. The paper never
  mentions whitening; it clearly whitened. *Plus* a sign error: the 6.9 ms detector shift was
  **added** instead of subtracted, putting the chirps 14 ms apart. Cross-correlating the
  whitened streams recovers −7.3 ms on its own — **the data will tell you the answer if you
  ask it.** Caught by a human eye. *No assertion in the repository would ever have fired.*
- **Failure 2** — plotted a Q-transform's **normalised energy** under a colour bar labelled
  **normalised amplitude** (the square root). The ceiling was then hand-set (`vmax=25`) until
  the picture looked right rather than matched to the paper. *Wrong in a way no test could see;
  only comparison against the original catches it.*
- **Failure 3 (the best one)** — the docstring says the frequency track is measured "from
  zero-crossings" as the paper prescribes; **the code below it takes the peak of each column of
  a Q-transform** — a different measurement of a different thing. The fitted chirp mass landed
  *inside* the paper's stated range. **A right answer obtained the wrong way is the hardest
  failure to catch, because the only thing you normally check is the answer.** Caught by
  nothing.
- **Failure 4 — the meta-failure.** The run log says **23/23 reported results reproduced**, and
  it is true. **Not one of those assertions touches any of the three figures.** The named
  "(Fig 3)" check is:

  ```python
  Mc = 30.0 * MSUN
  tt = np.linspace(-0.20, -0.005, 200)
  ff = B.track_f_of_t(tt, Mc, t_c=0.0)
  Mfit = B.fit_chirp_mass_from_track(tt, ff) / MSUN
  check("(Fig 3) f^-8/3 fit recovers M = 30 Msun", abs(Mfit - 30) < 1e-6, ...)
  ```

  **A function put against its own inverse.** No strain enters it; there is no arrangement of
  the data under which it would come out differently.

  **Two tells were on the page.** (i) The tolerance: `< 1e-6`. *Nothing measured off real noisy
  data comes back at 30.000000; only algebra does. A tolerance that tight is a statement about
  what is being compared.* (ii) Twelve lines below the green banner, the same run printed what
  the real Figure 3 got off the strain: **36 M☉**. *The number the run actually got and the
  number the check verified are different numbers, in the same file, twelve lines apart — and
  it is the check that passed.*

> **The count is over the checks that exist, and a check nobody wrote subtracts nothing.**
> Hence the standing rule: **make it state what it did not check.** "12/12 passing" is
> misleading if half the work has no assertions — *name the uncovered half.*

### 3.3 The check worth copying

The peak luminosity of GW150914 was computed to equal −dE/dt from Peters' circular orbital
decay, derived from the *other* paper, agreeing to **zero relative error**. Two papers fifty
years apart, the same formula from opposite ends.
**An independent route agreeing is worth more than any number of internal assertions.**

Also: reproducing Peters §V found that his own text rounds a separation ("about ten solar
radii" where Kepler gives 14 R☉; ten would be a 2.6 d period, not 4.5 d). Unimportant, and
completely characteristic — *the difference between a reproduction and a paraphrase.*

### 3.4 The working agreement (`day1/vault/AGENTS.md`) — a template to steal

The most valuable file in your working directory, and the one most worth changing.
Its rules, verbatim in substance:

- **Prefer a check to a claim.** Derive something → verify with `sympy` and assert the
  difference is zero. Compute a number a paper printed → assert you reproduced it.
  **A result with no assertion is a hypothesis; say so in those words.**
- **Say which parts you derived and which you recalled. Always, unprompted.** *They look
  identical on the page and only one of them survives a new problem.*
- **Never claim to have run something you did not run.** "The script produces X" and "I ran the
  script and it printed X" are different sentences. **Paste the output.**
- **Reproducing a figure means reproducing the figure** — same axes, same curve, from the same
  kind of input. Not describing it, not drawing something with the right shape. If you could
  not get the data, say the figure was not reproduced.
- **State what you did *not* check.**
- Formats: write-ups in **LaTeX → PDF** (not markdown); figures as a **marimo notebook (`.py`)
  exported to HTML** (not `.ipynb`); anything interactive as a **single self-contained HTML
  file**; markdown for the vault only.
- Do not write long prose summaries — *the one output that cannot be checked at a glance.*
- Do not install into the base environment; make a per-exercise one.
- Do not create files outside this directory. Do not `git commit`/`push` unless asked.

### 3.5 The vault (`day1/vault/README.md`)

Small interlinked markdown pages: one per paper, one per exercise, one per concept. Written
mostly *by* the agent, mostly *for* the agent.

- **There is nothing automatic about it.** No tool writes it. It happens because a file in the
  working directory tells the agent to, and that file is read at every session start. *If the
  agent does not write the page, that is a bug in `AGENTS.md`, not in the agent — fix the file.*
- **The one rule that makes it useful: a catalogue over artefacts, not a container.** The
  substance lives in `.py`, `.tex`, `.html`. The vault says what exists, where it came from,
  whether it was verified, and what it depends on. *If your pages start containing the
  derivation instead of pointing at it, you have built a worse version of a notebook.*
- Every exercise page carries a **run receipt** — date, command, environment, actual output.
- Link generously with `[[wiki-links]]`; a link to a page that does not exist yet is fine — it
  marks something worth writing later.

Templates ship at `day1/vault/templates/{paper,exercise,concept}.md`. The **paper** template's
core is a table with columns `§ | Claim | Their evidence | Ours` — where "Ours" is either a
link to an exercise page or the literal string **"not reproduced"**. The **exercise** template
has a required section: *"What was checked — and what was not."*

---

## 4. Day 3 — scaling out: written objectives, permissions, cost

### 4.1 A written objective (`day3/objective.txt`)

Handed to the agent as a *file*, invoked with `/goal read objective.txt and follow its
instructions. Do not stop until you are done.`

Two structural properties worth copying:

- **It asks for the report FIRST and the tool LAST**, "so that the hour always produces
  something."
- **It defines what a good report is**: *"A report that says the download failed and shows the
  traceback is a good report. One that implies things ran when they did not is the failure."*

Other reusable constraints from it: work only inside this directory; build your own conda
environment, do not modify one that exists; both HTML files must open from `file://` with no
network and no server; offer the expensive option **off by default, with what it will cost
stated next to the switch**; at the end, say the wall clock.

**Evidence that `/goal` + a file beats an inline prompt:** of the three runs handed the
objective via `/goal`, all three left all four deliverables; of the two given it inline, one
did. (Five runs is not a measurement, but it is why the invocation is written that way.)

**Permissions must be decided before you walk away.** *An agent that stops to ask is an agent
that is not running while you are in the corridor; an agent that never asks is one you have to
be able to let loose in that directory.* Pick on purpose, not at the first prompt.

### 4.2 Five independent programs, one event

Five agents, no communication, same prompt. Five different banks, five different winning
templates, and **chirp masses agreeing to 0.00014 M☉.**

> Writing the analysis a second independent way used to be the check you saved for a result you
> were going to defend for a decade; **here it is one more run of the same prompt, and the
> fifth one costs what the first one cost.**

This is arguably the single biggest *new capability* the course identifies.

### 4.3 Blind review by two independent reviewers

Five lettered folders, no run ids, no models, no token counts. Two fresh sessions, same brief,
could not see each other. *What "best" means was left to them on purpose — the criteria they
chose are as much the result as the ranking.*

- The winner was **unanimous**; they agreed on **2 of 5** places; the largest disagreement (A)
  was ranked 2nd by one and 5th by the other.
- **Same evidence, opposite sign.** Both found that A writes its results then dies importing a
  missing module. One ranked it 2nd (*"a packaging failure, not a program that was never
  run"*); the other ranked it last (*"writes new `results.json` and `reference.npz`, then fails
  instead of regenerating the report. The existing HTML can silently become stale."*) **Neither
  is wrong — they are answering different questions** (*is the analysis good* vs *is the folder
  safe to hand to somebody else*) **and the brief did not say which mattered, on purpose. You
  only find out that the criterion was doing the work when there are two of them.**

**Three findings from the review that generalise:**

1. **A guarantee about your sample is not a guarantee about the population.** 4 of 5 claimed a
   tighter template-bank loss bound than they had. The same slip every time: greedy placement
   guarantees every one of the 8192 *drawn candidates* is within *d* of a kept template — *that
   is not a statement about a signal from the continuum, which can sit further out.* Claimed
   1.0 %, measured 2.20 %; claimed 4 %, measured 6.07 %.
2. **A seed is not a name for the result.** `qmc.Halton(seed=…)` and `Halton(rng=…)` are both
   accepted and scramble differently. **Two programs printed the same seed and searched
   different banks.**
3. **An assumption wearing a measurement's clothes.** All five obtained an absolute GPS time by
   assuming the segment starts on an integer second, then quoted the residual against the
   published time as *agreement*. *"The residual is bounded by ±0.5 s by construction, because
   it is the rounding error of the integer-second assumption; it cannot be large, so it is not
   evidence."* The one that handled it best declined to play and said the time was **anchored,
   not recovered.**

**And the honest post-mortem of the review itself**: the folder-assembly script copied only
four files per program, dropping three files belonging to the two programs that were then
marked down for not regenerating their own reports. *"Their agreement says they were given the
same exhibit; it does not say the exhibit was right."* Also, the winner **had measured** the
right quantity and **did not read its own number** — the sentence it was praised for states the
guarantee, then states the measured worst case, and carries on. **The measurement was made and
the conclusion was not drawn.**

### 4.4 Two acceptance contracts, and what each buys

**Link 3 — bit-identical.** *"`reference.npz` must come back bit-for-bit identical: the same
three arrays, and the same sha256 digests. Faster and different is a failure, not a trade-off."*
A machine settles it; there is no tolerance in it. Plus: *"Everything must still be computed
from `strain.npy` on every run: caching a result to disk and reading it back is not a speed-up,
it is a different program."*

Result: 5.9× and 2.7× wall-clock speed-ups — but **read the user-CPU column: neither made the
computation cheaper. Both made it wider**, spreading arithmetic over idle cores (user CPU went
*up*). *A real result worth having, but not the same result as making the program better, and
only the second column would have told you which one you got.*

Also: **the contract holds only for the machine and library versions it was measured on. The
environment is part of the claim, the same way the machine is part of a wall clock.**

**Link 4 — the science gate.** The arithmetic is open; the *conclusions* must not change. Two
things are not open because they are the science rather than the implementation: **the search
must stay blind** (may not use knowledge of where the event is, must search the same parameter
space) and **everything must still be computed from the raw data on every run.**

The key paragraph — the most quotable thing in the course:

> **"`check_detection.py` … tests five things … That is the MINIMUM. Passing it is not the same
> as the science being unchanged, and I have deliberately not tried to write down every way it
> could differ. If you make a change whose effect on the result is not covered by those five
> checks, working out what else has to be true is your job. Write those checks, ship them as
> `check_extra.py`, and say what each one is for, what it would have caught, and what it did
> catch. If you conclude that nothing further is needed, say so and give the argument — that is
> an answer too, and I will read the reasoning rather than the verdict."**

Both runs answered the sentence rather than the number: they wrote 14 and 8 extra checks and
**reinstated constraints the gate had deliberately dropped** — that the bank is the same bank,
that the winning template is the same in *every* interval and not only at the event, and that
the background's whole shape is unchanged *since the false-alarm rate is a property of all of
it and no per-interval tolerance implies it.*

And the cost of the approximation stated in **physical terms rather than in ulps**:
*"SNR moves 7.9×10⁻⁸ (≈3 pc on a 40 Mpc distance whose measurement error is ±8 Mpc); 5 samples
in 9.55×10⁹ are vetoed differently."*

> **Deciding in advance what a wrong answer could still reproduce is the one check on the list
> that has not got cheaper. Everything else got cheaper.**

### 4.5 On comparing the two CLI tools

One was faster on every link and leaner in output tokens by more than it was faster — *and it
also does a bit less. Rarely on what it was explicitly asked for; usually on the extras, taking
whichever reading of the task is the shortest one.* But on the thing it *was* asked to do, its
program was the leanest of the lot. **Neither is a verdict on a brand. They are two shapes of
answer, and knowing which one you are being handed is the whole reason to run both.**

---

## 5. Day 4 — teams, roles, and an independent verifier

### 5.1 Structure

- **Role** = one prompt file describing one personality (`pi` = lead, `worker`, `verifier`).
- **Recipe** = roles cast into a team with an order of work and one deliverable.
  **A recipe that does not name a verifier is rejected.**
- **Job** = one concrete run of a recipe, in its own directory.
- **Round** = the lead reads state and writes a plan → workers execute → **the verifier goes at
  whatever they claimed and files `verified` / `refuted` / `unclear` on each one.**

Everything lands in `jobs/<id>/` as files: `spec.json` (what it was given), `state.json` (what
it ended up believing), `log.jsonl` (every call), `transcript/` (every role's whole answer every
round), `reports/`, `out/` (deliverables), `view.html` (its own monitor).
**That directory is the whole of what happened, and you are meant to read it.**

### 5.2 The staffing rule worth copying

> **The lead and the verifier run on one CLI/model family and the workers on the other, so the
> thing that checks a claim is a different model family from the thing that made it.**
> *Put the verifier on a different model family and the check stops being the same model
> marking its own homework.* (Jobs that took the default and ran everything on one tool cost
> the most per token.)

### 5.3 The prompt structure (`day4/prompts.txt`) — a template

Sections, in order:

1. **GOAL** — the question. For the open version: *"Decide for yourself what has to be
   computed."*
2. **SOURCES** — an explicit list of everything the job may read.
3. **FENCE — read nothing else** — named directories containing worked answers.
   *"Reading them voids the run. If you find yourself in either tree, stop and say so."*
4. **WHAT YOU CANNOT GET FROM THE SOURCES** — names the genuine judgement calls, so they are
   made consciously. Includes: *"Do not assume the framing is correct because we wrote it."*
5. **DISCIPLINE** — the environment; *"Derive it symbolically where you can, and say which steps
   are derived and which are recalled"*; *"Every number you report must come from code in this
   job, or be quoted from a named paper with its equation or page. Say which, every time"*;
   *"When two sources disagree, report the disagreement with both numbers. A few per cent is
   usually a modelling choice; an order of magnitude is usually a bug. Stop on the second
   kind"*; **"Say what you did NOT check."**
6. **DEFINITION OF DONE** — `out/notes.tex` where **every quantitative claim carries a
   `\src{key}` resolving in `out/provenance.json`**; `out/checks.py` that *"prints per-check
   coverage numbers, not verdicts"*; and *"paste the check output in the final report."*
   Also required: *"at least one case where the formula and the numerical method are made to
   **DISAGREE on purpose**, so the comparison is shown to be capable of failing."*
7. **DELIVERABLE** — `report.html` (what a physicist reads) **and** `agent.html` (the same
   content *plus* the provenance: every number resolving to the code that computed it, every
   figure to the script and inputs, the seeds and command lines, and the source of every script
   on the page). *"`out/provenance.json` is the registry both resolve against. Provenance is not
   optional in either format."*

Also: *"If you use the symbol C_l, say every time which of the two you mean"* — a notation
contract stated in the prompt because the wiki had settled it first.

### 5.4 What multi-agent runs actually did — the honest catalogue

**It works for the derivable part.** All three prompts had the exact closed-form result in
round one, *each from its own derivation, and each confirmed by a verifier working from an
implementation it wrote rather than the worker's.*

**It produced one genuinely new result the humans did not have** — a fourth-root rather than a
square-root relation. **It was not in the prompt, not in the papers, and not in the lead's own
plan** (the plan had written the square root); *the verifier corrected it* and it survived into
the deliverable.

**Six documented pathologies:**

1. **A refutation that never reached the round that had to act on it.** In round 2 the verifier
   refuted the run's own headline, naming the lines. In round 3 the run shipped that sentence
   anyway, in both deliverables. Cause: *"the engine pasted a filtered slice of the ledger into
   every prompt, and the filter kept **verified** claims, so refuted and unclear claims were
   stored and never shown to anyone."* → **Read `reports/` and `transcript/`, not the
   deliverable.**
2. **The fix reintroduced the bug one layer up.** The fix was to stop pasting and start
   pointing — but the pointer counts from a **capped summary**. On a six-round job the verifier
   filed 5 refuted / 30 unclear; `state.json` ended holding **0 refuted / 20 unclear**, and the
   index that tells the lead "n claims are unresolved" counts from the truncated list.
   → **Count the verdicts in `transcript/` yourself and compare with `state.json`.**
   (Disagreed on 2 of 6 jobs, always in the same direction.)
3. **Two words carrying four situations.** `refuted` and `unclear` is all a verifier can file,
   so *"a factor-of-three contradiction in a shipped number sits in the same bin as a missing
   browser"* — one needs adjudicating, the other needs everybody to stop trying.
4. **They are blind to their own coverage.** At round six the lead found *"two figures in
   [a supplied paper] that no role has opened in six rounds"* — and they were the external
   anchor for the very quantity the team had argued about since round two. Opening one settled
   it **against** the team.
5. **They price their own wasted effort late.** *"No browser backend exists in this environment;
   three roles have now tried. Record as a known unverified property and stop re-attempting."*
6. **Under a tight round budget, format is what goes.** A run's own verifier found both HTML
   deliverables emitted LaTeX as raw source with no renderer. The next lead ruled *"Structurally
   the deliverable is complete; bookkeeping is not what's blocking. Both tasks this round are
   physics."* **The defect shipped.** A human then steered the running job about exactly this
   and **the direction was never read** — three roles ran after it landed on disk.
   → **Send a direction with `job say` while it runs, then check that it arrived.** (It is read
   at the start of a round and not before.)

**On round budgets:** *the budget is an input the lead plans against, not just a cost.* A lead
with two rounds left reorders around the deadline rather than the physics, then closes enquiry
outright (*"No new science beyond what is listed; this is the last round"*) — and the wind-down
is visible in the timings before it is visible in the text (8.5 min against 37 for round one).
**But a stopped job can always be resumed, so no round is ever the last one.** The fix, now in
the lead's role file: *"produce the best deliverable the remaining rounds allow AND report what
further rounds would buy, since that is the input to the human's decision to resume"* — every
round's report ends with a section headed **WHAT MORE COULD BE DONE**.

**The result nobody reached, and why it matters most.** There was an answer key (a finished
paper with 106 stamped numbers). Three of four results were reached. The fourth — that the
published limit those jobs kept inverting *is close to a reproduction of the prior the analysis
went in with, so inverting it measures the prior, not the sky* — was **not** reached by either
prompt, six rounds, or 127.7 M tokens. **They came within one step of it twice and both times it
was filed and dropped**: a round-one verifier wrote *"Q3's inversion rests on this and must not
be presented as settled until it is checked"* → **logged `unclear` and never checked. The
inversion shipped.** And: the number the team spent three extra rounds chasing a discrepancy
against **is itself a simulation**, and the job says so in its own description of how it was
made. *"Nothing in the machinery asks what a thing you are measured against is made of. The
question does, and neither prompt asked it."*

**Well-posedness is the variable.** The loose prompt got one new result and then spent its last
round on packaging. The same subject written as **five numbered questions in the order they have
to be answered** produced the shape of a small paper. The fifth question is the one that is hard
to write and easy to leave out: *"Make everything you have found in 1–4 compatible with each
other and with the published results, and say plainly where it leaves the question."* **Without
it a team hands back four answers and no position.** And the escape hatch was kept open — *"if
answering them turns up something that makes one of them the wrong question, say so"* — and was
used: the closing section says the five questions are *"not quite the right five"* and says
which two it would replace. *"That is worth more than the factor of six, and it arrived in the
last round."*

**Best answers state their condition:** *"if the number of independent positions is bounded by
[X], then [Y] wins … The paper does not calibrate this effective count, so this job cannot
declare an unconditional winner."* **The answer is *it depends*, and here is the number nobody
has measured.**

---

## 6. PROVENANCE — the core of the final project

> **The PROVENANCE of a result is the record of where it came from — which file and which
> function produced it, what went in, what was written from scratch against what was called out
> of a library, and every choice that had a defensible alternative.**
> — `final-project.html`

The course implements this at three scales. **All three are the same object.**

### 6.1 The idea

> *"A standard part of reading a paper is working out what the claims are and what the evidence
> is. In this repository that is not left to the reader: it is written down, once, in a form
> both a person and an agent can read, and a script checks that the paper, the code and the data
> still agree with it."* — day 5

> Matias, 2026-08-18: *"A standard part of reading a paper is trying to figure out what the
> claims are and what the evidence is. Why have to guess? We can say it explicitly and the
> agents can understand it."*

**That is the "available now and was not before" claim of the whole week.**

### 6.2 Scale 1 — the laptop scaffold (`day5/exercise/`) — COPY THIS

The shape, stated in the hook's own docstring:

```
the convention lives in .md files, once
the skill points at them, so a fresh session follows it from the start
the hook checks the result, and on failure serves the .md that applies
```

*Nothing restates the format. Add a check, write its `.md`, and both delivery routes get it.*
**"A convention written twice drifts."**

Two artefacts the agent writes, in `provenance/`:

**`provenance/numbers.json`** — one entry per number reported, keyed by a short slug.
**All six fields required:**

```json
{
  "<slug>": {
    "value": <the number>,
    "statement": "<what this number is, in one line>",
    "produced_by": "<file>::<function that computed it>",
    "from_scratch": "<what you derived or implemented yourself>",
    "from_library": "<what you called, and from where>",
    "choices": ["<each decision that had a defensible alternative, and why you made it that way>"]
  }
}
```

*`choices` may be an empty list only if there was genuinely nothing to decide — write `[]`
explicitly rather than omitting it. **If a number appears in your reply and not here, the record
is incomplete.***

**`provenance/claims.yaml`** — what you are actually asserting, and what backs it:

```yaml
claims:
  - id: <slug>
    statement: >
      <the assertion, in one or two sentences, as you would say it out loud>
    evidence:
      - <file>::<function that establishes it>
      - <a citation, with the equation, section or figure number>
    numbers: [<slugs that appear in numbers.json>]

figures:
  - file: <the file as written on disk>
    produced_by: <file>::<function that draws it>
    shows: >
      <what is on the axes, and what a reader should take from it>
    from_scratch: <what you computed yourself>
    from_library: <what you called>
    choices: ["<each decision that had a defensible alternative>"]
    supports: [<claim ids this figure is evidence for>]
```

*Evidence is either code that ships here, or a citation precise enough to look up. Every slug
under `numbers:` must exist in `numbers.json`.*

> **"A claim is not 'what I did'. It is what you are asking the reader to believe."**
> **"A figure with no entry is a picture nobody can regenerate."**
> **"It is not a summary of what you did, and it is not a log. It is the map from each thing you
> assert to the thing that establishes it."**

**The enforcement** — `.claude/hooks/provenance_gate.py`, wired at two events by
`.claude/settings.json` (Claude Code) and `.codex/hooks.json` (Codex), **one script, two
wirings**:

- `SessionStart` (`--start`) — **stamp the time.** *Anything already in the tree at that moment
  was handed to the agent, not produced by it* — the source PDF must not be mistaken for a
  figure the agent drew. (Falls back to the scaffold's own `settings.json` mtime.)
- `Stop` — the check. **A non-zero exit puts the message on stderr into the conversation as a
  new turn, so the agent is not finished after all.**

What it checks: `numbers.json` parses, is non-empty, and every entry has all six fields;
`claims.yaml` parses, asserts something, every claim has a statement and evidence, and every
`numbers:` slug resolves; **every figure file newer than the session stamp has an entry.**

Two design details worth stealing:

- **The failure decides which instructions the agent is handed.** The gate maps each failing
  check to the `.md` that explains what to write, and prints *those files, whole, and nothing
  else.*
- **Loop guard.** `MAX_ATTEMPTS = 3`, tracked in a `.attempts` file. After three, it prints
  what is still missing and **exits 0 rather than looping.**

**The measured result (`a` = nothing, `b` = skill + hook, `c` = hook only):**

| cell | turns | gate fired | the record |
|---|---|---|---|
| a · nothing | 28 | — | **none** |
| b · skill and hook | 34 | **0 times** | 3 claims, 1 figure, 7 choices, 1 number |
| c · hook only | 22 | **1 time** | 2 claims, 1 figure, 7 choices, 2 numbers |

**All three agree on the physics to every digit written.** In `a/` the number is in a chat
message and dies with the terminal; in `b/` and `c/` it is in a file with the function that
computed it, the library call it rests on, and the choices that had an alternative, each with
its reason. Cost: **US$4.22 for all three**, 4–6 minutes each. *The price of a record is a few
tens of cents.*

- `a` vs `b`/`c` is **the real gap.** With nothing written down, the agent computes the right
  number, checks it a second way, says so — and the number lives in a chat message.
- `b` vs `c` is **not a quality difference. It is *when* the instruction arrives** — up front,
  or at the door — and both land.
- **The skill is the normal route; the hook is the one that still works when the skill does not
  get loaded.** *In a long run the agent may never load the skill. The hook runs whether or not
  it did, and that is what it is for.*

**Skill vs hook, stated generally** (`homework.html`): *a skill is loaded when the agent judges
the task matches — which means it can also go unloaded, and nothing announces that it did. A
hook does not depend on that judgement.* **They fail differently: a skill can go unloaded; a
hook cannot make an agent do something it does not already know how to do.**

Note also: **Codex will not run a hook it has not trusted in `/hooks`.** *A cell whose hook
silently never fires is a different experiment, and you would not know.*

### 6.3 Scale 2 — the published paper-as-repository

Five files, and this is the model for the final project's repository half.

| file | what it is |
|---|---|
| `structure/claims.yaml` | **the argument skeleton** — a `thesis:`, then one record per load-bearing claim: `statement`, the section and file it lives in, the `numbers` it quotes, its `evidence`, what it `depends_on`, and a `status`. (11 claims, 9 naming registry keys.) |
| `data/paper_numbers.json` | **the number registry** — 106 keys, **sole writer** `scripts/compute_paper_numbers.py` |
| `paper/paperclaims.sty` | **the provenance layer inside the prose** — 3 macros (`\dataref`, `\genby`, `\provnote`), **2 options (`draft`, `final`)** |
| `scripts/check_provenance.py` | **the gate** — 10 checks, ~1 second; the paper is not built until it prints `all checks pass` |
| `README.md` | **the reproducibility section** — 5 named categories, and what is *not* claimed |

**The rule the skeleton is held to:**
> *"Every `data_ref` must point to code that ships in this release and reproduces the result, or
> to a bib key in `paper/references.bib`. Nothing else is a valid data_ref."*

**Why the skeleton is worth writing:** *a reader normally has to reconstruct by guessing which
section carries a claim, which numbers it rests on, which script made each, and which other
claims have to be true first. Written this way, an agent handed the repository can answer "what
is the evidence for the third result" **by reading rather than by inferring** — and so can you.*

**The `draft`/`final` switch is the trick that makes it survivable:**
- `[draft]` renders every annotation inline, in colour, beside the number it belongs to —
  *"which is how the manuscript was reviewed."*
- `[final]` makes all of it vanish.
- *"The running text never names a script or a registry key outside these macros: in [final]
  mode the paper reads clean, and the machine-readable form of the same information is in
  `structure/` and `data/paper_numbers.json`."*

**→ The reviewed manuscript and the shipped PDF are the same file.** To trace a number: look the
key up in `paper_numbers.json` and search for it in the one script that writes that file.

**The ten checks** (none of them a check on the physics; all of them a check that the paper still
says what the code and the data say):

1. every `data_ref` in the skeleton resolves to a file or a bib key
2. every number key the skeleton names is in the registry
3. every figure has a source script and a PDF
4. every `\genby` names a file that exists
5. every path-like token in every shipped text file resolves
6. every `\dataref` key exists in the registry
7. **every displayed `\dataref` literal agrees with its registry value at the literal's
   displayed precision** — *not that the key exists (that is check 6), but that the digits
   printed on the page are the digits in the registry, to the precision the paper chose to
   print*
8. the citation file parses
9. every deposited array names a producer that ships
10. (+ layout checks)

**And a detail worth copying:** three of the ten only make sense in the public layout; against
another tree **each reports itself as *skipped* rather than passing silently.** It prints
`all checks pass (10 of 10 run)` and exits 0.

### 6.4 The reproducibility section — how to claim reproducibility honestly

The README **does not say the release is reproducible.** It says, in five named categories, what
was measured and how — *"That is not the same as byte equality across machines."*

1. **Bitwise, demonstrated.** `paper_numbers.json` regenerates byte-identically, and so do both
   table sidecars.
2. **Bitwise, with one caveat.** The eleven figure PDFs regenerate **raster-identically — zero
   differing pixels at 200 dpi — but not byte-identically, because matplotlib stamps wall-clock
   time into `/CreationDate` and its own version into `/Producer`.** *Fix: set
   `SOURCE_DATE_EPOCH` to a fixed value and, at a fixed matplotlib version, the PDFs become
   byte-identical too.*
3. **Within a stated tolerance, and measured.** Tests compare at `1e-12` of each array's own
   scale, **and that number was measured rather than chosen**: under a second OpenBLAS build
   (0.3.25/OpenMP vs 0.3.21) with every Python package version held identical, the worst
   disagreement over every reference array was **1.1e-14** of the array scale — *one hundred
   times inside the tolerance.*
4. **Within tolerance, not bitwise.** Monte Carlo products: *"distribution percentiles from
   independently drawn realizations agree only to Monte Carlo error"*, and *"every number quoted
   in the paper is rounded well inside that."*
5. **Not claimed anywhere:** *"that the deposited arrays can be reproduced byte-for-byte on a
   different platform. **The deposited arrays are what the paper used; the pipeline is how they
   were made.**"*

> *Nobody wrote a section like that by hand before, because nobody could afford to: every
> sentence in it is a run in a fresh environment against the deposited arrays, with the
> disagreement measured.*

### 6.5 A second, independently-arrived-at release (the same discipline, different choices)

- **Figures identified by their LaTeX reference-label key** (`fig:opt_depth`), *"not by figure
  number — numbers shift as the paper is revised, but the label key is stable."* 12 labels, each
  mapped to the PDFs it produces and the command that regenerates them.
- **One locked environment**: `pixi.toml` + `pixi.lock` — *a lockfile rather than a list of
  package names, so that the environment is a file and not a description of one.*
- **A staged rerun pipeline** (7 stages, producer scripts and SLURM templates included), with
  **intermediate products bundled** so results can be regenerated without rerunning the
  expensive PE and Monte-Carlo stages.

> *Two releases, two groups, and both ended up writing down what a reader would otherwise
> reconstruct: which script made which figure, from which inputs, in which environment.*

### 6.6 Scale 3 — the working wiki, where the skeleton comes from

`structure/claims.yaml` is *what is left of the wiki once the paper is done*. The wiki
(`day5/exercise-notes/wiki/`, 105 pages, a year of use) has:

`index.md` (read first, drill in), `conventions.md`, `formulas.md`, `bibliography.md`,
`environment.md`, `reproducibility.md`, `todo.md`, `log.md`, plus `concepts/` (15),
`sources/` (36, one per thing read), `code/` (one per module/script), `notebooks/`,
`figures/`, `results/`, `notes/`.

**`conventions.md` — the notation contract.** *"The contract that prevents notation drift.
Every concept page, source page, and paper section is held to these. If a source uses a
different convention, note the translation inline on that source page; never change the
canonical convention in the wiki."* It opens with **three distinct objects sharing one symbol**
and the instruction **"Never conflate."** It also records **symbols that must not be reused**,
with the date each collision was settled — and *"the bug that lived in the paper until
2026-04-13."*

**Its governing rule:** *"**Source wins.** If the wiki and a verified source disagree, the wiki
is wrong. Fix the wiki."*

**`log.md`** — append-only, entries prefixed `## [YYYY-MM-DD] <op> | <one-line description>`
**so `grep "^## \["` is a timeline.**

**`reproducibility.md` — the principle worth adopting verbatim:**
> *"Any file or directory that exists locally but is not tracked in git MUST have a
> reproducibility entry in this page. Creating new untracked content includes adding its entry
> here."*
Table columns: Category | Location | Typical size | **How to rebuild** (a literal command).

**A figure page** carries: file, ingest/update dates, **migration status** (which superseded
artefact must *not* be substituted, and why — e.g. an estimator mismatch), *What it shows*
(including **why the axes are what they are**), the caption as it appears in the paper,
**Generator** (script + exact command + what it writes), **Inputs**, concepts illustrated,
paper sections using it, related pages, and a **machine-readable JSON footer** with
`slug / concepts / generator / outputs / inputs / paper_uses`.

**Reachability is a separate problem from existence.** Notes are worth nothing if the agent
does not open them, *and it will not open a directory it has no reason to open.* Two mechanisms:
- a **`CLAUDE.md`** — always applies, costs some of the session's attention whether or not the
  question needed it;
- a **skill** — costs nothing until it fires, installs once for every project, **but can also
  fail to fire.**

The shipped `project-notes` skill is four lines of instruction; **its `description` is the whole
of what the agent sees before deciding to read the rest, which is why the description is the
part worth arguing about.** Measured (`homework.html`): *"`a` and `b` gave the same wrong answer
and neither opened anything. `b` began its reply by saying it would work the problem out
directly rather than reach for a skill, **with the notes sitting in the directory beside it**.
`c` and `d` read the notes and returned a result that is written down there and in no
publication. The skill in `d` fired unprompted, on a description that names no subject matter
and no filename."*

### 6.7 The course's own pages enforce it (`how-a-page-is-built/build_show.py`)

The session pages are built, not maintained, and **cannot drift from the record they report**:

- `prompts.txt` is **the single source of the prompt text**, parsed at build time — *"what you
  are reading is therefore the record, not a reconstruction of it."*
- **every image is base64-encoded from the file the code rendered**, so *"a figure and the page
  that shows it cannot be different objects"*;
- every run is read from its `meta.json`; the paths it names must exist, and **the lines it
  quotes are pulled out of the receipt at build time rather than typed into the page**;
- **rule 1c: every `[src:key]` written on a stage must resolve in a generated `provenance.json`
  registry, or the build fails.**

> *"A number or an output pasted into a page by hand is the failure all of this prevents."*
> *"A deck you maintain by hand drifts from the numbers in it, and a page you maintain by hand
> drifts from the runs it describes. Neither of these can — every number is re-read out of a
> file when it is built, and the build fails rather than printing a stale one."*

The `meta.json` schema is a good run-record template: `label, prompt, model, effort, format,
tokens_in/out/cached, wall_s, machine, turns, tool_calls, cost_usd, date, receipt, quote,
outputs`. *Everything optional except `label`; anything absent prints as "—", which is the
honest answer for a run made before the convention existed.* And the ordering principle:
**the table leads with tokens and wall time — those are durable; a price is a fact about a
price list on the day.**

---

## 7. The final project (`final-project.html`) — the brief

> *"Take a problem of your own and do the whole of it using agents, with the provenance written
> down as you go, and with enough evidence at the end that you would defend the result to
> somebody who did not watch you produce it."*

**Choosing the problem.** From your own research, or a GW problem based on the week. *Either is
fine and the first is usually better, **because you can tell whether the answer is right**.*
**Worth choosing one with many components — an analytic derivation, code, figures, data.**

**Doing the work.** Have the agent do the calculations and the figures, and **document the
provenance of every one of them as it goes, along with how each result was checked.**
> *"Both halves matter and the second is the one that gets skipped: **a figure with a recorded
> origin and no check behind it is still only an assertion with a picture beside it**."*

> *"Ultimately you are responsible for it being correct, exactly as you would be in any other
> piece of work. So produce enough artefacts that you are actually confident: **not enough to
> look thorough, enough that you would stand behind the result.** That judgement is yours and
> there is no rule for it, which is the point."*

**What you hand over.** Everything in a **GitHub repository**. Inside it:
1. an **HTML page** (will be linked from a general page with all projects), and
2. a **PDF** — the two things you present from,
3. **the rest of the repository behind them.**

> *"The split is deliberate. **The page and the PDF are for people; the repository is for
> machines.** Anyone who clones it should be able to reproduce all of it with an agent, which
> means the instructions, the environment, the data and the checks are in there and are readable
> by something that has never spoken to you."*

### 7.1 Checklist implied by the whole course

**Repository (for machines)**
- [ ] `CLAUDE.md` / `AGENTS.md` at the root — conventions, layout, what not to do (one real
      file + a one-line pointer)
- [ ] `structure/claims.yaml` — thesis + one record per load-bearing claim
      (`statement`, `section`, `file`, `numbers`, `evidence`, `depends_on`, `status`)
- [ ] a **number registry** JSON with a **single writer script**
- [ ] `scripts/check_provenance.py` — the gate; **build refuses unless it passes**; include the
      **displayed-precision** check (#7 above) and make skipped checks say *skipped*, not pass
- [ ] `.claude/hooks/` + `.claude/settings.json` + `.codex/hooks.json` — the Stop gate
- [ ] `.claude/skills/…/SKILL.md` (both harnesses) pointing at the convention `.md` files
- [ ] `checks.py` that **prints per-check coverage numbers, not verdicts**, and includes **at
      least one case made to fail on purpose**
- [ ] environment as a **lockfile**, not a list of package names
- [ ] `reproducibility.md`: every untracked artefact + the literal command that rebuilds it
- [ ] a `log.md` with a greppable date prefix
- [ ] run receipts: date, command, environment, actual pasted output, model, effort, tokens,
      wall clock, machine

**The write-up (for people)**
- [ ] LaTeX with a `draft`/`final` provenance-macro switch, so the reviewed manuscript and the
      shipped PDF are the same file
- [ ] a self-contained HTML page — **opens from `file://` with no network, no CDN, no external
      stylesheet, no remote font, no fetch; figures inline. Check this rather than assuming it.**
- [ ] optionally the `report.html` / `agent.html` pair: same content, second one carrying all
      provenance
- [ ] a reproducibility section in **named categories**, including **what is not claimed**
- [ ] an explicit statement of **what was not checked**
- [ ] every claim marked **derived** vs **recalled**
- [ ] every disagreement with a published value **reported with both numbers**, not tuned away

---

## 8. The transferable rules, condensed

1. **The loop is the difference.** A chatbot guesses about your files; an agent reads them, runs
   things, and can be asked for the receipt.
2. **Context is finite and cold starts are real.** Everything you want to survive must be on
   disk in a form that gets read next time.
3. **Formats decide whether you can review the output — and review, not generation, is the
   bottleneck.**
4. **You cannot tell a good derivation from a bad one by reading it. Ask for checks, not
   answers.**
5. **The harness is the part you control, and that is where the leverage is.**
6. **A result with no assertion is a hypothesis. Say so in those words.**
7. **A check nobody wrote subtracts nothing from a passing count. Name the uncovered half.**
8. **A check that cannot fail is not a check** — a function against its own inverse; a constant
   fitted to make the check pass; a residual bounded by construction.
9. **Prove your checks can fail**: ship at least one case where two methods are made to
   disagree on purpose.
10. **Check the artefact, not the summary.** Exit 0 and `"success"` mean nothing.
11. **Separate derived from recalled, and measured from assumed**, every time, unprompted.
12. **An independent second route agreeing beats any number of internal assertions** — and a
    second *implementation* is now one more run of the same prompt.
13. **Put the verifier on a different model family from the workers.**
14. **A verifier's finding is worthless if it does not reach the round that has to act on it.**
    Read `reports/` and `transcript/`, not the deliverable.
15. **Ask for the choices that could reasonably have gone another way** — and record the reason
    for each.
16. **Decide in advance what a wrong answer could still reproduce.** That is the one check that
    has not got cheaper.
17. **State the regime**: an acceptance contract holds only for the machine and library versions
    it was measured on. The environment is part of the claim.
18. **Measure your noise floor** — run the same thing twice unchanged — before believing any
    comparison. *If you change a setting and the answer moves by less than the noise floor, you
    have learned nothing, and you cannot know that until you have measured it.*
19. **One sample per cell is not a measurement.** Report what you saw, not what is true in
    general.
20. **Tokens and wall time are durable; prices are not.**
21. **A convention written twice drifts.** State it once; point at it from everywhere else.
22. **Skills can go unloaded; hooks always run.** Use both — the skill is the normal route, the
    hook is the floor.
23. **Source wins**: if your notes and a verified source disagree, the notes are wrong.
24. **A guarantee about your sample is not a guarantee about the population.**
25. **A seed is not a name for the result.**

---

## 9. Open questions to settle when the project problem is chosen

- Which of the three provenance scales fits the scope — the laptop scaffold alone, or the full
  paper-as-repository with `claims.yaml` + registry + `check_provenance.py`?
  (My default recommendation: **the full one**, since the brief explicitly asks for a repository
  a stranger's agent can reproduce, and the scaffold is only ~300 lines to lift.)
- Single-agent with a hook, or a lead/worker/verifier team with the verifier on a different
  model family?
- What is the **independent second route** for the headline number?
- What is the **acceptance contract** — bit-identical, or a science gate with a stated
  `check_extra.py`?
- What is the **noise floor** for this task (same prompt twice, unchanged)?

---

*Notes compiled from a full read of `GW-AI-course-main/`: `README.md`, `index.html`,
`using-the-tools.html`, `homework.html`, `final-project.html`, all five `day*/session-ai*.html`,
all `day*/README.md`, `day{2,3,4}/prompts.txt`, `day3/objective.txt`, `setup/{SETUP,README}.md`,
`day1/vault/` incl. templates, `day5/exercise/` (scaffold, conventions, skill, hook, wirings),
`day5/exercise-notes/` (README, prompt, CLAUDE.md, wiki index/conventions/log/reproducibility and
sample pages), `day5/skills/`, and `how-a-page-is-built/{page.py,build_show.py}`.
Physics and GW-detection content deliberately excluded except where it is the vehicle for a
method.*
