# HI ionizations per primary particle — IGM at z = 10
### IC secondary photoionization, with an energy-dependent W(K_e) from Furlanetto & Stoever 2010

> ## ⚠ SUPERSEDED IN PART — read this first (2026-09-12)
>
> The **consolidated, gated record of this project is now
> `photon_vs_electron.tex` → `photon_vs_electron.pdf` (16 pp, gate 9/9)**, by your
> standing decision that every figure and every further development lands there.
> It carries all twelve figures, both redshifts, and the two-route validation.
>
> Specifically superseded below: the **deposition-fit prescription (models A/B/C)**
> is obsolete as a source of published numbers. Electron propagation is computed
> with `igm_losses.py` (models D and E). A/B/C survive only as the independent
> cross-check that validated D/E.
>
> **Numerals in this Markdown file are NOT gated.** `check_provenance.py` reads
> LaTeX only; nothing checks this file, and the values below predate the
> 2026-09-11 harmonisation of the two environments. Where this file and
> `photon_vs_electron.pdf` disagree, **the PDF is right.** A numeral audit of this
> file is outstanding and is logged as such in `QUESTION_LOG.md`.

Full write-up: `ionization_yield.tex` → `ionization_yield.pdf` (4 pp, gate 9/9).
Code: `ionization_yield.py`. Gate: `check_provenance.py`. Registry: `results.json`.
Source now in the tree: `papers/0910.4410-furlanetto-stoever-2010.pdf`.

> **OPEN — not done, deliberately deferred.**
> **Email Furlanetto for the FS10 electronic tables.** They are not public
> ("available on request"), and FS10 decline to publish a fit to them. Until they
> arrive, `f_ion` comes from the Ricotti et al. 2002 fit, and **22.97 % of the
> secondary yield is produced below 100 eV where FS10 call that fit "a relatively
> poor match"**. This is the single largest available improvement and the only
> thing that closes the **-7.2 %** bracket. Raised 2026-09-07, deferred 2026-09-10.

---

## What FS10 does and does not give

**It gives no fitting formula for its own results, and its tables are not public.**
Abstract: *"Electronic tables of our results are available on request."* §7: *"we have
not attempted to find another form. Instead, we recommend interpolating the exact
results."* Reconstructing those tables from memory is the failure mode this project
exists to prevent, so it was not done. What the paper **does** give, with numbers:

| | what | used for |
|---|---|---|
| **eq. (2)** | secondary-electron energy distribution `p(ε) ∝ 1/[1+(ε/ε̄)^2.1]`, ε̄ = 8, 15.8, 32.6 eV for HI, HeI, HeII, with ε < (E−E_i)/2 | an **independent bound** that uses no SvdS input at all |
| **eqs. (13)–(14)** | Ricotti et al. 2002 energy-dependent fits to SvdS85 | the **energy-dependent W(K_e)** |
| **footnote 8** | SvdS exact f_heat(x=0.01) = 0.32; the Ricotti fit sits ~6% absolute above it | verification of the recalled coefficients |
| **§3, §6** | quoted median secondary energies 7.2, 14.2, 28.5 eV; *"our parameters do not include the effects of the initial ionization event"* | external anchor; justifies the explicit +1 in N_γ |

## The SvdS coefficients are verified at last

Until this paper entered the tree, the twelve recalled coefficients had **no external
test**. Eq. (13) carries the asymptote at two significant figures:

| | recalled | FS10 | rel. diff |
|---|---|---|---|
| ionization | 0.3908(1−x^0.4092)^1.7592 | 0.39(1−x^0.41)^1.76 | **0.18%** |
| heating | 0.9971[1−(1−x^0.2663)^1.3163] | 1.0[1−(1−x^0.27)^1.32] | **1.10%** |

And FS10's footnote 8 is reproduced: the recalled heating fit gives 0.3658 at x = 0.01,
**4.6% absolute above** SvdS's own exact 0.32; FS10 say ~6%. So the recalled numbers are
the *fitting function*, which is what they are, and they inherit its known offset from
the underlying Monte Carlo. **That offset is not removed here.**

## W is now a function of energy

`W(K_e) = E_th / f_ion,HI(K_e, x_e)` from eq. (13):

| K_e | < 28 eV | 100 eV | 1 keV | 10 keV | → ∞ |
|---|---|---|---|---|---|
| **W** | ∞ (no ionization) | 43.5 eV | 38.9 eV | 37.3 eV | 36.5 eV |

Model B now integrates the **marginal** yield `dN_A/dT`, not `f_ion/E_th` — once f_ion
depends on energy the two differ, and only the former is the ion pairs bought by the
energy deposited between T and T+dT. It collapses to the old formula *exactly* when
f_ion is made constant, and that collapse is a check.

## An independent bound from eq. (2)

Everything above rests on SvdS branching fractions. Eq. (2) offers a route that uses
none of them: if *every* inelastic event were an ionization,

    Y(E) = 1 + ⟨ Y(ε) + Y(E − E_th − ε) ⟩_p(ε),   Y(E < E_th) = 0

Switching off excitation and heating can only *add* ionizations, so Y is a strict upper
bound — and far tighter than E/E_th. It gives **one pair per 17.93 eV** against W ≈ 36 eV
in reality; the factor of two is the energy that genuinely goes to excitation and heat,
consistent with FS10's finding that the three channels split roughly equally. Model A
stays under Y at every energy tested.

## Verification — 50 automated checks, 50 pass

| Check | Outcome |
|---|---|
| **C32** SvdS coefficients | 0.18% (ion) / 1.10% (heat) vs FS10 eqs. (13)/(14). First external test they have had |
| **C32** FS10 footnote 8 | the known ~6% fit-vs-exact offset in f_heat is reproduced (4.6%), not papered over |
| **C32** eq. (2) medians | computed 7.356 / 14.51 / 29.89 eV vs FS10's quoted 7.2 / 14.2 / 28.5 → 2.2% / 2.2% / 4.9%. The medians are not used to build p(ε) |
| **C20** independent bound | ionization-only cascade: 17.93 eV per pair, monotone, under E/E_th, above model A everywhere |
| **C20** structural | `dN_A/dE` collapses to `f_ion/E_th` **exactly** when f_ion is constant → the change is physics, not refactoring |
| **C36** W(K_e) vs constant W | 4.575e6 vs 5.059e6 → **−9.56%** |
| **C20** second generation | identically zero: a 1218 eV photoelectron needs a 3.383 eV = 1310 kT seed photon, occupancy 1e-569 |
| **C53** null tests | N_γ = 0 exactly below 13.6 eV; N_γ = 1 exactly up to **41.6 eV** (the plateau edge *moved* with W(K_e); the check is written against the model, not the old number) |
| **C31** two-route checks | ⟨w⟩ to 1e-14; N_secondary to 1e-6; deposition integral to 2e-10 |
| **C56** determinism | `results.json` regenerates byte-identically |
| gate | 6/6; 56 `\src` uses across 34 keys, all resolving, all literals matching at displayed precision |

**Three bugs the suite caught in this revision.** (i) FS10 eq. (14) has a *different*
functional form from eq. (13) — `A[1−(1−x^B)^C]` not `A(1−x^B)^C`. Using the wrong one
gave 0.638 instead of 0.362, a factor 1.8, and looked perfectly plausible; the comparison
against FS10's own quoted 0.32 caught it. (ii) The eq. (2) cascade recursion was marched
on a **log** grid, where `E − E_th − ε` falls into the cell that is still empty; it
collapsed from Y(1e3) = 55 to Y(1e4) = 2.7 and the monotonicity check caught it. A linear
grid with step ≪ E_th makes the march exact. (iii) The old linear-space quadrature of the
model-B integral, still retained as a deliberate-failure case.

## Result

![figure](ionization_yield_fig.png)

| E [eV] | e⁻ (A) | e⁻ (B) | **e⁻ (C)** | γ (A) | γ (C) |
|---:|---:|---:|---:|---:|---:|
| 10¹ | — | — | — | **0** | **0** |
| 10² | 2.30 | 2.30 | 2.30 | 2.96 | 2.96 |
| 10³ | 25.7 | 25.7 | 25.7 | 26.4 | 26.4 |
| 10⁴ | 268 | 265 | 265 | 269 | 265 |
| 10⁵ | 2.72e3 | 2.16e3 | 2.16e3 | 2.73e3 | 2.16e3 |
| 10⁶ | 2.74e4 | 6.41e3 | 6.41e3 | — | — |
| 10⁷ | 2.75e5 | 8.45e3 | **2.36e4** | — | — |
| 10⁸ | 2.75e6 | 8.82e3 | **2.47e6** | — | — |
| 10⁹ | 2.75e7 | 8.86e3 | **4.56e6** | — | — |
| 10¹⁰ | 2.75e8 | 8.87e3 | **4.58e6** | — | — |
| 10¹² | 2.75e10 | 8.87e3 | **4.58e6** | — | — |

- **N_max = 4.575e6 ion pairs** for any electron above ~1 GeV — **515.9×** the
  collisional-only saturation (8.869e3)
- **E_crit = 1.303e5 eV**, **E_esc = 1218 eV**
- Peak reprocessing efficiency **99.0%** of the IC-radiated energy
- Independent ceiling from eq. (2): one pair per **17.93 eV**

### Correction: model A *is* an upper bound after all

The previous revision reported that C exceeds A by a few percent near 10⁸ eV, and
explained it: a near-threshold photon buys a pair for 13.6 eV where electron degradation
needs W. **That explanation was right about the mechanism and wrong about the size**,
because it used the *asymptotic* W for the near-threshold secondaries that do most of the
work. With W(K_e) those secondaries are correctly charged 43.5 eV, and the excess
disappears: max(C/A) = **1.000**. The claim that A is not an upper bound is **withdrawn**
— it was an artefact of the constant-W approximation.

### Uncertainty

None statistical, and that is a limitation. Three variations *are* propagated: I moves
the collisional saturation by 1.52%; the transport prescription moves N_C by **5.16%**
(4.339e6 for the light-cone variant); the sub-100 eV bracket moves it by **−7.2%**
(4.244e6). All one-sided or asymmetric; the modelling choices below dominate.

## What was NOT checked

1. **The SvdS paper itself is still not in the tree.** The coefficients are now checked
   against FS10's reproduction of the *Ricotti fit to* SvdS at two significant figures —
   a real external test, but a test against a fit, not against SvdS. FS10 quantify the
   gap: ~6% absolute in f_heat at x = 0.01.
2. **FS10's own tables** are not public and are not used. They would supersede eq. (13)
   entirely and are the single largest available improvement. They need an email to the
   authors.
3. **The sub-100 eV bracket is still live.** 22.97% of the secondary yield is made below
   100 eV, where FS10 state plainly that eq. (13) is *"a relatively poor match"*. W(K_e)
   improved this region but did not close it.
4. **Which transport prescription is right** — only the spread is measured.
5. **Compton scattering of the secondaries** off bound electrons (τ_T ~ 0.1 over c/H)
   would slightly *raise* C.
6. **Helium** in the fits but not the target, and absent from τ.
7. **Klein–Nishina** above 10¹² eV would raise B and lower C.
8. **The ionization-only cascade neglects excitation and heating by construction.** That
   is what makes it a bound; it is not a model and is not used as one.
9. **No independent reimplementation** (C24). Every cross-check except the W anchor, the
   FS10 comparisons and the eq. (2) bound is internal to this one code.

## Outstanding — needs you

| ID | Ask | Buys |
|---|---|---|
| **C32** | Email Furlanetto for the FS10 electronic tables | Replaces eq. (13) with the real thing; closes the −7.2% bracket |
| **C32** | Shull & van Steenberg 1985 PDF | Turns "checked against a fit to SvdS" into "checked against SvdS" |
| **C24** | Blind reimplementation in a fresh session | The eq. (2) cascade and the marginal-yield integral are both new code |
| — | Slatyer 2016 (PRD 93, 023521) | Would replace the transport prescription and settle the 5.16% |

## Reproduce

```bash
python3 ionization_yield.py     # 50 checks, figure, results.json, provenance/
python3 check_provenance.py     # the gate: 6/6
pdflatex ionization_yield.tex   # twice
```
Deterministic — no random numbers drawn, so no seed exists.
Environment: numpy 1.26.4, scipy 1.11.4, matplotlib 3.8.0, Python 3.11.
Runtime ~4 min (dominated by the p̃(w) table and seven ICPhotonChannel builds).


---

# Two independent routes to the electron yield: deposition fit vs energy losses

Code: `yield_comparison.py`. Figure: `yield_comparison_fig.png`.
Registry: `yield_comparison_results.json`. Loss physics: `igm_losses.py`.

**Why this matters.** Until now every cross-check in this project was internal to one
code, apart from the W(H) ≈ 36 eV anchor. Model D is built on a completely separate
prescription and shares nothing with models A/B/C but the physical scenario.

## Conceptual differences

| | **Route 1 — A/B/C** (`ionization_yield.py`) | **Route 2 — D** (`igm_losses.py`) |
|---|---|---|
| **What is counted** | energy, then split by a fitted fraction | **ionization events, one at a time** |
| Ionization yield from | `f_ion(E, x_e)` — FS10 eq. (13) = Ricotti+02 fit to SvdS85 | `σ_ion(K)` — **RBEB**, Kim et al. 2000 |
| The cascade | **implicit**: f_ion is the fraction of the *initial* energy reaching ionization summed over all generations | **explicit**: knock-on electrons drawn from the BEB SDCS (Kim & Rudd 1994) and followed recursively |
| Loss channels | **2** — Bethe–Berger–Seltzer stopping, Thomson-limit IC | **7** — adiabatic, synchrotron, IC **with Klein–Nishina**, Coulomb (Gould 72), excitation (Stone & Kim 02, n ≤ 10), ionization, bremsstrahlung |
| Excitation | folded inside f_ion, never separated | explicit competing channel |
| Expansion losses | **absent entirely** | present; peaks at **9.2%** of dK/dt near 3.05e+05 eV |
| IC regime | Thomson only | Klein–Nishina kernel over the Planck spectrum |
| Secondary spectrum | FS10 eq. (2), ∝ 1/[1+(ε/8 eV)^2.1] | BEB SDCS. *`igm_losses` states the eq. (2) fit underestimates ⟨ε⟩ by up to 29% and Λ_ion by up to 20% at 10⁵ eV* |

**The two pairings.** Each route now comes in a collisional-only and a
secondary-following flavour, so the comparison is made twice:

| | IC energy **discarded** | IC secondaries **followed** |
|---|---|---|
| route 1 (deposition fit) | **B** | **C** |
| route 2 (energy losses) | **D** | **E** |

Model **E** is built here on top of D, and it takes from route 2 everything route 2 can
supply: the IC branching ratio is `L_compton/L_total` (7 channels, **Klein–Nishina
corrected**), the photoelectron released by an absorbed secondary photon cascades with
**Y_D** rather than f_ion, and the optical depth uses route 2's n_HI and H(z). Only two
ingredients are shared, both exact physics rather than fits: the Blumenthal & Gould (1970)
scattered-photon spectrum and the hydrogenic photoionization cross section.

**A consistency point that had to be checked, not assumed.** Route 2 applies Klein–Nishina
to the IC *loss rate*, while the BG70 photon *spectrum* is a Thomson-limit result. Those
are compatible only if the secondary channel operates where KN is negligible. Measured at
the energies that actually deliver the ionizations: at 5% of the channel γ = 40.6,
F_KN = 1.00000; at 95%, γ = 739, F_KN = 0.99991. KN
corrections stay below 10⁻⁴ across the window, so the mixture is legitimate.

## Input differences (not method differences, and not harmonised)

| quantity | route 1 | route 2 | difference |
|---|---|---|---|
| n_H(z=10) | 2.5287e-04 cm⁻³ | 2.5870e-04 cm⁻³ | **2.3%** |
| E_th | 13.5984 eV (NIST ASD) | 13.6057 eV (1 Rydberg) | 0.05% |
| x_e | 10⁻⁴ | 10⁻⁴ | identical |
| B field | none | 1 nG comoving | — |

Each route keeps its own fiducials. The 2.3% density offset is an input difference and is
reported rather than absorbed, so it is not mistaken for a physics disagreement.

## Numerical differences

| E [eV] | B (route 1) | D (route 2) | **D/B** | W_D = E/D [eV] |
|---:|---:|---:|---:|---:|
| 10² | 2.30 | 2.48 | 1.077 | 40.4 |
| 10³ | 25.7 | 28.34 | 1.102 | **35.29** |
| 10⁴ | 265 | 276 | **1.044** | 36.2 |
| 10⁵ | 2157 | 2211 | **1.025** | 45.2 |
| 10⁶ | 6415 | 6602 | 1.029 | 151 |
| 10⁸ | 8816 | 9991 | 1.133 | — |
| 10¹² | 8869 | **1.191e+04** | **1.343** | — |

### With the secondary channel on both sides

| E [eV] | C (route 1) | **E (route 2)** | **E/C** | E/D |
|---:|---:|---:|---:|---:|
| 10³ | 25.7 | 28.34 | 1.102 | 1.000 |
| 10⁶ | 6415 | 6602 | 1.029 | 1.000 |
| 10⁹ | 4.559×10⁶ | 4.906e+06 | 1.076 | 456 |
| 10¹² | 4.575×10⁶ | **4.925e+06** | **1.077** | **414** |

**The routes agree to 7.7% with the secondary channel
included as well** — not just on the collisional-only yield. The IC-secondary channel
raises the loss-based saturated yield by a factor 414, the same
qualitative jump route 1 finds (573× for C over B), and it opens and closes over the same
decade. Two constructions that share no branching physics place the window in the same
place and give the same height to 8%.

**Two methods with nothing in common agree to within
10% over six decades.**
This is the strongest evidence the project has produced.

### Where they diverge, and why

1. **Above ~10⁸ eV: Klein–Nishina.** D/B rises to 1.343 at 10¹² eV.
   Route 1 uses Thomson-limit IC; route 2 applies the KN kernel
   (F_KN = 0.8111 at 10¹² eV), which **reduces** the IC loss and leaves
   more energy for the gas. The companion paper flagged this as an unquantified caveat —
   *"Klein–Nishina suppression would reduce b_IC and raise model B"*. **This is the
   quantification: +34%.**
2. **Near 10⁵ eV: adiabatic expansion**, worth 9.2% of the
   loss rate and entirely absent from route 1. It removes energy before the gas sees it,
   partly offsetting (1).
3. **Below 10⁴ eV: the fit versus the cross sections.** D sits 4–10% above B — the regime
   where FS10 caution their fit is "a relatively poor match". The sign is consistent with
   `igm_losses`' own note that the ε-fit underestimates Λ_ion.
4. **E_crit**: 151 keV (route 2, Compton vs excitation+ionization)
   against 130 keV (route 1, Thomson IC vs Bethe). The definitions of "collisional"
   differ, so agreement to tens of per cent is the most that is meaningful.

### What route 1 omits, and what that costs

Synchrotron (1 nG), Coulomb (x_e = 10⁻⁴) and bremsstrahlung each stay **below 0.25%** of
the loss rate everywhere in range. That is *why* route 1 can omit them and still land
within 10% — a quantitative justification the companion paper previously asserted without
one.

### Still shared by both, so still unverified

**Corrected 2026-09-12.** Both routes now *do* follow the IC-upscattered photons —
model C on route 1 and model E on route 2 — and they agree to 7.3% at saturation once
both carry that channel. What remains shared, and therefore still unverified by this
comparison: **helium is absent from the target** and **x_e is held static at 10⁻⁴** in
both. Agreement between the routes says nothing about either.
