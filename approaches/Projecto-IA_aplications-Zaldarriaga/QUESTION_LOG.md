# Question log — photon vs electron reionization

Required by `instructions_photon_vs_electron` §3: every question I ask, the options I
offered, and which one you chose. Newest at the bottom. Nothing here is a numerical value
I invented; each option carries its source.

## Literature checked before asking (instruction §2)

| Quantity | Source | Value |
|---|---|---|
| UV luminosity density at z = 10 | Donnan et al. 2024, JWST PRIMER, MNRAS, Table 3 | log10(ρ_UV / erg s⁻¹ Hz⁻¹ Mpc⁻³) = **25.12 (+0.07/−0.14)**; at z = 9, 25.29 ± 0.05 |
| ρ_UV → ρ_SFR conversion | Madau & Dickinson 2014, as used by Donnan+24 | K_UV = 1.15×10⁻²⁸ M⊙ yr⁻¹ / (erg s⁻¹ Hz⁻¹), **Salpeter (1955) IMF** |
| ⇒ ρ_SFR(z=10) | derived from the two rows above | **1.5×10⁻³ M⊙ yr⁻¹ Mpc⁻³** (Salpeter); ≈8.9×10⁻⁴ if rescaled to Chabrier |
| CR contribution to IGM ionization | Leite et al. 2017, MNRAS 469, 416 | *"CRs contribute negligibly to IGM ionization, but heat it substantially, raising its temperature by ΔT = 10–200 K by z = 10"* |
| CR preheating, Pop III SNe | Sazonov & Sunyaev 2015, MNRAS 454, 3464 | CRs can heat the IGM above T_CMB by z ≈ 15 (Pop III) / z ≈ 10 (Pop II); neither work finds more than a few per cent of the ionization |
| CR electron injection index | Leite et al. 2017 (secondary electrons) | Γ ≈ −2.1; steady-state steeper, from energy-dependent cooling |

**Order-of-magnitude expectation, written down BEFORE any computation (C30).** On the
strength of Leite+17 and Sazonov & Sunyaev+15, I expect **stellar UV to dominate CR
electrons as an ionizing agent by two to four orders of magnitude** at z = 10 for
canonical parameters, with the CR channel becoming competitive only if the CR injection
efficiency is pushed far above the SN-calibrated value, or f_esc is pushed far below it.
If the calculation says otherwise, that is a red flag to chase, not a discovery to report.

---

## Round 1 — 2026-09-10


| # | Question | Options offered | **Your answer** |
|---|---|---|---|
| 1.1 | Apply the Master Rules rigorously, or relax them? | Rigorously · Relax for this pass | **Rigorously** |
| 1.2 | Which ρ_SFR at z = 10 normalises the stellar UV source? | Donnan+24 JWST Salpeter (1.5e-3) · Donnan+24 rescaled to Chabrier (8.9e-4) · Madau & Dickinson 14 extrapolated · **Scan as a free parameter** | **Scan it as a free parameter**, fiducial anchored on Donnan+24 |
| 1.3 | Spectral shape of escaping stellar ionizing photons above 13.6 eV? | **Blackbody 5e4 K** · Blackbody 1e5 K (Pop III) · Power law · Both blackbodies as a bracket | **Blackbody 5×10⁴ K** |
| 1.4 | How to normalise the CR electron injection? | SN-calibrated from the same ρ_SFR · Match Leite+17 directly · **Scan CR efficiency as the free axis** · Fix the CR energy density | **Scan ε_CR × f_e as the free axis**, SN rate fixed from ρ_SFR |

**Consequence of 1.2 + 1.4.** Both source normalisations are now axes rather than fixed
numbers, so the figure becomes a competition map: ρ_SFR sets both channels together (it
feeds the SNe as well as the stars), and ε_CR × f_e slides the CR channel against the
stellar one. The fiducial point stays pinned to Donnan+24 so the map has an observed
anchor rather than floating free.


## Round 2 — 2026-09-10

| # | Question | Options offered | **Your answer** |
|---|---|---|---|
| 2.1 | Which ξ_ion? | log ξ_ion = 25.27 canonical · **25.28 JWST z~9** · 25.7 hard-SED · carry the 0.42 dex scatter as a band | **log₁₀(ξ_ion/Hz erg⁻¹) = 25.28** — Llerena et al. 2025, A&A 698, A302 (arXiv:2412.01358): median rises from 10^25.09 at z≈4.2 to 10^25.28 at the highest z probed |
| 2.2 | Which escape fraction? | f_esc = 0.10 · 0.20 · 0.05 · **scan as a third axis** | **Scan f_esc**, so the stellar channel is normalised by ξ_ion × f_esc with f_esc free |
| 2.3 | CR electron injection spectrum? | E^−2.1 1 MeV–1 TeV · E^−2.1 10 keV–1 TeV · E^−2.0 · E^−2.2 | **E^−2.1 from 1 keV to 1 TeV** — you widened the range below every option I offered |
| 2.4 | Which yield model for the electrons? | **Model C** · all three as a band · Model B · Model A | **Model C** — IC-limited collisional deposition plus photoionization by the absorbed IC-upscattered CMB photons |

**Note on 2.3 — WITHDRAWN AND CORRECTED after the calculation ran (2026-09-10).**

*What I told you before running:* that dropping E_min from 1 MeV to 1 keV would raise the
CR channel by ~10³·³ and was "the single most consequential number in the calculation."

*What the calculation shows:* **the effect on ζ_e is 5.159×**, not 10³·³.

| E_min | ζ_e [s⁻¹ per H atom] |
|---|---|
| 1 keV (your choice) | 6.411×10⁻²² |
| 10 keV | 3.950×10⁻²² |
| 1 MeV (my recommendation) | 1.243×10⁻²² |

*Where I went wrong:* I conflated the injected electron **number** with the observable.
The number really does rise by ~10³·³ at fixed CR energy — but ζ depends on the injected
**energy**, which is pinned by L_e, and the yield is close to N ≈ E/W. Those two cancel.

*Why the residual 5.159× is real physics and not an artefact:* model C **saturates** at
4.575×10⁶ ion pairs above ~1 GeV, because those electrons hand their energy to the CMB
rather than to the gas. Lowering E_min shifts weight to energies where electrons still
ionize efficiently. So E_min matters *because of the IC saturation*, not because of the
spectral divergence I cited.

**The conclusion is far more robust to your choice than I claimed.**

**Three scanned axes now:** ρ_SFR (feeds stars *and* supernovae), f_esc (stars only),
ε_CR × f_e (CRs only). ξ_ion is fixed at 10^25.28. The stellar/CR ratio therefore depends on
(ξ_ion f_esc) / (ε_CR f_e); ρ_SFR cancels from the ratio and sets only the absolute rates.

## Values I will DERIVE rather than adopt, per Master Rule 4

| Quantity | How |
|---|---|
| supernovae per solar mass formed | integrate the Salpeter (1955) IMF, m^−2.35 over 0.1–100 M⊙, progenitors ≥ 8 M⊙ — verified with sympy, not recalled |
| stellar ionizing photon spectrum | Planck function at T_eff = 5×10⁴ K, integrated above 13.598 eV — computed, not tabulated |
| ionization yields N_ion(E) | model C from `ionization_yield.py`, already built and checked (50/50) |

## Values still to be fixed, with sources, before the run

| Quantity | Provisional | Source |
|---|---|---|
| E_SN | 10⁵¹ erg | standard core-collapse energy budget |
| ε_CR (SN energy → CRs) | ~0.1 | canonical DSA efficiency; anchor of the scan |
| f_e (electron/proton) | ~0.01 | Galactic CR e/p ratio at GeV; anchor of the scan |
| ⇒ ε_CR × f_e anchor | ~10⁻³ | product is the scanned axis |

## Round 3 — 2026-09-10

| # | Question | Options offered | **Your answer** |
|---|---|---|---|
| 3.1 | Fiducial anchor for the ε_CR × f_e scan? | **ε_CR=0.1, f_e=0.01 → 10⁻³** · 0.3,0.01 → 3×10⁻³ · 0.1,0.001 → 10⁻⁴ · anchor on Leite+17 | **ε_CR = 0.1, f_e = 0.01 → 10⁻³** (10⁵⁰ erg per SN in CRs, 10⁴⁸ erg in electrons) |
| 3.2 | Fix x_e or scan it? | **Fixed at 10⁻⁴** · scan · two snapshots | **Fixed at 10⁻⁴** — same scenario as `ionization_yield_fig.png` |
| 3.3 | Comoving or proper volume for ṅ_ion? | **Comoving** · proper · comoving with the factor annotated | **Comoving** (Mpc⁻³), matching ρ_SFR from Donnan+24 |
| 3.4 | How to carry three scanned axes? | 2 panels + bands · **2 panels + competition map** · curve families · **two separate figures** | **Both layouts 2 and 4**, so you can choose after seeing them |

**Consequence of 3.4 — three figure files will be produced:**

| File | Layout |
|---|---|
| `photon_vs_electron_fig1.png` | option 2 — panels (a), (b), plus a ζ_γ/ζ_e competition map over (f_esc, ε_CR f_e) with the canonical point and the crossover line |
| `photon_vs_electron_fig2a.png` | option 4, part 1 — (a) and (b) only, at fiducial parameters, uncluttered |
| `photon_vs_electron_fig2b.png` | option 4, part 2 — the parameter exploration on its own |

**Sources for 3.1.** ε_CR ≈ 0.1: standard diffusive-shock-acceleration efficiency; SNR
modelling finds >10 per cent of the explosion energy can end up in cosmic rays.
f_e = K_ep ≈ 0.01: the observed Galactic cosmic-ray electron/proton ratio at 10 GeV.
E_SN = 10⁵¹ erg: standard core-collapse energy budget.

**No further questions.** Everything needed to run is now fixed or scanned.


## Round 4 — 2026-09-10 (modifications, no questions asked)

You specified both changes directly, so nothing needed asking.

| # | Change | Effect |
|---|---|---|
| 4.1 | Stellar SED: blackbody → literature SED for a z≈10 star-forming galaxy | ζ_γ **+2.9%** (4.499e-18 → 4.631e-18) |
| 4.2 | CR injection index −2.1 → **−2.2** | ζ_e **+23.2%** (6.411e-22 → 7.899e-22) |
| | **Net** | ratio 7018 → **5862**, i.e. 3.85 → **3.77 decades**. Conclusion unchanged. |

### The normalisation you asked about, for the record

**Energy-anchored, not number-anchored.** dN/dE = A·E^−p on [E_min, E_max] with A
fixed by ∫E·A E^−p dE = L_e = Ṙ_SN × (ε_CR f_e) × E_SN. The supernova energy budget
is the invariant; the electron count is derived as ṅ_e = L_e/⟨E⟩.

*Prediction registered before running the p = 2.2 case:* ζ_e would rise by **tens of
percent, not by the 1.63× the ⟨E⟩ change implies**, because ⟨E⟩ cancels against
N ≈ E/W and only the model-C IC saturation breaks the cancellation.
*Measured:* **+23.2%.** Held.

### The SED: what was found, and the limit that was not papered over

Identified population — U37126 at z = 10.255, our redshift (Marques-Chaves et al.
2026, arXiv:2602.02322, in `papers/`): **BPASS v2.2.1, imf135_300, Z = 0.003,
constant SFH, age 6.8 ± 1.6 Myr**, β_UV = −2.88 ± 0.10, log ξ_ion = 25.75 ± 0.09,
f_esc = 0.94 ± 0.06.

**The BPASS spectrum itself could not be obtained** — the tables are not reachable
from here and the paper quotes only Q_H. It was therefore **not invented**. The shape
is carried by the standard reionization-source parametrisation, f_ν ∝ ν^−α across
1–4 Ryd, with α **bracketed over 1–3** rather than asserted.

**Why that limit does not threaten the result.** ξ_ion carries the normalisation, so
the shape enters only through ⟨N_γ⟩, which has a hard floor of 1:

| shape | ⟨N_γ⟩ |
|---|---|
| blackbody 5×10⁴ K | 1.0063 |
| power law α = 1 | 1.0715 |
| power law α = 2 | 1.0327 |
| power law α = 3 | 1.0134 |

Worst-case spread **1.065** — under 7%, against a gap of nearly four decades.
⟨E_γ⟩ = 21.76 eV = 1.6 Ryd: every shape puts its photons within a factor of two of
threshold, where each makes exactly one ionization.

**Outstanding:** the BPASS v2.2.1 tables (Z = 0.003, imf135_300) would replace the
bracketed α with the real spectrum. Alongside the FS10 tables on the companion work.

## Round 5 — 2026-09-11 (audit of eq. 6, prompted by your question)

**Your question:** how much error does ṅ_e = L_e/⟨E⟩ introduce versus using the
energy-dependent distribution?

**Answer: none — it is an identity.** ⟨E⟩ is *defined* as the number-weighted mean of
the same distribution, so L_e/⟨E⟩ is that definition rearranged; sympy returns exactly 0
for the residual, for any normalisable f(E). Energy closure through ṅ_e × p(E) recovers
L_e to 1.8×10⁻¹⁶. The spectrum is reinstated immediately and integrated point by point.

**The approximation that would cost something** — evaluating the *yield* at ⟨E⟩ rather
than integrating it — is **+33.6%** at the fiducial (N_C(⟨E⟩) = 156.3 vs ⟨N⟩ = 117.0).
Jensen: N(E) is concave because it saturates above ~1 GeV. The sign flips with where ⟨E⟩
sits relative to the IC-secondary window: +132% at p = 2.0, +11% at p = 2.4, −81% at
E_min = 1 MeV. The code integrates and pays none of it.

### Three defects the question uncovered

| # | Defect | Consequence |
|---|---|---|
| 1 | `electron_pdf` / `mean_injected_electron_energy` bound `CR_INDEX`, `CR_E_MIN_EV` as **default arguments** (evaluated once at import) | the E_min sensitivity check mutated the globals and the spectrum never saw it — it varied the integration limits only. **Reported 10.63×; true value 2.542×.** A wrong number reached the paper |
| 2 | `p = 2.0` raised `ZeroDivisionError` — exponent 1−p = −1 makes the integral a logarithm | **that was one of the four indices offered in round 2.** Had it been chosen, the run would have died |
| 3 | The gate could not parse a literal followed by `\times$` or written as `10^{b}` | **9 of 41 `\src` tags were silently unchecked**, and the stale 10.63 rode through a "0 mismatches" report |

### Corrected E_min sensitivity

| E_min | ζ_e [s⁻¹] |
|---|---|
| 1 keV | 7.8989×10⁻²² |
| 10 keV | 6.3664×10⁻²² |
| 1 MeV | 3.1079×10⁻²² |

**Factor 2.542 between 1 keV and 1 MeV.** Both figures reported earlier (5.159× at
p = 2.1, 10.63× at p = 2.2) were artefacts of defect 1.

### Fixes

- Parameters resolved from the module globals **at call time**, with a structural check
  that fails if the spectrum ever stops tracking them.
- `_powint` handles the removable singularity at exponent −1; verified against log-space
  quadrature for p = 1.8 … 3.0 to ~10⁻¹⁶.
- Gate parser widened; **new check C2b fails if any `\src` tag carries a literal it
  cannot read**, so a silently-unchecked number is now a failure rather than a gap.

## Round 6 — 2026-09-11

**Standing decision (yours, unprompted):** all electron-propagation physics now uses
`igm_losses.py`, with environmental parameters adjustable. Implemented as `igm_config.py`:
overrides are applied, restored, and **verified** on exit; parameters baked into the cached
cross-section tables at import (`exc_nmax`, `fix_secondary_spectrum`, `fix_kn_kernel`) are
**refused** rather than silently ignored — the same failure mode as the default-argument
binding bug that put a wrong number in the paper in round 5.

| # | Question | Options offered | **Your answer** |
|---|---|---|---|
| 6.1 | Which figure to reproduce on the new prescription? | photon_vs_electron_fig1 · ionization_yield_fig · all three photon_vs_electron · **everything** | **Everything in the folder** |
| 6.2 | Replace model C, or keep it alongside? | **Replace — igm_losses becomes the yield** · both as a band · keep C, add E as a check | **Replace.** C is retained in code and registry as the cross-check that validated E, but no longer drives published numbers |

**Consequence.** ζ_e in the photon–electron competition is now computed from **model E**
(7 loss mechanisms, RBEB event counting, BEB secondary cascade, Klein–Nishina IC, plus the
IC-secondary photoionization channel) instead of model C. E/C ≈ 1.08, so ζ_e rises ~8% and
the competition ratio falls from 5862 to ≈ 5400. **The conclusion is unaffected**: photons
still lead by ~3.7 decades and CR parity still needs ε_CR f_e > 1.

**Flagged, not silently done.** `ionization_yield.tex` is a 10-page paper whose 56
`\src`-tagged literals are tied to the OLD prescription's registry. Regenerating its figure
on the new physics does not migrate the paper. That migration is a separate job and is
listed as outstanding rather than half-done.

## Round 7 — 2026-09-12

**Standing decision (yours):** every figure and every further development in this project
goes into `photon_vs_electron.tex`. That file is now the consolidated record; this round
added two sections to it rather than a new document.

**Prior decision carried out this round:** the two environments were harmonised on the
route-1 values ("Keep route 1 for both cases. Those are the values that should be used from
now on in future calculations"). `igm_losses.py` was edited: `THRESHOLD_EV_ION` moved from
one Rydberg to the NIST binding energy `13.598434599702` eV, and `N_HI_NORM` was
renormalised from Planck 2018 + PDG 2022. A backup sits at
`igm_losses.py.bak-before-harmonisation`.

| # | Question | Options offered | **Your answer** |
|---|---|---|---|
| 7.1 | Which figures at z = 20? | fig1 only · the three competition figures · the two yield figures · **all five** | **All five, at z = 20** |
| 7.2 | ρ_UV and ξ_ion have no measurement at z = 20. What should the absolute rates do? | (1) ratio only, no absolute axis · (2) extrapolate the UVLF · (3) hold the z=10 value silently · (4) scan ρ_SFR as a band | **1 + 4: ratio only, and scan ρ_SFR as a band** |
| 7.3 | Ratio-only treatment at z = 10 as well? | yes · no | **"Also, do 1 for z = 10 too. I'll analyze them later and conclude which ones are worth keeping"** |

### What was decided by me, and why — flagged rather than buried

- **Band endpoints** `ρ_SFR ∈ [10⁻³, 1] × its z=10 value`. Three decades down to one. This
  is an *ignorance interval chosen by hand*, not a measurement and not an error bar, and
  every figure carrying one says so in its own footer.
- **`Z_INIT` is not what "initial redshift" means here.** `igm_losses.py` has a global by
  that name, and it is *not* what varies: it sets the trajectory time grid and the range of
  the 2-D loss map, never the loss rates. Checked, not assumed — setting it from 10 to 20
  leaves the yield bit-identical at 10³, 10⁶, 10⁹ and 10¹² eV (max relative difference
  exactly 0). What varies is the snapshot redshift `Z_SNAP`.

### Two defects found and fixed this round

- **The model D/E caches were keyed without the redshift.** `set_redshift(20)` would have
  returned the z = 10 curves silently. Fixed by putting `Z_SNAP` in the cache key *and*
  clearing on `set_redshift`. Nothing published was affected — the bug was found before the
  first z = 20 figure was drawn.
- **Three captions and two assumption lines still said "blackbody"** after the SED became a
  parametrised LyC power law, and one said the environments were "deliberately not
  harmonised" after they had been harmonised. This is the fourth time caption prose has
  outlived the physics it describes. No check catches it; it remains an open gap.

### Still open

- **`H(z)` is not harmonised.** Route 1 evaluates it from the hard-coded Planck-18
  parameters of `ionization_yield.py`, route 2 from `astropy`'s `Planck18` inside
  `igm_losses.py`: `0.137%` apart at z = 10, `0.268%` at z = 20. Same cosmology, two
  evaluations. It reaches the yield only through the adiabatic term, so the effect is below
  10⁻⁴, but it is **your decision to make, and it has not been made.**
- `IONIZATION_YIELD_SUMMARY.md` still has ~50 numerals no gate guards; no check covers
  Markdown at all.

## Round 8 — 2026-09-12

You supplied the two APS papers (`Kim (1994).pdf`, `Kim (2000).pdf`) and asked for a
consistency check of the numerical values across the manuscript, the code and the
literature, with particular attention to the redshift dependence and to validity at
z = 10 and z = 20. No options needed choosing; the audit is in
`photon_vs_electron.pdf` §"Audit against the sources" and in `verify_kim_and_z.py`
(13/13, registry `provenance/audit_registry.json`).

### Decisions I made without asking, and why

- **Renamed the two PDFs** to `PhysRevA.50.3954-…` and `PhysRevA.62.052710-…`, because
  `Kim (1994).pdf` cannot be cited from LaTeX or resolved by the gate's path check
  (spaces and parentheses). Recorded in `papers/MANIFEST.md`.
- **Corrected `LOG_RHO_UV_LO` from −0.14 to −0.08 dex.** This is not a judgement call:
  Donnan+24 Table 3 reads 25.12 +0.07 −0.08 at z = 10, and −0.14 is the z = 11 row's
  *upper* error. A cited number that does not match its source is an error, so I fixed
  it rather than flagging it.
- **Corrected "RBEB" to "RBED" everywhere.** The code implements Kim 2000 eq. (20);
  RBEB is eq. (22). Since the two differ by up to 38%/55%, the label is not cosmetic.
  Figure footers carried the wrong name, so every figure was regenerated.
- **Left the B = NIST vs B = Rydberg inconsistency in place** and reported it instead:
  it is your harmonisation decision, and it costs at most 0.225% on σ.

### What I got wrong first, in my own checking

My first reimplementation "disagreed" with `igm_losses.py` by 29 orders of magnitude
and I nearly reported a catastrophic bug. The fault was mine: `scipy.quad` over
[0, (t−1)/2] with t up to 10⁹ silently returns ≈ 0 when the integrand lives near w ~ 1.
Substituting y = 1/(1+w) onto a bounded interval brought the agreement to 1.4×10⁻⁹.
Two further "failures" were also mine — comparing the code's 7 significant figures to
a table printed at 5, and predicting the H(z) gap from `Planck18.Onu0`, which counts
massive neutrinos as matter at z = 0 and is ~19× the relativistic density.
**Lesson recorded: a disagreement with a published formula is my bug until the
reimplementation is itself verified on a case where the answer is known.**

### Still open after this round

- The knock-on (SDCS) spectrum uses the **non-relativistic** BED differential cross
  section: +1.9% at 10 keV, +15.4% at 100 keV, +25.5% at 1 MeV on the relativistic
  factor. Shape only — the total comes from the relativistic eq. (20) — and above
  E_crit the cascade is IC-dominated. Quantified now, not corrected.
- **Salpeter's slope is used 0.60 dex below and 1.00 dex above its stated fitted
  range** (0.4–10 M☉). This propagates straight into ζ_e and is the largest
  source-attested weakness in the CR channel.
- `IONIZATION_YIELD_SUMMARY.md` numerals remain ungated.

## Round 9 — 2026-09-12

You asked whether the abstract's reionization-pace statement should be normalised by the
recombination time rather than the Hubble time, and what the two normalisations mean for
the total. **You were right, and the abstract was wrong in two independent ways.**

### The error

The abstract said: *"the stellar channel gives ζ_γ t_H = 0.1037 ionizations per atom per
Hubble time, the pace required for reionization to complete near z ≈ 6."*

1. **Wrong clock.** The standard criterion is the recombination time, not the Hubble time.
   Madau & Dickinson 2014 eq. (24) — already cited in this manuscript and in the tree —
   gives ⟨t_rec⟩ = (χ⟨n_H⟩α_B C_IGM)^-1 and the criterion ζ t_rec ≥ 1: ionizations must
   outpace recombinations for the gas to be *kept* ionized.
2. **Wrong magnitude, by 25×.** 0.1037 per Hubble time is not "the pace required"; the
   requirement is 1 + t_H/t_rec = 2.608 per Hubble time — which is exactly MD14's own
   "close to two ionizing photon per baryon are needed to keep the IGM ionized".

### What replaced it

| | z = 10 | z = 20 |
|---|---|---|
| t_rec/t_H | 0.6220 | 0.3448 |
| ζ_γ t_H | 0.1037 | 0.03934 |
| **ζ_γ t_rec** | **0.06450** | **0.01356** |
| ζ_e t_rec | 1.156×10⁻⁵ | 2.368×10⁻⁶ |
| required, 1 + t_H/t_rec | 2.608 | 3.900 |
| stellar shortfall | **25.1×** | 99.1× |
| CR shortfall | **1.403×10⁵** | 5.678×10⁵ |

The old check, C32, asserted only `0.01 < ζ_γ t_H < 3` — a window wide enough to accept
almost anything — and called the result "the right pace". It is replaced by three checks:
**C32** reproduces MD14 eq. (24) from our own n_H (3.149 Gyr vs their 3.2 Gyr; t_rec/t_H =
0.6220 vs their "~60%"); **C33** states ζ_γ t_rec < 1; **C34** states the shortfall.
21/21 checks now pass at both redshifts.

### Why the shortfall is the right answer, not a failure

ζ t_rec < 1 at z = 10 *is* what an incomplete reionization means: escaping stellar photons
cannot yet hold the IGM ionized. Completion near z ≈ 6 is done by the rise of ρ_SFR between
z = 10 and z ≈ 6, and **this document adopts no star-formation history**, so that closing is
reported as missing rather than asserted. The shortfall also sharpens the paper's own
conclusion: the channel that does reionize the universe is already 25× short at z = 10, so
the CR channel — a further 5579× below it — is not a candidate under any bookkeeping.

### Two imprecisions of my own, caught before publishing

- I first wrote t_rec/t_H ∝ (1+z)^-3/2, which would give 0.379 between the two redshifts;
  the measured factor is 0.554. The difference is C_IGM = 1 + 43 z^-1.71 falling from 1.838
  to 1.256, lengthening t_rec by 1.463: 0.379 × 1.463 = 0.555. Now stated as a checked
  decomposition rather than a proportionality.
- The z = 20 column of the two ζ rows inherits the unanchored ρ_UV, and the table now says
  so; only t_rec/t_H, C_IGM and the requirement are exact at both redshifts.

## Round 10 — 2026-09-12

You asked for the reionization-budget material to become its own section with the algebra
and the numbers shown, plus the required photons 1 + t_H/t_rec at z = 20, 10, 5.5 for
C = 1, 3, 12. Delivered as §"The reionization photon budget" (new), backed by
`reionization_budget.py` (5/5 checks, registry `provenance/reionization_budget.json`).
`photon_vs_electron.pdf` is now 25 pp, gate 9/9, 216 tagged literals.

### Required ionizing photons per H atom per Hubble time, N_req = 1 + t_H/t_rec

| z | t_H [Gyr] | n_H [cm⁻³] | C = 1 | C = 3 | C = 12 | C_IGM(z) |
|---|---|---|---|---|---|---|
| 20 | 0.2685 | 1.7595×10⁻³ | 3.302 | 7.907 | 28.63 | 3.892 (C = 1.256) |
| 10 | 0.7086 | 2.5287×10⁻⁴ | 1.873 | 3.620 | 11.48 | 2.606 (C = 1.838) |
| 5.5 | 1.556 | 5.2175×10⁻⁵ | 1.396 | 2.187 | 5.748 | 2.318 (C = 3.330) |

The requirement contains no ρ_UV and no star-formation history — only n_H(z), H(z), α_B(T)
and C — which is why it is quotable at z = 5.5 where this project has no measured source
normalisation.

### A defect in my first derivation, caught by sympy

I first wrote the balance as dx/dt = ζ(1−x) − α_B C χ n_H x², then tried to solve for ζ at
x = 1. `sympy` returned an empty solution set, because at x = 1 the supply term vanishes
identically and the question cannot be posed in that form. The budget literature uses the
**photon-counting** form, dx/dt = ζ − α_B C χ n_H x², valid because the IGM is optically
thick to LyC during reionization so every escaping photon is absorbed somewhere. In that
form the x = 1 steady state gives ζ_crit = α_B C χ n_H = 1/t_rec exactly — so ζ t_rec ≥ 1
is not a convention but the boundary condition itself.

### One inconsistency between two of my own sections, now removed

§5.1 computed t_H from route-1 H(z) (no radiation term) while the new section used astropy's
Planck 18. That put 2.608 in one section and 2.606 in the other for the same quantity.
Since the audit had already established astropy's H(z) as the complete one, `t_H` in
`photon_vs_electron.py` now uses it too. The two code paths — `reionization_budget.py` and
`photon_vs_electron.py` — now agree to **0 and 2×10⁻¹⁶ in double precision** at z = 10 and
z = 20 respectively, which is a cross-check rather than a restatement. Knock-on edits:
ζ_γ t_H 0.1037 → 0.1036, t_rec/t_H 0.6220 → 0.6228, requirement 2.608 → 2.606, stellar
shortfall 25.1 → 25.2. ζ itself is untouched — it contains no H.

## Round 11 — 2026-09-12

You asked for the budget table in a fully-ruled layout, in **both** normalisations. Done:
Tables 1 and 2 of §7.3, `\begin{tabular}{|c|c|c|c|c|c|c|}` with a rule between every row,
same seven columns in both. `reionization_budget.py` now 6/6 checks; paper 25 pp, gate 9/9,
222 tagged literals.

### What the second table required deriving first

"The same requirement per recombination time" is not a free choice of units — it follows:

    N_req(t_rec) = N_req(t_H) × (t_rec/t_H) = (1 + t_H/t_rec)(t_rec/t_H) = 1 + t_rec/t_H

so the two tables are mirror images. **Read term by term they say different things**, which
is the substance of your original question:

- eq. (13), Hubble-time form: the "1" is the single initial ionization, and the varying term
  counts recombinations.
- eq. (15), recombination-time form: the "1" **is** the recombination — exactly one per
  recombination time, by construction — and the varying term is that initial ionization
  amortised over one t_rec.

New check **B6** asserts this: across all nine (z, C) cells,
N_req(t_rec) − t_rec/t_H = 1 to 2×10⁻¹⁶. That is what choosing t_rec as the clock buys —
it normalises away the very quantity the criterion is about, leaving a number to compare
with unity. Hence the trends invert: Table 1 rises with C and z, Table 2 falls toward 1.

| z | C=1 | C=3 | C=12 | C_IGM(z) |
|---|---|---|---|---|
| **t_H normalisation** | | | | |
| 20 | 3.302 | 7.907 | 28.63 | 3.892 |
| 10 | 1.873 | 3.620 | 11.48 | 2.606 |
| 5.5 | 1.396 | 2.187 | 5.748 | 2.318 |
| **t_rec normalisation** | | | | |
| 20 | 1.434 | 1.145 | 1.036 | 1.346 |
| 10 | 2.145 | 1.382 | 1.095 | 1.623 |
| 5.5 | 3.527 | 1.842 | 1.211 | 1.759 |

### A layout defect I fixed twice

The fully-ruled tables overflowed the text block by 126 pt and 177 pt, because in provenance
draft mode every `\src` tag is printed next to its number and the tag names in this section
are long (`NreqTrec_z5p5_CMD14`). I first hand-tuned it — moved the C_IGM parentheticals
into the captions, dropped to `\footnotesize`, tightened `\tabcolsep` — which fixed Table 1
and left Table 2 over by 51 pt. Hand-tuning a width against a tag string is fragile, so both
tabulars are now wrapped in `\resizebox{\textwidth}{!}{...}`: they fit whatever the tags
turn out to be, including after a future rename. No overfull box remains in the section.

## Round 12 — 2026-09-12

You asked what each factor of eq. (8) means and where it comes from. Answering it properly
turned up something the shorthand had been hiding. Added as "Eq. (8) factor by factor" in
§5.1; `reionization_budget.py` now 11/11 checks, 71 registry keys; paper 26 pp, gate 9/9,
241 tagged literals. Two new sources fetched into `papers/`.

### Two factors I could derive instead of adopt

- **χ = 1.08 is not a fudge factor.** With Y_P = 0.245 and X_H = 0.755, the He/H number
  ratio is (Y_P/4)/X_H = 0.08113, so singly ionized helium gives χ = 1 + 0.08113 =
  **1.0811** — MD14's 1.08 to 0.1%. Doubly ionized helium would give 1.1623, correct after
  HeII reionization at z ≲ 3 and a 7% error here.
- **α_B(2×10⁴ K) = 1.43×10⁻¹³ cm³/s confirmed independently** against the Hui & Gnedin
  (1997) case-B fit, whose coefficients I read off the page (2.753e-14, 0.407, 2.242,
  T_HI = 157807 K): 1.4277×10⁻¹³ at 2×10⁴ K (0.16%) and 2.5918×10⁻¹³ at 10⁴ K against the
  textbook 2.59×10⁻¹³ (0.07%). Anchored at two temperatures, not one.

### What tracing "1 + 43 z^-1.71" to its source revealed

**Pawlik, Schaye & van Scherpenzeel 2009 never print that expression.** Their eq. (A1) is

    C(z) = z^β e^(−γz+δ) + α

with α = 1.00, β = −1.71, γ = 0.00, δ = 3.76 for C_100 in the r19.5L6N256 run (Table A1).
So **MD14's "43" is e^3.76 = 42.95**, and evaluating eq. (A1) directly agrees with the
shorthand to 0.084%. Two things the shorthand hides:

1. **It is specifically C_100** — the clumping of gas *below* overdensity 100, from the run
   reheated at z_r = 19.5. Their C_-1 (all gas) is roughly twice as large at z = 6, so
   MD14's choice is the **less demanding** one. That is exactly why the C = 3 and C = 12
   columns are in the tables.
2. **Its fitted range is 6 ≤ z ≤ 20.** z = 20 sits at the upper edge and z = 10 well inside,
   but **z = 5.5 is below it** — so the C_IGM entry at z = 5.5 is an extrapolation. Mild
   (0.5 in z) but real, and it is the only cell in either table that leaves a source's
   stated domain. New check **B10** asserts precisely this. The fixed-C columns assume
   nothing and are unaffected.

### One silent assumption now stated

α_B is the only factor with a free knob, and T is an assumption about photoionized gas, not
a measurement of this IGM. Since t_rec ∝ 1/α_B and α_B falls with T, a hotter assumed IGM
*lowers* the requirement. Between the two values in common use the recombination term
changes by 1.815: at z = 10 with the MD14 clumping, N_req goes from **2.606** at their
2×10⁴ K to **3.915** at 10⁴ K. So MD14's temperature is the milder choice, and T is the
second-largest lever after C (check B11).

### Three defects of my own, all in the provenance tagging

`\%` inside a tagged literal hides it from the gate's parser (third time); a tag after an
inequality (`$6 \le z \le 20$\src{...}`) has no bare numeral to bind to; and a closing brace
between numeral and tag (`$20$}\src{...}`) breaks their adjacency. All three made C2b fail
rather than pass silently — which is what C2b was added for. A slice-based insert had also
dropped the `sec:notchecked` label; caught by the undefined-reference count, now restored.

## Round 13 — 2026-09-12

You asked for one paragraph per figure — what is depicted and the main conclusion of each
plot — each calling `\ref{}`, with the instruction not to assume any conclusion without
asking first. Eleven paragraphs now sit inline after their figure environments. Paper 28 pp,
gate 9/9, 299 tagged literals, 0 undefined references. New: `figure_claims.py`, 5/5 checks,
`provenance/figure_claims.json` (57 keys).

| # | Question | Options offered | **Your answer** |
|---|---|---|---|
| 13.1 | What may the "conclusion" part state? | only already-gated conclusions · **gated + new, verified first** · gated + new, flagged as a reading | **Gated + new, verified first** — every new claim computed, gated, and listed for approval before insertion |
| 13.2 | What conclusion for the loss-branching lower panels? | IC takeover only · + excitation sink · + negligible channels · **all three** | **All three** |
| 13.3 | Panel (a): state that the channels occupy disjoint energy ranges? | **yes** · describe only | **Yes, state it** |
| 13.4 | Where do the paragraphs go? | new section · **inline after each figure** · merged into captions | **Inline after each figure** |
| 13.5 | Approve claims N1–N4? | all four · all but N2 · only N1 and N3 | **All four, "also mention the amount of energy that is lost to the adiabatic expansion"** |
| 13.6 | How should the six z = 20 twins read? | **only what differs from z = 10** · self-contained repeat | **Only what differs**, each pointing at its z = 10 twin with `\ref{}` |

### The five new claims, all gated before they went in

- **F1/F2 — excitation is the largest irrecoverable sink.** 0.3532 of dK/dt at 100 eV,
  151× the next non-ionizing channel; its energy leaves as Lyman-series photons, all below
  13.6 eV, so unlike the Compton share it can never return as an ionization.
- **F3 — the negligible channels.** Coulomb 0.2336%, bremsstrahlung 0.00416%, synchrotron
  1.18×10⁻⁵%: the quantitative licence for the superseded route to omit all three.
- **F4 — the bands are disjoint.** Stellar LyC spans 0.602 dex, CR electrons 9.00 dex
  (14.9× wider), with 1.264 dex of empty axis between them. Hence the paragraph for panel
  (a) says explicitly that comparing curve *heights* there concludes nothing.
- **F5 — energy lost to adiabatic expansion** (your addition). Integrated over the cascade,
  f_adiab(K) = (1/K)∫₀^K [L_adiab/L_tot] dK′ peaks at **7.696% near 1 MeV** at z = 10,
  falling to **1.849%** at z = 20 (factor 4.16). Non-monotonic and negligible at both ends:
  0.0145% at 1 keV, 0.000106% at 1 TeV where Compton takes 99.99% first.

### Two wording traps the checks caught

- **Excitation does not "peak" at 0.353.** 100 eV is the *grid edge* and the share is still
  rising as E falls; the registry key `peakE_excitation_eV = 100` is a boundary, not a
  turnover. F1 asserts this, and the paragraph says "reaches … at the 100 eV edge".
- **The adiabatic *energy* fraction is not the branching panel's 9.29%.** That 9.29% is an
  instantaneous rate share attained only near 3.05×10⁵ eV; the energy actually lost over the
  whole cascade is 7.696%. Both numbers now appear together so they cannot be confused. My
  first version of F5 also anchored on 1 TeV, comparing two numbers that are both ≈ 0 — the
  check was re-anchored on the real maximum.

### Three untagged numerals I had to close

My own paragraphs introduced "2.7%", "3×10⁵ eV" and "the bump near 10⁸ eV" as bare numbers.
The gate does not catch untagged literals — only tagged ones — so these were a C1 gap, not a
gate failure. All three are now tagged (`zeta_ratio_rise_z20_pct` added to the z20 registry;
the IC window quoted as 2.022×10⁷–3.554×10⁸ eV with the 1218.0 eV escape energy).

## Round 14 — 2026-09-13

You asked me to integrate the dζ/dlog₁₀E curves behind Fig. 3 and check them against the
manuscript. They did not match for the photon channel. You then said to fix the cut and
update all affected numbers. Both papers gate 9/9; `photon_vs_electron.pdf` 28 pp, 23/23
checks at both redshifts, 0 runtime warnings.

### The defect

The photon integral's support starts at `log10(E_TH_HI)`, and `10**log10(E_TH_HI)` is
**one ULP below** `E_TH_HI`. The exact `>=` tests in `photon_pdf` and `N_gamma` therefore
returned **zero at the first grid point**. Because the ionizing spectrum is steepest at
threshold, the dropped trapezoid element was the *largest* one: **0.35% of ζ_γ**, in every
number published between 2026-09-07 and 2026-09-13.

It was provable rather than arguable: the integral of Fig. 3(a) has a **closed form**
(ṅ_γ = f_esc ξ_ion ρ_UV = 2.51188643×10⁴⁹). The trapezoid gave 2.50258×10⁴⁹ — 0.37% low.
Nudging the lower limit one part in 10¹² off the edge reproduced the closed form to every
printed digit.

### What changed

| | before | after |
|---|---|---|
| ζ_γ | 4.631×10⁻¹⁸ | **4.647×10⁻¹⁸** |
| ζ_e | 8.299×10⁻²² | 8.299×10⁻²² (unchanged) |
| ζ_γ/ζ_e | 5579 | **5599** |
| decades | 3.747 | 3.748 |
| ε_CR f_e at parity | 5.579 | 5.599 |
| f_esc parity | 1.792% | 1.786% |
| ζ_γ t_H | 0.1036 | 0.1039 |
| ζ_γ t_rec | 0.06450 | 0.06472 |
| reionization shortfall | 25.16 | 25.07 |
| z = 20 ratio | 5728 | 5748 |

**No conclusion moved.** Photons still lead by 3.75 decades; parity still needs ε_CR f_e > 1.

### The fix, and the check that makes it permanent

Four gates now compare with a relative tolerance (`IY.at_or_above`, `IY.within_band`;
10⁻¹² of 13.6 eV is 1.4×10⁻¹¹ eV). A fifth latent instance in `ionization_yield_igm.py` was
hardened too — not triggered today, but the same pattern.

New check **C57** guards it by a route the code cannot argue with. The test is *not* "is the
residual small" — a 9-decade support at n=400 has a genuine 3×10⁻⁴ truncation error, and a
tolerance loose enough to accept that would also accept the bug. The test is **how the
residual scales**: trapezoid truncation falls as h², a dropped band edge only as h. Measured
after the fix: **order 2.00** both channels. With the bug reinstated: **order 1.00**. I
verified that discriminator empirically before relying on it.

### Three defects my own fix introduced, all caught before publishing

1. Admitting the sub-ULP point made `T_sec = E − E_th` **negative** (−1.8×10⁻¹⁵), which
   reached a `log` and raised a RuntimeWarning. Clamped at zero — physically exact, since
   the photoelectron has zero kinetic energy at threshold.
2. `T_sec = 0` then gave `log(0)`. `N_secondary` now clamps before the log; the clamp only
   touches entries a later `where()` already zeroes, so no value changed.
3. Raising `zeta_total`'s default n from 400 to 4001 desynchronised **C26**, which compares
   ζ by two code paths — one still had `400` hard-coded, so the check failed on a grid
   mismatch rather than on the physics it tests. Both now share one constant, `ZETA_N`.

The resolution was raised because n=400 left an 8×10⁻⁵ truncation error sitting on a rounding
boundary: the ratio came out 5599.54 and printed as **5600**, while the converged value is
5599.14 and prints as **5599**. It costs 0.6 s.

### Also mine

`sync_literals.py` strips trailing zeros, which silently cost six literals their significant
figures (5.600 → "5.6", 3.760 → "3.76", 1.740 → "1.74"). The gate passed them — it compares
at whatever precision is displayed — so this is a presentation regression a gate cannot see.
Restored by hand.

### Standing lesson

Every check in this project compared the code against itself, which is why a 0.35% error
survived six days and two independent-route validations. **C57 is the first check that
compares an integral against a closed form.** Where one exists, use it.

## Round 15 — 2026-09-14 (bibliography consolidation)

**Standing decision (yours):** `papers/references.bib` is the project's single
bibliography, and **every paper downloaded from now on is added to it as part of the same
piece of work**. Recorded in memory so it survives sessions.

### What was merged

Two bib files existed: yours at `papers/references.bib` (41 entries, `Author+Year` keys)
and one I had created at the project root (27 entries, short lowercase keys). Merged into
yours, which keeps the convention and the richer metadata. **59 entries, zero duplicates**
— verified four ways, by DOI, arXiv eprint, normalised title, and first-author+year.

Eleven entries were the same paper under two keys. Yours survived in every case, because
they carried DOIs and full page ranges: Blumenthal1970, Furlanetto2010, Gould1972,
Kim2000, KimRudd1994, Leite2017, Pawlik2009, Planck2020, Sazonov2015, Shull1985, Stone2002.
Eighteen entries were unique to mine and were converted to your convention and added.

All 15 PDFs in `papers/` now have a bib entry. The root `references.bib` was fully
contained in the merge and has been deleted (backup in `.bak-z20/`).

### A wrong citation the merge exposed

Only two of my entries failed the containment check — and both failed because **my titles
were wrong**:

- **`Gould1972`**: the paper is "Energy loss of fast electrons **and positrons** in a
  plasma". I had dropped "and positrons".
- **`Stone2002`**: I had it as "Electron-impact excitation cross sections, J. Phys. B 35,
  1675". **That journal is wrong.** Resolving your DOI gives
  **Stone, Kim & Desclaux (2002), J. Res. NIST 107, 327–337, doi:10.6028/jres.107.026** —
  three authors, different journal, different volume and page.

I had copied the bad reference from the header of `igm_losses.py` (line 25), which has
carried it since the module was written. **Corrected in the code header**, with the DOI
and a note saying what it used to say. This matters: excitation is not a minor channel —
it reaches 0.353 of dK/dt at 100 eV and is the largest irrecoverable energy sink — and it
is one of the prescriptions still marked `C` (cited, never verified against its paper) in
`inputs_table.tex`.

### Knock-on

`make_inputs_table.py` had 42 citation keys remapped to the new convention and now cites
`\bibliography{papers/references}`. `inputs_table.pdf` rebuilds with **0 undefined
citations** and 22 references. The two main papers are unaffected: they use embedded
`\bibitem` lists, so their keys are internal.
