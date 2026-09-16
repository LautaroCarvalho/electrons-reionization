# `papers/` — sources cited by `photon_vs_electron.tex`

Every PDF here was fetched from the URL given below and **verified against its own
title page** before being recorded (the filename is not the evidence; the title page
is). Text extracts (`*.txt`) are `pdftotext` output of the neighbouring PDF, kept so
the numbers quoted in the papers can be grepped without reopening a viewer.

Fetched 2026-09-12.

| bibitem | reference | file | source URL |
|---|---|---|---|
| `donnan24`  | Donnan et al. 2024, MNRAS, JWST PRIMER — arXiv:2403.03171 | `2403.03171-donnan-primer-uvlf.pdf` (+ `primer.txt`) | arXiv (fetched 2026-09-10) |
| `llerena25` | Llerena et al. 2025, A&A **698**, A302 — arXiv:2412.01358v2 | `2412.01358-llerena-xi-ion-z4-10.pdf` (+ `llerena.txt`) | `https://arxiv.org/pdf/2412.01358` |
| `md14`      | Madau & Dickinson 2014, ARA&A **52**, 415 — arXiv:1403.0007v3 | `1403.0007-madau-dickinson-sfh-review.pdf` (+ `madau-dickinson.txt`) | `https://arxiv.org/pdf/1403.0007` |
| `leite17`   | Leite, Evoli, D'Angelo, Ciardi, Sigl & Ferrara 2017, MNRAS **469**, 416 — arXiv:1703.09337v1 | `1703.09337-leite-cosmic-rays-heat-igm.pdf` (+ `leite.txt`) | `https://arxiv.org/pdf/1703.09337` |
| `ss15`      | Sazonov & Sunyaev 2015, MNRAS **454**, 3464 — arXiv:1509.08408v1 | `1509.08408-sazonov-sunyaev-cr-preheating.pdf` (+ `sazonov-sunyaev.txt`) | `https://arxiv.org/pdf/1509.08408` |
| `mc26`      | Marques-Chaves et al. 2026, A&A — arXiv:2602.02322 | `2602.02322-marques-chaves-u37126-z10.pdf` (+ `u37126.txt`) | arXiv (fetched 2026-09-10) |
| `salpeter`  | Salpeter 1955, ApJ **121**, 161 | `1955ApJ-121-161-salpeter-imf.pdf` (+ `salpeter.txt`) | `https://articles.adsabs.harvard.edu/pdf/1955ApJ...121..161S` (NASA ADS scanned-article service) |

Also in the folder, cited by the companion `ionization_yield.tex`:

| bibitem | reference | file | source |
|---|---|---|---|
| `fs10` | Furlanetto & Stoever 2010, MNRAS **404**, 1869 — arXiv:0910.4410 | `0910.4410-furlanetto-stoever-2010.pdf` (+ `fs10.txt`, `fs10_layout.txt`) | arXiv (fetched 2026-09-07) |

## Added 2026-09-12 for the reionization-budget section

| bibitem | reference | file | source URL |
|---|---|---|---|
| `hg97` | Hui & Gnedin 1997, MNRAS **292**, 27 — arXiv:astro-ph/9612232 | `astro-ph.9612232-hui-gnedin-1997-igm-eos.pdf` (+ `hui-gnedin.txt`) | `https://arxiv.org/pdf/astro-ph/9612232` |
| `pawlik09` | Pawlik, Schaye & van Scherpenzeel 2009, MNRAS **394**, 1812 — arXiv:0807.3963 | `0807.3963-pawlik-schaye-clumping-factor.pdf` (+ `pawlik.txt`) | `https://arxiv.org/pdf/0807.3963` |

Both were fetched to source the factors of Madau & Dickinson eq. (24):

- **Hui & Gnedin** supply the case-B α_B(T) fit. Their coefficients were read off
  the page — `2.753e-14`, `0.407`, `2.242`, `T_HI = 157807 K` — and used to confirm
  α_B(2×10⁴ K) = 1.43×10⁻¹³ cm³/s independently (0.16%), and α_B(10⁴ K) = 2.59×10⁻¹³
  (0.07%). Two temperatures, so the fit is anchored rather than merely consistent.
- **Pawlik, Schaye & van Scherpenzeel** are where MD14's `C_IGM = 1 + 43 z^-1.71`
  comes from — though **they never print that expression**. Their eq. (A1) is
  `C(z) = z^β e^(−γz+δ) + α`, and Table A1 gives α = 1.00, β = −1.71, γ = 0.00,
  δ = 3.76 for `C_100` in the `r19.5L6N256` run. So the "43" is **e^3.76 = 42.95**.
  Two consequences the shorthand hides: it is specifically `C_100` (gas *below*
  overdensity 100, the less demanding choice — their `C_-1` is ~2× larger at z = 6),
  and its **fitted range is 6 ≤ z ≤ 20**, which puts z = 5.5 outside it.

## The two APS papers — now supplied by the user (2026-09-12)

Both were provided directly and verified against their own title pages
(`PHYSICAL REVIEW A VOLUME 50, NUMBER 5` and `PHYSICAL REVIEW A, VOLUME 62,
052710`). They were renamed from `Kim (1994).pdf` / `Kim (2000).pdf` so the
provenance gate's path check (C8) can resolve them from the LaTeX.

| bibitem | reference | file |
|---|---|---|
| `kimrudd94` | Y.-K. Kim & M. E. Rudd 1994, *Phys. Rev. A* **50**, 3954 — `10.1103/PhysRevA.50.3954` | `PhysRevA.50.3954-kim-rudd-1994-bed.pdf` (+ `kim1994.txt`) |
| `kim00` | Y.-K. Kim, J. P. Santos & F. Parente 2000, *Phys. Rev. A* **62**, 052710 — `10.1103/PhysRevA.62.052710` | `PhysRevA.62.052710-kim-santos-parente-2000-rbed.pdf` (+ `kim2000.txt`) |

**The gap they closed.** Until they arrived, the collisional cross sections in
`igm_losses.py` had never been checked against the published equations. They
now have been, by independent reimplementation from the paper text
(`verify_kim_and_z.py`, 13/13 checks, registry
`provenance/audit_registry.json`). The result is reported in
`photon_vs_electron.pdf` §"Audit against the sources". In one line: the code
implements **Kim 2000 eq. (20), the RBED cross section**, to 1.4×10⁻⁹ — and the
project had been calling it **RBEB**, which is eq. (22), a different model
differing by up to tens of per cent. The physics was right; the name was not,
and it has been corrected throughout.

## Provenance checks performed while the papers were open

These are checks of the paper's own numbers against the sources, run at download time:

- **`K_UV = 1.15e-28`** — confirmed verbatim in Madau & Dickinson 2014, their
  Equation 10 and the captions of their Figures 8 and 9: "KFUV = 1.15 × 10⁻²⁸ … valid
  for a Salpeter IMF", quoted there for a Salpeter IMF over **0.1–100 M☉**. The project
  uses `M_MIN, M_MAX = 0.1, 100.0` with slope 2.35, so the conversion factor and the
  supernova-rate integral rest on the *same* IMF. Consistent.
- **`log ξ_ion = 25.28`** — confirmed in Llerena et al. 2025: the median of their sample
  at *z* ≈ 7.14, with an observed scatter of **0.42 dex** (both figures appear in their
  abstract). Two caveats the source states and the project should keep in view:
  their canonical comparison value is 25.27, and their own fit
  `log ξ_ion = (0.06 ± 0.012) z + (24.82 ± 0.07)` extrapolates to **25.42 at z = 10**.
  Using 25.28 at z = 10 is therefore *conservative for the photon channel* — it makes
  stellar UV look weaker than their fit would, so the photons-win conclusion is not an
  artefact of this choice.

---

### `1102.1891-mirabel-stellar-bh-dawn-2011.pdf`
Mirabel, Dijkstra, Laurent, Loeb & Pritchard (2011), A&A **528**, A149 —
*Stellar black holes at the dawn of the Universe*. Bib key `Mirabel2011`
(the entry predated the PDF; downloaded 2026-09-14 for the source-competition
figures). Numbers taken from it:

- **Conclusion 3:** an accreting BH in a high-mass binary emits a total number of
  ionizing photons *comparable to its progenitor star*, but one X-ray photon can
  ionize *several tens* of H atoms in a fully neutral medium. → the scaling used to
  place the microquasar box's photon channel.
- **§6 / conclusion 2:** SS 433 injects `>1e39 erg/s` into the ISM and the S26
  microquasar `>1e40 erg/s`; over a lifetime `>1e54 erg`, "orders of magnitude more
  than the photonic and baryonic energy from a typical core collapse supernova."
  → the microquasar box's *mechanical* power. NOTE this is total injected power, **not**
  the relativistic-electron share; the electron fraction is scanned, not assumed.
- **Footnote 1 (p. 4):** Cygnus X-1 as the moderate-accretion template — a UV/soft-X-ray
  bump from the disk at `kT ≈ 7 eV` plus a non-thermal hard-X-ray Comptonised power law
  from the corona and/or jet. ULX spectra break above `3 keV` (Gladstone, Roberts & Done 2009).
- **§6:** "the IGM at z = 10 is essentially transparent to the hard X-ray photons (> 1 keV)."
  → independently reproduced with this project's own `sigma_photoion`: over one Hubble
  length at z = 10, τ = 1.93 at 1 keV, 0.048 at 3 keV, 7.7e-4 at 10 keV. This is the
  basis for applying the `(1 - exp(-τ))` absorption weighting to the photon channel.
- **Eq. (6):** `f_X = 4.0 (f_2-10/0.1)(f_BH/0.01)(f_Edd/0.1)(f_bin/0.5)(t_acc/20 Myr)`.

### Not yet in the tree, cited from verified article pages
- `Telfer2002` — ApJ 565, 773. AGN EUV index, verified from the abstract page.
- `Lehmer2016` — ApJ 825, 7, eq. (6). HMXB L_X/SFR vs z, verified from the article page.

### `0905.4076-gladstone-roberts-done-ultraluminous-state-2009.pdf`
Gladstone, Roberts & Done (2009), MNRAS **397**, 1836 — *The ultraluminous state*.
Bib key `Gladstone2009`. Values read from the PDF:

- **Table 6 / §4.3:** a broken power law beats a single power law at `>98%` in **11 of 12**
  ULXs; break energies **3.5–7 keV**, steepening **ΔΓ ≈ 1–2**; `Γ₁ ≈ 1.38–3.1` (typical 2.1).
- **§4.1:** single power-law fits span `1.6 < Γ < 3.3`; `Γ < 2.1` corresponds to the
  low/hard state (citing McClintock & Remillard 2006) — the only first-hand constraint
  this project has on the BHB hard-state index.
- **§4.5.1 / Table 8:** preferred physical model is a **cool, optically thick** corona,
  `kT_e ~ 1–3 keV`, `τ ~ 6–80` (global χ² minimum in 10/12), versus a hot thin corona
  `kT_e ~ 50 keV`, `τ ≲ 1` as only a *local* minimum.
- **CAUTION:** fits are over **2–10 keV only**. The 13.6 eV–2 keV ionizing band is *not*
  constrained here; the cool disc (`kT_in ~ 0.17–1.7 keV`, Table 5) dominates there.
- **CAUTION:** `Mirabel2011` paraphrases the break as "above 3 keV"; the fitted range is
  3.5–7 keV. A citing paper reports "2–7 keV". Use the primary.

### AGN contribution to the ionizing budget — the two primaries
Both read from their abstract pages; **they agree**, contrary to a widely-circulated
"36–88%" figure that comes from a separate broad-line census with far larger systematics.

- `Jiang2025` — Jiang, Jiang, Sun, Liu & Fu, *Nature Astronomy* (2025). Point-source vs
  extended decomposition of JWST FUV imaging down to `M_UV ≈ −15`, over `7.15 ≤ z ≤ 7.75`:
  AGN supply **at most one third** of the LyC budget at z ~ 7.5. → upper edge of the AGN box.
- `Asthana2024` — Asthana, Haehnelt, Kulkarni, Bolton, Gaikwad, Keating & Puchwein.
  QSO-assisted model: AGN supply **17%** of the total hydrogen-ionizing emissivity.
  → lower edge of the AGN box.
- Both measured at z ~ 7.5. Carrying them to z = 10 is an extrapolation, and a
  **conservative** one for the photon channel: the AGN LF declines faster than the galaxy LF.

### `2602.02322-marques-chaves-u37126-z10.pdf` — additional value extracted
`M_UV = −20.10 ± 0.05` (AB) for U37126, `z = 10.255`, mildly lensed `μ ≈ 2.2`.
The project's own anchor galaxy, so it supplies the per-object star-forming point of the
source-competition figures without any new source.
