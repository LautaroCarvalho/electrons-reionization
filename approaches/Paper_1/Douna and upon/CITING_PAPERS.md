# Papers citing Douna et al. (2018)

**Cited paper:** Douna, V. M., Pellizza, L. J., Laurent, P., & Mirabel, I. F. (2018),
*Microquasars as heating sources of the intergalactic medium during reionization of the Universe*,
MNRAS **474**, 3488–3499. DOI: 10.1093/mnras/stx2983 — arXiv:1711.07374 — bibcode 2018MNRAS.474.3488D.

**Retrieved:** 2026-09-07 (this session).

## Sources queried
The citation list was first built as the **union** of four open citation indices, then corrected
against NASA ADS (token supplied 2026-09-08):

| Index | Endpoint | Citations found |
|---|---|---|
| Semantic Scholar Graph API | `DOI:10.1093/mnras/stx2983` and `ARXIV:1711.07374` | 12 |
| OpenAlex | `filter=cites:W2770391234` | 17 records → 14 unique works |
| OpenCitations COCI (Crossref-based) | `citations/10.1093/mnras/stx2983` | 13 |
| INSPIRE-HEP | `refersto recid 1637156` | 12 |
| **NASA ADS** (authoritative) | `citations(bibcode:2018MNRAS.474.3488D)` | **18** |

**Union of the four open indices: 14 unique citing works. ADS: 18 (see update below).** All 14 have been downloaded (13 from arXiv, 1 open-access
PDF from Cambridge University Press). All 14 are astro-ph papers.

**UPDATE 2026-09-08 — corrected against NASA ADS.** ADS reports **18** citing papers, not 14.
The four open indices jointly missed four *Boletin de la Asociacion Argentina de Astronomia*
proceedings that have no DOI and no arXiv posting, so no Crossref-derived index carries them.
All four are now downloaded into this folder:

| Bibcode | Paper |
|---|---|
| `2021BAAA...62..234G` | Garate Nunez, Escobar, Pellizza & Bosch-Ramon (2021) — *Reionizacion por jets de nucleos galacticos activos* |
| `2021BAAA...62..265E` | Escobar, Pellizza & Romero (2021) — *Microcuasares como fuentes de rayos cosmicos* |
| `2024BAAA...65..240C` | Carvalho, Escobar & Pellizza (2024) — *Cosmic rays at the epoch of reionization* |
| `2025BAAA...66..377P` | Pellizza, Badaracco, Carvalho, Escobar & Bignone (2025) — *X-ray binary populations in local galaxies* |

None of the four is cosmic-ray-electron-contradictory; the last two are your own group's
continuation of the line. Recall lesson: the open indices had complete recall on refereed,
DOI-bearing literature and zero recall on Spanish-language national proceedings.

## Cosmic-ray relevance flags

"CR mentions" = occurrences of "cosmic ray(s)" in the full text (hyphenation-normalized),
counted with `pdftotext`. Judgement column is from title/abstract content.

| # | File | Year | arXiv / DOI | CR mentions | Cosmic-ray relevance |
|---|---|---|---|---|---|
| 1 | Escobar (2021) - Cosmic-ray production from neutron escape in microquasar jets | 2021 | 2104.11975 / 10.1051/0004-6361/202039860 | 35 | **CORE** — CR production by microquasar jets |
| 2 | Escobar (2022) - Highly collimated microquasar jets as efficient cosmic-ray sources | 2022 | 2207.08633 / 10.1051/0004-6361/202142753 | 29 | **CORE** — microquasars as CR sources |
| 3 | Sotomayor Checa (2019) - A model for microquasars of Population III | 2019 | 1906.05184 / 10.1051/0004-6361/201834191 | 5 | **STRONG** — relativistic particle acceleration in Pop III microquasar jets |
| 4 | Bosch-Ramon (2018) - The role of AGN jets in the reionization epoch | 2018 | 1808.08911 / 10.1051/0004-6361/201833952 | 5 | **STRONG** — CR/non-thermal particle heating of the IGM at high z |
| 5 | Mirabel (2019) - Black Hole High Mass X-ray Binary Microquasars at Cosmic Dawn | 2019 | 1902.00511 / 10.1017/S1743921319002084 | 3 | **MODERATE** — review, CRs as one heating channel at cosmic dawn |
| 6 | Sazonov (2018) - Impact of ultraluminous X-ray sources on photoabsorption in the first galaxies | 2018 | 1712.04831 / 10.1093/mnras/sty442 | 1 | PASSING |
| 7 | Svoboda (2019) - Green Peas in X-rays | 2019 | 1810.09318 / 10.3847/1538-4357/ab2b39 | 1 | PASSING |
| 8 | Soria (2024) - A multiband look at ultraluminous X-ray sources in NGC 7424 | 2024 | 2402.09512 / 10.1093/mnras/stae551 | 1 | PASSING |
| 9 | Ross (2019) - Evaluating the QSO contribution to the 21-cm signal from the Cosmic Dawn | 2019 | 1808.03287 / 10.1093/mnras/stz1220 | 0 | none |
| 10 | Mirabel (2022) - Black holes at cosmic dawn in the redshifted 21cm signal of HI | 2022 | 2203.12741 / 10.1016/j.newar.2022.101642 | 0 | none |
| 11 | Plotkin (2019) - Radio Variability from a Quiescent Stellar Mass Black Hole Jet | 2019 | 1901.07776 / 10.3847/1538-4357/ab01cc | 0 | none |
| 12 | Artale (2019) - The High Mass X-ray Binaries in star-forming galaxies | 2019 | 1811.06291 / 10.1017/S1743921318007627 | 0 | none |
| 13 | Soria (2021) - The ultraluminous X-ray source bubble in NGC 5585 | 2021 | 2012.03970 / 10.1093/mnras/staa3784 | 0 | none |
| 14 | Sell (2019) - The X-ray binary populations of M81 and M82 | 2019 | (no arXiv) / 10.1017/S1743921318008190 | 0 | none |

Note: entries 9–14 return 0 hits for the literal string "cosmic ray", but several of them
(Ross, Mirabel 2022) still discuss non-thermal IGM heating; the flag is a text-based
indicator, not a substitute for reading the paper.

## Files
- PDFs are named `FirstAuthor (Year) - Title [arXiv id].pdf` in this folder.
- `citing_papers.bib` — BibTeX for all 14 citing papers plus Douna et al. (2018),
  fetched via Crossref DOI content negotiation.
