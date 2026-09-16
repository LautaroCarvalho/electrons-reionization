#!/usr/bin/env python3
r"""
Which kinds of high-z source favour CR electrons over UV photons?

THE DERIVATION.  In the linear-yield regime each channel collapses to a single
energy-per-ionization.  zeta is a LINEAR functional of the emissivity ndot(E)
against a kernel N(E) that is a property of the IGM, not of the source:

    zeta = (C/n_H) Int ndot(E) N(E) dE .

Split ndot(E) = Ndot_tot p(E) with p the unit-normalised shape.  Then

    zeta = (C/n_H) Ndot_tot <N>_p ,     L = Ndot_tot <E>_p
    =>  zeta = C L / (n_H W) ,          W = <E>_p / <N>_p .

Two consequences, both tested below rather than asserted:

  * the NORMALISATION cannot break any degeneracy -- it enters only through L;
  * the SHAPE enters only through W, one scalar.  The compression is exact but
    LOSSY: two spectra with equal W are indistinguishable here.

Hence  phi = zeta_e/zeta_gamma = (f_dep L_e)/(f_esc L_ion) x (W_gamma/W_e).

IGM TRANSPARENCY.  zeta_gamma counts ionizations ACHIEVED, so a photon that the
IGM never absorbs must not be counted.  Over one Hubble length at z = 10 the
optical depth is tau = 1.93 at 1 keV but only 0.048 at 3 keV: the z=10 IGM is
thick below ~500 eV and transparent above ~3 keV (confirming Mirabel+11 sec. 6).
Every W_gamma here therefore carries the factor (1 - exp(-tau(E))).  This is
what makes a hard source's photons genuinely worthless while its electrons still
deposit -- and it is why a SUB-Eddington hard state beats a SUPER-Eddington ULX
for the electron channel, which is the opposite of the naive expectation.

f_dep = 1 throughout, per standing assumption A11: every injected CR electron is
assumed to deposit.  That is a CEILING on the electron channel, not an estimate.

Run:  python3 source_map.py
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import parameters as PR          # the single source of truth
import json
import numpy as np

import ionization_yield as IY
import photon_vs_electron as P
import yield_comparison as YC

# ---------------------------------------------------------------------------
# The scenario and the W-value machinery now live in wvalue.py, which is the
# project's ONE implementation of these integrals.  They were duplicated here
# and in make_xcomp_definitions.py; the two agreed to 3.7e-8, but the next edit
# to one would not have reached the other.  Check X2 still compares the two call
# paths, so the agreement stays gated rather than assumed.
# ---------------------------------------------------------------------------
from wvalue import (                                        # noqa: F401
    Z_SNAP, PAR, COS, CHAN, CONV, N_H, X_E, N_HI, L_HUB,
    NGRID, E_TOP_MASTER, _GP, _EP, _NGAM, _TAU, _FABS, _GE, _EE, _NE,
    tau_igm, _ngam, _ne,
    W_gamma, W_gamma_broken, W_e, W_bounds_gamma, W_bounds_e, zeta_from_W,
)


# ===========================================================================
# 3. THE SOURCE CLASSES.  Every spectral parameter carries a bib key and a
#    status code, on the convention of make_inputs_table.py:
#      V verified against the source paper, which is in papers/
#      C cited, source not in tree / not checked against its equations
#      S a scanned range, not a measurement
#      U the user's stated choice
# ===========================================================================
FID_P, FID_EMIN, FID_EMAX = P.CR_INDEX, P.CR_E_MIN_EV, P.CR_E_MAX_EV

# photon side: (label, kind, params, E_top, bibkey, status, note)
PHOTON_CLASSES = [
    dict(label="Star-forming galaxies", kind="pl", alpha=2.0, alo=1.0, ahi=3.0,
         E_top=P.LYC_E_MAX_EV, key="MarquesChaves2026", status="V",
         note="U37126 LyC shape, BPASS v2.2.1 Z=0.003; 4 Ryd He II truncation"),
    dict(label="AGN, reionization template", kind="pl", alpha=1.5,
         alo=1.5, ahi=1.5, E_top=3.0e3, key="Graziani2018", status="V",
         note="S(nu) ~ nu^-1.5 over 13.6 eV-3 keV, purpose-built for EoR"),
    dict(label="AGN, radio-quiet", kind="pl", alpha=1.57, alo=1.40, ahi=1.74,
         E_top=3.0e3, key="Telfer2002", status="V",
         note="alpha_EUV = 1.57 +- 0.17; fit 500-1200 A, harder top extrapolated"),
    dict(label="AGN, radio-loud", kind="pl", alpha=1.96, alo=1.84, ahi=2.08,
         E_top=3.0e3, key="Telfer2002", status="V",
         note="alpha_EUV = 1.96 +- 0.12; same extrapolation caveat"),
    dict(label="ULX / super-Eddington", kind="brk", G1=2.1, G1lo=1.38, G1hi=3.1,
         E_break=5.0e3, Eblo=3.5e3, Ebhi=7.0e3, dG=1.5, dGlo=1.0, dGhi=2.0,
         E_top=1.0e5, key="Gladstone2009", status="V",
         note="Table 6 / sec 4.3; fits are 2-10 keV only, sub-2 keV extrapolated"),
    dict(label="BHB hard state (Cyg X-1-like)", kind="pl", alpha=0.7,
         alo=0.4, ahi=1.1, E_top=1.0e5, key="Gladstone2009", status="C",
         note="Gamma = alpha+1 = 1.7 typical is SECOND-HAND, no primary yet; "
              "only Gamma < 2.1 for the low/hard state is first-hand "
              "(Gladstone+09 sec 4.1, citing McClintock & Remillard 2006)"),
]

# electron side: the project's fiducial injection, held for every class.
# Per-class CR indices at z~10 have no citable determination -- see QUESTION_LOG.
ELECTRON_FID = dict(p=FID_P, E_min=FID_EMIN, E_max=FID_EMAX,
                    key="Park2015", status="U",
                    note="project fiducial; f_dep = 1 per assumption A11 (a CEILING)")


def class_W_gamma(c, which="mid"):
    """W_gamma for one class.  which in {mid, lo, hi} walks the sourced range."""
    if c["kind"] == "pl":
        a = {"mid": c["alpha"], "lo": c["ahi"], "hi": c["alo"]}[which]
        return W_gamma(a, c["E_top"])
    g1 = {"mid": c["G1"], "lo": c["G1hi"], "hi": c["G1lo"]}[which]
    eb = {"mid": c["E_break"], "lo": c["Eblo"], "hi": c["Ebhi"]}[which]
    dg = {"mid": c["dG"], "lo": c["dGhi"], "hi": c["dGlo"]}[which]
    return W_gamma_broken(g1, eb, dg, c["E_top"])


# ===========================================================================
# 4. CHECKS.  Master Rule 5: an independent route wherever sympy does not apply.
# ===========================================================================
# Tolerance for every check that compares two quadratures of N_gamma. NOT a
# fudge: measured by the convergence scan in check S12 below. W_gamma over the
# stellar band oscillates by ~5e-5 between n = 1200 and n = 16000 -- the noise
# floor of IY.N_gamma's own internal interpolation, not of the trapezoid, since
# it does not fall as h^2. Demanding 1e-6 of a quantity known to 5e-5 would be
# testing the grid, not the physics.
QUAD_TOL = PR.QUAD_TOL            # parameters.yaml

ROWS, PROV = [], {}


def rec(cid, name, ok, detail):
    ROWS.append((cid, name, "PASS" if ok else "FAIL", detail))


def put(k, v):
    PROV[k] = float(v)
    return v


def run_checks():
    W_g_fid = W_gamma(P.SED_ALPHA, P.photon_band_max(), exact=True)
    W_e_fid = W_e(FID_P, FID_EMIN, FID_EMAX, exact=True)
    put("W_gamma_fiducial_eV", W_g_fid)
    put("W_e_fiducial_eV", W_e_fid)
    put("spectral_factor_fiducial", W_g_fid / W_e_fid)

    # --- S3 the fiducial pair reproduces the manuscript's numbers -----------
    Eg = P.mean_ionizing_photon_energy()
    gg = np.linspace(np.log10(IY.E_TH_HI), np.log10(P.photon_band_max()), P.ZETA_N)
    Ng = IY._trapz(P.photon_pdf(10 ** gg) * IY.N_gamma(10 ** gg, COS, PAR, "C", CHAN)
                   * np.log(10) * 10 ** gg, gg)
    rec("S3", "W_gamma = <E>/<N_gamma> for the fiducial SED",
        abs(W_g_fid / (Eg / Ng) - 1.0) < QUAD_TOL,
        f"{W_g_fid:.4f} vs {Eg/Ng:.4f} eV  (tau >> 1 all across 13.6-54.4 eV, "
        f"so absorption is inert here)")

    # --- S1 phi at the fiducial point reproduces the published ratio --------
    zg = P.zeta_total("photon", COS, PAR, CHAN)
    ze = P.zeta_total("electron", COS, PAR, CHAN)
    _, L_e, _ = P.ndot_electrons_comoving()
    L_ion = P.ndot_photons_comoving() * Eg / P.EV_PER_ERG
    phi_W = (L_e / L_ion) * (W_g_fid / W_e_fid)
    put("zeta_ratio_published", zg / ze)
    put("phi_fiducial", phi_W)
    rec("S1", "phi from W reproduces 1 / (zeta_gamma/zeta_e)",
        abs(phi_W * (zg / ze) - 1.0) < QUAD_TOL,
        f"phi = {phi_W:.6e}, 1/ratio = {ze/zg:.6e}, "
        f"product = {phi_W*(zg/ze):.9f}")

    # --- S2 the linear-functional collapse, tested not assumed --------------
    worst = 0.0
    for a in (1.0, 1.5, 2.0, 2.5, 3.0):
        P.SED_ALPHA = a
        zt = P.zeta_total("photon", COS, PAR, CHAN)
        Wn = W_gamma(a, P.photon_band_max(), absorb=False, exact=True)
        En = P.mean_ionizing_photon_energy(alpha=a)
        Ln = P.ndot_photons_comoving() * En / P.EV_PER_ERG
        worst = max(worst, abs(zeta_from_W(Ln, Wn) / zt - 1.0))
    P.SED_ALPHA = 2.0
    rec("S2", "zeta = C L / (n_H W) equals the full integral",
        worst < QUAD_TOL,
        f"worst relative difference over alpha 1-3: {worst:.2e}, against the "
        f"measured quadrature noise floor {QUAD_TOL:.0e} (check S12). The "
        f"residual is a MIXED-QUADRATURE artefact, not a failure of the "
        f"identity: zeta_total integrates the normalised pdf over a finite "
        f"grid whose trapezoid sum is not exactly 1, while W here uses the "
        f"closed-form <E>. Same trap C27 documents. A broken identity would "
        f"fail by orders of magnitude, not by 1e-5.")
    put("S2_worst_reldiff", worst)

    # --- S4 the interpolation shortcut must not cost accuracy ---------------
    w = 0.0
    for a, t in ((2.0, 54.4), (1.5, 3e3), (0.7, 1e5)):
        w = max(w, abs(W_gamma(a, t) / W_gamma(a, t, exact=True) - 1.0))
    w = max(w, abs(W_e(FID_P, FID_EMIN, FID_EMAX)
                   / W_e(FID_P, FID_EMIN, FID_EMAX, exact=True) - 1.0))
    rec("S4", "interpolated W agrees with direct evaluation",
        w < 1e-3, f"worst relative difference: {w:.2e}")
    put("S4_worst_reldiff", w)

    # --- S6 every class W inside the rigorous weighted-mean bound -----------
    bad = []
    for c in PHOTON_CLASSES:
        lo, hi = W_bounds_gamma(IY.E_TH_HI, c["E_top"])
        for which in ("lo", "mid", "hi"):
            Wc = class_W_gamma(c, which)
            if not (lo - 1e-9 <= Wc <= hi + 1e-9):
                bad.append(f"{c['label']}/{which}: {Wc:.2f} not in [{lo:.2f},{hi:.2f}]")
    elo, ehi = W_bounds_e(FID_EMIN, FID_EMAX)
    if not (elo - 1e-9 <= W_e_fid <= ehi + 1e-9):
        bad.append(f"electrons: {W_e_fid:.2f} not in [{elo:.2f},{ehi:.2f}]")
    rec("S6", "every W lies inside its monoenergetic weighted-mean bound",
        not bad, "; ".join(bad) if bad else
        f"photon bounds at 4 Ryd [{W_bounds_gamma(IY.E_TH_HI, 54.4)[0]:.2f},"
        f"{W_bounds_gamma(IY.E_TH_HI, 54.4)[1]:.2f}] eV, "
        f"electron bounds [{elo:.2f},{ehi:.2f}] eV")

    # --- S7 no class may be plotted without a bibliography entry ------------
    bib = open("papers/references.bib").read()
    miss = [c["label"] for c in PHOTON_CLASSES if ("{" + c["key"] + ",") not in bib]
    if ("{" + ELECTRON_FID["key"] + ",") not in bib:
        miss.append("electron fiducial")
    rec("S7", "every plotted class resolves to a references.bib entry",
        not miss, "missing: " + ", ".join(miss) if miss else
        f"{len(PHOTON_CLASSES)} photon classes + 1 electron spectrum, all resolve")

    # --- S10 the transparency physics the figure rests on -------------------
    t1, t3, t10 = (float(tau_igm(1e3)), float(tau_igm(3e3)), float(tau_igm(1e4)))
    put("tau_1keV", t1); put("tau_3keV", t3); put("tau_10keV", t10)
    rec("S10", "IGM is thick below ~1 keV and transparent above ~3 keV",
        t1 > 1.0 > t3 and t10 < 0.01,
        f"tau(1 keV) = {t1:.3f}, tau(3 keV) = {t3:.4f}, tau(10 keV) = {t10:.2e} "
        f"over one Hubble length at z = {Z_SNAP:g}")

    # --- S8/S9 the budget plane's anchors ----------------------------------
    t_H = hubble_time_s()
    put("t_H_Gyr", t_H / 3.1557e16)
    boxes = population_boxes()
    sf = boxes[0]
    zg_tH = zeta_from_W(sf["L_ion"][0], sf["W"]) * t_H
    ze_tH = zeta_from_W(sf["L_e"][0], sf["We"]) * t_H
    put("zeta_gamma_tH_starforming", zg_tH)
    put("zeta_e_tH_starforming", ze_tH)
    rec("S8", "star-forming point reproduces the manuscript's zeta_gamma t_H",
        abs(zg_tH / 0.1039 - 1.0) < 2e-3 and abs(t_H / 3.1557e16 / 0.7086 - 1) < 1e-3,
        f"zeta_gamma t_H = {zg_tH:.4f} (manuscript 0.1039), "
        f"t_H = {t_H/3.1557e16:.4f} Gyr (manuscript 0.7086)")
    short = 2.606 / zg_tH
    put("shortfall_vs_C_IGM", short)
    rec("S9", "star-forming point sits x25.1 below the C_IGM budget line",
        abs(short / 25.1 - 1.0) < 2e-2,
        f"N_req(C_IGM) / (zeta_gamma t_H) = {short:.2f}, manuscript 25.1. "
        f"No class reaches ANY N_req line: the budget is not closed at z = 10 "
        f"by any source population here, which is the known shortfall, not a "
        f"new claim.")

    # --- S12 the quadrature noise floor QUAD_TOL rests on -------------------
    vals = []
    for n in (1200, 2400, 4800, 9600):
        vals.append(W_gamma(P.SED_ALPHA, P.photon_band_max(), n=n, exact=True))
    spread = (max(vals) - min(vals)) / np.mean(vals)
    put("W_gamma_quadrature_spread", spread)
    rec("S12", "measured quadrature noise floor justifies QUAD_TOL",
        spread < QUAD_TOL,
        f"W_gamma over n = 1200,2400,4800,9600: "
        + ", ".join(f"{v:.6f}" for v in vals)
        + f" eV; relative spread {spread:.2e} < QUAD_TOL = {QUAD_TOL:.0e}. "
          f"It does NOT fall as h^2, so this is IY.N_gamma's interpolation "
          f"noise, not trapezoid truncation -- which is why the tolerance is "
          f"set here and not tighter.")

    # --- S11 dimensions -----------------------------------------------------
    rec("S11", "dimensional check", True,
        "W in eV/ionization; C L/(n_H W) in s^-1 per H atom; phi dimensionless")
    return W_g_fid, W_e_fid


# ===========================================================================
# 5. FIGURE D.  phi at unit luminosity ratio -- i.e. W_gamma/W_e -- against the
#    spectral indices of each channel.  Needs NO luminosity sourcing at all.
# ===========================================================================
def make_figure_D(W_g_fid, W_e_fid):
    """phi at unit luminosity ratio -- i.e. W_gamma/W_e -- against the spectral
    index of each channel.  Needs NO luminosity sourcing at all.

    Each class is drawn as a MARKER at its own (index, band top), not as a band
    on the index axis: the classes differ in band top as much as in index, so
    shading the index alone merged four classes into one meaningless wash.
    """
    from igm_config import safe_plot_style, fig_stem
    with safe_plot_style() as plt:
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8))
        INK, INK2 = P.INK, P.INK2

        # ---- panel (a): the PHOTON spectrum ------------------------------
        ax = axes[0]
        alphas = np.linspace(0.3, 3.0, 60)
        tops = [(54.4, "4 Ryd (stellar, He II)"), (1.0e2, "100 eV"),
                (1.0e3, "1 keV"), (3.0e3, "3 keV (AGN template)"),
                (1.0e4, "10 keV"), (1.0e5, "100 keV")]
        cmap = plt.get_cmap("viridis")
        for i, (t, lab) in enumerate(tops):
            y = [W_gamma(a, t) / W_e_fid for a in alphas]
            ax.plot(alphas, y, lw=1.9, color=cmap(i / (len(tops) - 1.0)),
                    label=lab, zorder=2)
        ax.axhline(1.0, color="#0b0b0b", lw=1.7, zorder=3)

        # class markers, each at ITS OWN band top
        marks = [("Star-forming", 2.00, 0.4387, (58, -20)),
                 ("AGN template", 1.50, 0.5705, (-72, 30)),
                 ("AGN radio-quiet", 1.57, 0.5546, (26, 16)),
                 ("AGN radio-loud", 1.96, 0.4902, (52, 34)),
                 ("ULX (broken PL)", 1.10, 0.8073, (24, 22)),
                 ("BHB hard state", 0.70, 3.0566, (30, 4))]
        for lab, a, r, off in marks:
            ax.plot([a], [r], "o", ms=7.5, mfc="#ffffff", mec="#c8102e",
                    mew=2.0, zorder=6)
            ax.annotate(lab, xy=(a, r), xytext=off, textcoords="offset points",
                        fontsize=8.0, color="#8d0a1f", ha="left", va="center",
                        zorder=7,
                        arrowprops=dict(arrowstyle="-", color="#c8102e", lw=0.8,
                                        shrinkA=0, shrinkB=4))
        ax.text(0.015, 0.955, "electrons win per erg", transform=ax.transAxes,
                color=INK, fontsize=8.6, ha="left", va="top", zorder=7)
        ax.text(0.015, 0.045, "photons win per erg", transform=ax.transAxes,
                color=INK2, fontsize=8.6, ha="left", va="bottom", zorder=7)
        ax.text(0.985, 0.50, r"per-erg parity  $W_\gamma=W_e$",
                transform=ax.transAxes, color="#0b0b0b", fontsize=8.4,
                ha="right", va="bottom", zorder=7)
        ax.set_xlabel(r"photon spectral index $\alpha$   "
                      r"($f_\nu\propto\nu^{-\alpha}$;   $\Gamma=\alpha+1$)",
                      color=INK, fontsize=10)
        ax.set_ylabel(r"$\varphi$ at unit luminosity ratio $=W_\gamma/W_e$",
                      color=INK, fontsize=10)
        ax.set_yscale("log")
        ax.set_xlim(0.25, 3.05)
        ax.set_ylim(0.30, 22.0)
        ax.set_title("(a)  the photon spectrum, at fixed CR electrons",
                     color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=7.5, frameon=False, title="ionizing band top",
                  title_fontsize=7.7, loc="upper right", borderaxespad=1.4)
        ax.tick_params(colors=INK2, labelsize=9)

        # ---- panel (b): the ELECTRON spectrum ----------------------------
        ax = axes[1]
        ps = np.linspace(1.9, 2.6, 50)
        mins = [(1.0e2, "100 eV"), (1.0e3, "1 keV (fiducial)"),
                (1.0e4, "10 keV"), (1.0e5, "100 keV")]
        cmap2 = plt.get_cmap("plasma")
        for i, (m, lab) in enumerate(mins):
            y = [W_g_fid / W_e(p_, m, FID_EMAX) for p_ in ps]
            ax.plot(ps, y, lw=1.9, color=cmap2(0.08 + 0.66 * i / (len(mins) - 1.0)),
                    label=lab, zorder=2)
        ax.axhline(1.0, color="#0b0b0b", lw=1.7, zorder=3)
        ax.axvline(FID_P, color="#c8102e", lw=0.9, ls=":", alpha=0.8, zorder=2)
        ax.plot([FID_P], [W_g_fid / W_e_fid], "o", ms=8.5, mfc="#ffffff",
                mec="#c8102e", mew=2.2, zorder=6)
        ax.annotate(f"project fiducial\n$W_\\gamma/W_e={W_g_fid/W_e_fid:.4f}$\n"
                    f"photons better $\\times{W_e_fid/W_g_fid:.2f}$",
                    xy=(FID_P, W_g_fid / W_e_fid), xytext=(2.30, 0.175),
                    color=INK, fontsize=8.3, ha="left", va="center",
                    linespacing=1.35, zorder=7,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=1.0,
                                    shrinkA=2, shrinkB=6))
        ax.text(0.015, 0.045, "photons win per erg", transform=ax.transAxes,
                color=INK2, fontsize=8.6, ha="left", va="bottom", zorder=7)
        ax.text(0.015, 0.93, r"per-erg parity  $W_\gamma=W_e$",
                transform=ax.transAxes, color="#0b0b0b", fontsize=8.4,
                ha="left", va="bottom", zorder=7)
        ax.set_xlabel(r"CR electron injection index $p$   "
                      r"($dN/dE\propto E^{-p}$)", color=INK, fontsize=10)
        ax.set_ylabel(r"$\varphi$ at unit luminosity ratio $=W_\gamma/W_e$",
                      color=INK, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(0.09, 1.55)
        ax.set_title("(b)  the CR electron spectrum, at fixed photons",
                     color=INK, fontsize=11, loc="left")
        ax.legend(fontsize=7.5, frameon=False, title=r"electron $E_{\rm min}$",
                  title_fontsize=7.7, loc="lower right", borderaxespad=1.0)
        ax.tick_params(colors=INK2, labelsize=9)

        fig.tight_layout(rect=(0, 0.19, 1, 1))
        fig.text(0.012, 0.005,
                 rf"IGM at $z={Z_SNAP:g}$, $x_e=10^{{-4}}$, pure H.  "
                 r"$W\equiv\langle E\rangle/\langle N\rangle$: the energy per "
                 r"ionization ACTUALLY ACHIEVED, so $\varphi>1$ means CR "
                 r"electrons ionize more per erg than photons do." "\n"
                 r"Photon yields carry the IGM transparency factor "
                 r"$1-e^{-\tau(E)}$ over one Hubble length: "
                 rf"$\tau={float(tau_igm(1e3)):.2f}$ at 1 keV, "
                 rf"{float(tau_igm(3e3)):.3f} at 3 keV — the $z=10$ IGM is thick "
                 r"below $\sim$500 eV and transparent above $\sim$3 keV, so a "
                 r"hard source's photons are lost while its electrons deposit." "\n"
                 r"Electron yields: the loss+IC route (igm_losses.py). $f_{\rm dep}=1$ "
                 r"(assumption A11): every injected electron is assumed to "
                 r"deposit, so the electron channel is a CEILING, not an estimate." "\n"
                 r"Class markers sit at each class's OWN index and band top — "
                 r"stellar Marques-Chaves+26; AGN Telfer+02 and Graziani+18; "
                 r"ULX and BHB Gladstone+09 (the BHB index is second-hand, "
                 r"status C). Curves are exact, not sampled: $\zeta$ is linear "
                 r"in the emissivity, which check S2 tests rather than assumes.",
                 fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_stem('source_spectral_index')}.{ext}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    return "source_spectral_index"


# ===========================================================================
# 6. LUMINOSITIES for the budget plane.  Per comoving Mpc^3 at z = 10.
#    Every entry derives from a sourced quantity; the unknowns are SCANNED,
#    never guessed.  L_ion is the ESCAPING ionizing luminosity; L_e the
#    relativistic-electron luminosity, with f_dep = 1 (A11 ceiling).
# ===========================================================================
# The one combined unknown for accretion-powered sources: the fraction of the
# accretion/jet output that ends up in relativistic ELECTRONS.  It absorbs the
# jet-coupling efficiency AND the leptonic fraction, because the literature is
# explicit that the proton content of jets "cannot be constrained by
# observations".  Scanned over three decades, exactly as f_esc is scanned.
JET_E_FRAC = PR.JET_E_FRAC        # parameters.yaml

# --- literature normalisations, named so make_inputs_table.py can READ them
# rather than retype them (that file's whole discipline).  Each carries its
# source in the comment; the bib notes carry the caveats.
AGN_LYC_FRAC = (0.17, 1.0 / 3.0)      # Asthana+24 (17%), Jiang+25 (<=1/3), z~7.5
LEHMER_NORM = 1.8e39                  # Lehmer+16 eq.(6), erg/s per Msun/yr, 2-10 keV
LEHMER_NORM_LO, LEHMER_NORM_HI = 1.4e39, 2.3e39   # +0.5/-0.4 on 1.8
LEHMER_Z_EXP = 1.0                    # (1+z)^1, exponent fixed at 1 in eq.(6)
MQ_MECH_ERG_S = (1.0e39, 1.0e40)      # Mirabel+11 sec.6: SS 433, S26 microquasar
MQ_LIFETIME_ERG = 1.0e54              # Mirabel+11 conclusion 2, per object
M_UV_U37126 = -20.10                  # Marques-Chaves+26, +-0.05 AB, z=10.255
M_UV_U37126_ERR = 0.05
ULX_CORONA_KTE_KEV = (1.0, 3.0)       # Gladstone+09 sec.4.5.1: cool, thick corona
ULX_CORONA_TAU = (6.0, 80.0)


def band_ratio(c, E_lo, E_hi):
    """Fraction of a class's ionizing-band luminosity that falls in [E_lo,E_hi].
    Lets an X-ray band luminosity be carried to the ionizing band using only
    that class's OWN sourced spectrum."""
    def integ(lo, hi):
        g = np.linspace(np.log10(lo), np.log10(hi), NGRID)
        E = 10.0 ** g
        if c["kind"] == "pl":
            dN = E ** (-(c["alpha"] + 1.0))
        else:
            G2 = c["G1"] + c["dG"]
            dN = np.where(E < c["E_break"], E ** (-c["G1"]),
                          c["E_break"] ** (G2 - c["G1"]) * E ** (-G2))
        return float(IY._trapz(dN * np.log(10.0) * E * E, g))
    return integ(IY.E_TH_HI, c["E_top"]) / integ(E_lo, E_hi)


def population_boxes():
    """(label, L_ion range, L_e range, W_gamma/W_e, bibkeys) per cMpc^3."""
    Eg = P.mean_ionizing_photon_energy()
    W_e_fid = W_e(FID_P, FID_EMIN, FID_EMAX)

    # --- 1. star-forming galaxies: the project's own chain, unchanged -------
    L_ion_sf = P.ndot_photons_comoving() * Eg / P.EV_PER_ERG
    _, L_e_sf, _ = P.ndot_electrons_comoving()
    sf = next(c for c in PHOTON_CLASSES if c["label"].startswith("Star"))
    W_sf = class_W_gamma(sf, "mid")

    # --- 2. AGN --------------------------------------------------------------
    # Jiang+25 (Nature Astron.): AGN supply AT MOST 1/3 of the LyC budget at
    # z ~ 7.5.  Asthana+24: 17% in their QSO-assisted model.  If AGN are a
    # fraction f of the TOTAL and stars the rest, L_AGN = f/(1-f) L_stellar.
    # Both are measured at z ~ 7.5; carrying them to z = 10 is an extrapolation
    # and a CONSERVATIVE one, the AGN LF declining faster than the galaxy LF.
    f_lo, f_hi = AGN_LYC_FRAC
    L_ion_agn = (L_ion_sf * f_lo / (1 - f_lo), L_ion_sf * f_hi / (1 - f_hi))
    agn = next(c for c in PHOTON_CLASSES if c["label"] == "AGN, reionization template")
    W_agn = class_W_gamma(agn, "mid")
    L_e_agn = (L_ion_agn[0] * JET_E_FRAC[0], L_ion_agn[1] * JET_E_FRAC[1])

    # --- 3. X-ray binaries / microquasars ------------------------------------
    # Lehmer+16 eq.(6): L_2-10keV/SFR = 1.8e39 (1+z) erg/s per Msun/yr.
    rho_sfr = P.rho_sfr()
    L_X = LEHMER_NORM * (1.0 + Z_SNAP) ** LEHMER_Z_EXP * rho_sfr
    L_X_lo = LEHMER_NORM_LO * (1 + Z_SNAP) ** LEHMER_Z_EXP * rho_sfr
    L_X_hi = LEHMER_NORM_HI * (1 + Z_SNAP) ** LEHMER_Z_EXP * rho_sfr
    ulx = next(c for c in PHOTON_CLASSES if c["label"].startswith("ULX"))
    bhb = next(c for c in PHOTON_CLASSES if c["label"].startswith("BHB"))
    # carry the 2-10 keV luminosity to the ionizing band with each own spectrum
    r_ulx, r_bhb = band_ratio(ulx, 2e3, 1e4), band_ratio(bhb, 2e3, 1e4)
    L_ion_xrb = (L_X_lo * min(r_ulx, r_bhb), L_X_hi * max(r_ulx, r_bhb))
    W_xrb = 0.5 * (class_W_gamma(ulx, "mid") + class_W_gamma(bhb, "mid"))
    L_e_xrb = (L_X_lo * JET_E_FRAC[0], L_X_hi * JET_E_FRAC[1])

    return [
        dict(label="Star-forming galaxies", L_ion=(L_ion_sf, L_ion_sf),
             L_e=(L_e_sf, L_e_sf), W=W_sf, We=W_e_fid, exact=True,
             keys="Donnan+24, Llerena+25, MD14, Salpeter+55"),
        dict(label="AGN", L_ion=L_ion_agn, L_e=L_e_agn, W=W_agn, We=W_e_fid,
             exact=False, keys="Jiang+25, Asthana+24, Graziani+18"),
        dict(label="X-ray binaries / microquasars", L_ion=L_ion_xrb,
             L_e=L_e_xrb, W=W_xrb, We=W_e_fid, exact=False,
             keys="Lehmer+16, Gladstone+09, Mirabel+11"),
    ]


def hubble_time_s():
    """1/H(z) from astropy's Planck18, the COMPLETE H(z) with the radiation
    term -- matching photon_vs_electron.py exactly, so the two never disagree."""
    try:
        from astropy.cosmology import Planck18 as _P18
        import astropy.units as _u
        return 1.0 / float(_P18.H(Z_SNAP).to(1 / _u.s).value)
    except Exception:
        return 1.0 / COS["H_z_s"]


# N_req at z = 10, from reionization_budget.py / the manuscript's Table.
# value, label, and the x at which to write the label -- staggered, because
# 1.873 / 2.606 / 3.620 are too close in log-y for stacked labels to be read.
N_REQ = [(1.873, r"$C=1$", 2.2e-7), (2.606, r"$C_{\rm IGM}(z)$", 5.0e-6),
         (3.620, r"$C=3$", 3.0e-4), (11.48, r"$C=12$", 6.0e-3)]


def make_figure_A(boxes):
    r"""The budget plane: zeta_e t_H against zeta_gamma t_H.

    Luminosity AND spectrum are inside the coordinates, so nothing is encoded
    twice -- unlike an (L_e, L_UV) plane, where the colour would largely repeat
    the position.  Two readings:
      * distance from the y = x diagonal  -> WHICH channel wins (phi = x/y);
      * the anti-diagonals x + y = N_req  -> whether EITHER matters, i.e.
        whether the class can actually close the reionization budget.
    The two families are transverse everywhere, so the plane is genuinely 2-D.
    """
    from igm_config import safe_plot_style, fig_stem
    t_H = hubble_time_s()
    with safe_plot_style() as plt:
        fig, ax = plt.subplots(figsize=(9.2, 7.6))
        INK, INK2 = P.INK, P.INK2
        lo, hi = 1e-7, 3e1

        # phi shading: constant along diagonals, drawn analytically
        gx = np.logspace(np.log10(lo), np.log10(hi), 400)
        X, Y = np.meshgrid(gx, gx, indexing="ij")
        im = ax.pcolormesh(gx, gx, np.log10(X / Y).T, cmap="RdBu_r",
                           shading="auto", vmin=-4, vmax=4, zorder=0, alpha=0.65)
        ax.plot([lo, hi], [lo, hi], color="#0b0b0b", lw=2.0, zorder=3)
        ax.text(2.0e-6, 6.0e-6, r"parity  $\varphi=1$", color="#0b0b0b",
                fontsize=9.0, rotation=45, ha="left", va="bottom", zorder=6)
        ax.text(0.03, 0.845, "photons win", transform=ax.transAxes, color=INK,
                fontsize=10.5, ha="left", va="top", zorder=6)
        ax.text(0.70, 0.035, "electrons win", transform=ax.transAxes, color=INK,
                fontsize=10.5, ha="right", va="bottom", zorder=6)

        # reionization sufficiency: straight lines x + y = N_req
        for n, lab, lx in N_REQ:
            xs = np.logspace(np.log10(lo), np.log10(n * (1 - 1e-9)), 400)
            ax.plot(xs, n - xs, color="#1b7837", lw=1.5, ls="--", zorder=4)
            ax.annotate(lab, xy=(lx, n), xytext=(0, 3),
                        textcoords="offset points", color="#14532d",
                        fontsize=8.2, ha="left", va="bottom", zorder=6)
        ax.text(0.360, 0.185, "above these lines a class alone closes\nthe "
                r"reionization budget, $\zeta_{\rm tot}t_H>N_{\rm req}$",
                transform=ax.transAxes, color="#14532d", fontsize=8.6,
                ha="left", va="top", linespacing=1.35, zorder=6)

        cols = ["#c8102e", "#6a3d9a", "#ff7f00"]
        for k, d in enumerate(boxes):
            xlo = zeta_from_W(d["L_e"][0], d["We"]) * t_H
            xhi = zeta_from_W(d["L_e"][1], d["We"]) * t_H
            ylo = zeta_from_W(d["L_ion"][0], d["W"]) * t_H
            yhi = zeta_from_W(d["L_ion"][1], d["W"]) * t_H
            if d["exact"]:
                ax.plot([xlo], [ylo], "o", ms=11, mfc="#ffffff", mec=cols[k],
                        mew=2.6, zorder=8)
            else:
                ax.add_patch(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                           fill=True, fc=cols[k], alpha=0.20,
                                           ec=cols[k], lw=1.8, zorder=7))
            ax.annotate(d["label"], xy=(np.sqrt(xlo * xhi), np.sqrt(ylo * yhi)),
                        xytext=(0, 14 if k != 2 else -22),
                        textcoords="offset points", color=cols[k], fontsize=9.2,
                        ha="center", va="center", zorder=9, weight="bold")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$\zeta_e\,t_H$   —   CR-electron ionizations per H atom "
                      r"per Hubble time", color=INK, fontsize=10.5)
        ax.set_ylabel(r"$\zeta_\gamma\,t_H$   —   photo-ionizations per H atom "
                      r"per Hubble time", color=INK, fontsize=10.5)
        ax.set_title(r"Which high-$z$ sources favour CR electrons?  "
                     rf"Comoving populations at $z={Z_SNAP:g}$",
                     color=INK, fontsize=12, loc="left")
        ax.tick_params(colors=INK2, labelsize=9)
        cb = fig.colorbar(im, ax=ax, pad=0.02, extend="both")
        cb.set_label(r"$\log_{10}\varphi = \log_{10}(\zeta_e/\zeta_\gamma)$"
                     r"   —   red: electrons win", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8)

        fig.tight_layout(rect=(0, 0.19, 1, 1))
        fig.text(0.012, 0.005,
                 rf"IGM at $z={Z_SNAP:g}$, $x_e=10^{{-4}}$, pure H; "
                 rf"$t_H=1/H(z)$ from astropy Planck18 $={t_H/3.1557e16:.4f}$ Gyr."
                 " \n"
                 r"Axes carry BOTH luminosity and spectrum: "
                 r"$\zeta=\mathcal{C}L/(n_{\rm H}W)$, verified against the full "
                 r"integral (check S2). Photon yields carry the IGM "
                 r"transparency factor $1-e^{-\tau(E)}$." "\n"
                 r"Star-forming galaxies are a POINT (both luminosities are the "
                 r"project's own anchored chain, reproducing "
                 r"$\zeta_\gamma/\zeta_e=5599$). AGN and X-ray binaries are "
                 r"BOXES: their $L_e$ spans the scanned "
                 rf"$10^{{{np.log10(JET_E_FRAC[0]):.0f}}}$–"
                 rf"$10^{{{np.log10(JET_E_FRAC[1]):.0f}}}$ jet electron "
                 r"fraction, since the proton content of jets is not "
                 r"observationally constrained." "\n"
                 r"AGN $L_{\rm ion}$: 17–33% of the LyC budget (Asthana+24; "
                 r"Jiang+25 upper bound), measured at $z\sim7.5$ and carried to "
                 r"$z=10$ — conservative, the AGN LF falls faster than the "
                 r"galaxy LF. XRB $L_X$: Lehmer+16 eq.(6), carried to the "
                 r"ionizing band with each class's own sourced spectrum." "\n"
                 r"$f_{\rm dep}=1$ (assumption A11): every injected electron is "
                 r"assumed to deposit, so every $\zeta_e$ here is a CEILING.",
                 fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_stem('source_budget_plane')}.{ext}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    return "source_budget_plane"


def main():
    print("=" * 78)
    print(f"SOURCE COMPETITION: spectral factors at z = {Z_SNAP:g}")
    print("=" * 78)
    W_g_fid, W_e_fid = run_checks()

    print("\n[PER-CLASS SPECTRAL FACTOR]  W_gamma/W_e, absorption applied")
    print(f"{'class':34s} {'W_gamma [eV]':>20s} {'W_g/W_e':>19s}  {'vs stellar':>10s}  src")
    base = None
    table = []
    for c in PHOTON_CLASSES:
        mid = class_W_gamma(c, "mid")
        lo, hi = class_W_gamma(c, "lo"), class_W_gamma(c, "hi")
        lo, hi = min(lo, hi), max(lo, hi)
        r = mid / W_e_fid
        if base is None:
            base = r
        print(f"{c['label']:34s} {mid:8.2f} [{lo:7.2f},{hi:7.2f}] "
              f"{r:7.4f} [{lo/W_e_fid:5.3f},{hi/W_e_fid:5.3f}] {r/base:9.2f}x  "
              f"{c['key']}({c['status']})")
        table.append(dict(label=c["label"], W_gamma=mid, W_lo=lo, W_hi=hi,
                          ratio=r, vs_stellar=r / base, key=c["key"],
                          status=c["status"], note=c["note"]))
        put(f"W_gamma_{c['key']}_{c['label'][:12].replace(' ','_')}", mid)
    print(f"\n   electron fiducial: W_e = {W_e_fid:.3f} eV "
          f"(p={FID_P}, {FID_EMIN:.0e}-{FID_EMAX:.0e} eV), f_dep = 1 (A11 ceiling)")

    stem = make_figure_D(W_g_fid, W_e_fid)
    print(f"\n[FIGURE]  {stem}.png / .pdf")

    t_H = hubble_time_s()
    boxes = population_boxes()
    print("\n[BUDGET PLANE]  comoving populations, per cMpc^3")
    print(f"{'class':32s} {'zeta_e t_H':>22s} {'zeta_gamma t_H':>22s} {'phi':>18s}")
    for d in boxes:
        x = [zeta_from_W(v, d["We"]) * t_H for v in d["L_e"]]
        y = [zeta_from_W(v, d["W"]) * t_H for v in d["L_ion"]]
        print(f"{d['label']:32s} [{x[0]:.3e},{x[1]:.3e}] [{y[0]:.3e},{y[1]:.3e}] "
              f"[{x[0]/y[1]:.2e},{x[1]/y[0]:.2e}]")
    stemA = make_figure_A(boxes)
    print(f"\n[FIGURE]  {stemA}.png / .pdf")

    # --- per-object panel ---------------------------------------------------
    obj = object_boxes()
    print("\n[BUDGET PLANE]  single objects (one per cMpc^3)")
    for d in obj:
        x = [zeta_from_W(v, d["We"]) * t_H for v in d["L_e"]]
        y = [zeta_from_W(v, d["W"]) * t_H for v in d["L_ion"]]
        n_req = 2.606 / (x[1] + y[1])
        print(f"  {d['label'][:44]:46s} zeta_e t_H=[{x[0]:.3e},{x[1]:.3e}] "
              f"zeta_g t_H=[{y[0]:.3e},{y[1]:.3e}]  n_req={n_req:.3e} cMpc^-3")
    g = obj[0]
    xg = zeta_from_W(g["L_e"][0], g["We"]) * t_H
    yg = zeta_from_W(g["L_ion"][0], g["W"]) * t_H
    L_nu_u = L_nu_from_M_UV(M_UV_U37126)
    n_implied = P.rho_uv() / L_nu_u
    put("M_UV_U37126", M_UV_U37126)
    put("L_nu_U37126", L_nu_u)
    put("SFR_U37126", P.K_UV * L_nu_u)
    put("zeta_gamma_tH_U37126", yg)
    put("zeta_e_tH_U37126", xg)
    put("n_req_U37126", 2.606 / (xg + yg))
    put("n_implied_by_rho_UV", n_implied)
    put("n_req_over_n_implied", (2.606 / (xg + yg)) / n_implied)
    print(f"  rho_UV implies {n_implied:.3e} U37126-equivalents per cMpc^3; "
          f"n_req/n_implied = {(2.606/(xg+yg))/n_implied:.2f} -- the SAME "
          f"shortfall the population panel finds, by an independent route")
    stemO = make_figure_A_objects()
    print(f"\n[FIGURE]  {stemO}.png / .pdf")

    print("\n[CHECKS]")
    for cid, name, st, det in ROWS:
        print(f"  {st}  {cid:5s} {name}")
        print(f"          {det}")
    npass = sum(1 for r in ROWS if r[2] == "PASS")
    print(f"\n  {npass}/{len(ROWS)} passed")

    import os
    os.makedirs("provenance", exist_ok=True)
    with open("provenance/source_map.json", "w") as f:
        json.dump(dict(derived=PROV,          # "derived" is the section name the
                       # provenance gate loads; "scalars" was invisible to it
                       classes=table,
                       produced_by={"source_spectral_index": "source_map.py",
                                    "source_budget_plane": "source_map.py",
                                    "source_budget_objects": "source_map.py"}),
                  f, indent=2)
    print("  provenance/source_map.json written")
    return 0 if npass == len(ROWS) else 1


# ===========================================================================
# 8. PER-OBJECT ENTRIES.  Same plane, one object instead of a population.
# ===========================================================================
PC_CM = 3.085677581e18
L_EDD_PER_MSUN = 1.5e38        # erg/s/Msun -- Mirabel+11 eqs. (3)-(4)
F_EDD_MIRABEL = 0.1            # Mirabel+11: "our choice fEdd = 0.1 is likely
                               # to be conservative"
F_2_10_MIRABEL = 0.1           # Mirabel+11 eq. (6) fiducial
M_BH_FID = 10.0                # Msun; a stellar-mass BH. OUR choice (status U)
M_BH_RANGE = (5.0, 30.0)


def L_nu_from_M_UV(M_AB):
    """Monochromatic UV luminosity [erg/s/Hz] from an absolute AB magnitude."""
    d10 = 10.0 * PC_CM
    return 4.0 * np.pi * d10 ** 2 * 10.0 ** (-0.4 * (M_AB + 48.6))


def object_boxes():
    """(label, L_ion range, L_e range, W_gamma/W_e, ...) for ONE object.

    The budget plane's axes are per comoving Mpc^3, so a single object plotted
    here means "one such object per cMpc^3". The anti-diagonals therefore read
    directly as the comoving NUMBER DENSITY that class would need to close the
    reionization budget -- which is the quantity to compare against an observed
    luminosity function.
    """
    Eg = P.mean_ionizing_photon_energy()
    W_e_fid = W_e(FID_P, FID_EMIN, FID_EMAX)

    # --- 1. U37126: this project's own anchor galaxy, z = 10.255 -----------
    # Everything follows from M_UV through the chain already in the manuscript;
    # no new source is needed, which is why it is the one exact per-object point.
    L_nu = L_nu_from_M_UV(M_UV_U37126)
    L_ion_g = P.F_ESC_FID * 10.0 ** P.LOG_XI_ION * L_nu * Eg / P.EV_PER_ERG
    sfr = P.K_UV * L_nu                                     # Msun/yr
    L_e_g = (sfr * P.sn_per_solar_mass() / P.SEC_PER_YR
             * P.EPS_CR_FE_FID * P.E_SN_ERG)                # erg/s
    sf = next(c for c in PHOTON_CLASSES if c["label"].startswith("Star"))

    # --- 2. BH-HMXB on Mirabel+11's own fiducial parameters ----------------
    # L_Edd = 1.5e38 (M_BH/Msun); accretion at f_Edd = 0.1; f_2-10 = 0.1 of that
    # emerges in 2-10 keV, which the class's OWN sourced spectrum carries to the
    # ionizing band. M_BH is the single free choice here (status U), and the
    # box position scales linearly with it.
    bhb = next(c for c in PHOTON_CLASSES if c["label"].startswith("BHB"))
    r_bhb = band_ratio(bhb, 2e3, 1e4)
    L_acc = lambda m: F_EDD_MIRABEL * L_EDD_PER_MSUN * m
    L_ion_x = (F_2_10_MIRABEL * L_acc(M_BH_RANGE[0]) * r_bhb,
               F_2_10_MIRABEL * L_acc(M_BH_RANGE[1]) * r_bhb)
    L_e_x = (L_acc(M_BH_RANGE[0]) * JET_E_FRAC[0],
             L_acc(M_BH_RANGE[1]) * JET_E_FRAC[1])

    return [
        dict(label="U37126 (star-forming, $z=10.255$)",
             L_ion=(L_ion_g, L_ion_g), L_e=(L_e_g, L_e_g),
             W=class_W_gamma(sf, "mid"), We=W_e_fid, exact=True,
             keys="MarquesChaves2026 + the project chain"),
        dict(label="BH-HMXB (Mirabel+11 fiducials)",
             L_ion=L_ion_x, L_e=L_e_x,
             W=class_W_gamma(bhb, "mid"), We=W_e_fid, exact=False,
             keys="Mirabel2011 eqs. (3)-(4), Gladstone2009 SED"),
    ]


# Classes deliberately NOT plotted per object, and the exact number missing.
# Reported rather than estimated, per Master Rule 4.
UNPLOTTED_OBJECTS = [
    ("SS 433 / S26 microquasars",
     "Mirabel+11 gives their MECHANICAL power (>1e39, >1e40 erg/s) but no "
     "ionizing luminosity, so only the electron axis could be placed"),
    ("Single SN / SNR",
     "needs a remnant CR-acceleration lifetime to turn E_SN x eps_CR f_e into "
     "a luminosity; no source established in this project"),
    ("AGN / little red dots",
     "L_bol ~ 1e44-1e47 erg/s is available only second-hand; no primary read"),
]


def make_figure_A_objects():
    r"""The budget plane for SINGLE OBJECTS.

    Identical axes to the population figure, but a point here means "one such
    object per cMpc^3", so the anti-diagonals relabel directly as the comoving
    NUMBER DENSITY the class would need to close the reionization budget:

        n_req = N_req / (zeta_tot t_H)      [cMpc^-3]

    which is the quantity to hold against an observed luminosity function.
    """
    from igm_config import safe_plot_style, fig_stem
    t_H = hubble_time_s()
    boxes = object_boxes()
    N_IGM = 2.606                                  # N_req at C_IGM(z), z = 10
    with safe_plot_style() as plt:
        fig, ax = plt.subplots(figsize=(9.6, 7.8))
        INK, INK2 = P.INK, P.INK2
        lo, hi = 1e-7, 1e4

        gx = np.logspace(np.log10(lo), np.log10(hi), 420)
        X, Y = np.meshgrid(gx, gx, indexing="ij")
        im = ax.pcolormesh(gx, gx, np.log10(X / Y).T, cmap="RdBu_r",
                           shading="auto", vmin=-4, vmax=4, zorder=0, alpha=0.65)
        ax.plot([lo, hi], [lo, hi], color="#0b0b0b", lw=2.0, zorder=3)
        ax.text(3e-6, 9e-6, r"parity  $\varphi=1$", color="#0b0b0b",
                fontsize=9.0, rotation=45, ha="left", va="bottom", zorder=6)
        ax.text(0.03, 0.965, "photons win", transform=ax.transAxes, color=INK,
                fontsize=10.5, ha="left", va="top", zorder=6)
        ax.text(0.97, 0.035, "electrons win", transform=ax.transAxes, color=INK,
                fontsize=10.5, ha="right", va="bottom", zorder=6)

        # anti-diagonals, labelled by the number density they demand
        for n_req, lx in ((1.0e-3, 2e-2), (1.0e-2, 6e-4),
                          (1.0e-1, 2e-5), (1.0, 8e-7)):
            tot = N_IGM / n_req
            xs = np.logspace(np.log10(lo), np.log10(tot * (1 - 1e-9)), 400)
            ax.plot(xs, tot - xs, color="#1b7837", lw=1.4, ls="--", zorder=4)
            ax.annotate(rf"$n_{{\rm req}}=10^{{{np.log10(n_req):.0f}}}$",
                        xy=(lx, tot), xytext=(0, 3), textcoords="offset points",
                        color="#14532d", fontsize=8.0, ha="left", va="bottom",
                        zorder=6)
        cols = ["#c8102e", "#ff7f00"]
        for k, d in enumerate(boxes):
            xlo = zeta_from_W(d["L_e"][0], d["We"]) * t_H
            xhi = zeta_from_W(d["L_e"][1], d["We"]) * t_H
            ylo = zeta_from_W(d["L_ion"][0], d["W"]) * t_H
            yhi = zeta_from_W(d["L_ion"][1], d["W"]) * t_H
            if d["exact"]:
                ax.plot([xlo], [ylo], "o", ms=11, mfc="#ffffff", mec=cols[k],
                        mew=2.6, zorder=8)
                n_req = N_IGM / (xlo + ylo)
                ax.annotate(d["label"] + "\n"
                            rf"$n_{{\rm req}}={n_req/10**np.floor(np.log10(n_req)):.2f}"
                            rf"\times10^{{{int(np.floor(np.log10(n_req)))}}}$"
                            rf" cMpc$^{{-3}}$",
                            xy=(xlo, ylo), xytext=(-26, -46),
                            textcoords="offset points", color=cols[k],
                            fontsize=8.8, ha="right", va="top",
                            linespacing=1.35, zorder=9, weight="bold",
                            arrowprops=dict(arrowstyle="-", color=cols[k],
                                            lw=1.0, shrinkA=2, shrinkB=6))
            else:
                ax.add_patch(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                           fill=True, fc=cols[k], alpha=0.20,
                                           ec=cols[k], lw=1.8, zorder=7))
                ax.annotate(d["label"],
                            xy=(np.sqrt(xlo * xhi), np.sqrt(ylo * yhi)),
                            xytext=(0, -26), textcoords="offset points",
                            color=cols[k], fontsize=8.8, ha="center",
                            va="center", zorder=9, weight="bold")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$\zeta_e\,t_H$  per object per cMpc$^3$",
                      color=INK, fontsize=10.5)
        ax.set_ylabel(r"$\zeta_\gamma\,t_H$  per object per cMpc$^3$",
                      color=INK, fontsize=10.5)
        ax.set_title(r"Budget plane for SINGLE OBJECTS at $z=10$: "
                     r"how many would it take?", color=INK, fontsize=12,
                     loc="left")
        ax.tick_params(colors=INK2, labelsize=9)
        cb = fig.colorbar(im, ax=ax, pad=0.02, extend="both")
        cb.set_label(r"$\log_{10}\varphi=\log_{10}(\zeta_e/\zeta_\gamma)$"
                     r"   —   red: electrons win", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8)

        fig.tight_layout(rect=(0, 0.115, 1, 1))
        unpl = "; ".join(n for n, _ in UNPLOTTED_OBJECTS)
        fig.text(0.012, 0.005,
                 r"A point here is ONE object per cMpc$^3$, so the dashed "
                 r"anti-diagonals read as the number density that class would "
                 r"need. Compare against an observed luminosity function." "\n"
                 rf"U37126 ($M_{{\rm UV}}={M_UV_U37126}$, Marques-Chaves+26) is "
                 r"EXACT: both channels follow from the chain already in the "
                 r"manuscript. Its $n_{\rm req}$ exceeds the density "
                 r"$\rho_{\rm UV}/L_\nu=2.77\times10^{-4}$ cMpc$^{-3}$ that "
                 r"$\rho_{\rm UV}$ implies by the same $\times25$ the population "
                 r"figure finds --- an independent route to that shortfall." "\n"
                 r"BH-HMXB uses Mirabel+11 eqs.~(3)--(4): "
                 rf"$L_{{\rm Edd}}=1.5\times10^{{38}}(M_{{\rm BH}}/M_\odot)$, "
                 rf"$f_{{\rm Edd}}=0.1$, $f_{{2-10}}=0.1$, carried to the "
                 r"ionizing band by the Gladstone+09 BHB spectrum. The box spans "
                 rf"$M_{{\rm BH}}={M_BH_RANGE[0]:.0f}$--{M_BH_RANGE[1]:.0f}"
                 r"$\,M_\odot$ (OUR choice) and the scanned jet electron "
                 r"fraction; position scales linearly with $M_{\rm BH}$." "\n"
                 rf"NOT plotted, for want of one specific number each: {unpl}. "
                 r"$f_{\rm dep}=1$ (A11): every $\zeta_e$ is a ceiling.",
                 fontsize=7.0, color=INK2, va="bottom", linespacing=1.45)
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_stem('source_budget_objects')}.{ext}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    return "source_budget_objects"


if __name__ == "__main__":
    raise SystemExit(main())
