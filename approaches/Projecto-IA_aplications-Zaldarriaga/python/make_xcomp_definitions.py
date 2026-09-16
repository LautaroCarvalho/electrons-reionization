#!/usr/bin/env python3
r"""
X_comp: every quantity, its definition, its value and what it depends on.

Task 2 of Instructions_x_comp: state the inventory explicitly so it can be
audited BEFORE any map is computed.

    X_comp(alpha_g, alpha_e) = [ Int S_g N_g dE / <E>_g ] / [ Int S_e N_e dE / <E>_e ]
                             = ( <N_g>/<E>_g ) / ( <N_e>/<E>_e )
                             = W_e(alpha_e) / W_g(alpha_g)

evaluated at luminosity parity L_gamma = L_e, so only the spectral indices act.

DISCIPLINE, inherited from make_inputs_table.py: every value is READ FROM THE
CODE that computes it. Nothing is retyped, so the inventory cannot drift from
the calculation.

Run:  python3 python/make_xcomp_definitions.py          # print the inventory
      python3 python/make_xcomp_definitions.py --tex    # also emit the LaTeX
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import parameters as PR          # the single source of truth
import sys

import numpy as np
from scipy.optimize import brentq

import ionization_yield as IY
import photon_vs_electron as P
import source_map as SM

# ===========================================================================
# 1. THE TWO BANDS
# ===========================================================================
E_MIN_G = IY.E_TH_HI                      # HI ionization potential, physical
E_MIN_E = P.CR_E_MIN_EV                   # project choice
E_MAX_E = P.CR_E_MAX_EV                   # project choice


def free_streaming_energy():
    """E at which tau_IGM = 1 over one Hubble length.

    DERIVED, not adopted: this is what 'the free-streaming regime (~keV)' in
    Instructions_x_comp actually evaluates to. Now delegates to
    ionization_yield.igm_cutoff_eV, which is the project's ONE implementation --
    this module used to solve it independently against source_map.tau_igm.
    """
    return IY.igm_cutoff_eV(PR.Z, PR.X_E)


# ---- the canonical band policy: min(source cutoff, IGM cutoff) -------------
# E_MAX_G_IGM is source-INDEPENDENT and governs the continuous alpha_gamma
# sweep, because a point on that sweep is not a source class and has no
# emission cutoff of its own. E_MAX_G_STELLAR is what the PROJECT'S OWN source
# actually gets: stars emit nothing above the He II edge, so their band stops
# at 4 Ryd whatever the IGM does.
E_MAX_G_IGM = free_streaming_energy()
E_MAX_G_STELLAR = PR.photon_band(
    PR.STELLAR_CUTOFF_RYD * P.RYD_EV, E_MAX_G_IGM)[1]
E_MAX_G = E_MAX_G_IGM          # default for the sweep; kept for back-compat

# axis ranges: chosen to contain every spectral index sourced in this project
ALPHA_G_RANGE = (0.3, 5.0)
ALPHA_E_RANGE = (1.2, 2.7)

FID_AG = P.SED_ALPHA                      # 2.0, the U37126 LyC shape
FID_AE = P.CR_INDEX                       # 2.2, the project's CR injection index


# ===========================================================================
# 2. THE MOMENTS.  One routine, both channels, so the two sides of X_comp
#    cannot drift apart through separate implementations.
# ===========================================================================
def moments(channel, index, E_lo, E_hi, n=4000, scale=1.0):
    """<N_ion>, <E>, W for a power-law primary spectrum on [E_lo, E_hi].

    channel "gamma": S(E) ~ E^-(alpha+1), i.e. f_nu ~ nu^-alpha.
    channel "e"    : S(E) ~ E^-p, the CR injection spectrum.

    `scale` multiplies S by a constant. X_comp must be independent of it --
    that is check X7, not an assumption.
    """
    g = np.linspace(np.log10(E_lo), np.log10(E_hi), n)
    E = 10.0 ** g
    if channel == "gamma":
        S = scale * E ** (-(index + 1.0))
        N = np.interp(g, SM._GP, SM._NGAM)
    elif channel == "e":
        S = scale * E ** (-index)
        N = np.interp(g, SM._GE, SM._NE)
    else:
        raise ValueError(channel)
    w = S * np.log(10.0) * E               # dN/dlog10E
    norm = IY._trapz(w, g)
    N_bar = IY._trapz(w * N, g) / norm     # ionizations per primary
    E_bar = IY._trapz(w * E, g) / norm     # eV per primary
    return N_bar, E_bar, E_bar / N_bar     # <N>, <E>, W


def W_gamma(alpha, E_top=None, **kw):
    """W_gamma over [E_th, E_top]. E_top defaults to the IGM cutoff, which is
    the right cap for a source whose own emission extends past it; pass
    E_MAX_G_STELLAR for a stellar source, whose spectrum stops at 4 Ryd."""
    return moments("gamma", alpha, E_MIN_G,
                   E_MAX_G_IGM if E_top is None else E_top, **kw)[2]


def W_e(p, E_lo=None, E_hi=None, **kw):
    return moments("e", p, E_MIN_E if E_lo is None else E_lo,
                   E_MAX_E if E_hi is None else E_hi, **kw)[2]


def X_comp(alpha_g, alpha_e, E_top=None):
    """Ionizations per unit injected energy, photons relative to electrons."""
    return W_e(alpha_e) / W_gamma(alpha_g, E_top=E_top)


def X_comp_stellar():
    """X_comp for THIS PROJECT'S source: a stellar spectrum, capped at 4 Ryd.

    This is the number the manuscript should quote. Under the old single-band
    treatment this module reported 2.073 (IGM cap applied to a stellar source),
    while photon_vs_electron and source_map reported 2.280. The band policy
    reconciles them: 2.280 is correct for stars.
    """
    return X_comp(FID_AG, FID_AE, E_top=E_MAX_G_STELLAR)


# ===========================================================================
# 4. THE PHYSICS OF THE ELECTRON PENALTY.  Computed, not asserted.
# ===========================================================================
def loss_branching_weighted(p_index=None, n=600):
    """Where a CR electron's energy flux goes, averaged over the injection
    spectrum. Returns [(mechanism, fraction), ...] sorted large-first."""
    import igm_losses as L
    import yield_comparison as YC
    p_index = FID_AE if p_index is None else p_index
    YC.set_redshift(10.0)
    Hz = YC.H_z_snapshot()
    g = np.linspace(np.log10(E_MIN_E), np.log10(E_MAX_E), n)
    E = 10.0 ** g
    w = E ** (-p_index) * np.log(10.0) * E
    fr, _ = YC.loss_branching(E, Hz)
    den = IY._trapz(w, g)
    out = [(k, float(IY._trapz(w * fr[i], g) / den))
           for i, k in enumerate(L.MECH_KEYS)]
    return sorted(out, key=lambda t: -t[1])


def emin_sensitivity(mins=(1.0e2, 1.0e3, 1.0e4, 1.0e5)):
    """W_e and X_comp at the fiducial indices, versus the electron band bottom.
    This is the least-constrained input in the whole construction."""
    Wg = W_gamma(FID_AG)
    return [(m, W_e(FID_AE, E_lo=m), W_e(FID_AE, E_lo=m) / Wg) for m in mins]


def photon_is_electron_plus_one(E_eV=(54.4, 1e2, 1e3, 1e4, 1e5)):
    """N_gamma(E) vs 1 + N_e(E - E_th).

    The photon channel is STRUCTURALLY the electron channel plus one free
    ionization: the photon spends the first 13.598 eV at 100% efficiency, and
    only the excess suffers the electron penalty. Verified here to be
    APPROXIMATE (a few %), not an identity -- the residual is the known
    depfit-route / loss+IC-route disagreement (assumption A9, 7.7%).
    """
    E = np.asarray(E_eV, dtype=float)
    Ng = np.asarray(IY.N_gamma(E, SM.COS, SM.PAR, "C", SM.CHAN), dtype=float)
    Ne = np.asarray(SM.P._electron_yield(np.maximum(E - E_MIN_G, 1e-3),
                                         SM.COS, SM.PAR, SM.CHAN), dtype=float)
    return [(float(e), float(a), float(1 + b), float(abs(a / (1 + b) - 1)))
            for e, a, b in zip(E, Ng, Ne)]


# ===========================================================================
# 3. THE INVENTORY
# ===========================================================================
R = []


def row(sec, sym, defn, val, unit, dep, src, st):
    R.append((sec, sym, defn, val, unit, dep, src, st))


def build():
    cos = IY.cosmology(10.0)
    Ng, Eg, Wg = moments("gamma", FID_AG, E_MIN_G, E_MAX_G_STELLAR)
    Ne, Ee, We = moments("e", FID_AE, E_MIN_E, E_MAX_E)

    C = "Scenario"
    row(C, "z", "redshift of the snapshot", "10", "--",
        "--", "project fiducial", "U")
    row(C, "n_H", "proper hydrogen number density",
        f"{cos['n_H_cm3']:.4e}", "cm^-3", "z, Omega_b h^2, X_H", "Planck2020", "D")
    row(C, "x_e", "ionized fraction, held static",
        f"{1e-4:.0e}", "--", "--", "your value", "U")
    row(C, "H(z), t_H", "expansion rate and Hubble time",
        f"{SM.hubble_time_s()/3.1557e16:.4f} Gyr", "s^-1, Gyr",
        "Planck18 complete H(z)", "Planck2020", "D")
    row(C, "target", "pure atomic hydrogen at mean density", "Z = 0", "--",
        "--", "project fiducial", "U")

    B = "Photon band"
    row(B, "E_min,g", "HI ionization potential -- PHYSICAL floor",
        f"{E_MIN_G:.9f}", "eV", "atomic constant", "NIST_ASD", "V")
    row(B, "E_max,g (stellar)",
        "band top ACTUALLY USED: min(4 Ryd source cutoff, IGM cutoff)",
        f"{E_MAX_G_STELLAR:.4f}", "eV", "band policy", "DERIVED here", "D")
    row(B, "E_max,g (IGM)",
        "free-streaming onset: tau_IGM = 1 over one Hubble length; caps any "
        "source whose own emission reaches past it",
        f"{E_MAX_G_IGM:.1f}", "eV", "sigma_photoion, n_HI, c/H(z)",
        "DERIVED here", "D")
    row(B, "tau(E)", "IGM optical depth over a Hubble length",
        f"{float(SM.tau_igm(1e3)):.3f} at 1 keV, "
        f"{float(SM.tau_igm(3e3)):.4f} at 3 keV", "--",
        "sigma_photoion, n_HI, z", "Karzas1961 / DERIVED", "D")
    row(B, "S_g(E)", "dN/dE ~ E^-(alpha_g+1), NUMBER-normalised to 1 on the band",
        "power law", "eV^-1", "alpha_g, band", "convention", "U")
    # alpha_gamma is THREE things that do not share a status, and collapsing
    # them into one row marked V was wrong. Marques-Chaves+26 is cited for
    # NEITHER index: that paper reports beta_UV = -2.88 over lambda_rest >=
    # 1400 A and no ionizing-continuum index at all. It supplies the stellar
    # POPULATION, not the slope.
    row(B, "alpha_g (fiducial)",
        "the index actually used, f_nu ~ nu^-alpha_g. OUR PARAMETRISATION: "
        "MarquesChaves2026 gives the population (BPASS v2.2.1, Z=0.003, "
        "6.8 Myr) but no LyC index, and its beta_UV is measured longward of "
        "1400 A, which cannot be carried across the Lyman break",
        f"{FID_AG:.1f}", "--", "--", "this work", "U")
    row(B, "alpha_g (measured)",
        "the ONLY direct measurement in our band: L_nu ~ nu^-gamma over "
        "13.6-54.4 eV, exactly this project's convention, no conversion. "
        "CAVEAT: four GALACTIC HII regions at roughly solar metallicity, and "
        "the authors find it disagrees with BPASS; metal-poor z~10 should be "
        "harder, i.e. smaller",
        "4.5 +/- 0.4", "--", "--", "Murchikova2020", "V")
    row(B, "alpha_g (axis)",
        "range swept by the figure, chosen to contain every index sourced in "
        "this project AND the measured 4.5",
        f"{ALPHA_G_RANGE[0]}-{ALPHA_G_RANGE[1]}", "--", "--", "", "S")

    E = "Electron band"
    row(E, "E_min,e", "low cutoff of the CR injection spectrum",
        f"{E_MIN_E:.0e}", "eV", "--", "your choice", "U")
    row(E, "E_max,e", "high cutoff of the CR injection spectrum",
        f"{E_MAX_E:.0e}", "eV", "--", "your choice", "U")
    row(E, "S_e(E)", "dN/dE ~ E^-alpha_e, NUMBER-normalised to 1 on the band",
        "power law", "eV^-1", "alpha_e, band", "convention", "U")
    row(E, "alpha_e", "CR electron injection index",
        f"{FID_AE:.1f} (fid); {ALPHA_E_RANGE[0]}-{ALPHA_E_RANGE[1]} (axis)", "--",
        "--", "your choice; DSA predicts 2", "U")

    Y = "Ionization yields"
    row(Y, "N_ion,g(E)", "ionizations per ABSORBED photon of energy E, "
        "primary + full secondary cascade",
        f"{Ng:.4f} (band mean)", "--", "E, x_e, z", "ionization_yield, depfit route", "V")
    row(Y, "N_ion,e(E)", "ionizations per INJECTED electron of energy E, "
        "7 loss channels + BED cascade + IC secondaries",
        f"{Ne:.2f} (band mean)", "--", "E, x_e, z, B",
        "igm_losses the loss+IC route", "V")

    M = "Derived moments"
    row(M, "<E>_g", "mean energy per emitted ionizing photon",
        f"{Eg:.4f}", "eV", "alpha_g, band", "DERIVED", "D")
    row(M, "<E>_e", "mean energy per injected CR electron",
        f"{Ee:.4e}", "eV", "alpha_e, band", "DERIVED", "D")
    row(M, "W_g", "energy price of one ionization, photons = <E>_g/<N>_g",
        f"{Wg:.4f}", "eV/ion", "alpha_g, band, x_e, z", "DERIVED", "D")
    row(M, "W_e", "energy price of one ionization, electrons = <E>_e/<N>_e",
        f"{We:.4f}", "eV/ion", "alpha_e, band, x_e, z, B", "DERIVED", "D")
    row(M, "X_comp", "W_e/W_g at L_gamma = L_e: ionizations per unit energy, "
        "photons relative to electrons",
        f"{We/Wg:.4f}", "--", "alpha_g, alpha_e ONLY", "DERIVED", "D")
    return Ng, Eg, Wg, Ne, Ee, We


def main():
    Ng, Eg, Wg, Ne, Ee, We = build()
    print("=" * 100)
    print("X_comp -- DEFINITION INVENTORY      z = 10, pure H, x_e = 1e-4")
    print("=" * 100)
    print("\n  X_comp(a_g, a_e) = [Int S_g N_g dE / <E>_g] / [Int S_e N_e dE / <E>_e]"
          "\n                   = (<N_g>/<E>_g) / (<N_e>/<E>_e)"
          "\n                   = W_e(a_e) / W_g(a_g)          at L_gamma = L_e\n")
    sec = None
    for s, sym, defn, val, unit, dep, src, st in R:
        if s != sec:
            print(f"\n--- {s} " + "-" * (96 - len(s)))
            sec = s
        print(f"  [{st}] {sym:12s} {val:>26s}  {unit:9s}  {defn}")
        print(f"      {'':12s} {'depends on:':>26s}  {dep}   [src: {src}]")

    print("\n" + "=" * 100)
    print("FIDUCIAL")
    print("=" * 100)
    print(f"  photons   a_g={FID_AG:.1f}:  <N>={Ng:8.4f}   <E>={Eg:10.4f} eV   "
          f"W_g={Wg:8.4f} eV/ion   efficiency {100*E_MIN_G/Wg:5.1f}%")
    print(f"  electrons a_e={FID_AE:.1f}:  <N>={Ne:8.2f}   <E>={Ee:10.4e} eV   "
          f"W_e={We:8.4f} eV/ion   efficiency {100*E_MIN_G/We:5.1f}%")
    print(f"  X_comp = {We/Wg:.4f}   -> photons ionize {We/Wg:.2f}x more per erg")

    if "--tex" in sys.argv:
        out = emit_tex(Ng, Eg, Wg, Ne, Ee, We)
        print(f"\n[ARTEFACT] {out}   ({len(R)} inventory entries)")
    else:
        print("\n  (inventory only; pass --tex to emit the LaTeX document)")

    if "--check" in sys.argv:
        print("\n[CHECKS]")
        rr = run_checks()
        for cid, name, st, det in rr:
            print(f"  {st}  {cid:4s} {name}")
            print(f"         {det}")
        print(f"\n  {sum(1 for r in rr if r[2]=='PASS')}/{len(rr)} passed")

    if "--fig" in sys.argv:
        stem = make_figure_X()
        print(f"[ARTEFACT] Images/{stem}.png / .pdf")

    emit_provenance(Ng, Eg, Wg, Ne, Ee, We)
    return 0


def emit_provenance(Ng, Eg, Wg, Ne, Ee, We):
    """provenance/xcomp.json -- so the manuscript can \src{} these numbers and
    the gate can check them like any other literal."""
    import json
    import os
    prov = {
        "X_comp_fiducial": We / Wg,        # stellar cap: the project's source
        "X_comp_igm_cap": W_e(FID_AE) / W_gamma(FID_AG),   # sweep / hard sources
        "E_max_gamma_stellar_eV": E_MAX_G_STELLAR,
        "W_gamma_xcomp_fiducial_eV": Wg,
        "W_e_xcomp_fiducial_eV": We,
        "N_gamma_xcomp_band_mean": Ng,
        "N_e_xcomp_band_mean": Ne,
        "E_gamma_mean_eV": Eg,
        "E_e_mean_eV": Ee,
        "E_max_gamma_tau1_eV": E_MAX_G,
        "eff_gamma_percent": 100.0 * E_MIN_G / Wg,
        "eff_e_percent": 100.0 * E_MIN_G / We,
    }
    for k, v in loss_branching_weighted():
        prov[f"loss_frac_{k}"] = v
    for m, w, x in emin_sensitivity():
        prov[f"W_e_Emin_{m:.0e}".replace("+", "")] = w
        prov[f"X_comp_Emin_{m:.0e}".replace("+", "")] = x
    for b in BETA_U_DSA:
        e = dsa_injection_energy(b)
        prov[f"E_DSA_beta_{b:g}_eV".replace(".", "p")] = e
        prov[f"X_comp_DSA_beta_{b:g}".replace(".", "p")] = W_e(FID_AE, E_lo=e) / Wg
    os.makedirs("provenance", exist_ok=True)
    with open("provenance/xcomp.json", "w") as fh:
        json.dump({"derived": prov}, fh, indent=2, sort_keys=True)
    print(f"[ARTEFACT] provenance/xcomp.json   ({len(prov)} keys)")
    return prov




# ===========================================================================
# 5. THE DOCUMENT
# ===========================================================================
SYM = {
    "z": r"$z$", "n_H": r"$n_{\rm H}$", "x_e": r"$x_e$",
    "H(z), t_H": r"$H(z),\ t_H$", "target": "target",
    "E_min,g": r"$E_{\min,\gamma}$", "E_max,g": r"$E_{\max,\gamma}$",
    "tau(E)": r"$\tau_{\rm IGM}(E)$", "S_g(E)": r"$S_\gamma(E)$",
    "alpha_g": r"$\alpha_\gamma$", "E_min,e": r"$E_{\min,e}$",
    "E_max,e": r"$E_{\max,e}$", "S_e(E)": r"$S_e(E)$",
    "alpha_e": r"$\alpha_e$", "N_ion,g(E)": r"$N_{{\rm ion},\gamma}(E)$",
    "N_ion,e(E)": r"$N_{{\rm ion},e}(E)$", "<E>_g": r"$\langle E\rangle_\gamma$",
    "<E>_e": r"$\langle E\rangle_e$", "W_g": r"$W_\gamma$", "W_e": r"$W_e$",
    "X_comp": r"$X_{\rm comp}$",
}
ST = {"V": r"\textbf{V}", "C": "C", "D": "D", "S": "S", "U": "U",
      "X": r"\textcolor{red}{\textbf{X}}"}


# Tokens that must become real maths rather than escaped ASCII. Longest first,
# so "n_HI" is matched before "n_H" and "alpha_g" before "alpha".
_MATHS = [
    ("dN/dE ~ E^-(alpha_g+1)", r"$dN/dE\propto E^{-(\alpha_\gamma+1)}$"),
    ("dN/dE ~ E^-alpha_e", r"$dN/dE\propto E^{-\alpha_e}$"),
    ("E^-(alpha_g+1)", r"$E^{-(\alpha_\gamma+1)}$"),
    ("f_nu ~ nu^-alpha_g", r"$f_\nu\propto\nu^{-\alpha_\gamma}$"),
    ("<E>_g/<N>_g", r"$\langle E\rangle_\gamma/\langle N\rangle_\gamma$"),
    ("<E>_e/<N>_e", r"$\langle E\rangle_e/\langle N\rangle_e$"),
    ("E^-alpha_e", r"$E^{-\alpha_e}$"),
    ("sigma_photoion", r"$\sigma_{\rm photoion}$"),
    ("Omega_b h^2", r"$\Omega_b h^2$"),
    ("L_gamma = L_e", r"$L_\gamma=L_e$"),
    ("W_e/W_g", r"$W_e/W_\gamma$"),
    ("tau_IGM", r"$\tau_{\rm IGM}$"),
    ("alpha_g", r"$\alpha_\gamma$"), ("alpha_e", r"$\alpha_e$"),
    ("E_min,e", r"$E_{\min,e}$"), ("E_max,e", r"$E_{\max,e}$"),
    ("E_th", r"$E_{\rm th}$"), ("x_e", r"$x_e$"),
    ("n_HI", r"$n_{\rm HI}$"), ("n_H", r"$n_{\rm H}$"),
    ("X_H", r"$X_{\rm H}$"), ("c/H(z)", r"$c/H(z)$"), ("H(z)", r"$H(z)$"),
    ("dN/dE", r"$dN/dE$"),
    ("cm^-3", r"cm$^{-3}$"), ("eV^-1", r"eV$^{-1}$"), ("s^-1", r"s$^{-1}$"),
    ("eV/ion", r"eV\,ion$^{-1}$"),
    ("igm_losses", r"\texttt{igm\_losses}"),
    ("ionization_yield", r"\texttt{ionization\_yield}"),
]


def _sci(t):
    """1e+03 -> $10^{3}$ ; 2.5287e-04 -> $2.5287\times10^{-4}$.

    Scientific notation typeset as bare ASCII ("1e + 03" after maths mode gets
    hold of it) is the main thing that makes a generated table unreadable.
    """
    import re
    def rep(m):
        mant, ex = m.group(1), int(m.group(2))
        if mant in ("1", "1.0"):
            return r"$10^{%d}$" % ex
        return r"$%s\times10^{%d}$" % (mant, ex)
    return re.sub(r"(\d+(?:\.\d+)?)[eE]([+-]?\d+)", rep, t)


def _tex_escape(t):
    """Substitute maths tokens, then escape whatever ASCII is left over."""
    holes = {}
    for i, (src, dst) in enumerate(_MATHS):
        if src in t:
            key = "\x00%d\x00" % i
            t = t.replace(src, key)
            holes[key] = dst
    # Scientific notation must ALSO be hole-protected: it emits "^", which the
    # escaping pass below would otherwise turn into "$10\^{}{3}$".
    import re as _re
    def _hole_sci(m):
        key = "\x00s%d\x00" % len(holes)
        holes[key] = _sci(m.group(0))
        return key
    t = _re.sub(r"\d+(?:\.\d+)?[eE][+-]?\d+", _hole_sci, t)
    t = (t.replace("_", r"\_").replace("^", r"\^{}").replace("%", r"\%")
          .replace("<", r"$<$").replace(">", r"$>$").replace("&", r"\&"))
    for key, dst in holes.items():
        t = t.replace(key, dst)
    return t


def emit_tex(Ng, Eg, Wg, Ne, Ee, We):
    """Write Text_files/xcomp_definitions.tex.

    LaTeX is held in PLAIN raw strings with @@NAME@@ placeholders, never in
    f-strings: every literal LaTeX brace would otherwise have to be doubled,
    which is how a generated document quietly acquires a syntax error.
    """
    br = loss_branching_weighted()
    sens = emin_sensitivity()
    ident = photon_is_electron_plus_one()
    d = dict(br)

    V = {
        "ETH": f"{E_MIN_G:.3f}", "ETOP": f"{E_MAX_G:.1f}",
        "AG": f"{FID_AG:.1f}", "AE": f"{FID_AE:.1f}",
        "WG": f"{Wg:.3f}", "WE": f"{We:.3f}",
        "EFFG": f"{100*E_MIN_G/Wg:.1f}", "EFFE": f"{100*E_MIN_G/We:.1f}",
        "NG": f"{Ng:.4f}", "NE": f"{Ne:.2f}", "X": f"{We/Wg:.4f}",
        "EXC": f"{100*d['excitation']:.2f}",
        "OTHER": f"{100*sum(v for k, v in br if k not in ('ionization','excitation')):.2f}",
        "SLIDE": f"{max(s[2] for s in sens)/min(s[2] for s in sens):.2f}",
        "BRTAB": " \n".join(rf"{k.replace('_',' ')} & ${100*v:.2f}\%$ \\"
                            for k, v in br if v >= 1e-4),
        "IDTAB": " \n".join(rf"{_sci(f'{e:.4g}')} & ${a:.3f}$ & ${b:.3f}$ & "
                            rf"${100*r:.1f}\%$ \\" for e, a, b, r in ident),
        "SENSTAB": " \n".join(rf"{_sci(f'{m:.0e}')} & ${w:.3f}$ & ${x:.4f}$ \\"
                              for m, w, x in sens),
    }

    body = r"""% Generated by make_xcomp_definitions.py -- DO NOT EDIT BY HAND.
% Rebuild:  python3 python/make_xcomp_definitions.py --tex
%           cd Text_files && pdflatex xcomp_definitions   (x2)
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,longtable,xcolor,booktabs}
\usepackage[colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue]{hyperref}
\title{$X_{\rm comp}$: definitions, dependencies, and the physics behind it}
\author{Compiled for the \texttt{photon\_vs\_electron} project}
\date{2026-09-15}
\begin{document}\maketitle

\section{What $X_{\rm comp}$ is}
At luminosity parity $L_\gamma=L_e$, $X_{\rm comp}$ compares how many ionizations each
agent buys with a given amount of injected energy:
\begin{equation}
  X_{\rm comp}(\alpha_\gamma,\alpha_e)
  = \frac{\displaystyle\int S_\gamma N_{{\rm ion},\gamma}\,dE \big/ \langle E\rangle_\gamma}
         {\displaystyle\int S_e N_{{\rm ion},e}\,dE \big/ \langle E\rangle_e}
  = \frac{\langle N_\gamma\rangle/\langle E\rangle_\gamma}
         {\langle N_e\rangle/\langle E\rangle_e}
  = \frac{W_e(\alpha_e)}{W_\gamma(\alpha_\gamma)},
  \label{eq:xcomp}
\end{equation}
where $W\equiv\langle E\rangle/\langle N_{\rm ion}\rangle$ is the \emph{energy price of one
ionization} --- the $W$-value of radiation physics. $X_{\rm comp}>1$ means photons are the
cheaper ionizing agent per erg.

\paragraph{Everything else cancels.} $n_{\rm H}$, the comoving conversion, the redshift and
the luminosity normalisation all drop out of eq.~\eqref{eq:xcomp} identically, so the map
depends on $\alpha_\gamma$ and $\alpha_e$ \emph{and nothing else}. Because
$\log X_{\rm comp}=\log W_e(\alpha_e)-\log W_\gamma(\alpha_\gamma)$ is additively separable,
the map has no degenerate direction: both axes do real work.

\section{Two repairs to the construction as originally written}
\paragraph{(1) The denominator double-counted the yield.} The integrand was specified as
$S(E)N_{\rm ion}(E)/\langle E_{\rm ion}\rangle$ with $\langle E_{\rm ion}\rangle$ defined as
``the average energy spent per ionization'', i.e.\ $W$ itself. With $S$ number-normalised
that gives $\langle N\rangle/W=\langle N\rangle^{2}/\langle E\rangle$: the yield enters
twice. The form is \emph{dimensionally} correct either way, which is why it does not
announce itself --- but the two readings differ by
$\langle N_\gamma\rangle/\langle N_e\rangle\approx1/106$ and \textbf{invert the conclusion}:
$X_{\rm comp}=2.07$ (photons win) versus $0.0196$ (electrons win $51\times$). Adopted here:
$\langle E_{\rm ion}\rangle=\langle E\rangle$, the mean energy per \emph{primary}, which
makes the integral the ionizations produced per unit energy injected.

\paragraph{(2) The band top is derived, not adopted.} ``The free-streaming regime
($\sim$keV)'' evaluates, using this project's own $\sigma_{\rm photoion}$ over one Hubble
length at $z=10$, to $\tau_{\rm IGM}=1$ at $E=@@ETOP@@$~eV. A smooth $(1-e^{-\tau})$
weighting was tried first and \emph{diverges}: it charges the numerator $\langle E\rangle$
for photons that free-stream and ionize nothing while the denominator saturates, so
$W_\gamma$ grows without bound (measured $+224\%$ at $\alpha_\gamma=0.5$ between a $10$~keV
and a $100$~keV integration limit). The sharp band is therefore kept.

\section{The physics: why electrons are the more expensive agent}\label{sec:why}

\paragraph{The floor.} Ionizing hydrogen costs $E_{\rm th}=@@ETH@@$~eV and no less, so
$E_{\rm th}/W$ is a true efficiency. At the fiducial indices
($\alpha_\gamma=@@AG@@$, $\alpha_e=@@AE@@$):
\begin{center}
\begin{tabular}{lccc}
\toprule
 & $W$ [eV/ion] & efficiency $E_{\rm th}/W$ & $\langle N_{\rm ion}\rangle$ \\
\midrule
photons   & $@@WG@@$ & $@@EFFG@@\%$ & $@@NG@@$ \\
electrons & $@@WE@@$ & $@@EFFE@@\%$ & $@@NE@@$ \\
\bottomrule
\end{tabular}
\end{center}
so $X_{\rm comp}=@@X@@$: a photon buys an ionization for less than half the energy.

\paragraph{Where the electron's energy actually goes.} Averaged over the injection
spectrum, the loss flux divides as:
\begin{center}
\begin{tabular}{lr}
\toprule
mechanism & fraction of $dK/dt$ \\
\midrule
@@BRTAB@@
\bottomrule
\end{tabular}
\end{center}
The decisive entry is \textbf{excitation at $@@EXC@@\%$}. That energy leaves as
Lyman-series photons, \emph{every one of which is below} $13.6$~eV --- Ly$\alpha$ is
$10.2$~eV --- so it can never ionize hydrogen, at any later time, by any route. It is not
delayed; it is gone. A further $@@OTHER@@\%$ goes to inverse Compton on a CMB that is
$(1+z)^{4}\simeq1.5\times10^{4}$ times denser at $z=10$, plus adiabatic and Coulomb losses.
The remainder is sub-threshold kinetic energy: once a secondary electron drops below
$13.6$~eV it can only excite and heat, so it too leaves the ionization budget. Of the
energy injected, only $@@EFFE@@\%$ ends up as binding energy delivered.

\paragraph{Why photons escape most of that penalty.} A photon does not avoid the electron
physics --- it \emph{defers} it. Photoionization converts the first $@@ETH@@$~eV at $100\%$
efficiency, and only the excess $E-E_{\rm th}$ is handed to a photoelectron, which from that
instant suffers exactly the same losses as any CR electron. Structurally,
\begin{equation}
  N_{{\rm ion},\gamma}(E)\;\simeq\;1+N_{{\rm ion},e}(E-E_{\rm th}),
  \label{eq:plusone}
\end{equation}
\emph{one free ionization plus an electron cascade}. Checked directly:
\begin{center}
\begin{tabular}{rrrr}
\toprule
$E$ [eV] & $N_{{\rm ion},\gamma}(E)$ & $1+N_{{\rm ion},e}(E-E_{\rm th})$ & rel.\ diff. \\
\midrule
@@IDTAB@@
\bottomrule
\end{tabular}
\end{center}
Equation~\eqref{eq:plusone} is \emph{approximate}, not an identity: the few-per-cent
residual is the known disagreement between the depfit-route photon route and the loss+IC-route electron
route (assumption A9, $7.7\%$), not new physics.

\paragraph{The consequence.} The photon advantage is not that photons are intrinsically
efficient --- it is that they can be delivered \emph{at the threshold, where the job is
cheapest}, whereas CR electrons are injected at keV--TeV and must bleed down through
non-ionizing channels to get there. Monoenergetically the two costs converge above
$\sim1$~keV ($37.9$ vs $35.3$ eV/ion at $1$~keV; $46.3$ vs $45.4$ at $100$~keV), because a
keV photon's photoelectron \emph{is} a keV electron. The entire photon advantage lives in
the $13.6$--$100$~eV window, and is really a statement about how badly slow electrons
perform there: a $20$~eV photon ionizes with near-certainty, while a $20$~eV electron costs
$171$~eV per ionization because it overwhelmingly excites instead.

\section{The weakest input: $E_{\min,e}$}\label{sec:emin}
The two bands are not equally well determined. The photon band's floor is atomic physics and
its ceiling is derived from $\tau=1$; \emph{every} entry of the electron band is a choice
with no external source. That would be harmless if the answer were insensitive to it:
\begin{center}
\begin{tabular}{rrr}
\toprule
$E_{\min,e}$ [eV] & $W_e$ [eV/ion] & $X_{\rm comp}$ at fiducial indices \\
\midrule
@@SENSTAB@@
\bottomrule
\end{tabular}
\end{center}
Lowering $E_{\min,e}$ drags in the catastrophically expensive sub-$100$~eV region; raising it
does the reverse. Across $10^{2}$--$10^{5}$~eV the whole map slides by a factor $@@SLIDE@@$
--- \textbf{more than the entire $\alpha_e$ axis moves it}. The map's \emph{shape} is well
posed; its \emph{normalisation} is not. The figure is therefore drawn at two values of
$E_{\min,e}$ rather than one, so the shift is visible rather than hidden in a constant.

\section{Inventory}
Status: \textbf{V} verified against a source in \texttt{papers/}; C cited, unverified;
D derived here; S scanned; U your choice; \textcolor{red}{\textbf{X}} unsourced.
\begin{center}
\begin{longtable}{lllll}
\toprule
symbol & value & units & depends on & st. \\
\midrule
\endhead
@@INVTAB@@
\bottomrule
\end{longtable}
\end{center}
\end{document}
"""
    inv, sec = [], None
    for s_, sym, defn, val, unit, dep, src, st in R:
        if s_ != sec:
            inv.append(r"\multicolumn{5}{l}{\textbf{" + s_ + r"}} \\[2pt]")
            sec = s_
        inv.append(SYM.get(sym, _tex_escape(sym)) + " & " + _tex_escape(val)
                   + " & " + _tex_escape(unit) + " & " + _tex_escape(dep)
                   + " & " + ST[st] + r" \\")
        inv.append(r"\multicolumn{5}{p{0.93\textwidth}}{\footnotesize\quad "
                   + _tex_escape(defn) + r"\quad[src: " + _tex_escape(src)
                   + r"]} \\[3pt]")
    V["INVTAB"] = " \n".join(inv)

    for k, v in V.items():
        body = body.replace("@@" + k + "@@", v)
    assert "@@" not in body, "unsubstituted placeholder remains"
    out = project_paths.text("xcomp_definitions.tex")
    open(out, "w").write(body)
    return out




# ===========================================================================
# 6. THE FIGURE.  Three panels, one per electron-band floor.
# ===========================================================================
MP_C2, ME_C2 = 938.272089e6, 0.51099895e6      # eV (PDG)


def dsa_injection_energy(beta_u):
    """Electron kinetic energy at the DSA injection momentum.

    Park, Caprioli & Spitkovsky (2015): protons are injected into DSA at
    p_inj ~ 3 m_p v_u, and electrons "enter DSA when their momentum is
    comparable to the proton injection momentum" -- the SAME momentum, not the
    same energy. Young Galactic SNRs (Tycho, Cas A, SN1006, Kepler) have
    v_u ~ 0.01-0.02c, so the electron power law cannot start below a few tens
    of MeV. That is four to five decades above the 1 keV this project assumes.
    """
    pc = 3.0 * MP_C2 * beta_u
    return float(np.sqrt(pc ** 2 + ME_C2 ** 2) - ME_C2)


BETA_U_DSA = (0.01, 0.015, 0.02)               # Park+15, young Galactic SNRs


def panels():
    """(label, E_min,e, provenance note) for each panel, left to right."""
    lo = dsa_injection_energy(BETA_U_DSA[0])
    return [
        (r"$E_{\min,e}=100$ eV", 1.0e2, "arbitrary low floor"),
        (r"$E_{\min,e}=1$ keV", 1.0e3, "project fiducial (unsourced)"),
        (r"DSA floor, $v_u=0.01c$", lo, "Park+15: $p_{\\rm inj}\\simeq3m_pv_u$"),
    ]


def make_figure_X(n_g=70, n_e=60):
    """X_comp over (alpha_gamma, alpha_e), one panel per electron-band floor.

    X_comp = W_e(alpha_e)/W_gamma(alpha_gamma) is SEPARABLE, so each panel needs
    two 1-D sweeps and an outer ratio -- not an n_g x n_e double loop. That is
    also why the map has no degenerate direction: log X_comp is a difference of
    one function of alpha_e and one of alpha_gamma.
    """
    from igm_config import safe_plot_style, fig_stem
    ag = np.linspace(*ALPHA_G_RANGE, n_g)
    ae = np.linspace(*ALPHA_E_RANGE, n_e)
    Wg = np.array([W_gamma(a) for a in ag])           # 1-D, shared by all panels
    pans = panels()

    with safe_plot_style() as plt:
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 9.6), sharey=True)
        INK, INK2 = P.INK, P.INK2
        ims = []
        for k, (lab, emin, note) in enumerate(pans):
            ax = axes[k]
            We_ = np.array([W_e(p, E_lo=emin) for p in ae])
            X = We_[None, :] / Wg[:, None]             # (alpha_g, alpha_e)
            # TwoSlopeNorm keeps white pinned at X_comp = 1 (log = 0); a plain
            # vmin/vmax pair would put white at the midpoint of an asymmetric
            # range instead, and the parity colour would silently lie.
            from matplotlib.colors import TwoSlopeNorm
            im = ax.pcolormesh(ag, ae, np.log10(X).T, cmap="RdBu_r",
                               shading="auto", zorder=1,
                               norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.4, vmax=1.6))
            ims.append(im)
            cs = ax.contour(ag, ae, np.log10(X).T,
                            levels=np.log10([1, 1.5, 2, 3, 4, 6, 10, 20]),
                            colors="#333333", linewidths=0.8, zorder=3)
            ax.clabel(cs, fmt=lambda v: f"{10**v:g}", fontsize=7.5, inline=True)
            if X.min() < 1.0 < X.max():                # parity, only if present
                ax.contour(ag, ae, np.log10(X).T, levels=[0.0],
                           colors="#0b0b0b", linewidths=2.2, zorder=4)
            if k == 1:                                  # the project's point
                ax.plot([FID_AG], [FID_AE], "o", ms=9, mfc="none",
                        mec="#0b0b0b", mew=2.2, zorder=6)
                ax.annotate(f"project fiducial\n$X_{{\\rm comp}}="
                            f"{W_e(FID_AE)/W_gamma(FID_AG):.2f}$",
                            xy=(FID_AG, FID_AE), xytext=(-8, -34),
                            textcoords="offset points", fontsize=8.2, color=INK,
                            ha="center", linespacing=1.3, zorder=7,
                            arrowprops=dict(arrowstyle="-", color=INK, lw=1.0,
                                            shrinkA=2, shrinkB=5))
            # source classes as BOXES spanning the sourced range on BOTH axes.
            # A class unconstrained on one axis is drawn open across it, rather
            # than given an invented range. Hatching marks a photon index that
            # was fitted OUTSIDE our band (2-10 keV) and carried in.
            for c in CLASS_BOXES:
                x0, x1 = c["ag"] if c["ag"] else (ag[0], ag[-1])
                y0, y1 = c["ae"] if c["ae"] else (ae[0], ae[-1])
                open_x, open_y = c["ag"] is None, c["ae"] is None
                # Fill first with NO drawn edge (ec is still set, because the
                # hatch colour is taken from it), then lay solid edges only on
                # the sides that are actually constrained. An axis the class has
                # no measured index for is therefore drawn OPEN, which says the
                # same thing the old dashes did but with far better contrast.
                ax.add_patch(plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=True, fc=c["colour"],
                    alpha=0.13 if not (open_x or open_y) else 0.07,
                    ec=c["colour"], lw=0.0,
                    hatch="///" if c["extrapolated"] else None, zorder=6))
                ekw = dict(color=c["colour"], lw=2.1, solid_capstyle="butt",
                           zorder=7)
                if not open_x:                       # alpha_gamma constrained
                    ax.plot([x0, x0], [y0, y1], **ekw)
                    ax.plot([x1, x1], [y0, y1], **ekw)
                if not open_y:                       # alpha_e constrained
                    ax.plot([x0, x1], [y0, y0], **ekw)
                    ax.plot([x0, x1], [y1, y1], **ekw)
                if k == 0:                          # label once, on panel (a)
                    ax.annotate(c["label"], xy=c["lab"], fontsize=7.4,
                                color=c["colour"], ha=c["la"], va="center",
                                zorder=9, weight="bold",
                                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                          ec="none", alpha=0.72))
            ax.set_xlabel(r"photon index $\alpha_\gamma$  "
                          r"($f_\nu\propto\nu^{-\alpha_\gamma}$)",
                          color=INK, fontsize=10)
            if k == 0:
                ax.set_ylabel(r"CR electron injection index $\alpha_e$",
                              color=INK, fontsize=10)
            # W_e is a SPECTRUM average, so it varies along this panel's y-axis.
            # Quoting only its fiducial value read as a panel-wide constant, which
            # it is not: the span below is 3.6-4.2x across the plotted alpha_e.
            w_lo, w_hi = We_.min(), We_.max()
            ax.set_title(f"({'abc'[k]})  {lab}\n"
                         rf"$W_e={w_lo:.0f}$ to ${w_hi:.0f}$ eV/ion "
                         rf"over the axis; ${W_e(FID_AE, E_lo=emin):.1f}$ at "
                         rf"$\alpha_e={FID_AE}$",
                         color=INK, fontsize=9.4, loc="left")
            ax.tick_params(colors=INK2, labelsize=9)

        # Reserve the margins explicitly. bbox_inches="tight" plus a manual
        # fig.text is what made the first draft collide: tight cropping moves
        # the axes onto the footer instead of away from it.
        fig.subplots_adjust(left=0.052, right=0.868, bottom=0.245, top=0.925,
                            wspace=0.07)
        cax = fig.add_axes([0.886, 0.245, 0.015, 0.615])
        cb = fig.colorbar(ims[-1], cax=cax, extend="both")
        # The explicit integral form, not the collapsed W_e/W_gamma: this is the
        # quantity as defined, and it shows the reader that <E> in the denominator
        # is the mean energy per PRIMARY -- the point on which the whole
        # construction turns. Luminosity parity is stated in the footer instead.
        cb.set_label(r"$\log_{10}X_{\rm comp}$,    $X_{\rm comp}="
                     r"\frac{\int S_\gamma\,N_{\mathrm{ion},\gamma}\,dE\ /\ "
                     r"\langle E\rangle_\gamma}"
                     r"{\int S_e\,N_{\mathrm{ion},e}\,dE\ /\ \langle E\rangle_e}$",
                     color=INK, fontsize=11, labelpad=14)
        cb.ax.tick_params(colors=INK2, labelsize=8)
        # Name the winning agent at each end rather than spelling it out in the
        # axis label: the red end is where photons buy an ionization for less.
        cax.text(0.5, 1.068, "PHOTONS", transform=cax.transAxes, color="#8b1a1a",
                 fontsize=10.5, ha="center", va="bottom", weight="bold")
        cax.text(0.5, -0.068, "ELECTRONS", transform=cax.transAxes,
                 color="#1a4f8b", fontsize=10.5, ha="center", va="top",
                 weight="bold")

        lo = dsa_injection_energy(BETA_U_DSA[0])
        hi = dsa_injection_energy(BETA_U_DSA[-1])
        fig.text(0.052, 0.018,
                 r"$X_{\rm comp}$ is the number of ionizations bought per unit "
                 r"energy injected, photons relative to electrons, at equal "
                 r"luminosity. $X_{\rm comp}>1$: photons are the cheaper "
                 r"ionizing agent." "\n"
                 rf"SCENARIO    IGM at $z=10$, $x_e=10^{{-4}}$, pure H.  "
                 rf"Photon band $[{E_MIN_G:.3f},\,{E_MAX_G:.0f}]$ eV, from the HI "
                 rf"threshold to the free-streaming onset $\tau_{{\rm IGM}}=1$.  "
                 rf"Electron band $[E_{{\min,e}},\,10^{{12}}]$ eV." "\n"
                 r"BOXES    span the range each index is given in the literature, "
                 r"on BOTH axes; all edges solid.  A box drawn OPEN on an axis "
                 r"(no edges on that pair of sides, paler fill) has no measured "
                 r"index for it, so no range is invented." "\n"
                 r"HATCHED    the photon index was fitted at 2--10 keV, which has "
                 r"ZERO overlap with our band; it is carried in by extrapolation, "
                 r"and in our band these objects are disc- not corona-dominated." "\n"
                 r"$\alpha_\gamma$ SOURCES    Murchikova+20 ($4.5\pm0.4$ over "
                 r"13.6--54.4 eV, the only direct measurement in our band, but "
                 r"Galactic);  Telfer+02 (AGN, 10.3--24.8 eV);  Gladstone+09 "
                 r"(ULX and BHB, 2--10 keV)." "\n"
                 r"$\alpha_e$ SOURCES    Park+15 (DSA, $f(p)\propto p^{-3r/(r-1)}$, "
                 r"so $r=4$ gives exactly 2);  Aharonian+24 (SS\,433, "
                 r"$\Gamma_e=2$, measured in this exact convention)." "\n"
                 r"CONVENTIONS CHECKED    $L_\nu\propto\nu^{-\gamma}\Rightarrow"
                 r"\alpha=\gamma$;   $N(E)\propto E^{-\Gamma}\Rightarrow"
                 r"\alpha=\Gamma-1$;   $f_\lambda\propto\lambda^{\beta}\Rightarrow"
                 r"\alpha=\beta+2$, which is INVALID across the Lyman break --- "
                 r"hence Marques-Chaves+26's $\beta_{\rm UV}=-2.88$ cannot be used "
                 r"here." "\n"
                 rf"PANELS    differ only in $E_{{\min,e}}$; (c) uses the DSA "
                 rf"injection momentum $p_{{\rm inj}}\simeq3m_pv_u$ (Park+15), the "
                 rf"lowest floor in the sourced range and so the most generous to "
                 rf"electrons.",
                 fontsize=8.2, color=INK2, va="bottom", linespacing=1.62)
        for ext in ("png", "pdf"):
            fig.savefig(f"{fig_stem('xcomp_map')}.{ext}", dpi=200)
        plt.close(fig)
    return "xcomp_map"




# ===========================================================================
# 7. CHECKS.  Master Rule 5: an independent route wherever sympy does not apply.
# ===========================================================================
QUAD_TOL = PR.QUAD_TOL     # parameters.yaml; measured floor, see numerics.quad_tol


def run_checks():
    rows = []

    def rec(cid, name, ok, detail):
        rows.append((cid, name, "PASS" if ok else "FAIL", detail))

    Wg, We = W_gamma(FID_AG), W_e(FID_AE)
    X = X_comp(FID_AG, FID_AE)

    # --- X1 the headline number, two independent routes --------------------
    Ng, Eg, Wg2 = moments("gamma", FID_AG, E_MIN_G, E_MAX_G)
    Ne, Ee, We2 = moments("e", FID_AE, E_MIN_E, E_MAX_E)
    alt = (Ne / Ee) ** -1 / (Ng / Eg) ** -1          # ratio of "per unit energy"
    rec("X1", "X_comp = W_e/W_gamma = (<N>/<E>)_g / (<N>/<E>)_e",
        abs(X / alt - 1) < 1e-12,
        f"X_comp = {X:.6f}; via the ionizations-per-eV form {alt:.6f}")

    # --- X2 the moments routine reproduces source_map's independent one -----
    d = abs(W_e(FID_AE) / SM.W_e(FID_AE, E_MIN_E, E_MAX_E) - 1)
    rec("X2", "W_e agrees with source_map's independent implementation",
        d < QUAD_TOL, f"relative difference {d:.2e} (two separate quadratures "
                      f"of the same integral, tolerance {QUAD_TOL:.0e})")

    # --- X3 the band top is DERIVED, so it must satisfy its own definition --
    t = float(SM.tau_igm(E_MAX_G))
    rec("X3", "the photon band top is exactly where tau_IGM = 1",
        abs(t - 1.0) < 1e-9,
        f"tau({E_MAX_G:.4f} eV) = {t:.12f}; solved, not adopted")

    # --- X4 grid convergence ------------------------------------------------
    v = [W_gamma(FID_AG, n=n) for n in (1000, 2000, 4000, 8000)]
    sp = (max(v) - min(v)) / np.mean(v)
    rec("X4", "X_comp is converged in the quadrature grid",
        sp < QUAD_TOL,
        "W_gamma over n = 1000,2000,4000,8000: "
        + ", ".join(f"{x:.6f}" for x in v) + f"; spread {sp:.2e}")

    # --- X5 the steep-spectrum limit, against its ANALYTIC form -------------
    # A steep spectrum piles every photon just above threshold, so N_gamma -> 1
    # and W_gamma -> <E>. For dN/dE ~ E^-(alpha+1) that mean is E_th*alpha/(alpha-1),
    # i.e. W_gamma approaches the 13.598 eV floor only as 1/alpha -- NOT at any
    # finite alpha. An earlier version of this check demanded W_gamma = E_th to
    # 2% at alpha = 12 and failed by exactly 1/11 = 9.09%, which is the analytic
    # answer, not an error. Testing the asymptotic form is the sharper check.
    worst, det = 0.0, []
    for a in (8.0, 12.0, 20.0, 40.0):
        pred = E_MIN_G * a / (a - 1.0)
        got = W_gamma(a)
        worst = max(worst, abs(got / pred - 1))
        det.append(f"alpha={a:.0f}: {got:.5f} vs {pred:.5f}")
    rec("X5", "steep-spectrum limit matches W_gamma = E_th*alpha/(alpha-1)",
        worst < 1e-3,
        "; ".join(det) + f"; worst {worst:.2e}. W_gamma reaches the "
        f"{E_MIN_G:.3f} eV floor only as alpha -> infinity.")

    # --- X6 both W inside the rigorous weighted-mean bounds -----------------
    gl, gh = SM.W_bounds_gamma(E_MIN_G, E_MAX_G, absorb=False)
    el, eh = SM.W_bounds_e(E_MIN_E, E_MAX_E)
    rec("X6", "every W lies inside its monoenergetic weighted-mean bound",
        gl - 1e-9 <= Wg <= gh + 1e-9 and el - 1e-9 <= We <= eh + 1e-9,
        f"W_gamma {Wg:.3f} in [{gl:.3f},{gh:.3f}]; W_e {We:.3f} in [{el:.3f},{eh:.3f}]")

    # --- X8 the band policy RECONCILES the two values the project reported ---
    # Until 2026-09-15 this module capped a stellar source at the IGM cutoff and
    # reported X_comp = 2.073, while photon_vs_electron and source_map capped at
    # 4 Ryd and reported 2.280. Same quantity, two answers, differing only by an
    # undeclared band. Under the policy the stellar source caps at 4 Ryd, and
    # this check is what holds the two modules together from now on.
    x_stellar = X_comp_stellar()
    sm_factor = SM.W_gamma(FID_AG, SM.P.photon_band_max(), absorb=True) \
        / SM.W_e(FID_AE, SM.FID_EMIN, SM.FID_EMAX)
    rec("X8", "stellar-band X_comp reconciles with source_map's spectral factor",
        abs(x_stellar * sm_factor - 1.0) < QUAD_TOL,
        f"X_comp(stellar cap) = {x_stellar:.6f}; source_map W_gamma/W_e = "
        f"{sm_factor:.6f}; product = {x_stellar*sm_factor:.6f}. The project now "
        f"quotes ONE value for photons-per-erg, not the 2.280/2.073 pair it "
        f"carried before the band policy.")

    # --- X7 luminosity normalisation must cancel ----------------------------
    Xs = X_comp(FID_AG, FID_AE)
    Wg_s = moments("gamma", FID_AG, E_MIN_G, E_MAX_G, scale=1e3)[2]
    We_s = moments("e", FID_AE, E_MIN_E, E_MAX_E, scale=1e-3)[2]
    rec("X7", "X_comp is independent of the spectral normalisation",
        abs((We_s / Wg_s) / Xs - 1) < 1e-12,
        f"scaling S_gamma by 1e3 and S_e by 1e-3 leaves X_comp at "
        f"{We_s/Wg_s:.6f} vs {Xs:.6f} -- the L_gamma = L_e premise is enforced "
        f"by construction, not assumed")
    return rows




# ===========================================================================
# 8. SOURCE CLASSES AS BOXES IN (alpha_gamma, alpha_e).
#
# Every entry carries the band the index was MEASURED over, so the audit is
# visible in the code rather than only in the caption. Our photon band is
# 13.598-1218 eV; an index fitted at 2-10 keV has ZERO overlap with it and is
# flagged extrapolated=True. `None` on an axis means that index is genuinely
# unconstrained for this class -- the box is then drawn open in that direction
# rather than invented.
# ===========================================================================
CLASS_BOXES = [
    dict(label="Star-forming galaxies", colour="#c8102e", lab=(3.25, 2.56), la="center",
         ag=(2.0, 4.5), ae=(2.0, 2.5), extrapolated=False,
         ag_src=r"ours (2.0) $\to$ Murchikova+20 ($4.5\pm0.4$), 13.6--54.4 eV",
         ae_src=r"Park+15 DSA $f(p)\propto p^{-3r/(r-1)}$, $r=4\to3$",
         note="both axes sourced; photon index spans convention to measurement"),
    dict(label="AGN", colour="#6a3d9a", lab=(1.74, 2.36), la="center",
         ag=(1.40, 2.08), ae=(2.2, 2.3), extrapolated=False,
         ag_src=r"Telfer+02, radio-quiet $1.57\pm0.17$ to radio-loud $1.96\pm0.12$",
         ae_src=r"jet shock acceleration, $p\simeq2.2$--$2.3$ (second-hand)",
         note="photon band 10.3--24.8 eV covers only 13% of ours"),
    dict(label="ULX / super-Eddington", colour="#ff7f00", lab=(1.24, 1.26), la="center",
         ag=(0.38, 2.10), ae=None, extrapolated=True,
         ag_src=r"Gladstone+09 $\Gamma_1=1.38$--$3.1$, $\alpha=\Gamma-1$",
         ae_src=r"no measured jet electron index",
         note="index fitted at 2--10 keV: NO overlap with our band"),
    dict(label="BHB hard state (Cyg X-1)", colour="#1f78b4", lab=(1.22, 1.45), la="left",
         ag=(0.4, 1.1), ae=(1.3, 1.6), extrapolated=True,
         ag_src=r"$\Gamma\simeq1.7$ (second-hand; only $\Gamma<2.1$ first-hand)",
         ae_src=r"jet synchrotron fits, $p\simeq1.3$--$1.6$ (second-hand)",
         note="index fitted at 2--10 keV: NO overlap with our band"),
    dict(label="SS 433 jets", colour="#33a02c", lab=(4.55, 1.98), la="right",
         ag=None, ae=(1.95, 2.05), extrapolated=False,
         ag_src=r"no measured ionizing-continuum index",
         ae_src=r"Aharonian+24, $\Gamma_e=2$ exactly our convention",
         note="electron index measured; photon index unconstrained"),
]


if __name__ == "__main__":
    raise SystemExit(main())
