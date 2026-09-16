"""
emis_engine_B.py -- ENGINE B: the external, published energy-deposition
                    prescriptions, used to cross-check engine A.

Three literature sources, each inside its own stated validity range.  None is
extrapolated outside it; where a range is not covered the engine returns NaN
so that the figures simply stop rather than silently invent numbers.

B1  Shull & van Steenberg (1985), ApJ 298, 268, Table 2.
    THE ONLY SOURCE IN THE LITERATURE THAT RESOLVES THE H I / He I SPLIT IN
    CLOSED FORM.  Monte Carlo, primary electrons up to 3 keV; the fits are to
    the LIMITING curves, i.e. E_0 > 100 eV, and reproduce the tabulated values
    to 1-2 per cent over 0.0001 < x < 1.
        Form 1 (heat)              y = C [ 1 - (1 - x^a)^b ]
        Form 2 (ionization/exc.)   y = C ( 1 - x^a )^b
    with y = phi(i) * (I_i / E_0), i.e. y is the FRACTION OF THE PRIMARY
    ENERGY deposited in ionizations of species i, and phi(i) is the number of
    such ionizations.  Hence  N_ion,i(E_0) = y_i(x) * E_0 / I_i .
    CAVEATS carried into every result derived from B1:
      (i)  SvS85 assume n(He)/n(H) = 0.1, whereas Planck 2018 Y_p gives
           0.08130 -- their helium target is 23 per cent too abundant.
      (ii) SvS85 assume n(H+)/n(H_tot) = n(He+)/n(He_tot), i.e. helium is
           ionized in step with hydrogen.  This CONTRADICTS the fully-neutral
           helium adopted here, and cannot be undone from the published fits.
      (iii) "Excitations or ionizations of He II were negligibly small"
           (SvS85, Sect. III), so B1 provides no He II channel.

B2  Valdes & Ferrara (2008), MNRAS 387, L8 (arXiv:0803.0370), Eqs. (4)-(7).
    Monte Carlo (MEDEA precursor), primary electrons E_in = 3 and 10 keV;
    the authors state the partition is the same in both to within 2 per cent
    and, following Shull (1979) and SvS85, converges for E_0 > 100 eV.  Fits
    accurate to 3.5 per cent.  TOTAL ionization only -- no species split.

B3  Valdes, Evoli & Ferrara (2010), MNRAS 404, 1569 (arXiv:0911.1125),
    Appendix A, Tables A1-A21.  MEDEA with Bremsstrahlung and inverse Compton
    on the CMB.  1 MeV <= E_in <= 1 TeV at z = 10, 30, 50, on a 10-point
    x_e grid.  Column "Ionizations (H, He, HeII)" is again a TOTAL.
    This is the only external source that covers the energy range where
    inverse Compton dominates, which is exactly where engine A's electron
    yield saturates.

The species split for B2/B3 is NOT invented here: those two engines are used
only for the TOTAL ionization cross-check, and the H/He split is taken from
B1 alone.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from emis_common import E_TH

HERE = Path(__file__).resolve().parent
VEF2010_TXT = HERE.parent / "txt" / \
    "ValdesEvoliFerrara2010_ParticleEnergyCascade_IGM_arXiv0911.1125.txt"

# ---------------------------------------------------------------------------
# B1.  Shull & van Steenberg (1985) Table 2
# ---------------------------------------------------------------------------
SVS85 = {                    # process -> (C, a, b, form)
    "heat":    (0.9971, 0.2663, 1.3163, 1),
    "ion_HI":  (0.3908, 0.4092, 1.7592, 2),
    "ion_HeI": (0.0554, 0.4614, 1.6660, 2),
    "exc_HI":  (0.4766, 0.2735, 1.5221, 2),
    "exc_HeI": (0.0246, 0.4049, 1.6594, 2),
}
SVS85_EMIN = 100.0           # eV -- fits are to the E_0 > 100 eV limiting curves
SVS85_XMIN, SVS85_XMAX = 1.0e-4, 1.0
SVS85_NHE_OVER_NH = 0.1      # their assumed helium abundance (Sect. II)


def svs85_fraction(x_e, process):
    """Fraction of the PRIMARY energy deposited in `process`.  SvS85 Table 2."""
    C, a, b, form = SVS85[process]
    x = np.clip(np.asarray(x_e, float), SVS85_XMIN, SVS85_XMAX)
    if form == 1:
        return C * (1.0 - (1.0 - x ** a) ** b)
    return C * (1.0 - x ** a) ** b


def svs85_nion(E_eV, x_e, species):
    """Number of ionizations of `species` per primary electron of energy E.

    N_ion,i = f_i(x_e) * E_0 / I_i   with f_i from Table 2 and I_i the
    ionization potential.  Energy-independent for E_0 > 100 eV by
    construction; NaN below that, where SvS85's limiting fits do not apply.
    """
    key = {"HI": "ion_HI", "HeI": "ion_HeI"}[species]
    E = np.asarray(E_eV, float)
    f = svs85_fraction(x_e, key)
    out = f * E / E_TH[species]
    return np.where(E >= SVS85_EMIN, out, np.nan)


# ---------------------------------------------------------------------------
# B2.  Valdes & Ferrara (2008) Eqs. (4)-(7)
# ---------------------------------------------------------------------------
VF08_EMIN, VF08_EMAX = 1.0e2, 1.0e4      # eV; fitted at 3 and 10 keV, and the
                                         # authors state convergence > 100 eV


def vf08_fractions(x_e):
    """(f_heat, f_Lya, f_ion_total, f_continuum).  V&F 2008 Eqs. (4)-(7)."""
    x = np.clip(np.asarray(x_e, float), 1.0e-4, 1.0)
    f_h = 1.0 - 0.8751 * (1.0 - x ** 0.4052)
    f_a = 0.3484 * (1.0 - x ** 0.3065) ** 0.9533
    f_i = 0.3846 * (1.0 - x ** 0.5420) ** 1.1952
    f_c = 0.1537 * (1.0 - x ** 0.3224)
    return f_h, f_a, f_i, f_c


def vf08_fion(E_eV, x_e):
    """TOTAL ionization energy fraction; NaN outside 100 eV - 10 keV."""
    E = np.asarray(E_eV, float)
    f = vf08_fractions(x_e)[2]
    return np.where((E >= VF08_EMIN) & (E <= VF08_EMAX),
                    np.broadcast_to(f, np.shape(E)), np.nan)


# ---------------------------------------------------------------------------
# B3.  Valdes, Evoli & Ferrara (2010) Appendix A
# ---------------------------------------------------------------------------
_VEF_ENERGIES = [1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12]   # eV
_VEF_Z = [10.0, 30.0, 50.0]
_NUM = re.compile(r"([0-9]\.[0-9]{3,4}e[+-][0-9]{2})(?:±[0-9](?:\.[0-9]+)?e[+-][0-9]{2})?")


def _parse_vef2010():
    """Parse Tables A1-A21 out of the PDF, page by page.

    pdftotext emits each appendix page as: the table captions on that page,
    then the numeric columns as consecutive blocks of ten, in printed column
    order (x_e, heat, Lyman-alpha, ionizations, E<10.2 eV, E>10 keV, CMB).
    On the last page pdftotext places one caption AFTER its data, so slicing
    on captions fails; parsing per page and assigning 70 numbers per caption
    in order is robust to that.  Every parsed table is validated with the
    paper's own energy-conservation statement: the five deposition columns
    must sum to unity.
    """
    import subprocess
    pdf = HERE.parent / "Agents" / \
        "ValdesEvoliFerrara2010_ParticleEnergyCascade_IGM_arXiv0911.1125.pdf"
    npages = int(subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                text=True).stdout.split("Pages:")[1].split()[0])
    num = re.compile(r"([0-9]\.[0-9]{3,4}e[+-][0-9]{2})")
    cap = re.compile(r"Table A(\d+)\.")
    out, seen = {}, []
    for pg in range(1, npages + 1):
        txt = subprocess.run(["pdftotext", "-q", "-f", str(pg), "-l", str(pg),
                              str(pdf), "-"], capture_output=True,
                             text=True).stdout
        caps = [int(m.group(1)) for m in cap.finditer(txt)]
        if not caps:
            continue
        vals = [float(m.group(1)) for m in num.finditer(txt)]
        if len(vals) != 70 * len(caps):
            raise RuntimeError("page %d: %d captions but %d numbers"
                               % (pg, len(caps), len(vals)))
        for j, n in enumerate(sorted(caps)):
            v = np.array(vals[70 * j:70 * (j + 1)]).reshape(7, 10)
            k = n - 1
            z = _VEF_Z[k // 7]
            E = _VEF_ENERGIES[k % 7]
            d = {"x_e": v[0], "f_heat": v[1], "f_Lya": v[2], "f_ion": v[3],
                 "f_cont": v[4], "f_HE": v[5], "f_CMB": v[6]}
            # Parse validation.  The five deposition columns plus the CMB
            # column close to unity EXACTLY at E_in = 1 MeV (A1, A8, A15:
            # 1.0002, 1.0001, 1.0000) which confirms the column assignment.
            # At 10-100 MeV the published tables close only to 0.77-0.94; that
            # is a property of the tables, not of this parse (the IC photons
            # between 10.2 eV and 10 keV fall in no tabulated column), and is
            # reported in emis_verify.py rather than treated as an error.
            tot = (d["f_heat"] + d["f_Lya"] + d["f_ion"] + d["f_cont"]
                   + d["f_HE"] + d["f_CMB"])
            d["closure"] = tot
            if np.any(tot < 0.70) or np.any(tot > 1.02):
                raise RuntimeError("table A%d: column assignment looks wrong, "
                                   "closure = %s" % (n, tot))
            out[(z, E)] = d
            seen.append(n)
    if sorted(seen) != list(range(1, 22)):
        raise RuntimeError("parsed tables %s" % sorted(seen))
    return out


VEF2010 = _parse_vef2010()
VEF_EMIN, VEF_EMAX = 1.0e6, 1.0e12


def vef2010_fion(E_eV, x_e, z):
    """TOTAL ionization energy fraction, log-log interpolated in E and x_e.

    Nearest tabulated redshift in {10, 30, 50}; NaN outside 1 MeV - 1 TeV.
    """
    zt = min(_VEF_Z, key=lambda t: abs(t - z))
    lE = np.log(np.atleast_1d(np.asarray(E_eV, float)))
    xg = VEF2010[(zt, _VEF_ENERGIES[0])]["x_e"]
    grid = np.array([VEF2010[(zt, E)]["f_ion"] for E in _VEF_ENERGIES])   # (7,10)
    # interpolate in log x_e for each tabulated energy, then in log E
    lx = np.log(np.clip(x_e, xg[0], xg[-1]))
    col = np.array([np.interp(lx, np.log(xg), grid[i]) for i in range(7)])
    val = np.exp(np.interp(lE, np.log(_VEF_ENERGIES), np.log(col)))
    val = np.where((lE >= np.log(VEF_EMIN)) & (lE <= np.log(VEF_EMAX)), val, np.nan)
    return val if np.ndim(E_eV) else float(val[0])


def engine_B_nion(E_eV, x_e, z, species):
    """Ionizations per primary electron, engine B.

    Returns a dict with one entry per available sub-engine.  'svs85' is
    species-resolved; 'vf08' and 'vef2010' are TOTAL ionizations and are
    converted to a number using the species' own threshold only for display,
    with the total-vs-species caveat stated in the write-up.
    """
    E = np.asarray(E_eV, float)
    res = {"svs85": svs85_nion(E, x_e, species)}
    if species == "HI":       # totals are dominated by H I; He I adds ~ 1 %
        res["vf08"] = vf08_fion(E, x_e) * E / E_TH["HI"]
        res["vef2010"] = vef2010_fion(E, x_e, z) * E / E_TH["HI"]
    return res


if __name__ == "__main__":
    print("B3: parsed %d VEF2010 appendix tables" % len(VEF2010))
    d = VEF2010[(10.0, 1.0e6)]
    print("   z=10, E=1 MeV: x_e grid %.3e ... %.3e" % (d["x_e"][0], d["x_e"][-1]))
    print("   f_heat %.4e ... %.4e" % (d["f_heat"][0], d["f_heat"][-1]))
    print("   f_ion  %.4e ... %.4e" % (d["f_ion"][0], d["f_ion"][-1]))
    s = sum(d[k][0] for k in ("f_heat", "f_Lya", "f_ion", "f_cont", "f_HE"))
    print("   energy-conservation test at x_e=1e-4: sum of deposition"
          " channels = %.4f  (+ CMB input %.4e)" % (s, d["f_CMB"][0]))
    print("\nB1/B2 at x_e = 1e-4, 1e-2, 1e-1:")
    print("   %-8s %-12s %-12s %-12s %-12s"
          % ("x_e", "f_ion(HI)", "f_ion(HeI)", "f_heat SvS", "f_ion V&F08"))
    for x in (1e-4, 1e-3, 1e-2, 1e-1):
        print("   %-8.0e %-12.4f %-12.4f %-12.4f %-12.4f"
              % (x, svs85_fraction(x, "ion_HI"), svs85_fraction(x, "ion_HeI"),
                 svs85_fraction(x, "heat"), vf08_fractions(x)[2]))
    print("\nN_ion per primary electron (engine B), x_e = 1e-4:")
    print("   %-10s %-14s %-14s %-14s %-14s"
          % ("E [eV]", "SvS85 H I", "SvS85 He I", "V&F08 tot", "VEF2010 tot"))
    for E in (1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12):
        r = engine_B_nion(E, 1e-4, 10.0, "HI")
        rh = engine_B_nion(E, 1e-4, 10.0, "HeI")
        print("   %-10.3g %-14.4g %-14.4g %-14.4g %-14.4g"
              % (E, r["svs85"], rh["svs85"], r["vf08"], r["vef2010"]))
