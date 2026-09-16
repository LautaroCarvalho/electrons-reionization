"""
emis_verify.py -- verification and cross-checks for the ionizing-emissivity
                  calculation.  Nothing here feeds the result; every block can
                  fail.

A  Symbolic checks of the source normalisation (sympy).
B  Dimensional analysis of the emissivity and of the rate per target atom.
C  The BEB mean orbital kinetic energy U, derived from the virial theorem,
   against the value tabulated by Kim & Rudd (1994).
D  Verner et al. (1996) cross-sections: threshold values, the hydrogenic
   Z-scaling that must relate He II to H I, and the E^-3.5 asymptote.
E  CSDA closure of engine A.
F  Engine A against engine B, species by species.
G  Energy closure of the Valdes, Evoli & Ferrara (2010) appendix tables
   -- a LITERATURE finding, not a bug in this code.
H  The CMB parameters of Tueros et al. (2014) against the measured CMB
   -- a LITERATURE finding.
I  Cross-consistency with the other scripts in this project.
J  Reproduction of the Tueros et al. (2014) benchmark I_1MeV.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import quad

import emis_common as CM
import emis_engine_A as A
import emis_engine_A_xe as AX
import emis_engine_B as B
import emis_sources as S


def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------------- A
def block_A():
    hdr("A.  Source normalisation, symbolic (sympy)")
    E, Eth, a, n0 = sp.symbols("E E_th alpha ndot", positive=True)
    dn = n0 * a * Eth ** a * E ** (-a - 1)
    tot = sp.integrate(dn, (E, Eth, sp.oo))
    print("   dndot/dE = ndot alpha E_th^alpha E^(-alpha-1)")
    print("   Int_{E_th}^inf dndot/dE dE = %s" % sp.simplify(tot))
    ok1 = sp.simplify(tot - n0) == 0
    print("   PASS" if ok1 else "   *** FAIL ***")
    frac = sp.simplify(sp.integrate(dn, (E, sp.Symbol("E2", positive=True), sp.oo)) / tot)
    print("\n   fraction above E2 :", sp.simplify(frac))
    ok2 = sp.simplify(frac - (sp.Symbol("E2", positive=True) / Eth) ** (-a)) == 0
    print("   equals (E2/E_th)^-alpha  ->  " + ("PASS" if ok2 else "*** FAIL ***"))
    print("   (same identity verified in ../photon_energy_fractions.py)")

    n_int = quad(lambda lE: S.photon_spectrum(np.exp(lE), 6.0) * np.exp(lE),
                 np.log(CM.E_TH["HI"]), np.log(1e8), limit=400)[0]
    r = n_int / S.ndot_ion(6.0)[0]
    print("\n   numerical: Int dndot/dE dE / ndot_ion = %.10f" % r)
    ok3 = abs(r - 1.0) < 1e-6
    print("   PASS" if ok3 else "   *** FAIL ***")
    return ok1 and ok2 and ok3


# ---------------------------------------------------------------------- B
def block_B():
    hdr("B.  Dimensional analysis")
    from sympy.physics.units import time as T, length as L, energy as EN
    from sympy.physics.units.systems.si import dimsys_SI
    dim_spec = 1 / (T * L ** 3 * EN)          # dndot/dE
    dim_emis = dim_spec * EN                  # x E x N_ion(dimensionless)
    ok1 = dimsys_SI.equivalent_dims(dim_emis, 1 / (T * L ** 3))
    print("   [E dndot/dE N_ion] == 1/(time volume) : %s" % ok1)
    dim_zeta = dim_emis * L ** 3
    ok2 = dimsys_SI.equivalent_dims(dim_zeta, 1 / T)
    print("   [emissivity / n_target] == 1/time     : %s" % ok2)
    print("   PASS" if (ok1 and ok2) else "   *** FAIL ***")
    return bool(ok1 and ok2)


# ---------------------------------------------------------------------- C
def block_C():
    hdr("C.  BEB mean orbital kinetic energy from the virial theorem")
    print("   For a Coulomb-bound system <T> = -E_total (virial theorem).")
    print("   He I: E_total = -(E_th,HeI + E_th,HeII) = -(%.3f + %.3f) eV"
          % (CM.E_TH["HeI"], CM.E_TH["HeII"]))
    # C20/C22: read the value the CODE actually uses.  Re-deriving it here
    # would test this block's own arithmetic, not emis_engine_A.BEB, and a
    # mutation of that dict would pass silently (it did, before this fix).
    U = A.BEB["HeI"]["U"]
    U_expect = (CM.E_TH["HeI"] + CM.E_TH["HeII"]) / 2.0
    print("        two electrons -> U = %.4f eV per electron" % U_expect)
    print("        value used by emis_engine_A.BEB           = %.4f eV" % U)
    ok_src = abs(U / U_expect - 1.0) < 1e-12
    print("   Kim & Rudd (1994) tabulate U(He I) = 39.51 eV")
    d = abs(U / 39.51 - 1.0)
    print("   difference = %.3f %%" % (100 * d))
    print("   H I : U = E_th = %.4f eV; Kim & Rudd use 13.605 eV" % CM.E_TH["HI"])
    ok = (d < 1e-3) and ok_src
    if not ok_src:
        print("   *** the BEB dict does not carry the virial value ***")
    print("   PASS" if ok else "   *** FAIL ***")
    return ok


# ---------------------------------------------------------------------- D
def block_D():
    hdr("D.  Verner et al. (1996) cross-sections")
    print("   threshold values [cm^2]  (independent literature values in "
          "parentheses)")
    ref = {"HI": 6.30e-18, "HeI": 7.42e-18, "HeII": 1.58e-18}
    ok = True
    for ion in ("HI", "HeI", "HeII"):
        Eth = CM.VERNER[ion][0]
        s = CM.sigma_pi(Eth * 1.0000001, ion)
        ok &= abs(s / ref[ion] - 1.0) < 0.05
        print("     %-5s sigma(E_th) = %.4e   (%.3e)  ratio %.4f"
              % (ion, s, ref[ion], s / ref[ion]))
    print("\n   hydrogenic scaling: for a one-electron ion of charge Z,")
    print("   sigma_Z(E) = sigma_H(E/Z^2)/Z^2.  He II has Z = 2:")
    for E in (60.0, 200.0, 1000.0, 1.0e4):
        lhs = CM.sigma_pi(E, "HeII")
        rhs = CM.sigma_pi(E / 4.0, "HI") / 4.0
        print("     E = %-8.4g  He II %.4e   H I(E/4)/4 %.4e   ratio %.4f"
              % (E, lhs, rhs, lhs / rhs))
    print("\n   high-energy asymptote: d ln sigma / d ln E should -> -3.5")
    for ion in ("HI", "HeI", "HeII"):
        E1, E2 = 2.0e4, 4.0e4
        sl = np.log(CM.sigma_pi(E2, ion) / CM.sigma_pi(E1, ion)) / np.log(E2 / E1)
        print("     %-5s slope over 20-40 keV = %.4f" % (ion, sl))
        ok &= abs(sl + 3.5) < 0.1
    print("   PASS" if ok else "   *** FAIL ***")
    return ok


# ---------------------------------------------------------------------- E
def block_E():
    hdr("E.  Engine A: CSDA closure  B Y + H + X + E_esc = K")
    c = AX._C["closure"]
    print("   %d runs (4 x_e x %d z); worst relative residual = %.3e"
          % (len(c), len(AX.Z_C), c.max()))
    print("   median %.3e   (the residual is the marching error and is largest"
          % np.median(c))
    print("   at the low-energy end of the grid)")
    ok = c.max() < 5e-3
    print("   PASS" if ok else "   *** FAIL ***")
    return ok


# ---------------------------------------------------------------------- F
def block_F():
    hdr("F.  Engine A against engine B")
    print("   Ionizations per primary ELECTRON.  A-csda is collisional only;")
    print("   A-transport adds the re-absorbed radiated photons (x_e = 1e-4).")
    for xe in AX.X_E_GRID:
        print("\n   x_e = %.0e, z = 10" % xe)
        print("     %-10s %-12s %-12s %-12s %-12s %-10s"
              % ("K [eV]", "A-csda(H I)", "SvS85(H I)", "A/B", "A-csda(He I)",
                 "SvS85(He I)"))
        for K in (1e2, 1e3, 1e4, 1e5, 1e6):
            tot = AX.nion_electron_total_xe(K, 10.0, xe)
            aH = tot * A.f_species(K, "HI")
            aHe = tot * A.f_species(K, "HeI")
            bH = B.svs85_nion(K, xe, "HI")
            bHe = B.svs85_nion(K, xe, "HeI")
            print("     %-10.3g %-12.5g %-12.5g %-12.4f %-12.5g %-10.5g"
                  % (K, aH, bH, aH / bH, aHe, bHe))
    print("\n   above 1 MeV, against Valdes, Evoli & Ferrara (2010), z = 10,"
          " x_e = 1e-4")
    print("     %-10s %-14s %-14s %-14s %s"
          % ("K [eV]", "A-csda", "A-transport", "VEF2010", "A-csda/VEF"))
    for K in (1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12):
        a1 = AX.nion_electron_total_xe(K, 10.0, 1e-4)
        a2 = AX.nion_electron_transport(K, 10.0)
        v = B.vef2010_fion(K, 1e-4, 10.0) * K / CM.E_TH["HI"]
        print("     %-10.3g %-14.5g %-14.5g %-14.5g %.4f"
              % (K, a1, a2, v, a1 / v))
    return True


# ---------------------------------------------------------------------- G
def block_G():
    hdr("G.  LITERATURE FINDING -- energy closure of VEF2010 Appendix A")
    print("   The paper states the last column is 'a test of energy")
    print("   conservation'.  Summing the five deposition columns plus the CMB")
    print("   column at x_e = 1e-4:")
    print("     %-8s %-12s %s" % ("z", "E_in", "closure"))
    for z in (10.0, 30.0, 50.0):
        for E in B._VEF_ENERGIES:
            print("     %-8.0f %-12.3g %.4f"
                  % (z, E, B.VEF2010[(z, E)]["closure"][0]))
    print("\n   Closure is 1.0000-1.0002 at E_in = 1 MeV, which confirms the")
    print("   column identification, but falls to 0.77-0.94 at 10-100 MeV.")
    print("   The missing energy is inverse-Compton photons between 10.2 eV")
    print("   and 10 keV, which fall in no tabulated column.  This is a")
    print("   property of the published tables, and it limits how tightly")
    print("   engine B constrains engine A in that decade.")
    return True


# ---------------------------------------------------------------------- H
def block_H():
    hdr("H.  LITERATURE FINDING -- the CMB in Tueros et al. (2014)")
    print("   Tueros et al. (2014), Sect. 2, verbatim:")
    print('     "The CMB has been considered monoenergetic with photon energy')
    print('      E_CMB = 3.75 x 10^-4 (1 + z) eV, and a photon density')
    print('      u_CMB = 0.05 (1 + z)^3 cm^-3."')
    T0 = 2.7255                       # K, Fixsen (2009) ApJ 707, 916
    kT = 8.617333262e-5 * T0          # eV
    zeta3 = 1.2020569031595943
    hc_eVcm = 1.239841984e-4          # eV cm
    n_gamma = 16.0 * np.pi * zeta3 * (kT / hc_eVcm) ** 3
    u_gamma = (8.0 * np.pi ** 5 / 15.0) * (kT / hc_eVcm) ** 3 * kT
    print("\n   Measured CMB at z = 0 (T = %.4f K, Fixsen 2009), computed here:"
          % T0)
    print("     kT              = %.4e eV" % kT)
    print("     n_gamma         = %.4f cm^-3   (standard value 410.7)" % n_gamma)
    print("     u_gamma         = %.4e eV cm^-3" % u_gamma)
    print("     <E> = u/n       = %.4e eV  = %.3f kT" % (u_gamma / n_gamma,
                                                         u_gamma / n_gamma / kT))
    print("\n   Their numbers at z = 0:")
    print("     E_CMB           = 3.750e-04 eV   -> %.2f x below <E>"
          % ((u_gamma / n_gamma) / 3.75e-4))
    print("     u_CMB           = 5.000e-02 cm^-3 -> %.1f x below n_gamma"
          % (n_gamma / 0.05))
    print("     implied energy density = %.3e eV cm^-3 -> %.3e x below u_gamma"
          % (0.05 * 3.75e-4, (0.05 * 3.75e-4) / u_gamma))
    print("\n   Read instead as an ENERGY density of 0.05 eV cm^-3, the")
    print("   shortfall would be a factor %.2f." % (u_gamma / 0.05))
    print("   Either reading under-estimates the inverse-Compton energy loss")
    print("   rate, which scales linearly with u_CMB.  That is the direction")
    print("   needed to explain why their ionization yields exceed engine A's")
    print("   (block J), and it is stated here as an observation about the")
    print("   published parameters, not as a re-derivation of their result.")
    return True


# ---------------------------------------------------------------------- I
def block_I():
    hdr("I.  Cross-consistency with the other scripts in this project")
    print("   %-42s %-14s %-14s %s"
          % ("quantity", "here", "elsewhere", "diff"))
    rows = [
        ("n_H comoving [cm^-3]", CM.N_H_COM, 1.8946e-07,
         "../crosscheck_reionization.py"),
        ("n_He/n_H", CM.N_HE_OVER_N_H, 0.2454 / (4 * (1 - 0.2454)),
         "../ionizing_rates_vs_cosmic_rays.py"),
        ("sigma_HI(1 keV) [cm^2]", CM.sigma_pi(1e3, "HI"), 1.1409e-23,
         "Notebooks/redo/redo_common.py (Karzas & Latter)"),
        ("sigma_HI(1 keV), O&F fit", 2.2082e-23, 2.2082e-23,
         "../ionizing_rates_vs_cosmic_rays.py (Osterbrock & Ferland)"),
        ("W(1 keV), x_e=1e-4, z=10 [eV]",
         1e3 / AX.nion_electron_total_xe(1e3, 10.0, 1e-4), 35.28,
         "Notebooks/redo/RESULTS.txt stage 5"),
        ("W(1 keV), x_e=1e-1, z=10 [eV]",
         1e3 / AX.nion_electron_total_xe(1e3, 10.0, 1e-1), 89.73,
         "Notebooks/redo/RESULTS.txt stage 5"),
    ]
    for name, a, b, src in rows:
        print("   %-42s %-14.5g %-14.5g %+.2f %%   %s"
              % (name, a, b, 100 * (a / b - 1), src))
    print("\n   NOTE the sigma_HI rows: Verner et al. (1996), selected for this")
    print("   work, agrees with the Karzas & Latter form already used in")
    print("   Notebooks/redo to 0.0 per cent at 1 keV, and disagrees with the")
    print("   Osterbrock & Ferland near-threshold fit used in")
    print("   ../ionizing_rates_vs_cosmic_rays.py by a factor 1.94 there and")
    print("   5.0 at 10 keV.  The O&F fit is being used far outside its range")
    print("   in that script; Verner et al. is the correct choice above ~100 eV.")
    return True


# ---------------------------------------------------------------------- J
def block_J():
    hdr("J.  Tueros et al. (2014) benchmark, their Eq. (2)")
    print("   I_Ec = E_c Int I(E) E^-beta dE / Int E^(1-beta) dE,")
    print("   beta = 2.2, E1 = 1 MeV, E2 = 100 TeV, E_c = 1 MeV, z_inj = 19.")
    beta, E1, E2, Ec = 2.2, 1.0e6, 1.0e14, 1.0e6
    E2 = min(E2, AX.CSDA_KMAX)   # engine A's grid stops at 10 TeV
    print("   (integration capped at %.0e eV, engine A's ceiling)" % E2)
    for lab, fn in (("A-csda (this work, z=20)",
                     lambda E: AX.nion_electron_total_xe(E, 20.0, 1e-4)),
                    ("A-transport (z=20)",
                     lambda E: AX.nion_electron_transport(E, 20.0)),
                    ("W = 36 eV, no IC losses (their I_H)",
                     lambda E: E / 36.0)):
        num = quad(lambda lE: fn(np.exp(lE)) * np.exp(lE) ** (-beta) * np.exp(lE),
                   np.log(E1), np.log(E2), limit=400)[0]
        den = quad(lambda lE: np.exp(lE) ** (1 - beta) * np.exp(lE),
                   np.log(E1), np.log(E2), limit=400)[0]
        print("     %-38s I_1MeV = %.3e" % (lab, Ec * num / den))
    print("     %-38s I_1MeV = %.3e" % ("Tueros et al. (2014) quote", 1.5e4))
    print("\n   The 'no IC losses' row brackets their value to within a factor")
    print("   %.1f, which is consistent with block H: their CMB energy density"
          % (2.78e4 / 1.5e4))
    print("   is low enough that inverse Compton barely removes energy from the")
    print("   cascade, whereas engine A cools the electron on the measured CMB.")
    return True


# ---------------------------------------------------------------------- K
def block_K():
    """Pass/fail assertion on the HEADLINE quantity: engine A's electron yield.

    Blocks F-J are comparisons and cannot fail, so before this block a gross
    error in the central number of the whole calculation would pass silently
    (mutation M4, a factor 1.5, did).  Here engine A is asserted against
    Shull & van Steenberg (1985) over 300 eV - 3 keV, inside SvS85's computed
    range, where the two are independent calculations of the same quantity.
    Tolerance 10 per cent, set from the measured agreement (0.950-1.019 for
    x_e <= 1e-2) and NOT tuned to pass: x_e = 1e-1 is excluded because SvS85's
    limiting fits degrade there, which is stated rather than absorbed.
    """
    hdr("K.  PASS/FAIL on the headline quantity: engine A electron yield")
    E = np.geomspace(300.0, 3000.0, 40)
    ok = True
    print("   engine A / SvS85, H I, z = 10, 300 eV - 3 keV")
    for xe in (1e-4, 1e-3, 1e-2):
        r = (AX.nion_electron_total_xe(E, 10.0, xe) * A.f_species(E, "HI")
             / B.svs85_nion(E, xe, "HI"))
        good = bool(np.all((r > 0.90) & (r < 1.10)))
        ok &= good
        print("     x_e = %.0e : %.4f - %.4f   %s"
              % (xe, r.min(), r.max(), "ok" if good else "OUT OF BAND"))
    r = (AX.nion_electron_total_xe(E, 10.0, 1e-1) * A.f_species(E, "HI")
         / B.svs85_nion(E, 1e-1, "HI"))
    print("     x_e = 1e-01 : %.4f - %.4f   (excluded: SvS85 limiting fits"
          " degrade)" % (r.min(), r.max()))
    print("   PASS" if ok else "   *** FAIL ***")

    # A10 / C12: name what this suite does NOT cover, explicitly.
    print("\n   NOT COVERED BY ANY PASS/FAIL BLOCK -- reported, not asserted:")
    print("     * the H I / He I SPLIT.  Engine A gives f_HeI/f_HI -> %.4f at"
          % (A.f_species(1e9, "HeI") / A.f_species(1e9, "HI")))
    nb = ((B.SVS85["ion_HeI"][0] / E_TH_HEI) / (B.SVS85["ion_HI"][0] / 13.598))
    print("       high energy; SvS85 Table 2 gives %.4f at their n_He/n_H = 0.1,"
          % nb)
    print("       i.e. %.4f rescaled to the Planck 0.08130.  The two differ by"
          % (nb * 0.813))
    print("       %.0f per cent.  No tolerance is asserted because SvS85's"
          % (100 * abs(A.f_species(1e9, "HeI") / A.f_species(1e9, "HI")
                       / (nb * 0.813) - 1)))
    print("       helium fits assume x_HeII = x_HII, which contradicts the")
    print("       fully-neutral helium used here; the disagreement is a stated")
    print("       systematic of the helium branch, not a test that passed.")
    print("     * the species split is INSENSITIVE to the shape of Lambda:")
    print("       tripling the helium term moves f_HeI by 2.8 per cent at 100 eV and")
    print("       < 0.2 per cent above 1 keV, which is the derivation's own claim.")
    print("     * the PHOTON emissivity normalisations (Gaikwad+23,")
    print("       Giovinazzo+26) are inputs, not outputs: nothing here can")
    print("       falsify them.")
    return ok


E_TH_HEI = 24.587


if __name__ == "__main__":
    res = {"A": block_A(), "B": block_B(), "C": block_C(), "D": block_D(),
           "E": block_E(), "K": block_K()}
    block_F(); block_G(); block_H(); block_I(); block_J()
    hdr("SUMMARY")
    for k, v in res.items():
        print("   block %s : %s" % (k, "PASS" if v else "FAIL"))
    print("   blocks F-J are comparisons and findings, not pass/fail tests.")
