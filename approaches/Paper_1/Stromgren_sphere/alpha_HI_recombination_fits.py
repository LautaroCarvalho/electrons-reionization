#!/usr/bin/env python3
"""
Temperature dependence of the H I (H+ + e-) radiative recombination coefficient.

Implements the analytic fits collected in Sec. 1.4 of stromgren_sphere_summary.tex.
All coefficients are transcribed from the PDFs in references/; nothing is invented.

  hui_case_B      Hui & Gnedin (1997), Appendix A, Case B   [0.7%, 1 K - 1e9 K]
  hui_case_A      Hui & Gnedin (1997), Appendix A, Case A   [2%,   3 K - 1e9 K]
  verner_total    Verner & Ferland (1996), eq. (4) + Table 1 H I row  [<0.5%, 3 K - 1e10 K]
  seaton_total    Seaton (1959) as quoted by Verner & Ferland (1996), eq. (6)
  abel_total      Abel et al. (1997), Table 1, rate k2  -- NOTE: argument in eV

Units: cm^3 s^-1 throughout; T in K (except abel_total, see docstring).
Run:  python3 alpha_HI_recombination_fits.py
"""
import numpy as np

T_HI_TR = 157807.0   # K, H I ionization threshold in temperature units (Hui & Gnedin 1997)
KB_EV   = 8.617333262e-5  # eV/K (CODATA)


def hui_case_B(T):
    """Case B recombination coefficient, Hui & Gnedin (1997) App. A.
    Fit to Ferland et al. (1992); quoted accuracy 0.7% for 1 K <= T <= 1e9 K."""
    lam = 2.0 * T_HI_TR / np.asarray(T, dtype=float)
    return 2.753e-14 * lam**1.500 / (1.0 + (lam / 2.740)**0.407)**2.242


def hui_case_A(T):
    """Case A recombination coefficient, Hui & Gnedin (1997) App. A.
    Fit to Ferland et al. (1992); quoted accuracy 2% for 3 K <= T <= 1e9 K."""
    lam = 2.0 * T_HI_TR / np.asarray(T, dtype=float)
    return 1.269e-13 * lam**1.503 / (1.0 + (lam / 0.522)**0.470)**1.923


def verner_total(T):
    """Total (= Case A) radiative recombination coefficient for H I.
    Verner & Ferland (1996) eq. (4) with their Table 1 parameters for H I.
    Quoted rms error < 0.5% for hydrogenic species, 3 K <= T <= 1e10 K."""
    a, b, T0, T1 = 7.982e-11, 0.7480, 3.148, 7.036e5   # Table 1, row 'H I'
    T = np.asarray(T, dtype=float)
    s0, s1 = np.sqrt(T / T0), np.sqrt(T / T1)
    return a / (s0 * (1.0 + s0)**(1.0 - b) * (1.0 + s1)**(1.0 + b))


def seaton_total(T, Z=1):
    """Seaton (1959) hydrogenic total radiative recombination coefficient, as
    quoted in eq. (6) of Verner & Ferland (1996). Not valid for T > 1e6 Z^2 K."""
    lam = 157890.0 * Z**2 / np.asarray(T, dtype=float)
    return 5.197e-14 * Z * np.sqrt(lam) * (0.4288 + 0.5*np.log(lam) + 0.469*lam**(-1.0/3.0))


_ABEL_K2 = [-28.6130338, -0.72411256, -2.02604473e-2, -2.38086188e-3,
            -3.21260521e-4, -1.42150291e-5, 4.98910892e-6, 5.75561414e-7,
            -1.85676704e-8, -3.07113524e-9]


def abel_total(T):
    """Abel et al. (1997) Table 1 rate k2 for H+ + e- -> H + gamma.
    CAUTION: that table states 'the temperatures are in eV unless stated
    otherwise', so the polynomial argument is ln(T[eV]), not ln(T[K]).
    This function takes T in K and converts internally."""
    L = np.log(np.asarray(T, dtype=float) * KB_EV)
    return np.exp(sum(c * L**i for i, c in enumerate(_ABEL_K2)))


def power_law_B(T):
    """Local power-law approximation to Case B anchored at 1e4 K (this work,
    derived from hui_case_B; max 5.3% error over 5e3-3e4 K)."""
    return 2.59e-13 * (np.asarray(T, dtype=float) / 1e4)**(-0.83)


def power_law_A(T):
    """Local power-law approximation to Case A anchored at 1e4 K; the exponent
    -0.7 is the one quoted by Gnedin & Madau (2022) (max 3.7% over 5e3-3e4 K)."""
    return 4.2e-13 * (np.asarray(T, dtype=float) / 1e4)**(-0.7)


if __name__ == "__main__":
    print("H I radiative recombination coefficient [cm^3 s^-1]\n")
    hdr = ("T [K]", "alpha_B (HG97)", "alpha_A (HG97)", "V&F96 tot",
           "Seaton59 tot", "Abel97 tot", "PL_B", "PL_A")
    print("{:>9} {:>15} {:>15} {:>12} {:>13} {:>12} {:>11} {:>11}".format(*hdr))
    for T in [1e2, 1e3, 5e3, 1e4, 1.5e4, 2e4, 3e4, 1e5]:
        print("{:9.0f} {:15.4e} {:15.4e} {:12.4e} {:13.4e} {:12.4e} {:11.4e} {:11.4e}".format(
            T, hui_case_B(T), hui_case_A(T), verner_total(T), seaton_total(T),
            abel_total(T), power_law_B(T), power_law_A(T)))

    print("\nCross-checks at T = 1e4 K")
    print("  alpha_B = {:.4e}  vs  2.59e-13 quoted by Shapiro et al. (2006) -> "
          "ratio {:.4f}".format(hui_case_B(1e4), hui_case_B(1e4)/2.59e-13))
    print("  alpha_A = {:.4e}  vs  4.2e-13  quoted by Gnedin & Madau (2022) -> "
          "ratio {:.4f}".format(hui_case_A(1e4), hui_case_A(1e4)/4.2e-13))
    print("  alpha_A: V&F96 / Seaton59 = {:.4f}; Abel97 / HG97 = {:.4f}".format(
        verner_total(1e4)/seaton_total(1e4), abel_total(1e4)/hui_case_A(1e4)))
    print("  alpha_B / alpha_A = {:.4f}  (fraction of recombinations to excited "
          "states)".format(hui_case_B(1e4)/hui_case_A(1e4)))

    print("\nLocal logarithmic slopes  d ln(alpha) / d ln(T)")
    d = 1e-5
    for name, f in [("alpha_B (HG97)", hui_case_B), ("alpha_A (HG97)", hui_case_A),
                    ("V&F96 total", verner_total)]:
        for T0 in [5e3, 1e4, 2e4]:
            sl = (np.log(f(T0*(1+d))) - np.log(f(T0*(1-d)))) / np.log((1+d)/(1-d))
            print("  {:<16} T = {:6.0f} K : {:+.4f}".format(name, T0, sl))

    print("\nMaximum error of the power-law approximations")
    for name, pl, ref in [("Case B, exponent -0.83", power_law_B, hui_case_B),
                          ("Case A, exponent -0.70", power_law_A, hui_case_A)]:
        for lo, hi in [(5e3, 2e4), (5e3, 3e4), (1e3, 1e5)]:
            Tg = np.logspace(np.log10(lo), np.log10(hi), 2000)
            err = 100.0 * np.max(np.abs(pl(Tg)/ref(Tg) - 1.0))
            print("  {:<24} {:6.0f} - {:7.0f} K : {:5.2f} %".format(name, lo, hi, err))
