#!/usr/bin/env python3
r"""
The reionization photon budget: algebra, then numbers.

WHAT THIS ANSWERS
    How many ionizing photons per hydrogen atom, per Hubble time, are needed to
    KEEP the IGM ionized -- the quantity 1 + t_H/t_rec -- at z = 20, 10 and 5.5,
    for clumping factors C = 1, 3 and 12, plus the Madau & Dickinson
    prescription C_IGM(z) = 1 + 43 z^-1.71.

WHY IT NEEDS NO rho_UV
    The requirement is set by cosmology and atomic physics alone: n_H(z), H(z),
    alpha_B(T) and C. No star-formation history and no UV luminosity density
    enter it. That is why it can be quoted at z = 5.5, where this project has no
    measured rho_UV, while zeta itself cannot.

SOURCES
    Madau & Dickinson 2014, ARA&A 52, 415, eq. (24)   [papers/1403.0007-...pdf]
        <t_rec> = (chi <n_H> alpha_B C_IGM)^-1
                ~= 3.2 Gyr ((1+z)/7)^-3 C_IGM^-1
        chi = 1.08 (photoelectrons from singly ionized He), T = 2e4 K,
        C_IGM = 1 + 43 z^-1.71  (Pawlik et al. 2009)
    Osterbrock & Ferland 2006, Table 2.1: alpha_B(2e4 K) = 1.43e-13 cm^3/s
    Planck 18 via astropy for H(z); n_H0 from ionization_yield.cosmology, which
        is the harmonised route-1 value used everywhere else in this project.
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import json
import numpy as np
import sympy as sp
import ionization_yield as IY

CHI_HE = 1.08
ALPHA_B_2E4 = 1.43e-13           # cm^3 s^-1, T = 2e4 K
SEC_GYR = 3.155693e16
ZS = (20.0, 10.0, 5.5)
CS = (1.0, 3.0, 12.0)

rows, prov = [], {}
def rec(cid, name, ok, detail):
    rows.append((cid, name, "PASS" if ok else "FAIL", detail))
def put(k, v):
    prov[k] = float(v); return v


def H_astropy(z):
    from astropy.cosmology import Planck18
    import astropy.units as u
    return float(Planck18.H(z).to(1 / u.s).value)


def C_MD14(z):
    return 1.0 + 43.0 * z ** -1.71


# =========================================================================
# 1. ALGEBRA  (sympy; every step verified, nothing asserted)
# =========================================================================
print("=" * 78)
print("1. DERIVATION")
print("=" * 78)

zeta, aB, C, chi, nH, nH0, z, H0, Om, t = sp.symbols(
    "zeta alpha_B C chi n_H n_H0 z H_0 Omega_m t", positive=True)
x = sp.Symbol("x", positive=True)

print("""
(1a) Ionization balance for hydrogen at mean density, in the PHOTON-COUNTING
     form used for reionization budgets (Madau, Haardt & Rees 1999). During
     reionization the IGM is optically thick to Lyman-continuum radiation, so
     every escaping ionizing photon is absorbed somewhere and the supply term
     is the full zeta, not zeta(1-x). Per unit volume:

         supply         S = zeta n_H
         recombinations R = alpha_B C n_e n_HII
                          = alpha_B C chi n_H^2 x^2

     with x = n_HII/n_H and n_e = chi n_HII (chi = 1.08 counts the electrons
     donated by singly ionized helium). Dividing by n_H,

         dx/dt = zeta - alpha_B C chi n_H x^2
""")
balance = zeta - aB * C * chi * nH * x**2
print("     dx/dt =", balance)

print("""
(1b) The MAINTENANCE condition. Ask what zeta must be for a FULLY ionized IGM
     to be held in steady state: set dx/dt = 0 and x = 1. (Note the x -> 1
     limit is only meaningful in the photon-counting form above; written with a
     zeta(1-x) absorption term the supply vanishes at x = 1 and the question
     cannot be posed -- which is why the budget literature uses this form.)
""")
zeta_crit = sp.solve(sp.Eq(balance.subs(x, 1), 0), zeta)[0]
print("     zeta_crit =", zeta_crit)
t_rec_expr = 1 / (chi * nH * aB * C)
print("     t_rec     =", t_rec_expr, "   (Madau & Dickinson eq. 24)")
check_1b = sp.simplify(zeta_crit - 1 / t_rec_expr)
print("     zeta_crit - 1/t_rec =", check_1b, " -> the criterion is zeta t_rec >= 1")
rec("B1", "zeta t_rec >= 1 IS the steady-state maintenance condition",
    check_1b == 0,
    f"solving dx/dt = 0 at x = 1 gives zeta_crit = {zeta_crit}, which is exactly "
    f"1/t_rec with t_rec of MD14 eq. (24). sympy residual {check_1b}. So the "
    f"recombination-time normalisation is not a convention: it is the boundary "
    f"at which ionizations can just hold the gas ionized.")

print("""
(1c) Photons per atom over an interval. Integrating the balance, each atom needs
     one ionization plus one per recombination it suffers:

         N_req = 1 + integral dt / t_rec

     Per Hubble time (t_H = 1/H) that is the quantity asked for:

         N_req = 1 + t_H / t_rec = 1 + chi n_H(z) alpha_B C / H(z)
""")
N_req = 1 + chi * nH * aB * C / sp.Symbol("H", positive=True)
print("     N_req =", N_req)

print("""
(1d) Explicit z dependence. n_H(z) = n_H0 (1+z)^3, and in matter domination
     H(z) -> H0 sqrt(Om) (1+z)^(3/2), so
""")
Hmd = H0 * sp.sqrt(Om) * (1 + z) ** sp.Rational(3, 2)
N_md = sp.simplify(1 + chi * nH0 * (1 + z) ** 3 * aB * C / Hmd)
print("     N_req =", N_md)
excess = sp.simplify(N_md - 1)
print("     N_req - 1 =", excess, "   -> linear in C, rising as (1+z)^(3/2)")
rec("B2", "the excess over unity scales as C (1+z)^(3/2), analytically",
    sp.simplify(excess - chi * nH0 * aB * C * (1 + z) ** sp.Rational(3, 2)
                / (H0 * sp.sqrt(Om))) == 0,
    f"N_req - 1 = chi n_H0 alpha_B C (1+z)^(3/2) / (H0 sqrt(Omega_m)) -- "
    f"linear in the clumping factor and rising as (1+z)^(3/2). sympy-verified. "
    f"CONSEQUENCE: the requirement gets HARDER at higher redshift, because the "
    f"gas is denser (t_rec ~ (1+z)^-3) faster than the clock is short "
    f"(t_H ~ (1+z)^-3/2).")

# =========================================================================
# 1e. EVERY FACTOR OF EQ. (8) DERIVED, NOT RESTATED
# =========================================================================
print()
print("=" * 78)
print("1e. THE FACTORS OF <t_rec> = (chi <n_H> alpha_B C_IGM)^-1")
print("=" * 78)

# --- chi: why 1.08 ------------------------------------------------------
# When helium is singly ionized, every He atom donates one electron as well:
#     n_e = n_HII + n_HeII = n_H + n_He = n_H (1 + n_He/n_H)
# and n_He/n_H = (Y_P/4)/X_H from the primordial mass fractions.
Y_P = IY.Y_P
X_H = IY.X_H
he_per_h = (Y_P / 4.0) / X_H
chi_derived = 1.0 + he_per_h
put("Y_P", Y_P); put("X_H", X_H)
put("n_He_over_n_H", he_per_h); put("chi_derived", chi_derived)
put("chi_HeIII", 1.0 + 2.0 * he_per_h)   # if helium were DOUBLY ionized
print(f"   Y_P = {Y_P:.6f}, X_H = {X_H:.6f}  ->  n_He/n_H = (Y_P/4)/X_H = {he_per_h:.6f}")
print(f"   chi = 1 + n_He/n_H = {chi_derived:.6f}   vs MD14's quoted {CHI_HE}")
rec("B7", "chi = 1.08 DERIVED from the primordial mass fractions",
    abs(chi_derived - CHI_HE) / CHI_HE < 0.01,
    f"with Y_P = {Y_P:.4f} and X_H = {X_H:.4f}, singly ionized helium gives "
    f"n_e/n_HII = 1 + (Y_P/4)/X_H = {chi_derived:.5f}, against the 1.08 quoted "
    f"in MD14 eq. (24) -- agreement to {100*abs(chi_derived/CHI_HE-1):.2f}%. So "
    f"chi is not a fudge factor: it is the electron-per-proton ratio of a gas "
    f"whose helium is singly ionized. Were helium DOUBLY ionized it would be "
    f"1 + 2(Y_P/4)/X_H = {1+2*he_per_h:.5f}, which is the value appropriate "
    f"after HeII reionization (z < 3) and NOT here.")

# --- alpha_B: verify the value against an independent analytic fit ------
# Hui & Gnedin (1997) fit to the case-B coefficient:
#    alpha_B(T) = 2.753e-14 lambda^1.5 / [1 + (lambda/2.74)^0.407]^2.242,
#    lambda = 2 x 157807 K / T
def alpha_B_HG97(T):
    lam = 2.0 * 157807.0 / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74) ** 0.407) ** 2.242

print()
print("   alpha_B, case B, against the Hui & Gnedin (1997) analytic fit:")
for T, ref, who in ((1.0e4, 2.59e-13, "Osterbrock & Ferland Table 2.1"),
                    (2.0e4, 1.43e-13, "the value used here, MD14's T")):
    got = alpha_B_HG97(T)
    print(f"      T = {T:.0e} K: fit {got:.4e}  vs {ref:.3e} ({who})"
          f"   diff {100*abs(got/ref-1):.2f}%")
put("alpha_B_HG97_1e4", alpha_B_HG97(1.0e4))
put("alpha_B_HG97_2e4", alpha_B_HG97(2.0e4))
rec("B8", "alpha_B(2e4 K) = 1.43e-13 confirmed by an independent fit",
    abs(alpha_B_HG97(2.0e4) / ALPHA_B_2E4 - 1) < 0.02
    and abs(alpha_B_HG97(1.0e4) / 2.59e-13 - 1) < 0.02,
    f"the Hui & Gnedin (1997) case-B fit gives "
    f"{alpha_B_HG97(2.0e4):.4e} cm^3/s at 2e4 K against the "
    f"{ALPHA_B_2E4:.2e} used here ({100*abs(alpha_B_HG97(2e4)/ALPHA_B_2E4-1):.2f}%), "
    f"and {alpha_B_HG97(1.0e4):.4e} at 1e4 K against the textbook 2.59e-13 "
    f"({100*abs(alpha_B_HG97(1e4)/2.59e-13-1):.2f}%). Two temperatures, so the "
    f"fit is anchored and not just consistent at one point. CASE B, not A: "
    f"recombinations straight to the ground state are excluded because the "
    f"photon they emit is reabsorbed on the spot and ionizes another atom, so "
    f"it is not a net recombination.")
print()
print(f"   alpha_B falls with T, so a HOTTER assumed IGM gives a LONGER t_rec")
print(f"   and a SMALLER requirement: at 1e4 K instead of 2e4 K the factor is "
      f"{alpha_B_HG97(1e4)/alpha_B_HG97(2e4):.3f}.")
put("alpha_B_ratio_1e4_over_2e4", alpha_B_HG97(1e4) / alpha_B_HG97(2e4))

# --- C_IGM: where "1 + 43 z^-1.71" actually comes from -------------------
# Pawlik, Schaye & van Scherpenzeel 2009 do NOT print "1 + 43 z^-1.71". Their
# eq. (A1) is
#       C(z) = z^beta exp(-gamma z + delta) + alpha
# and for C_100 in their r19.5L6N256 run (reheating at z_r = 19.5), Table A1
# gives alpha = 1.00, beta = -1.71, gamma = 0.00, delta = 3.76. Substituting:
#       C_100(z) = 1 + exp(3.76) z^-1.71
# so MD14's "43" is exp(delta). Checked rather than assumed.
PAWLIK = dict(alpha=1.00, beta=-1.71, gamma=0.00, delta=3.76)
print()
print("   C_IGM: Pawlik+09 eq. (A1), C_100 of the r19.5L6N256 run")
coef = np.exp(PAWLIK["delta"])
put("pawlik_exp_delta", coef)
print(f"      C(z) = z^beta exp(-gamma z + delta) + alpha, with "
      f"alpha={PAWLIK['alpha']}, beta={PAWLIK['beta']}, "
      f"gamma={PAWLIK['gamma']}, delta={PAWLIK['delta']}")
print(f"      exp(delta) = exp({PAWLIK['delta']}) = {coef:.4f}  ->  MD14's '43'")

def C_pawlik(z):
    return (z ** PAWLIK["beta"] * np.exp(-PAWLIK["gamma"] * z + PAWLIK["delta"])
            + PAWLIK["alpha"])

worst = max(abs(C_pawlik(zz) / C_MD14(zz) - 1.0) for zz in ZS)
put("C_pawlik_vs_MD14_maxreldiff", worst)
put("C_pawlik_vs_MD14_maxreldiff_pct", 100.0 * worst)
for zz in ZS:
    print(f"      z = {zz:4.1f}:  Pawlik A1 = {C_pawlik(zz):.4f}   "
          f"MD14's 1+43z^-1.71 = {C_MD14(zz):.4f}")
rec("B9", "MD14's '1 + 43 z^-1.71' IS Pawlik+09 eq. (A1) for C_100, r19.5 run",
    worst < 0.005,
    f"exp(delta) = exp(3.76) = {coef:.3f}, which is MD14's 43; evaluating "
    f"Pawlik's eq. (A1) with (alpha, beta, gamma, delta) = (1.00, -1.71, 0.00, "
    f"3.76) against MD14's shorthand agrees to {100*worst:.3f}% at z = 20, 10 "
    f"and 5.5. So the '43' is not a fitted constant of MD14's: it is e^delta "
    f"from Pawlik's Table A1, and the clumping factor used here is "
    f"specifically C_100 -- gas BELOW overdensity 100 -- from the run reheated "
    f"at z_r = 19.5. Pawlik's C_-1 (all gas) is about twice as large at z = 6, "
    f"so this choice is the LESS demanding one.")

# --- and the fit's range of validity ------------------------------------
# Pawlik Appendix A1: "we fit the evolution of the clumping factors C_-1 and
# C_100 over the redshift range 6 <= z <= 20".
PAWLIK_ZLO, PAWLIK_ZHI = 6.0, 20.0
put("pawlik_fit_z_lo", PAWLIK_ZLO); put("pawlik_fit_z_hi", PAWLIK_ZHI)
outside = [zz for zz in ZS if not (PAWLIK_ZLO <= zz <= PAWLIK_ZHI)]
print()
print(f"      Pawlik's stated fit range is {PAWLIK_ZLO:g} <= z <= {PAWLIK_ZHI:g}")
print(f"      of our redshifts, OUTSIDE it: {outside}")
rec("B10", "the clumping fit is used inside its range at z = 20 and 10, "
           "OUTSIDE it at z = 5.5",
    outside == [5.5],
    f"Pawlik+09 Appendix A1 fits C over {PAWLIK_ZLO:g} <= z <= {PAWLIK_ZHI:g}. "
    f"z = 20 sits exactly at the upper edge and z = 10 well inside, so the "
    f"C_IGM column is on solid ground there. z = 5.5 is BELOW the fitted range: "
    f"C_IGM(5.5) = {C_MD14(5.5):.3f} is an extrapolation, mild (0.5 in z) but "
    f"real, and it is the only cell of the tables that leaves a source's stated "
    f"domain. The fixed-C columns are unaffected -- they assume nothing.")

# =========================================================================
# 2. NUMBERS
# =========================================================================
print()
print("=" * 78)
print("2. NUMERICAL EVALUATION")
print("=" * 78)
cos10 = IY.cosmology(10.0)
n_H0 = cos10["n_H_cm3"] / 11.0 ** 3
put("n_H0_cm3", n_H0); put("chi_He", CHI_HE); put("alpha_B_2e4K", ALPHA_B_2E4)
print(f"   n_H0 = {n_H0:.6e} cm^-3 (comoving; = n_H(z=10)/11^3, harmonised value)")
print(f"   chi  = {CHI_HE}     alpha_B(2e4 K) = {ALPHA_B_2E4:.3e} cm^3/s")

# --- reproduce MD14's own normalisation, as a check on our inputs ---------
t_rec_z6_C1 = 1.0 / (CHI_HE * n_H0 * 7.0**3 * ALPHA_B_2E4 * 1.0)
put("t_rec_z6_C1_Gyr", t_rec_z6_C1 / SEC_GYR)
rec("B3", "MD14 eq. (24) 3.2 Gyr normalisation reproduced from our own n_H0",
    abs(t_rec_z6_C1 / SEC_GYR / 3.2 - 1.0) < 0.10,
    f"at (1+z)/7 = 1 and C_IGM = 1: t_rec = {t_rec_z6_C1/SEC_GYR:.4f} Gyr "
    f"against the 3.2 Gyr printed in their eq. (24). Agreement to "
    f"{100*abs(t_rec_z6_C1/SEC_GYR/3.2-1):.1f}%.")

def table(zs, cs):
    out = {}
    for zz in zs:
        H = H_astropy(zz)
        t_H = 1.0 / H
        nHz = n_H0 * (1.0 + zz) ** 3
        for cc in list(cs) + ["MD14"]:
            Cv = C_MD14(zz) if cc == "MD14" else cc
            t_rec = 1.0 / (CHI_HE * nHz * ALPHA_B_2E4 * Cv)
            # N_H   : photons per atom per HUBBLE time      = 1 + t_H/t_rec
            # N_rec : photons per atom per RECOMBINATION time = 1 + t_rec/t_H
            # The second is the first rescaled by t_rec/t_H, which is what
            # changing the clock does and nothing more:
            #     (1 + t_H/t_rec) x (t_rec/t_H) = t_rec/t_H + 1.
            # Read term by term: the "1" is the recombination that must be
            # undone within this interval -- exactly one, by construction, which
            # is the whole point of using t_rec as the clock -- and the
            # t_rec/t_H is the single initial ionization amortised over one
            # recombination time.
            out[(zz, cc)] = dict(C=Cv, t_H=t_H, t_rec=t_rec,
                                 ratio=t_rec / t_H,
                                 N=1.0 + t_H / t_rec,
                                 N_rec=1.0 + t_rec / t_H)
    return out

T = table(ZS, CS)
print()
print("   REQUIRED IONIZING PHOTONS PER H ATOM PER HUBBLE TIME,  N_req = 1 + t_H/t_rec")
print(f"   {'z':>5} {'t_H [Gyr]':>10} {'n_H [cm^-3]':>12} |"
      + "".join(f"{('C='+str(c)):>12}" for c in CS) + f"{'C_MD14(z)':>16}")
print("   " + "-" * 78)
for zz in ZS:
    H = H_astropy(zz); nHz = n_H0 * (1 + zz) ** 3
    line = f"   {zz:5.1f} {1.0/H/SEC_GYR:10.4f} {nHz:12.4e} |"
    for cc in CS:
        line += f"{T[(zz,cc)]['N']:12.3f}"
    line += f"{T[(zz,'MD14')]['N']:11.3f} (C={T[(zz,'MD14')]['C']:.3f})"
    print(line)
print()
print("   SAME REQUIREMENT IN THE t_rec NORMALISATION,  N_req = 1 + t_rec/t_H")
print(f"   {'z':>5} {'t_H [Gyr]':>10} {'n_H [cm^-3]':>12} |"
      + "".join(f"{('C='+str(c)):>12}" for c in CS) + f"{'C_MD14(z)':>16}")
print("   " + "-" * 78)
for zz in ZS:
    H = H_astropy(zz); nHz = n_H0 * (1 + zz) ** 3
    line = f"   {zz:5.1f} {1.0/H/SEC_GYR:10.4f} {nHz:12.4e} |"
    for cc in CS:
        line += f"{T[(zz,cc)]['N_rec']:12.3f}"
    line += f"{T[(zz,'MD14')]['N_rec']:11.3f} (C={T[(zz,'MD14')]['C']:.3f})"
    print(line)
print()
print(f"   {'z':>5} | " + "  ".join(f"t_rec/t_H (C={c:g})" for c in CS))
for zz in ZS:
    print(f"   {zz:5.1f} | " + "  ".join(f"{T[(zz,c)]['ratio']:16.4f}" for c in CS))

# the maintenance part of the t_rec-normalised requirement is exactly 1
maint = [T[(zz, cc)]["N_rec"] - T[(zz, cc)]["ratio"] for zz in ZS for cc in CS]
put("NreqTrec_maintenance_term", max(maint))
rec("B6", "in t_rec units the recombination term is exactly 1, by construction",
    max(abs(m - 1.0) for m in maint) < 1e-12,
    f"across all nine (z, C) cells, N_req(t_rec) - t_rec/t_H = 1 to "
    f"{max(abs(m-1.0) for m in maint):.1e}. THAT is what choosing the "
    f"recombination time as the clock buys: the recombination part of the "
    f"budget becomes unity everywhere, and the whole (z, C) dependence moves "
    f"into the amortised initial ionization, t_rec/t_H. In the Hubble-time "
    f"normalisation it is the other way round -- the '1' is the initial "
    f"ionization and the recombination term carries all the variation.")

for zz in ZS:
    tag = f"{zz:g}".replace(".", "p")
    for cc in list(CS) + ["MD14"]:
        ct = f"C{cc:g}".replace(".", "p") if cc != "MD14" else "CMD14"
        put(f"Nreq_z{tag}_{ct}", T[(zz, cc)]["N"])
        put(f"NreqTrec_z{tag}_{ct}", T[(zz, cc)]["N_rec"])
        put(f"trec_over_tH_z{tag}_{ct}", T[(zz, cc)]["ratio"])
    put(f"C_MD14_z{tag}", C_MD14(zz))
    put(f"t_H_Gyr_z{tag}", 1.0 / H_astropy(zz) / SEC_GYR)
    put(f"n_H_cm3_z{tag}", n_H0 * (1.0 + zz) ** 3)

# --- the (1+z)^3/2 law, checked against the exact table ------------------
pred = (21.0 / 11.0) ** 1.5
got = (T[(20.0, 1.0)]["N"] - 1.0) / (T[(10.0, 1.0)]["N"] - 1.0)
put("Nreq_excess_ratio_z20_z10_C1", got)
put("Nreq_excess_ratio_predicted", pred)
rec("B4", "the exact table obeys the analytic (1+z)^(3/2) law",
    abs(got / pred - 1.0) < 0.05,
    f"at fixed C = 1 the excess N_req - 1 grows by {got:.4f} from z = 10 to "
    f"z = 20 against the matter-domination prediction (21/11)^1.5 = {pred:.4f}, "
    f"a {100*abs(got/pred-1):.2f}% departure -- that residual is Lambda and "
    f"radiation in the exact Planck-18 H(z), which the power law omits.")
# --- how much harder the requirement gets from z=5.5 to z=20 -------------
for cc in CS:
    put(f"Nreq_rise_z5p5_to_z20_C{cc:g}",
        T[(20.0, cc)]["N"] / T[(5.5, cc)]["N"])
put("Nreq_rise_z5p5_to_z20_CMD14",
    T[(20.0, "MD14")]["N"] / T[(5.5, "MD14")]["N"])
print()
print("   rise in N_req from z = 5.5 to z = 20:")
for cc in list(CS) + ["MD14"]:
    k = f"C{cc:g}" if cc != "MD14" else "C_IGM(z)"
    print(f"      {k:>10}: x{T[(20.0, cc)]['N']/T[(5.5, cc)]['N']:.3f}")

# --- temperature sensitivity: alpha_B is the one factor with a free knob --
# t_rec ~ 1/alpha_B and alpha_B FALLS with T, so a hotter assumed IGM lengthens
# t_rec and LOWERS the requirement. MD14 assume 2e4 K; 1e4 K is the other value
# in common use, and it is the more demanding one.
fac_T = alpha_B_HG97(1.0e4) / alpha_B_HG97(2.0e4)
put("alpha_B_ratio_1e4_over_2e4", fac_T)
N10 = T[(10.0, "MD14")]["N"]
N10_1e4 = 1.0 + (N10 - 1.0) * fac_T
put("Nreq_z10_CMD14_at_1e4K", N10_1e4)
print()
print(f"   temperature sensitivity: alpha_B(1e4)/alpha_B(2e4) = {fac_T:.4f}, so")
print(f"   at z = 10 with C_IGM the requirement moves {N10:.3f} -> {N10_1e4:.3f}")
print(f"   if T = 1e4 K is assumed instead of MD14's 2e4 K.")
rec("B11", "the gas temperature is a real knob, and MD14's choice is the milder one",
    N10_1e4 > N10,
    f"alpha_B falls with T, so t_rec ~ 1/alpha_B lengthens and the requirement "
    f"drops as the assumed IGM gets hotter. Between the two values in common "
    f"use the recombination term changes by {fac_T:.3f}: at z = 10 with "
    f"C_IGM = {C_MD14(10.0):.3f}, N_req goes from {N10:.3f} at MD14's 2e4 K to "
    f"{N10_1e4:.3f} at 1e4 K. T is an ASSUMPTION about photoionized gas, not a "
    f"measurement of this IGM, and it is the second-largest lever after C.")

# --- linearity in C, exactly ---------------------------------------------
lin = ((T[(10.0, 12.0)]["N"] - 1.0) / (T[(10.0, 1.0)]["N"] - 1.0))
put("Nreq_excess_linearity_C12_over_C1", lin)
rec("B5", "N_req - 1 is exactly linear in the clumping factor",
    abs(lin - 12.0) < 1e-9,
    f"(N_req-1)|C=12 / (N_req-1)|C=1 = {lin:.10f} at z = 10, against the "
    f"analytic 12. Exact, as eq. (1d) requires.")

print()
for cid, name, st, det in rows:
    print(f"   [{st}] {cid:3s} {name}\n         {det}")
print(f"\n   {sum(1 for r in rows if r[2]=='PASS')}/{len(rows)} checks pass.")

with open("provenance/reionization_budget.json", "w") as fh:
    json.dump({"derived": prov,
               "checks": [{"id": c, "name": n, "state": s, "detail": d}
                          for c, n, s, d in rows]}, fh, indent=2, sort_keys=True)
print(f"[ARTEFACT] provenance/reionization_budget.json  --  {len(prov)} keys")
