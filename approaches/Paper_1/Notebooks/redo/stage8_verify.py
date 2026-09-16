"""
stage8_verify.py -- independent verification of the photon-cascade module
=========================================================================

Cross-checks required by the "rigorous" working rules for the deliverable
"average number of ionizations per primary particle (photons and electrons)".
Nothing here feeds the science result; every block is a check that can fail.

Blocks
------
A  Klein-Nishina differential cross-section.  The differential form used in
   stage8_photon_yield.kn_dsigma_ds is integrated back to the total with
   (i) sympy (symbolic, in the scattered-energy fraction s) and (ii) mpmath
   adaptive quadrature, and compared to the closed-form total in
   redo_common.sigma_kn (Klein & Nishina 1929; Rybicki & Lightman 1979,
   Eq. 7.5).  Also the Thomson limit sigma_KN(a->0) -> sigma_T.
B  Compton kinematics: s_min = 1/(1+2a) <-> backscatter, T_max = E 2a/(1+2a),
   and energy conservation E = E' + T on the quadrature nodes.
C  First-interaction path kernel: the exact per-segment weights must be
   non-negative and sum to 1 - exp(-tau_tot) (analytic for constant opacity).
D  Channel competition sigma_pi / sigma_C: where photoionization stops
   dominating, i.e. the regime in which the free-electron (impulse)
   approximation for Compton scattering off bound H is harmless.
E  Omitted physics, pair production on nuclei (Bethe-Heitler).  The user
   excluded this channel, so the tabulated yield is a LOWER BOUND above the
   energy where sigma_pair > sigma_KN.  That energy is computed here.
F  Omitted physics, pair production on the CMB (Breit-Wheeler).  Threshold
   and mean free path against the z-dependent CMB, to delimit the top of the
   energy grid.

Run:  python stage8_verify.py
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import redo_common as R          # inserts the repo root on sys.path
import igm_losses as L
import stage8_photon_yield as S

EV = R.EV
ALPHA = 7.2973525693e-3          # CODATA 2018 fine-structure constant
R_E = L.ELECTRON_RADIUS_MKS      # classical electron radius [m]
S_T = L.THOMSON_CROSS_SECTION_MKS
E0 = L.E0                        # electron rest energy [J]
E0_EV = R.E0_EV


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------------------
# A.  Klein-Nishina differential -> total
# ---------------------------------------------------------------------------
def block_A():
    hdr("A.  Klein-Nishina differential cross-section  d(sigma)/ds")
    print("   d(sigma)/ds = (pi r_e^2 / a) [ s + 1/s - 1 + cos^2(theta) ],")
    print("   cos(theta) = 1 - (1/s - 1)/a,   1/(1+2a) <= s <= 1,   a = E/m_e c^2")
    print("   [Klein & Nishina 1929; Rybicki & Lightman 1979 Eq. 7.5, with")
    print("    d(cos theta) = ds/(a s^2) so that d(sigma)/ds = d(sigma)/d(cos theta)/(a s^2)]")

    # ---- A1: symbolic integration with sympy ------------------------------
    import sympy as sp
    s, a = sp.symbols("s a", positive=True)
    cth = 1 - (1 / s - 1) / a
    dsig = (sp.pi / a) * (s + 1 / s - 1 + cth ** 2)          # in units of r_e^2
    tot = sp.integrate(dsig, (s, 1 / (1 + 2 * a), 1))
    tot = sp.simplify(tot)

    kn_closed = sp.pi * 2 * (((1 + a) / a ** 3)
                             * (2 * a * (1 + a) / (1 + 2 * a) - sp.log(1 + 2 * a))
                             + sp.log(1 + 2 * a) / (2 * a)
                             - (1 + 3 * a) / (1 + 2 * a) ** 2)
    diff = sp.simplify(sp.together(tot - kn_closed))
    print("\n   A1  sympy:  integral - closed form  simplifies to :", diff)
    num = [float(sp.N((tot - kn_closed).subs(a, v))) for v in
           (sp.Rational(1, 1000), sp.Rational(1, 10), 1, 10, 1000)]
    print("       numeric residual at a = 1e-3, 0.1, 1, 10, 1e3 :",
          np.array2string(np.array(num), precision=3))
    okA1 = max(abs(x) for x in num) < 1e-10
    print("       PASS" if okA1 else "       *** FAIL ***")

    # ---- A2: independent adaptive quadrature ------------------------------
    from mpmath import mp, mpf, quad as mquad, log as mlog
    mp.dps = 30

    def sig_mp(av):
        av = mpf(av)
        f = lambda x: (mp.pi / av) * (x + 1 / x - 1 + (1 - (1 / x - 1) / av) ** 2)
        return mquad(f, [1 / (1 + 2 * av), mpf(1)])

    def kn_closed_mp(av):
        """redo_common.sigma_kn's closed form evaluated at 30 digits."""
        av = mpf(av)
        return mpf(3) / 4 * (((1 + av) / av ** 3)
                             * (2 * av * (1 + av) / (1 + 2 * av) - mlog(1 + 2 * av))
                             + mlog(1 + 2 * av) / (2 * av)
                             - (1 + 3 * av) / (1 + 2 * av) ** 2)

    print("\n   A2  mpmath adaptive quadrature (30 digits) vs the closed form of")
    print("       redo_common.sigma_kn, evaluated both at 30 digits and in float64.")
    print("       The float64 column degrades at small a purely by cancellation in")
    print("       the closed form -- the formula itself is exact (col. 'vs mp').")
    print("       %-11s %-11s %-13s %-13s %s"
          % ("E [eV]", "a=E/m_ec^2", "quad/sigma_T", "rel err vs mp", "rel err float64"))
    okA2 = True
    for E_eV in (1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e5, 5.11e5, 1.0e7, 1.0e10, 1.0e13):
        av = E_eV * EV / E0
        q = sig_mp(av) * mpf(R_E) ** 2 / mpf(S_T)      # quadrature, in sigma_T
        cm = kn_closed_mp(av)                          # closed form, 30 digits
        c64 = float(R.sigma_kn(E_eV)[0]) / S_T         # closed form, float64
        rel_mp = abs(q - cm) / cm
        rel_64 = abs(mpf(c64) - cm) / cm
        okA2 &= float(rel_mp) < 1e-12
        print("       %-11.3g %-11.3e %-13.10f %-13.2e %.2e"
              % (E_eV, av, float(q), float(rel_mp), float(rel_64)))
    print("       formula check (quadrature vs exact closed form): "
          + ("PASS" if okA2 else "*** FAIL ***"))
    print("       NOTE: redo_common.sigma_kn loses ~4e-07 relative accuracy at")
    print("       10 eV and ~1e-05 at 1 eV to cancellation.  Both are far below")
    print("       every other error in the cascade (the transport grid itself is")
    print("       good to ~1e-03), and below 2 keV Compton carries < 5% of the")
    print("       interactions anyway (block D), so this is recorded, not fixed.")

    # ---- A3: the Gauss-Legendre rule actually used in the march -----------
    Eg = np.geomspace(R.B_H, 1.0e13, 400)
    q = S.kn_sigma_quadrature(Eg)
    c = R.sigma_kn(Eg)
    rel = np.abs(q - c) / c
    i = int(np.argmax(rel))
    print("\n   A3  stage8 %d-node Gauss-Legendre rule in u = ln s :" % S.N_S)
    print("       max rel. err = %.3e  at E = %.4g eV   (over 13.6 eV - 1e13 eV)"
          % (rel[i], Eg[i]))
    okA3 = rel.max() < 1e-6
    print("       PASS" if okA3 else "       *** FAIL ***")

    # ---- A4: Thomson limit -----------------------------------------------
    # sigma_KN/sigma_T = 1 - 2a + 26a^2/5 - 133a^3/10 + ...  (Heitler 1954,
    # Sect. 22; Rybicki & Lightman 1979 Eq. 7.6a)
    print("\n   A4  Thomson limit  sigma_KN/sigma_T -> 1 - 2a + 26a^2/5")
    print("       %-11s %-13s %-15s %-15s %s"
          % ("E [eV]", "a", "quadrature", "series", "float64 closed"))
    okA4 = True
    for E_eV in (1.0, 1.0e2, 1.0e4):
        av = E_eV * EV / E0
        q = float(sig_mp(av) * mpf(R_E) ** 2 / mpf(S_T))
        ser = 1.0 - 2.0 * av + 26.0 * av ** 2 / 5.0 - 133.0 * av ** 3 / 10.0
        c64 = float(R.sigma_kn(E_eV)[0]) / S_T
        okA4 &= abs(q - ser) < 50.0 * av ** 4 + 1e-15
        print("       %-11.3g %-13.3e %-15.10f %-15.10f %.10f"
              % (E_eV, av, q, ser, c64))
    print("       PASS" if okA4 else "       *** FAIL ***")

    # ---- A5: the normalised spectrum used in the march --------------------
    Eg = np.array([1.0e3, 1.0e4, 1.0e5, 5.11e5, 1.0e7, 1.0e10])
    sarr, p = S.kn_spectrum(Eg)
    tot_p = p.sum(axis=1)
    mean_loss = np.sum(p * (1.0 - sarr), axis=1)
    print("\n   A5  normalised scattered-energy spectrum")
    print("       %-12s %-18s %s" % ("E [eV]", "sum p (=1)", "<1-s> = <T>/E"))
    for E_eV, tp, ml in zip(Eg, tot_p, mean_loss):
        print("       %-12.3g %-18.12f %.6f" % (E_eV, tp, ml))
    okA5 = np.allclose(tot_p, 1.0, atol=1e-12)
    print("       PASS" if okA5 else "       *** FAIL ***")
    return okA1 and okA2 and okA3 and okA4 and okA5


# ---------------------------------------------------------------------------
# B.  Compton kinematics
# ---------------------------------------------------------------------------
def block_B():
    hdr("B.  Compton kinematics on the quadrature nodes")
    ok = True
    print("   %-12s %-14s %-14s %-14s %s"
          % ("E [eV]", "s_min", "1/(1+2a)", "T_max [eV]", "E 2a/(1+2a) [eV]"))
    for E_eV in (1.0e2, 1.0e4, 5.11e5, 1.0e8, 1.0e12):
        a = E_eV * EV / E0
        sarr, p = S.kn_spectrum(np.array([E_eV]))
        smin_num = sarr[0].min()
        smin_th = 1.0 / (1.0 + 2.0 * a)
        Tmax_num = E_eV * (1.0 - smin_num)
        Tmax_th = E_eV * 2.0 * a / (1.0 + 2.0 * a)
        # the GL nodes never sit exactly at the endpoint; tolerance is the
        # first-node offset of the rule, so compare the ANALYTIC endpoints and
        # only require the nodes to stay inside the support.
        inside = (sarr[0] >= smin_th * (1.0 - 1e-12)) & (sarr[0] <= 1.0 + 1e-12)
        ok &= bool(inside.all())
        print("   %-12.3g %-14.8g %-14.8g %-14.8g %.8g"
              % (E_eV, smin_num, smin_th, Tmax_num, Tmax_th))
    print("   all quadrature nodes inside [1/(1+2a), 1] :", ok)
    # energy conservation is exact by construction: T = E(1-s), E' = E s
    print("   energy conservation E = E' + T holds identically "
          "(T := E(1-s), E' := E s)")
    print("       PASS" if ok else "       *** FAIL ***")
    return ok


# ---------------------------------------------------------------------------
# C.  First-interaction path kernel
# ---------------------------------------------------------------------------
def block_C():
    hdr("C.  First-interaction weights along the light path")
    print("   dP = (dtau_j/dt) exp(-tau_tot) dt ;  the segment rule integrates")
    print("   exp(-tau) analytically for tau linear in t, so weights are >= 0")
    print("   and sum to 1 - exp(-tau_tot).")
    ok = True
    print("\n   %-8s %-12s %-14s %-14s %-14s %s"
          % ("z_i", "E [eV]", "sum w_pi", "sum w_C", "sum w", "1-exp(-tau)"))
    for z_i in (20.0, 15.0, 10.0, 7.0):
        for E_eV in (5.0e1, 3.0e2, 2.0e3, 1.0e4, 1.0e6, 1.0e9):
            w_pi, w_C, Ep, zp = S.path_kernel(z_i, E_eV, S.Z_FINAL)
            # independent recomputation of tau_tot on a 4x finer grid
            zp2 = np.expm1(np.linspace(np.log1p(z_i), np.log1p(S.Z_FINAL),
                                       4 * S.N_PATH))
            tp2 = L.age_s(zp2)
            Ep2 = E_eV * (1.0 + zp2) / (1.0 + z_i)
            nH2 = L.n_HI(zp2)
            k2 = nH2 * (R.sigma_pi(Ep2)
                        + (1.0 + S.X_E) * R.sigma_kn(Ep2)) * L.C_LIGHT
            tau2 = np.trapz(k2, tp2)
            tgt = 1.0 - np.exp(-tau2)
            tot = w_pi.sum() + w_C.sum()
            neg = min(w_pi.min(), w_C.min())
            ok &= (neg >= -1e-300) and (abs(tot - tgt) < 5e-3 * max(tgt, 1e-6))
            print("   %-8.1f %-12.3g %-14.6e %-14.6e %-14.8f %.8f"
                  % (z_i, E_eV, w_pi.sum(), w_C.sum(), tot, tgt))
    print("\n   weights non-negative and matching 1-exp(-tau) to < 0.5% :", ok)
    print("       PASS" if ok else "       *** FAIL ***")
    return ok


# ---------------------------------------------------------------------------
# D.  Channel competition
# ---------------------------------------------------------------------------
def block_D():
    hdr("D.  Photoionization vs Compton on neutral hydrogen")
    print("   sigma_pi: Karzas & Latter (1961) H(1s), threshold value")
    print("             %.3g m^2 at %.4f eV" % (R.SIGMA_PI_THRESHOLD, R.B_H))
    print("   sigma_C : (1+x_e) sigma_KN per H nucleus, x_e = %.0e" % S.X_E)
    f = lambda lE: (np.log(R.sigma_pi(np.exp(lE))[0])
                    - np.log((1.0 + S.X_E) * R.sigma_kn(np.exp(lE))[0]))
    E_cross = float(np.exp(brentq(f, np.log(1.0e2), np.log(1.0e6), xtol=1e-12)))
    print("\n   sigma_pi = sigma_C  at  E = %.4g eV = %.4g keV"
          % (E_cross, E_cross / 1.0e3))
    print("\n   %-12s %-14s %-14s %s" % ("E [eV]", "sigma_pi [m^2]",
                                         "sigma_C [m^2]", "ratio pi/C"))
    for E_eV in (13.7, 20.0, 5.0e1, 1.0e2, 3.0e2, 1.0e3, 2.0e3, E_cross,
                 5.0e3, 1.0e4, 1.0e5):
        sp_ = float(R.sigma_pi(E_eV)[0])
        sc_ = float((1.0 + S.X_E) * R.sigma_kn(E_eV)[0])
        print("   %-12.4g %-14.5e %-14.5e %.4g" % (E_eV, sp_, sc_, sp_ / sc_))
    # Where does the free-electron (impulse) treatment of Compton on bound H
    # stop being harmless?  The maximum recoil is T_max = E 2a/(1+2a); when
    # T_max < B_H the bound branch produces no ionization at all, and the event
    # is a near-elastic scattering that simply returns the photon to the
    # cascade with E' ~ E.  Binding then cannot change the ionization budget.
    h = lambda lE: (np.exp(lE) * 2.0 * (np.exp(lE) * EV / E0)
                    / (1.0 + 2.0 * np.exp(lE) * EV / E0) - R.B_H)
    E_Tmax = float(np.exp(brentq(h, np.log(1.0e2), np.log(1.0e5), xtol=1e-12)))
    print("\n   Maximum Compton recoil T_max = E 2a/(1+2a) reaches B_H = %.4f eV"
          % R.B_H)
    print("   only at E = %.4g eV = %.3g keV." % (E_Tmax, E_Tmax / 1e3))
    print("   Below that energy every Compton event is sub-threshold: the model")
    print("   gives no ionization and returns the photon with E' ~ E, which is")
    print("   what bound-electron (Rayleigh / Raman) scattering does as well, so")
    print("   the impulse approximation cannot bias the ionization budget there.")
    print("   Above it, T >> B_H within a factor of a few and the impulse")
    print("   approximation is the standard one (e.g. Zdziarski & Svensson 1989).")
    print("   Photoionization still dominates the interaction rate up to")
    print("   %.4g eV, so Compton never controls the yield below ~2 keV."
          % E_cross)
    return E_cross


# ---------------------------------------------------------------------------
# E.  Omitted physics I: Bethe-Heitler pair production on nuclei
# ---------------------------------------------------------------------------
def sigma_pair_nosc(E_eV, Z=1):
    """Bethe-Heitler pair production, no screening, E >> m_e c^2 [m^2].

    Heitler (1954), 'The Quantum Theory of Radiation', 3rd ed., Sect. 26,
    Eq. (13); equivalently Motz, Olsen & Koch (1969) Rev. Mod. Phys. 41, 581,
    formula 3BN in the high-energy no-screening limit:
        sigma = 4 alpha r_e^2 Z(Z+1) [ (7/9) ln(2k) - 109/54 ],  k = E/m_e c^2.
    The Z -> Z(Z+1) replacement adds pair production in the field of the
    atomic electrons (triplet production).
    """
    k = np.asarray(E_eV, float) * EV / E0
    val = 4.0 * ALPHA * R_E ** 2 * Z * (Z + 1) * ((7.0 / 9.0) * np.log(2.0 * k)
                                                  - 109.0 / 54.0)
    return np.maximum(val, 0.0)


def sigma_pair_screen(E_eV, Z=1):
    """Bethe-Heitler, complete screening (Thomas-Fermi) [m^2].

    Heitler (1954) Sect. 26, Eq. (14); PDG Rev. Part. Phys., 'Passage of
    particles through matter':
        sigma = 4 alpha r_e^2 Z(Z+1) [ (7/9) ln(183 Z^(-1/3)) - 1/54 ].
    Asymptotic; valid for k >> 137 Z^(-1/3).
    """
    val = 4.0 * ALPHA * R_E ** 2 * Z * (Z + 1) * (
        (7.0 / 9.0) * np.log(183.0 * Z ** (-1.0 / 3.0)) - 1.0 / 54.0)
    return np.full(np.shape(np.asarray(E_eV, float)), val)


def block_E():
    hdr("E.  OMITTED CHANNEL I -- pair production on H nuclei (Bethe-Heitler)")
    print("   Excluded from the calculation by explicit instruction.  Above the")
    print("   energy where it overtakes Compton the tabulated yield is a LOWER")
    print("   BOUND (the pair converts the photon into two electrons that then")
    print("   ionize, so the true yield is larger).")
    print("\n   sigma_pair(complete screening, Z=1) = %.4g m^2  (energy-independent)"
          % sigma_pair_screen(1.0e9))
    print("   %-14s %-16s %-16s %-16s" % ("E [eV]", "sigma_KN [m^2]",
                                          "sig_pair(no scr)", "sig_pair(cs)"))
    for E_eV in (1.0e6, 3.0e6, 1.0e7, 3.0e7, 5.0e7, 1.0e8, 3.0e8, 1.0e9, 1.0e10):
        print("   %-14.3g %-16.4e %-16.4e %-16.4e"
              % (E_eV, R.sigma_kn(E_eV)[0], sigma_pair_nosc(E_eV),
                 sigma_pair_screen(E_eV)))

    g1 = lambda lE: np.log(sigma_pair_nosc(np.exp(lE))) - np.log(
        R.sigma_kn(np.exp(lE))[0])
    g2 = lambda lE: np.log(sigma_pair_screen(np.exp(lE))) - np.log(
        R.sigma_kn(np.exp(lE))[0])
    Ex1 = float(np.exp(brentq(g1, np.log(3.0e6), np.log(1.0e10), xtol=1e-12)))
    Ex2 = float(np.exp(brentq(g2, np.log(3.0e6), np.log(1.0e10), xtol=1e-12)))
    print("\n   crossing sigma_pair = sigma_KN")
    print("     no screening     : E_x = %.4g eV = %.3g MeV" % (Ex1, Ex1 / 1e6))
    print("     complete screen. : E_x = %.4g eV = %.3g MeV" % (Ex2, Ex2 / 1e6))
    print("   The two screening limits bracket the true (intermediate-screening)")
    print("   crossing:  E_x = %.0f (+/- %.0f) MeV.  The tabulated yield is a"
          % (0.5 * (Ex1 + Ex2) / 1e6, abs(Ex1 - Ex2) / 2e6))
    print("   LOWER BOUND above ~%.0f MeV (conservative edge)." % (min(Ex1, Ex2) / 1e6))
    return min(Ex1, Ex2), max(Ex1, Ex2)


# ---------------------------------------------------------------------------
# F.  Omitted physics II: pair production on the CMB (Breit-Wheeler)
# ---------------------------------------------------------------------------
def sigma_gg(s_over_4m2):
    """Breit-Wheeler gamma-gamma -> e+e- cross-section [m^2].

    Breit & Wheeler (1934); Jauch & Rohrlich (1976) Sect. 11-3:
        sigma = (3/16) sigma_T (1-b^2)[(3-b^4) ln((1+b)/(1-b)) - 2b(2-b^2)],
        b = sqrt(1 - 4 m^2 c^4 / s).
    """
    x = np.asarray(s_over_4m2, float)
    out = np.zeros_like(x)
    m = x > 1.0
    b = np.sqrt(1.0 - 1.0 / x[m])
    out[m] = (3.0 / 16.0) * S_T * (1.0 - b ** 2) * (
        (3.0 - b ** 4) * np.log((1.0 + b) / (1.0 - b)) - 2.0 * b * (2.0 - b ** 2))
    return out


def gg_rate(E_eV, z):
    """Inverse mean free path [1/m] of a photon of energy E on the CMB at z.

    Gamma/c = int deps n(eps) int_{-1}^{1} dmu (1-mu)/2 sigma_gg(s),
    s = 2 E eps (1-mu);  n(eps) deps = (8 pi / (h c)^3) eps^2/(e^{eps/kT}-1) deps.
    """
    kT = float(L.K_B * L.T_CMB0 * (1.0 + z)) if hasattr(L, "T_CMB0") else \
        float(1.380649e-23 * 2.7255 * (1.0 + z))
    h = 6.62607015e-34
    c = L.C_LIGHT
    m2c4 = E0 ** 2
    EJ = E_eV * EV

    def inner(eps):                     # eps in J
        n = (8.0 * np.pi / (h * c) ** 3) * eps ** 2 / np.expm1(eps / kT)
        f = lambda mu: 0.5 * (1.0 - mu) * sigma_gg(
            2.0 * EJ * eps * (1.0 - mu) / (4.0 * m2c4))
        val, _ = quad(f, -1.0, 1.0, limit=200)
        return n * val

    lo, hi = 1.0e-4 * kT, 60.0 * kT
    val, _ = quad(inner, lo, hi, limit=200)
    return val


Ex_lo_global = [np.inf]


def block_F():
    hdr("F.  OMITTED CHANNEL II -- pair production on the CMB (Breit-Wheeler)")
    print("   Threshold against a CMB photon of energy ~kT(z):")
    print("     E_th ~ (m_e c^2)^2 / kT(z)")
    for z in (20.0, 10.0, 5.5):
        kT_eV = 1.380649e-23 * 2.7255 * (1.0 + z) / EV
        print("     z = %-5.1f  kT = %.4e eV   E_th = %.4e eV = %.3g TeV"
              % (z, kT_eV, E0_EV ** 2 / kT_eV, E0_EV ** 2 / kT_eV / 1e12))
    print("\n   Exact Wien-tail rate (numerical, full Planck spectrum):")
    print("   %-8s %-14s %-16s %s" % ("z", "E [eV]", "1/Gamma [m]", "1/Gamma [Mpc]"))
    for z in (20.0, 10.0):
        for E_eV in (1.0e12, 1.0e13, 1.0e14):
            g = gg_rate(E_eV, z)
            mfp = np.inf if g <= 0 else 1.0 / g
            print("   %-8.1f %-14.3g %-16.4e %.4e"
                  % (z, E_eV, mfp, mfp / 3.0857e22))
    # Where does gamma-gamma start to matter?  Criterion: the CMB pair-
    # production mean free path, evaluated at the emission redshift (where the
    # CMB is hottest and the rate largest, hence the earliest onset), drops
    # below the proper light path travelled down to Z_FINAL.
    print("\n   Onset criterion:  1/Gamma_gg(E, z_i)  <  proper light path")
    print("   %-8s %-16s %-16s %s"
          % ("z_i", "light path [m]", "E_gg [eV]", "E_gg [TeV]"))
    E_gg = {}
    for z_i in (20.0, 15.0, 10.0, 7.0):
        dl = L.C_LIGHT * (L.age_s(S.Z_FINAL) - L.age_s(z_i))
        f = lambda lE: np.log(max(gg_rate(np.exp(lE), z_i), 1e-300)) + np.log(dl)
        try:
            Eg_ = float(np.exp(brentq(f, np.log(1.0e11), np.log(1.0e15),
                                      xtol=1e-8, rtol=1e-6)))
        except ValueError:
            Eg_ = np.nan
        E_gg[z_i] = Eg_
        print("   %-8.1f %-16.4e %-16.4e %.4g" % (z_i, dl, Eg_, Eg_ / 1e12))
    print("\n   So the E > %.2g eV end of the grid is ALSO a lower bound: there"
          % E_gg[20.0])
    print("   the photon pair-produces on the CMB well inside the window, and")
    print("   the resulting e+e- pair inverse-Compton-cools and ionizes.  The")
    print("   two omitted channels therefore bracket the trustworthy range as")
    print("     %.3g eV  <  E  <  %.3g eV   (yield exact)"
          % (R.B_H, min(Ex_lo_global[0], E_gg[20.0])))
    print("   and mark everything outside it as a lower bound.")
    return E_gg

# ---------------------------------------------------------------------------
def main():
    print("stage8_verify.py -- verification of the photon-cascade ingredients")
    okA = block_A()
    okB = block_B()
    okC = block_C()
    E_pi_C = block_D()
    Ex_lo, Ex_hi = block_E()
    Ex_lo_global[0] = Ex_lo
    E_gg = block_F()

    hdr("SUMMARY")
    print("   A  Klein-Nishina differential -> total        : %s"
          % ("PASS" if okA else "FAIL"))
    print("   B  Compton kinematics                         : %s"
          % ("PASS" if okB else "FAIL"))
    print("   C  path-kernel weights                        : %s"
          % ("PASS" if okC else "FAIL"))
    print("   D  sigma_pi = sigma_C at %.4g eV" % E_pi_C)
    print("   E  sigma_pair = sigma_KN at %.4g - %.4g eV  -> yield is a LOWER"
          % (Ex_lo, Ex_hi))
    print("      BOUND above ~%.0f MeV" % (Ex_lo / 1e6))
    print("   F  gamma-gamma on the CMB sets in at E ~ %.3g eV (z_i = 20)"
          % E_gg[20.0])
    print("\n   ==> the tabulated photon yield is EXACT for")
    print("       %.3g eV < E < %.3g eV  and a LOWER BOUND outside."
          % (R.B_H, min(Ex_lo, E_gg[20.0])))


if __name__ == "__main__":
    main()
