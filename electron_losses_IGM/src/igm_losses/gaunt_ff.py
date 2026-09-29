"""Exact non-relativistic free-free Gaunt factor for an electron in a pure Coulomb field (e–p bremsstrahlung).

Purpose
    Electron–proton (and electron–ion of charge Z) bremsstrahlung for the ionized fraction x_e of the IGM,
    in the non-relativistic regime (assumption A15, decision D18, erratum E07).

Physics implemented
    [KarzasLatter1961 Eq. 16]  g_ff = 2√3/(π η_i η_f) [(η_i² + η_f² + 2η_i²η_f²) I_0
                                        − 2 η_i η_f (1+η_i²)^½ (1+η_f²)^½ I_1] I_0
    [KarzasLatter1961 Eq. 9]   I_l = ¼ [4k_ik_f/(k_i−k_f)²]^(l+1) e^(π|η_i−η_f|/2)
                                     |Γ(l+1+iη_i) Γ(l+1+iη_f)| / Γ(2l+2) · G_l
    [KarzasLatter1961 Eq. 10]  G_l = |(k_f−k_i)/(k_f+k_i)|^(iη_i+iη_f)
                                     ₂F₁(l+1−iη_f, l+1−iη_i; 2l+2; −4k_ik_f/(k_i−k_f)²)
    [KarzasLatter1961 Eq. 17]  η² = Z² Ry / E   (η = Z e²/ħv, Eq. 8)
    The formulas are written for ABSORPTION E_i → E_f = E_i + hν. Emission of a photon hν by an electron of
    energy E_0 uses g_ff(E_i = E_0 − hν, E_f = E_0) [KarzasLatter1961 Eq. 22].
    Only ratios of the wavenumbers enter, so k = 1/η is used (units of Z/a_0).

Energy loss
    From [KarzasLatter1961 Eq. 22], ∫ hν dσ = (8π/(3√3)) α r_0² Z² m c² ⟨g⟩, with ⟨g⟩ = ∫_0^1 g_ff(E_0, x E_0) dx
    and x = hν/E_0 (derived here; checked in tests/test_gaunt_ff.py against the Born limit, which must give
    φ_rad = 16/3). `mean_gff` returns ⟨g⟩; the conversion to a loss rate lives in losses.py (F4).

Regime of validity
    Non-relativistic (β ≪ 1), pure point-Coulomb field (no screening), first-order QED (dipole, one photon).
    Not valid for γ ≳ 1.1 (T ≳ 50 keV); there B&G 1970 Eq. (3.53) takes over (empalme pending).

Numerics (from scratch; library: mpmath for complex Γ, ₂F₁ with analytic continuation, and tanh-sinh quad)
    Arbitrary precision (default 30 digits). The result is real to ~1e-29 (checked); the imaginary part is
    discarded after an explicit check. No number is hard-coded: the only inputs are η_i and η_f.

Checks (tests/test_gaunt_ff.py)
    Born limit, Born × Elwert (η ≪ 1), symmetry η_i ↔ η_f, precision convergence (30 vs 60 digits),
    ∫g_Born dx = 2√3/π → φ_rad = 16/3.
"""

import mpmath as mp

DEFAULT_DPS = 30
IMAG_TOL = mp.mpf("1e-20")   # |Im g| allowed before the result is rejected (relative to |g|)


def _I_l(l, eta_i, eta_f):
    """K&L Eqs. (9)–(10) for multipole l, with k_i = 1/η_i and k_f = 1/η_f (dimensionless)."""
    k_i, k_f = 1 / eta_i, 1 / eta_f
    x = 4 * k_i * k_f / (k_i - k_f) ** 2
    prefactor = (mp.mpf(1) / 4 * x ** (l + 1) * mp.e ** (mp.pi * abs(eta_i - eta_f) / 2)
                 * abs(mp.gamma(l + 1 + 1j * eta_i) * mp.gamma(l + 1 + 1j * eta_f)) / mp.gamma(2 * l + 2))
    phase = (abs(k_f - k_i) / (k_f + k_i)) ** (1j * eta_i + 1j * eta_f)
    G_l = phase * mp.hyp2f1(l + 1 - 1j * eta_f, l + 1 - 1j * eta_i, 2 * l + 2, -x)
    return prefactor * G_l


def gff_eta(eta_i, eta_f, dps=DEFAULT_DPS):
    """Free-free Gaunt factor g_ff(η_i, η_f), K&L Eq. (16). η = Z e²/(ħ v), dimensionless, η_i ≠ η_f.

    Returns an mpmath real. Raises ValueError if the result is not real to IMAG_TOL (numerical failure).
    """
    with mp.workdps(dps):
        eta_i, eta_f = mp.mpf(eta_i), mp.mpf(eta_f)
        if eta_i == eta_f:
            raise ValueError("g_ff is undefined for η_i = η_f (zero photon energy)")
        # Soft photons (η_i ≈ η_f) make |z| = 4k_ik_f/(k_i−k_f)² huge: the ₂F₁ continuation cancels ~log10|z|
        # digits, so the working precision is raised by that amount (plus a margin of 10 digits).
        z_abs = 4 / (eta_i * eta_f) / (1 / eta_i - 1 / eta_f) ** 2
        extra = int(mp.log10(z_abs)) + 10 if z_abs > 1 else 10
    with mp.workdps(dps + extra):
        eta_i, eta_f = mp.mpf(eta_i), mp.mpf(eta_f)
        I0, I1 = _I_l(0, eta_i, eta_f), _I_l(1, eta_i, eta_f)
        bracket = ((eta_i ** 2 + eta_f ** 2 + 2 * eta_i ** 2 * eta_f ** 2) * I0
                   - 2 * eta_i * eta_f * mp.sqrt(1 + eta_i ** 2) * mp.sqrt(1 + eta_f ** 2) * I1)
        g = 2 * mp.sqrt(3) / (mp.pi * eta_i * eta_f) * bracket * I0
        if abs(g.imag) > IMAG_TOL * abs(g):
            raise ValueError(f"g_ff not real: {g}")
    with mp.workdps(dps):
        return +g.real


def gff_emission(E0, hnu, dps=DEFAULT_DPS):
    """g_ff for EMISSION of a photon hν by an electron of kinetic energy E0 (both in units of Z² Ry), 0 < hν < E0.

    Uses the absorption formula with E_i = E0 − hν, E_f = E0 (K&L Eq. 22) and η² = 1/E (K&L Eq. 17).
    """
    with mp.workdps(dps):
        E0, hnu = mp.mpf(E0), mp.mpf(hnu)
        if not (0 < hnu < E0):
            raise ValueError("need 0 < hν < E0")
        return gff_eta(1 / mp.sqrt(E0 - hnu), 1 / mp.sqrt(E0), dps=dps)


X_MIN = mp.mpf("1e-15")      # lower cut of the x = hν/E0 integral (see mean_gff)


def mean_gff(E0, dps=DEFAULT_DPS, return_cut_bound=False):
    """⟨g⟩ = ∫_0^1 g_ff(E0, x·E0) dx for E0 in units of Z² Ry.

    Both endpoints are cut at a distance X_MIN, because tanh-sinh quadrature samples points so close to them
    that the hypergeometric continuation fails: near x = 0 (soft photon, η_i → η_f) E0 − xE0 rounds to E0,
    and near x = 1 (final energy → 0, η_i → ∞) ₂F₁ gets huge parameters. The integrand is log-singular at
    x = 0 and finite at x = 1, so the neglected pieces are bounded by
        X_MIN·(g(X_MIN) + √3/π)   and   X_MIN·g(1 − X_MIN)
    (g grows like −(√3/π) ln x as x → 0 and tends to a finite limit as x → 1, checked numerically).
    With return_cut_bound=True the sum of both bounds is returned as well.
    """
    with mp.workdps(dps):
        E0 = mp.mpf(E0)
        f = lambda x: gff_emission(E0, x * E0, dps=dps)
        val = mp.quad(f, [X_MIN, mp.mpf(1) / 2, 1 - X_MIN])
        if return_cut_bound:
            bound = X_MIN * (f(X_MIN) + mp.sqrt(3) / mp.pi) + X_MIN * f(1 - X_MIN)
            return val, bound
        return val


def g_born(E0, hnu):
    """Non-relativistic Born free-free Gaunt factor (√3/π) ln[(√E0 + √(E0−hν))/(√E0 − √(E0−hν))].

    Used ONLY as a test reference (recalled standard result, limit η → 0; not from a file in references/).
    """
    a, b = mp.sqrt(mp.mpf(E0)), mp.sqrt(mp.mpf(E0) - mp.mpf(hnu))
    return mp.sqrt(3) / mp.pi * mp.log((a + b) / (a - b))


def elwert_factor(E0, hnu):
    """Elwert Coulomb correction (η_f/η_i)(1−e^{−2πη_i})/(1−e^{−2πη_f}) in emission notation (η_i for E0).

    Used ONLY as a test reference (recalled; not from a file in references/).
    """
    e0, e1 = 1 / mp.sqrt(mp.mpf(E0)), 1 / mp.sqrt(mp.mpf(E0) - mp.mpf(hnu))
    return (e1 / e0) * (1 - mp.e ** (-2 * mp.pi * e0)) / (1 - mp.e ** (-2 * mp.pi * e1))
