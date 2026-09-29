"""Fraction of the energy loss that goes to each cooling process, at fixed redshift (objective 1; A19, A20).

Instantaneous (objective 1a)
    φ_i(K) = L_i(K) / Σ_j L_j(K),   L_i = −dE/dt of process i from igm_losses.losses.rates (one implementation each).
Integrated (objective 1b), continuous slowing down at fixed z (A11, A19): while the electron goes from K to K − dK,
process i takes dE_i = φ_i(K) dK. Hence the fraction of K_ini lost to process i before reaching K_floor (A07) is
    f_i(K_ini) = (1/K_ini) ∫_{K_floor}^{K_ini} φ_i(K) dK            [derived; Σ_i f_i + K_floor/K_ini = 1]
No time integration is needed at fixed z; tests compare with integrate.evolve(fixed_z=True, augmented=True), an
independent route (ODE in time).
Units: K in eV at the interface, rates in J s^-1 inside.
Numerics: trapezoid in K (not ln K: with Σ_i φ_i = 1 the budget Σ_i f_i + K_floor/K_ini = 1 then closes to rounding)
on the grid of parameters.yaml → figures.<slug>.grid (D22); the integrand φ_i is bounded
in [0, 1], and the rates have steps (excitation threshold, Coulomb switch, E12/E18), so the grid error is checked
by halving the spacing in the tests.
"""

import numpy as np

from . import _paths  # noqa: F401  (no bytecode inside electron_losses_IGM; igm_losses on sys.path)
from igm_losses import constants as K
from igm_losses import losses

P = K._P


def energy_grid(spec):
    """Log grid [eV] from K_floor to spec['grid']['K_max_eV'] with spec['grid']['points_per_decade'] points (D22)."""
    g = spec["grid"]
    lo, hi = np.log10(P["K_floor"]), np.log10(g["K_max_eV"])
    n = int(np.ceil((hi - lo) * g["points_per_decade"])) + 1
    return np.logspace(lo, hi, n)


# [A19] fixed redshift: every rate is evaluated in the medium of z = z_init
def rate_fractions(K_eV, x_e, z=None, include=losses.PROCESSES):
    """{process: array φ_i(K)} for the kinetic energies K_eV [eV] at redshift z (default z_init)."""
    z = P["z_init"] if z is None else z
    out = {p: np.empty(len(K_eV)) for p in include}
    for j, Ke in enumerate(K_eV):
        r = losses.rates(Ke * K.e, z, x_e, include)
        tot = sum(r.values())
        for p in include:
            out[p][j] = r[p] / tot
    return out


# [A11] [A20] continuous slowing down: dE_i = φ_i dK
def integrated_fractions(K_eV, x_e, z=None, include=losses.PROCESSES):
    """{process: array f_i(K_ini)} for K_ini on the grid K_eV (which must start at K_floor); f_i(K_floor) = 0."""
    K_eV = np.asarray(K_eV, float)
    if abs(K_eV[0] / P["K_floor"] - 1) > 1e-12:
        raise ValueError("the grid must start at K_floor")
    phi = rate_fractions(K_eV, x_e, z, include)
    dK = np.diff(K_eV)
    out = {}
    for p in include:
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (phi[p][1:] + phi[p][:-1]) * dK)))
        out[p] = cum / K_eV
    return out
