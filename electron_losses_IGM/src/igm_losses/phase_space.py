"""Maps of the loss rates on a (z, K) grid: dominant process (cells 28–35) and adiabatic fraction Γ_ad (cell 41).

Uses only losses.rates (one implementation per process). The maps are pure evaluations of the rates, no ODE.
"""

import numpy as np

from . import constants as K_
from . import losses


def rate_stack(z_grid, K_eV_grid, x_e, include=losses.PROCESSES, **opts):
    """Array [n_process, n_K, n_z] of −dE/dt [J s^-1]."""
    names = list(include)
    out = np.zeros((len(names), len(K_eV_grid), len(z_grid)))
    for j, z in enumerate(z_grid):
        for i, Ke in enumerate(K_eV_grid):
            r = losses.rates(Ke * K_.e, z, x_e, names, **opts)
            out[:, i, j] = [r[n] for n in names]
    return names, out


def dominant_process(z_grid, K_eV_grid, x_e, include=losses.PROCESSES, **opts):
    """(names, index map [n_K, n_z]) of the process with the largest loss rate."""
    names, st = rate_stack(z_grid, K_eV_grid, x_e, include, **opts)
    return names, np.argmax(st, axis=0)


def adiabatic_fraction(z_grid, K_eV_grid, x_e, include=losses.PROCESSES, **opts):
    """Γ_ad = L_ad / Σ L_i on the grid (cell 41)."""
    names, st = rate_stack(z_grid, K_eV_grid, x_e, include, **opts)
    return st[names.index("adiabatic")] / st.sum(axis=0)
