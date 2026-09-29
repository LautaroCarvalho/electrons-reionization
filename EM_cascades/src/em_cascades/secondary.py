"""Energy distribution of the secondary electron of a collisional ionization (A14, A20; E17).

p(ε | T) ∝ 1/[1 + (ε/ε̄)^p] on ε ∈ [0, X], X = (T − B)/2   [FurlanettoStoever2010 Eq. 2, p. 1871; ε̄, p and B from
parameters.yaml: secondary_E_bar, secondary_exponent, B_ion_RBED — the same spectrum igm_losses.ionization uses for ⟨ε⟩].
Normalization (closed form, as in igm_losses.ionization, D07): Z(X) = X ₂F₁(1, 1/p; 1 + 1/p; −(X/ε̄)^p).
mean_over(T, g) = ∫_{ε_min}^{X} g(ε) p(ε) dε / Z(X) for a function g that is zero below ε_min (the cascade
quantities vanish at and below K_floor, A20).
"""

import numpy as np
from scipy.special import hyp2f1

from . import _paths  # noqa: F401
from igm_losses import constants as K
from igm_losses import ionization

EPS_BAR = K._P["secondary_E_bar"]
P_EXP = K._P["secondary_exponent"]
B_EV = ionization.B_EV


def x_max(T_eV):
    """Largest secondary energy X = (T − B)/2 [eV] (0 below threshold)."""
    return max((T_eV - B_EV) / 2.0, 0.0)


def weight(eps_eV):
    """Unnormalized p(ε)."""
    return 1.0 / (1.0 + (np.asarray(eps_eV, float) / EPS_BAR) ** P_EXP)


def norm(X_eV):
    """Z(X) = ∫_0^X p(ε) dε [eV] (closed form)."""
    if X_eV <= 0:
        return 0.0
    return X_eV * hyp2f1(1.0, 1.0 / P_EXP, 1.0 + 1.0 / P_EXP, -(X_eV / EPS_BAR) ** P_EXP)


def mean_over(T_eV, eps_grid, g_grid, eps_min):
    """⟨g⟩ over p(ε|T), for g tabulated on eps_grid (ascending, eV) and g = 0 for ε ≤ eps_min.

    Trapezoid on the tabulated points inside [eps_min, X] plus the two end points (g linearly interpolated in ln ε).
    """
    X = x_max(T_eV)
    Z = norm(X)
    if X <= eps_min or Z == 0.0:
        return 0.0
    inside = (eps_grid > eps_min) & (eps_grid < X)
    e = np.concatenate(([eps_min], eps_grid[inside], [X]))
    lg = np.log(eps_grid)
    g = np.interp(np.log(e), lg, g_grid, left=0.0)
    g[0] = 0.0
    y = g * weight(e)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(e)) / Z)
