"""Photoionization of H(1s) by secondary photons: cross section and ionization window (A22, A24).

Cross section  [Verner1996 Eq. (1), p. 488; H I parameters Table 1, p. 489, read from provenance/data/verner1996_HI.yaml]
    σ(E) = σ0 F(y) [Mb],  x = E/E0 − y0,  y = √(x² + y1²),  F(y) = [(x − 1)² + yw²] y^(0.5P − 5.5) (1 + √(y/ya))^(−P).
    Class A (hydrogenic, rms < 0.2 %, p. 495); valid for E_th ≤ E ≤ E_max = 50 keV (p. 494). Between R_H (A12,
    the threshold used by the window) and E_th (13.60 eV, 2e-3 eV above R_H) σ is taken at E_th (A24 regime note).
Window (A22)   a photon ionizes if R_H ≤ E_γ ≤ E_max, where n_HI σ(E_max) c = H(z) (mean interaction time = Hubble
    time); σ decreases monotonically above threshold, so the root is unique. If n_HI σ(E_th) c ≤ H the window is empty.
Units          E in eV at the interface, σ in m².
"""

import math
import pathlib

import yaml
from scipy.optimize import brentq

from . import _paths  # noqa: F401
from igm_losses import constants as K
from igm_losses import cosmology
from igm_losses.medium import Medium

ROOT = pathlib.Path(__file__).resolve().parents[3]
V = yaml.safe_load((ROOT / "provenance" / "data" / "verner1996_HI.yaml").read_text(encoding="utf-8"))
MB = 1.0e-22                                   # m² per Mb (1 Mb = 1e-18 cm², Verner p. 488)
P = K._P


# [A24] Verner et al. 1996 fit for H I
def sigma(E_eV):
    """σ_pi(E) [m²] for photon energy E [eV]; 0 below R_H; ValueError above the fit's E_max."""
    if E_eV < K.R_H_eV:
        return 0.0
    if E_eV > V["E_max_eV"]:
        raise ValueError("above the validity of the Verner et al. 1996 fit")
    E = max(E_eV, V["E_th_eV"])
    x = E / V["E0_eV"] - V["y0"]
    y = math.sqrt(x * x + V["y1"] ** 2)
    F = ((x - 1.0) ** 2 + V["yw"] ** 2) * y ** (0.5 * V["P"] - 5.5) * (1.0 + math.sqrt(y / V["ya"])) ** (-V["P"])
    return V["sigma0_Mb"] * F * MB


# [A22] ionization window: R_H ≤ E_γ ≤ E_max with n_HI σ(E_max) c = H(z)
def window(x_e, z=None):
    """(E_lo, E_max) [eV]; E_max = E_lo when the window is empty."""
    z = P["z_init"] if z is None else z
    m = Medium(z, x_e)
    H = float(cosmology.hubble(z))
    f = lambda lnE: math.log(m.n_HI * sigma(math.exp(lnE)) * K.c / H)
    lo = K.R_H_eV
    a, b = math.log(V["E_th_eV"]), math.log(V["E_max_eV"]) - 1e-12      # b: stay inside the fit's range (rounding)
    if f(a) <= 0.0:
        return lo, lo
    if f(b) > 0.0:
        raise ValueError("the window extends beyond the validity of the Verner et al. 1996 fit (50 keV)")
    return lo, math.exp(brentq(f, a, b, xtol=1e-12))
