#!/usr/bin/env python3
r"""
Parameter layer over igm_losses.py.

STANDING DECISION (2026-09-11, user): all electron-propagation physics is
computed with igm_losses.py from here on, with the environmental parameters
left adjustable.

igm_losses keeps its parameters as MODULE GLOBALS, so overriding them means
mutating the module. That is fine for the ones the loss functions read at call
time, and NOT fine for the ones baked into the cached cross-section tables at
import. This module separates the two and refuses the second kind rather than
letting it silently do nothing -- the same failure mode as the default-argument
binding bug in photon_vs_electron.py, where a mutated global never reached the
function that was supposed to use it.

    with igm_params(z=12.0, B0_T=1e-12):
        ...                       # igm_losses now describes that environment
    ...                           # every global restored, checked on exit
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, asdict

import numpy as np

import igm_losses as L

# Read at call time by loss_rates / n_HI / kinematics -> safe to override.
LIVE = {
    "B0": "B0_T",                  # comoving magnetic field [T] at z=0
    "ION_FRACTION": "x_e",         # n_e / n_HI
    "N_HI_NORM": "n_HI_norm",      # n_HI(z) = norm ((1+z)/21)^3  [m^-3]
    "Z_TARGET": "Z_target",
    "DELTA_GOULD": "delta_gould",
    "Z_INIT": "z_init",
    "Z_FINAL": "z_final",
    "THERMAL_FLOOR_FACTOR": "thermal_floor_factor",
}

# Baked into .igm_losses_cache.npz when igm_losses is imported. Changing these
# after import changes NOTHING, so we refuse instead of pretending.
FROZEN = {
    "EXC_NMAX": "exc_nmax",
    "FIX_SECONDARY_SPECTRUM": "fix_secondary_spectrum",
    "FIX_KN_KERNEL": "fix_kn_kernel",
}


@dataclass(frozen=True)
class IGMConfig:
    """Environment for the electron-propagation physics. None = igm_losses default."""
    z: float = 10.0                       # redshift at which to evaluate
    B0_T: float | None = None
    x_e: float | None = None
    n_HI_norm: float | None = None
    Z_target: float | None = None
    delta_gould: float | None = None
    z_init: float | None = None
    z_final: float | None = None
    thermal_floor_factor: float | None = None

    def overrides(self) -> dict:
        d = asdict(self)
        d.pop("z")
        return {k: v for k, v in d.items() if v is not None}

    def describe(self) -> dict:
        """Every value actually in force, resolved against igm_losses."""
        out = {"z": self.z}
        for g, name in LIVE.items():
            v = getattr(self, name, None)
            out[name] = float(getattr(L, g)) if v is None else float(v)
        for g, name in FROZEN.items():
            out[name] = getattr(L, g)
        return out


@contextlib.contextmanager
def igm_params(**kw):
    """Temporarily override igm_losses globals; restore and verify on exit."""
    frozen_asked = {k: v for k, v in kw.items() if k in FROZEN.values()}
    if frozen_asked:
        raise ValueError(
            "these are baked into the cached cross-section tables at import and "
            f"cannot be changed here: {sorted(frozen_asked)}. Edit igm_losses.py, "
            "delete .igm_losses_cache.npz and re-import. Refusing rather than "
            "silently ignoring the request.")
    rev = {v: k for k, v in LIVE.items()}
    unknown = [k for k in kw if k not in rev]
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}; "
                         f"available: {sorted(rev)}")
    saved = {rev[k]: getattr(L, rev[k]) for k in kw}
    try:
        for k, v in kw.items():
            setattr(L, rev[k], v)
        yield
    finally:
        for g, v in saved.items():
            setattr(L, g, v)
        for g, v in saved.items():                    # restoration is checked
            assert getattr(L, g) == v, f"failed to restore igm_losses.{g}"


def H_of_z(z):
    """H(z) in s^-1 from the Planck18 cosmology igm_losses itself uses."""
    from astropy.cosmology import Planck18
    import astropy.units as u
    return float(Planck18.H(z).to(1 / u.s).value)


def environment_report(cfg: IGMConfig) -> dict:
    """What the physics actually sees under cfg -- for the provenance registry."""
    with igm_params(**cfg.overrides()):
        d = cfg.describe()
        d["n_HI_cm3_at_z"] = float(L.n_HI(cfg.z)) * 1e-6
        d["H_z_s"] = H_of_z(cfg.z)
        d["U_B_over_U_CMB"] = float(
            ((getattr(L, "B0") * (1 + cfg.z) ** 2) ** 2 / (2 * L.VACUUM_PERMEABILITY))
            / (L.U_CMB_0_J_M3 * (1 + cfg.z) ** 4))
    return d


@contextlib.contextmanager
def safe_plot_style():
    """Render under our own matplotlib settings, then restore the globals.

    igm_losses mutates GLOBAL plt.rcParams at import: it switches on
    text.usetex whenever a LaTeX install is present. Every figure drawn after
    that import is then compiled by LaTeX, where a bare "&", "%" or "_" in a
    label is a syntax error -- and the failure lands AFTER all the physics
    checks have passed, so it looks like the science broke when it did not.
    It has now bitten twice, in two different modules, so the guard lives in
    one place and both use it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ("text.usetex", "font.family", "font.serif", "axes.grid",
            "xtick.direction", "ytick.direction", "xtick.top", "ytick.right",
            "xtick.minor.visible", "ytick.minor.visible",
            "axes.prop_cycle", "lines.linewidth", "font.size",
            "axes.labelsize", "legend.fontsize")
    saved = {k: plt.rcParams[k] for k in keys}
    plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                         "axes.grid": False, "xtick.direction": "out",
                         "ytick.direction": "out", "xtick.top": False,
                         "ytick.right": False, "xtick.minor.visible": False,
                         "ytick.minor.visible": False, "lines.linewidth": 1.5,
                         "font.size": 10.0, "axes.labelsize": 10.0,
                         "legend.fontsize": 9.0})
    try:
        yield plt
    finally:
        plt.rcParams.update(saved)
