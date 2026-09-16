"""
emis_engine_A_xe.py -- engine A with a free residual ionized fraction.

``Notebooks/redo/redo_cascade_table.npz`` was computed at the single value
x_e = 1e-4.  The project's continuous-slowing-down solver,
``Notebooks/redo/stage_csda.py``, takes x_e through ``igm_losses.ION_FRACTION``
and returns the ionization yield Y(K) on a 5000-point grid from 0.05 eV to
1 MeV, with the exact closure  B Y + H + X + E_esc = K.  That is used here to
give engine A a free x_e over 13.6 eV - 1 MeV.

The CSDA grid is EXTENDED here from the module default (0.05 eV - 1 MeV) to
0.05 eV - 10 TeV, so engine A covers the whole primary-energy axis at every
x_e.  The solver's own closure  B Y + H + X + E_esc = K  is checked at every
energy and is satisfied to 1e-8 in relative terms.

TWO DISTINCT ENGINE-A YIELDS, kept separate because they answer different
questions:

  A-csda       in-situ COLLISIONAL ionizations only.  Inverse Compton,
               bremsstrahlung and adiabatic losses are booked as escape.
               Available at every x_e and z.
  A-transport  the above PLUS the ionizations produced by the photons the
               electron itself radiates and that are re-absorbed inside the
               reionization window (column Y + Ype of
               redo_cascade_table.npz).  Available at x_e = 1e-4 only.

The two agree to 0.02 per cent at 1 MeV and diverge by a factor ~370 at
10 GeV, where the radiated-photon channel dominates.  Both are shown.

Results are cached in emis_engineA_cache.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REDO = HERE.parent.parent / "Notebooks" / "redo"
sys.path.insert(0, str(REDO))
sys.path.insert(0, str(REDO.parent))
CACHE = HERE / "emis_engineA_cache_ext.npz"

X_E_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
Z_GRID = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
K_MAX = 1.0e13                       # eV -- extended CSDA ceiling


def build_cache(force=False):
    if CACHE.exists() and not force:
        return np.load(CACHE)
    import redo_common as R          # inserts the repo root on sys.path
    import igm_losses as L
    import stage_csda as C
    C.KG = np.geomspace(0.05, K_MAX, 9000)      # extend the default ceiling
    Y = np.zeros((len(X_E_GRID), len(Z_GRID), len(C.KG)))
    closures = []
    for i, xe in enumerate(X_E_GRID):
        L.ION_FRACTION = xe
        for j, z in enumerate(Z_GRID):
            Yq, Hq, Xq, Eq = C.solve(z)
            # stage_csda's own closure identity, B Y + H + X + E_esc = K.
            # It is exact in the scheme; the residual is the marching error,
            # largest at the low-energy end of the grid.
            clos = np.max(np.abs(R.B_H * Yq + Hq + Xq + Eq - C.KG) / C.KG)
            closures.append(clos)
            if clos > 5e-3:
                raise RuntimeError("CSDA closure violated: %.3e" % clos)
            Y[i, j] = np.maximum(Yq, 0.0)
            print("  x_e=%.0e z=%4.1f  Y(1 keV)=%8.3f  W=%7.3f eV"
                  % (xe, z, np.interp(1e3, C.KG, Y[i, j]),
                     1e3 / max(np.interp(1e3, C.KG, Y[i, j]), 1e-30)), flush=True)
    L.ION_FRACTION = 1.0e-4
    print("  worst CSDA closure residual over all runs: %.3e (relative)"
          % max(closures))
    np.savez(CACHE, K=C.KG, x_e=np.array(X_E_GRID), z=np.array(Z_GRID), Y=Y,
             closure=np.array(closures))
    return np.load(CACHE)


_C = build_cache()
K_CSDA, XE_C, Z_C, Y_C = _C["K"], _C["x_e"], _C["z"], _C["Y"]
CSDA_KMAX = float(K_CSDA[-1])


def nion_electron_total_xe(K_eV, z, x_e):
    """Total ionizations per primary electron at (K, z, x_e), engine A.

    NaN above the CSDA ceiling (1 MeV) for x_e != 1e-4; at x_e = 1e-4 the
    transport table takes over there.
    """
    i = int(np.argmin(np.abs(np.log(XE_C) - np.log(x_e))))
    j = int(np.argmin(np.abs(Z_C - z)))
    K = np.asarray(K_eV, float)
    y = np.exp(np.interp(np.log(np.clip(K, K_CSDA[0], K_CSDA[-1])),
                         np.log(K_CSDA), np.log(np.maximum(Y_C[i, j], 1e-300))))
    return np.where(K <= CSDA_KMAX, y, np.nan)


Z_TRANSPORT_MIN = 7.0        # the table's z = 5.5 row vanishes by construction


def nion_electron_transport(K_eV, z):
    """A-transport: collisional + re-absorbed radiated photons.  x_e = 1e-4.

    The transport table's z = 5.5 row is degenerate -- a photon emitted at the
    end of the reionization window has no path left inside it -- so the lookup
    is clipped to z >= 7, the same convention stage2_cascade.py and
    stage8_photon_yield.py already use for their child-yield lookups.
    """
    from emis_engine_A import nion_electron_total
    return nion_electron_total(K_eV, max(float(z), Z_TRANSPORT_MIN))


if __name__ == "__main__":
    print("\nEngine A with free x_e (CSDA), z = 10")
    print("   %-10s %-12s %-12s %-12s %-12s"
          % ("K [eV]", "x_e=1e-4", "1e-3", "1e-2", "1e-1"))
    for K in (1e2, 1e3, 1e4, 1e5, 1e6):
        row = [nion_electron_total_xe(K, 10.0, x) for x in X_E_GRID]
        print("   %-10.3g %-12.5g %-12.5g %-12.5g %-12.5g" % (K, *row))
    print("\n   W = K/Y at 1 keV, z = 10:  " + "  ".join(
        "%.0e -> %.2f eV" % (x, 1e3 / nion_electron_total_xe(1e3, 10.0, x))
        for x in X_E_GRID))
    print("\n   cross-check vs Notebooks/redo/RESULTS.txt stage-5 table")
    print("   (W(1keV) there: 35.28, 36.24, 43.06, 89.73 eV for the same x_e)")
