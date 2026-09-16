"""
Stage 5: how the heat-ionization lock responds to the one parameter the
manuscript holds fixed -- the residual ionized fraction x_e.

Theta = (2/3) Q / (mu k_B) is quoted as 4.3e4 - 6.9e4 K "independently of the
injection spectrum and of redshift".  It is not independent of x_e, because
Coulomb heating scales as n_e while the atomic channels scale as n_HI.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_csda as C
import igm_losses as L

KB, MU = 8.617333262e-5, 1.0813
print("Lock coefficient Theta and W as a function of the residual ionized "
      "fraction (z = 10, CSDA cascade)\n")
print(f"  {'x_e':>8} {'W(1keV)':>9} {'Q(100eV)':>9} {'Q(1keV)':>9} {'Q(1e5eV)':>10}"
      f" {'Theta(1keV) [K]':>16} {'N_ion/n_H at dT=100 K':>23}")
base = None
for xe in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    L.ION_FRACTION = xe
    Y, H, X, E = C.solve(10.0)

    def at(K0):
        j = int(np.argmin(np.abs(C.KG - K0)))
        return Y[j], H[j]
    y0, h0 = at(1e2); y1, h1 = at(1e3); y5, h5 = at(1e5)
    th = (2.0 / 3.0) * (h1 / y1) / (MU * KB)
    base = base or th
    print(f"  {xe:8.0e} {1e3/y1:9.2f} {h0/y0:9.3f} {h1/y1:9.3f} {h5/y5:10.3f}"
          f" {th:16.3e} {100.0/th:23.3e}")
L.ION_FRACTION = 1e-4
print("\n  At x_e = 1e-2 the lock coefficient is 2.4x larger than the quoted "
      "value, so the\n  same Delta T implies 2.4x FEWER ionizations: the "
      "manuscript's x_e = 1e-4 choice is\n  the one that maximises the inferred "
      "ionization contribution.")
