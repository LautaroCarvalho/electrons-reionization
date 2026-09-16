"""
Stage 3c: how good is the headline assumption of Fig. 4 -- that every IC photon
between 13.6 eV and 1 keV makes exactly one hydrogen photoionization?

For a photon of energy E emitted at z, compare the photoionization mean free
path with the Hubble length, and give the fraction of photons that are actually
absorbed (rather than redshifted out of the band or Compton-scattered).
"""
import numpy as np
import redo_common as R
import igm_losses as L

print(f"  {'z':>5} {'E [eV]':>9} {'lambda_pi/(c/H)':>16} {'lambda_tot/(c/H)':>17} "
      f"{'P(absorbed)':>12} {'PI share':>9}")
for z in (20.0, 10.0):
    lh = R.hubble_length(z)
    for E in (13.7, 20.0, 50.0, 1e2, 3e2, 1e3, 1272.0, 1820.0, 3e3):
        nH = L.n_HI(z)
        kpi = nH * float(R.sigma_pi(E)[0])
        ktot = kpi + (1.0 + L.ION_FRACTION) * nH * float(R.sigma_kn(E)[0])
        lpi, ltot = 1.0 / kpi, 1.0 / ktot
        # optical depth accumulated over one Hubble length (crude but bounding)
        P = 1.0 - np.exp(-lh / ltot)
        print(f"  {z:5.1f} {E:9.1f} {lpi/lh:16.3e} {ltot/lh:17.3e} "
              f"{P:12.6f} {kpi/ktot:9.4f}")
    print()
