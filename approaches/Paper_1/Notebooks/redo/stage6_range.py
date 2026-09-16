"""
Stage 6: how far does an electron get from the place where it was injected?

Same skeleton as ``stage2_cascade.py`` -- the same K grid, the same fiducial
IGM, the same seven loss channels from ``igm_losses.loss_rates``, the same
stiff integration from an injection redshift down to z = 5.5 with a terminal
event at the thermal floor 1.5 k_B T_CMB(z).  The only difference is what is
accumulated along the way.  There is no cascade recursion here: the question
"where does this electron end up" is a property of the PRIMARY trajectory
alone, so a single ODE per (K_ini, z_i) node is enough.

States
------
  y0  K       kinetic energy of the primary                       [J]
  y1  ell     PROPER path length,      d(ell)/dt = v              [cMpc-scaled]
  y2  chi     COMOVING path length,    d(chi)/dt = v (1+z)        [cMpc]

Both are integrated in comoving-Mpc units so that the state vector stays well
scaled across the thirteen decades of injection energy.

What "distance from the source" means here
------------------------------------------
Three different numbers are produced and they must not be confused.

1. ``chi``  -- the BALLISTIC comoving range.  This is how far the electron
   would get if it moved in a straight line at speed v(K) until it thermalized,
   with the expansion of the universe divided out.  It is the arc length of the
   trajectory, so it is a strict UPPER BOUND on the displacement from the
   source.  This is the quantity plotted as "the distance travelled".

2. ``r_g`` -- the gyroradius p/(eB) in the fiducial field B(z) = 1 nG (1+z)^2
   that ``igm_losses`` already assumes for the synchrotron channel.  The
   electron is magnetized: it spirals, so the ballistic range is only reached
   if the field happens to be coherent over the whole path.

3. ``chi_rw`` -- the displacement of a random walk whose step is the comoving
   coherence length lambda_c of the field, sqrt(lambda_c * chi), capped at
   ``chi``.  This is the realistic estimate when the field is tangled on a
   scale much shorter than the ballistic range.  lambda_c is a free parameter
   of the IGM, not something this project fixes, so it is scanned over two
   decades (1 comoving kpc to 1 comoving Mpc) and drawn as a band.

The shape of chi(K) is the physical content of the figure.  Below ~10 keV the
electron is stopped by collisional ionization and excitation of the neutral gas
within a few comoving kpc, and the range grows steeply with K because the
collisional cross-sections fall.  Above ~1 MeV the range SATURATES: an electron
injected at 10^13 eV is cooled by inverse Compton down to a few MeV almost
instantly (t_IC ~ 1/gamma), covering a negligible distance while it does so,
and then travels exactly the same MeV track as an electron that was injected
at a few MeV in the first place.  The plateau is therefore the range of the
MeV electron, not of the primary, and it is set by the slowest channel there
(adiabatic + inverse Compton), i.e. by a sizeable fraction of the Hubble time.
That is why the plateau sits at tens to hundreds of comoving Mpc.

Truncation
----------
A trajectory that has not reached the thermal floor by z = 5.5 is cut by the
end of the integration window, not by physics.  Those nodes are flagged
(``cooled == False``) and drawn with open markers, because their range is a
lower bound on the true one.
"""
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import redo_common as R
import igm_losses as L

EV = R.EV
MPC = L.MEGAPARSEC_MKS
OUT = Path(__file__).resolve().parent / "redo_range_table.npz"

# Same K grid as stage2_cascade.py / cascade_traj.py, so the range table can be
# read side by side with the yield table element by element.
K_LOW = np.geomspace(0.30, R.B_H * 0.995, 9)
K_MAIN = np.concatenate([np.geomspace(R.B_H * 1.02, 1e3, 22),
                         np.geomspace(1.3e3, 1e13, 42)])
K_GRID = np.concatenate([K_LOW, K_MAIN])
NLOW = len(K_LOW)

# Finer than the cascade grid: nothing here is expensive, and the second panel
# of the figure is a cut at fixed K through the redshift axis, which wants more
# than eight nodes to be readable.
Z_GRID = np.array([5.75, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0,
                   12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
Z_FINAL = 5.5

# Comoving coherence lengths of the IGM field used for the random-walk band.
LAMBDA_C = (1.0e-3, 1.0)          # comoving Mpc


def gyroradius(K_eV, z):
    """Proper gyroradius [m] of an electron of kinetic energy K_eV at z.

    B(z) = B0 (1+z)^2 with B0 = 1 nG, the same field ``igm_losses`` uses in the
    synchrotron rate.  r_g = p / (e B), with p from the exact kinematics.
    """
    K = np.asarray(K_eV, float) * EV
    _, _, p2c2, _ = L.kinematics(K)
    p = np.sqrt(p2c2) / L.C_LIGHT
    Bz = L.B0 * (1.0 + np.asarray(z, float)) ** 2
    return p / (L.ELECTRON_CHARGE_MKS * Bz)


def rhs(t, y):
    """[dK/dt, d(ell)/dt, d(chi)/dt] with the lengths in comoving Mpc."""
    KJ = max(float(y[0]), 0.0)
    if KJ <= 0.0:
        return [0.0, 0.0, 0.0]
    z = float(L.z_interp(t))
    lr = L.loss_rates(z, KJ, float(L.H_interp(t)), L.toggles())
    tot = float(np.asarray(lr, float).sum())
    _, _, _, v = L.kinematics(KJ)
    v = float(v)
    return [-tot, v / MPC, v * (1.0 + z) / MPC]


def _event(t, y):
    return y[0] - float(L.thermal_floor(L.z_interp(t)))


_event.terminal, _event.direction = True, -1


def run():
    t0 = time.time()
    nK, nZ = len(K_GRID), len(Z_GRID)
    ell = np.zeros((nZ, nK))          # proper path length      [cMpc units]
    chi = np.zeros((nZ, nK))          # comoving path length    [cMpc]
    zend = np.zeros((nZ, nK))         # redshift where it stopped
    tcool = np.zeros((nZ, nK))        # elapsed time            [s]
    cooled = np.zeros((nZ, nK), bool)  # reached the thermal floor?

    t_f = L.age_s(Z_FINAL)
    for iz, z_i in enumerate(Z_GRID):
        t_i = L.age_s(z_i)
        for ik, K0 in enumerate(K_GRID):
            sol = solve_ivp(rhs, (t_i, t_f), [K0 * EV, 0.0, 0.0],
                            events=_event, method="Radau",
                            rtol=1e-6, atol=[1e-30, 1e-18, 1e-18])
            ell[iz, ik] = sol.y[1, -1]
            chi[iz, ik] = sol.y[2, -1]
            tcool[iz, ik] = sol.t[-1] - t_i
            zend[iz, ik] = float(L.z_interp(sol.t[-1]))
            cooled[iz, ik] = bool(sol.t_events[0].size)
        n_cut = int((~cooled[iz]).sum())
        print(f"[{time.time()-t0:6.1f}s] z_i={z_i:5.2f} | "
              f"chi(1 keV)={chi[iz, np.argmin(np.abs(K_GRID-1e3))]:9.3e} cMpc "
              f"chi(1 MeV)={chi[iz, np.argmin(np.abs(K_GRID-1e6))]:9.3e} cMpc "
              f"chi_max={chi[iz].max():9.3e} cMpc @ K={K_GRID[chi[iz].argmax()]:9.3e} eV"
              f" | {n_cut} window-truncated", flush=True)

    rg = gyroradius(K_GRID[None, :], Z_GRID[:, None])          # proper [m]
    np.savez(OUT, K=K_GRID, z=Z_GRID, nlow=NLOW, z_final=Z_FINAL,
             ell=ell, chi=chi, zend=zend, tcool=tcool, cooled=cooled,
             rg_proper_m=rg, lambda_c=np.array(LAMBDA_C))
    print("saved", OUT, f"in {time.time()-t0:.1f} s")
    return OUT


if __name__ == "__main__":
    run()
