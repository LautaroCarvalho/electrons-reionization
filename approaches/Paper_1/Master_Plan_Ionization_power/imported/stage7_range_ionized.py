# Copied from Notebooks/redo/stage7_range_ionized.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""
Stage 7: distance travelled from the source in a COMPLETELY IONIZED medium.

Companion to ``stage6_range.py``.  Same skeleton, same K grid, same cosmology,
same three ODE states, same terminal-event integration -- the only thing that
changes is the medium the electron is moving through.  The two tables are
therefore directly comparable node by node.

The medium
----------
Same baryons, different ionization state.  The hydrogen density law of the rest
of the project,

    n_H(z) = 1.8e3 ((1+z)/21)^3  m^-3,

is kept, but now x_e = 1 instead of x_e = 1e-4, so

    n_e = n_p = n_H(z),      n_HI = 0.

This is the inside of an H II region, or the post-reionization IGM: the same
gas that stage6 treated as neutral, fully stripped.

What changes in the loss budget
-------------------------------
The seven channels are returned in the same order as ``igm_losses.loss_rates``
so that the two runs can be compared channel by channel.

  0 adiabatic       unchanged; H(z) p^2c^2/E does not care about ionization.
  1 synchrotron     unchanged; same fiducial B(z) = 1 nG (1+z)^2.
  2 inverse Compton unchanged; the CMB does not care either.
  3 Coulomb         SAME Gould (1972) expression, but n_e is now the full gas
                    density instead of 1e-4 of it: a factor 1e4 stronger.  This
                    is the whole story below ~1 MeV.
  4 collisional excitation of H(1s)   ZERO.  There are no bound electrons.
  5 collisional ionization of H(1s)   ZERO.  Same reason.
  6 bremsstrahlung  Rewritten.  ``igm_losses`` uses Z^2 with the neutral-atom
                    screening interpolation, appropriate for an electron
                    passing a NEUTRAL hydrogen atom.  In a fully ionized plasma
                    the nucleus is bare, so there is no atomic screening, and
                    the plasma electrons radiate too; the standard result
                    (Blumenthal & Gould 1970) is Z(Z+1) = 2 for hydrogen with
                    the unscreened factor ln(2 gamma) - 1/3.

So: channels 4 and 5 are switched off, channel 3 is amplified by 1e4, channel 6
is doubled and de-screened, and channels 0-2 are untouched.

The thermal floor
-----------------
stage6 stops the trajectory at 1.5 k_B T_CMB(z), which is the right floor for
gas that is thermally coupled to nothing warmer.  A fully ionized medium is
not that: it is photoionized and sits near T_gas ~ 1e4 K, so the floor here is

    K_term = 1.5 k_B T_GAS = 1.29 eV     (T_GAS = 1e4 K)

and it is redshift independent.  This is not cosmetic.  The Gould Coulomb loss
rate is a fast-test-particle expression: it is only valid while the electron is
well above the thermal speed of the plasma, and its Coulomb logarithm goes
through zero near thermal energies.  Stopping at 1.5 k_B T_GAS is what keeps
the integration inside the domain where the rate means anything.  Nodes with
K_ini below that floor are integrated over a zero-length interval and are
reported as zero; the figure masks them.

Everything else -- the states, the units, the grids, the flags -- is identical
to stage6_range.py, and the same three caveats apply: chi is an arc length and
therefore an upper bound on the displacement, the electron is magnetized, and
trajectories that had not cooled by z = 5.5 are flagged.
"""
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import redo_common as R
import igm_losses as L

EV = R.EV
MPC = L.MEGAPARSEC_MKS
OUT = Path(__file__).resolve().parent / "redo_range_table_ionized.npz"

T_GAS = 1.0e4                       # K; photoionized-plasma temperature
K_TERM_J = 1.5 * L.K_B * T_GAS      # thermal floor, redshift independent
Z_TARGET = 1.0                      # hydrogen

# Identical grids to stage6_range.py.
K_LOW = np.geomspace(0.30, R.B_H * 0.995, 9)
K_MAIN = np.concatenate([np.geomspace(R.B_H * 1.02, 1e3, 22),
                         np.geomspace(1.3e3, 1e13, 42)])
K_GRID = np.concatenate([K_LOW, K_MAIN])
NLOW = len(K_LOW)
Z_GRID = np.array([5.75, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0,
                   12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
Z_FINAL = 5.5
LAMBDA_C = (1.0e-3, 1.0)            # comoving Mpc, for the random-walk band


def n_e(z):
    """Electron (= proton) density [m^-3] of the fully ionized gas."""
    return L.N_HI_NORM * ((1.0 + np.asarray(z, float)) / 21.0) ** 3


def loss_rates_ionized(z, K_J, H_z):
    """Positive |dK/dt| [J/s] for the seven channels, in MECH_KEYS order."""
    E, gamma, p2c2, v = L.kinematics(K_J)
    K = E - L.E0
    ne = n_e(z)

    # 0) adiabatic expansion
    L_ad = H_z * p2c2 / E

    # 1) synchrotron on the fiducial field
    U_B = (L.B0 * (1.0 + z) ** 2) ** 2 / (2.0 * L.VACUUM_PERMEABILITY)
    L_sy = ((4.0 / 3.0) * L.THOMSON_CROSS_SECTION_MKS * L.C_LIGHT
            * U_B / L.E0 ** 2 * p2c2)

    # 2) inverse Compton on the CMB, Klein-Nishina corrected
    U_cmb = L.U_CMB_0_J_M3 * (1.0 + z) ** 4
    b_kn = 4.0 * gamma * L.K_B * L.T_CMB_0 * (1.0 + z) / L.E0
    L_ic = ((4.0 / 3.0) * L.THOMSON_CROSS_SECTION_MKS * L.C_LIGHT
            * U_cmb / L.E0 ** 2 * p2c2 * L.F_KN(b_kn))

    # 3) Coulomb on the plasma (Gould 1972) -- verbatim the expression used by
    #    igm_losses.loss_rates, evaluated at the full electron density.
    w_p = np.sqrt(ne * L.ELECTRON_CHARGE_MKS ** 2
                  / (L.VACUUM_PERMITTIVITY * L.ELECTRON_MASS_MKS))
    x = (np.sqrt(K / L.E0) * v * L.C_LIGHT * L.ELECTRON_MASS_MKS
         * np.sqrt(2.0 * L.DELTA_GOULD)
         / (L.PLANCK_CONSTANT_REDUCED_MKS * w_p))
    x_safe = np.maximum(x, 1e-50)
    fb = (np.log(x_safe)
          + np.log(1.0 - L.DELTA_GOULD) * (0.5 + 1.0 / gamma - 0.5 / gamma ** 2)
          + 0.5 * L.DELTA_GOULD / (1.0 - L.DELTA_GOULD)
          + 0.25 * (1.0 - 1.0 / gamma) ** 2 * L.DELTA_GOULD ** 2)
    # The Coulomb logarithm goes negative once the test particle stops being
    # fast compared with the plasma; the thermal floor is meant to be reached
    # long before that, but the rate is clipped so the ODE can never gain
    # energy if a stiff step overshoots.
    force = (ne * Z_TARGET ** 2 * L.ELECTRON_CHARGE_MKS ** 4 * np.maximum(fb, 0.0)
             / (4.0 * np.pi * L.VACUUM_PERMITTIVITY ** 2
                * L.ELECTRON_MASS_MKS * np.maximum(v, 1e-30) ** 2))
    L_co = np.where(x > 1e-40, force * v, 0.0)

    # 4), 5) no bound electrons left
    zero = np.zeros(np.broadcast(np.asarray(z, float), E).shape)
    L_ex = zero
    L_io = zero

    # 6) bremsstrahlung on bare protons, plus the electron-electron term,
    #    unscreened: Z(Z+1) with phi = ln(2 gamma) - 1/3.
    phi = np.maximum(np.log(2.0 * gamma) - 1.0 / 3.0, 0.0)
    L_br = (L.PREFACTOR_BREMS * v * ne * Z_TARGET * (Z_TARGET + 1.0) * E * phi)

    return np.stack(np.broadcast_arrays(L_ad, L_sy, L_ic, L_co, L_ex, L_io, L_br))


def rhs(t, y):
    """[dK/dt, d(ell)/dt, d(chi)/dt] with the lengths in comoving Mpc."""
    KJ = max(float(y[0]), 0.0)
    if KJ <= 0.0:
        return [0.0, 0.0, 0.0]
    z = float(L.z_interp(t))
    lr = loss_rates_ionized(z, KJ, float(L.H_interp(t)))
    tot = float(np.asarray(lr, float).sum())
    _, _, _, v = L.kinematics(KJ)
    v = float(v)
    return [-tot, v / MPC, v * (1.0 + z) / MPC]


def _event(t, y):
    return y[0] - K_TERM_J


_event.terminal, _event.direction = True, -1


def run():
    t0 = time.time()
    nK, nZ = len(K_GRID), len(Z_GRID)
    ell = np.zeros((nZ, nK))
    chi = np.zeros((nZ, nK))
    zend = np.zeros((nZ, nK))
    tcool = np.zeros((nZ, nK))
    cooled = np.zeros((nZ, nK), bool)
    below = K_GRID * EV <= K_TERM_J          # injected already thermal

    print(f"fully ionized medium: x_e = 1, T_gas = {T_GAS:.3g} K, "
          f"K_term = {K_TERM_J/EV:.4f} eV ({int(below.sum())} grid nodes "
          f"start below it)", flush=True)

    t_f = L.age_s(Z_FINAL)
    for iz, z_i in enumerate(Z_GRID):
        t_i = L.age_s(z_i)
        for ik, K0 in enumerate(K_GRID):
            if below[ik]:
                continue
            sol = solve_ivp(rhs, (t_i, t_f), [K0 * EV, 0.0, 0.0],
                            events=_event, method="Radau",
                            rtol=1e-6, atol=[1e-30, 1e-18, 1e-18])
            ell[iz, ik] = sol.y[1, -1]
            chi[iz, ik] = sol.y[2, -1]
            tcool[iz, ik] = sol.t[-1] - t_i
            zend[iz, ik] = float(L.z_interp(sol.t[-1]))
            cooled[iz, ik] = bool(sol.t_events[0].size)
        i1k = int(np.argmin(np.abs(K_GRID - 1e3)))
        i1m = int(np.argmin(np.abs(K_GRID - 1e6)))
        print(f"[{time.time()-t0:6.1f}s] z_i={z_i:5.2f} | "
              f"chi(1 keV)={chi[iz, i1k]:9.3e} cMpc "
              f"chi(1 MeV)={chi[iz, i1m]:9.3e} cMpc "
              f"chi_max={chi[iz].max():9.3e} cMpc @ K={K_GRID[chi[iz].argmax()]:9.3e} eV"
              f" | {int((~cooled[iz] & ~below).sum())} window-truncated", flush=True)

    rg = np.zeros((nZ, nK))
    KJ = K_GRID[None, :] * EV
    _, _, p2c2, _ = L.kinematics(np.broadcast_to(KJ, (nZ, nK)))
    Bz = L.B0 * (1.0 + Z_GRID[:, None]) ** 2
    rg = (np.sqrt(p2c2) / L.C_LIGHT) / (L.ELECTRON_CHARGE_MKS * Bz)

    np.savez(OUT, K=K_GRID, z=Z_GRID, nlow=NLOW, z_final=Z_FINAL,
             ell=ell, chi=chi, zend=zend, tcool=tcool, cooled=cooled,
             below=below, rg_proper_m=rg, lambda_c=np.array(LAMBDA_C),
             t_gas=T_GAS, k_term_eV=K_TERM_J / EV, x_e=1.0)
    print("saved", OUT, f"in {time.time()-t0:.1f} s")
    return OUT


if __name__ == "__main__":
    run()
