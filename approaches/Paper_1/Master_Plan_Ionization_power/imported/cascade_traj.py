# Copied from Notebooks/cascade_traj.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""
cascade_traj.py -- self-consistent ionization yield Y(K, z) of an injected
electron INCLUDING every generation of the secondary shower, with each
generation followed along its own cosmological trajectory.

Fixed point solved by marching upward in K (a secondary always has
eps <= (K - B)/2 < K, so the recursion is causal):

    dK/dt    = -sum_i L_i(z, K)                       [the seven channels]
    dN_1/dt  =  n_HI v sigma_ion(K)
    dY/dt    =  n_HI v sigma_ion(K) [ 1 + <Y(eps, z)>_{p(eps|K)} ]

p(eps|K) is the BEB singly-differential cross-section rebuilt from
``igm_losses._DIPOLE_FIT`` (see cascade_yield.py), so the secondary spectrum
is energy dependent rather than a fixed shape, and Y(K) is genuinely
non-linear rather than K / constant.

Unlike the CSDA version this places no ceiling on the secondary energy: a
secondary born above K_4 loses its energy to escaping photons exactly as a
primary does, because it is integrated with the same loss model.

Output: cascade_traj_table.npz  (K grid, z grid, N_primary, Y_total)
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, interp1d

import igm_losses as L
from cascade_yield import secondary_pdf, sigma_ion, B

EV = L.EV_MKS
OUT = Path(__file__).resolve().parent / "cascade_traj_table.npz"

K_GRID = np.concatenate([
    np.geomspace(B * 1.02, 1e3, 22),
    np.geomspace(1.3e3, 1e13, 42),
])
Z_GRID = np.array([5.5, 7.0, 8.5, 10.0, 12.0, 14.0, 17.0, 20.0])
Z_FINAL = 5.5


def _mean_shower(K_eV, z, Ytab, n_done, ms_interp):
    """<Y(eps, z)> over the BEB secondary spectrum; 0 before anything is known."""
    if ms_interp is None:
        return 0.0
    zc = float(np.clip(z, Z_GRID[0], Z_GRID[-1]))
    lk = float(np.clip(np.log(max(K_eV, K_GRID[0])),
                       np.log(K_GRID[0]), np.log(K_GRID[n_done - 1])))
    return max(float(ms_interp([[zc, lk]])[0]), 0.0)


def _build_ms_interp(Ytab, n_done):
    """Tabulate <Y(eps,z)> as a function of the PRIMARY energy, on K_GRID[:n_done]."""
    if n_done < 2:
        return None
    grid = np.zeros((len(Z_GRID), n_done))
    Kk = K_GRID[:n_done]
    for ik in range(n_done):
        e, p = secondary_pdf(K_GRID[ik])
        if e is None:
            continue
        m = e >= B
        if not m.any():
            continue
        for iz in range(len(Z_GRID)):
            Yv = np.interp(e[m], Kk, Ytab[iz, :n_done], left=0.0)
            grid[iz, ik] = np.trapz(p[m] * Yv, e[m])
    return RegularGridInterpolator((Z_GRID, np.log(Kk)), grid,
                                   bounds_error=False, fill_value=None)


def run():
    nK, nZ = len(K_GRID), len(Z_GRID)
    N1 = np.zeros((nZ, nK))
    Y = np.zeros((nZ, nK))
    ms = None

    for ik, K0 in enumerate(K_GRID):
        for iz, z_i in enumerate(Z_GRID):
            t_i, t_f = L.age_s(z_i), L.age_s(Z_FINAL)

            def rhs(t, y):
                K = max(float(y[0]), 0.0)
                if K <= 0.0:
                    return [0.0, 0.0, 0.0]
                z = float(L.z_interp(t))
                tot = float(L.total_loss(z, K, float(L.H_interp(t)),
                                         L.toggles()))
                _, _, _, v = L.kinematics(K)
                rate = L.n_HI(z) * float(v) * float(sigma_ion(K / EV))
                sec = _mean_shower(K / EV, z, Y, ik, ms) if ik else 0.0
                return [-tot, rate, rate * (1.0 + sec)]

            def ev(t, y):
                return y[0] - float(L.thermal_floor(L.z_interp(t)))
            ev.terminal, ev.direction = True, -1

            sol = solve_ivp(rhs, (t_i, t_f), [K0 * EV, 0.0, 0.0], events=ev,
                            method="Radau", rtol=1e-6, atol=1e-25)
            N1[iz, ik] = sol.y[1][-1]
            Y[iz, ik] = sol.y[2][-1]

        ms = _build_ms_interp(Y, ik + 1)
        if ik % 4 == 0 or ik == nK - 1:
            print(f"  K = {K0:10.4g} eV   Y(z=10) = {Y[3, ik]:12.5g}"
                  f"   Y(z=20) = {Y[7, ik]:12.5g}", flush=True)

    np.savez(OUT, K=K_GRID, z=Z_GRID, N1=N1, Y=Y)
    print("saved", OUT)


if __name__ == "__main__":
    run()
