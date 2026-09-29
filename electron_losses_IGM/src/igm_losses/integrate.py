"""Single ODE driver for the electron energy K(t) between z_init and z_final (guideline 4; D05, D06, D09, D13).

    dK/dt = − Σ_i L_i(K, z(t))                  (state = kinetic energy K [J], D13)
    augmented=True adds dE_i/dt = L_i (energy lost to each process, cells 36–39) and the path lengths
    dD/dt = v (proper) and dχ/dt = v (1+z) (comoving) (A09).

Numerics   scipy solve_ivp, method Radau, rtol and atol from parameters.yaml (D05: 1e-8, 1e-25 J; distances
           of the augmented mode use atol_distance = 1 m).
Stop       terminal event K = floor (default K_floor = 10.2 eV, A07; per figure the floor may be a process
           threshold, D09). After the event K is held at the floor on the output grid.
Grid       union of A (elapsed time, log from t_grid_A_start) and B (cosmic time, log), sizes N_A, N_B (D06).
fixed_z    z held at z_init (cell 12, "UTOPIA" comparison).
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as K_
from . import cosmology, kinematics, losses

P = K_._P


def time_grid(z_init, z_final):
    t0, t1 = float(cosmology.age(z_init)), float(cosmology.age(z_final))
    elapsed = np.logspace(np.log10(P["t_grid_A_start"] * K_.year), np.log10(t1 - t0), int(P["N_time_grid_A"]))
    A = t0 + elapsed
    B = np.logspace(np.log10(t0), np.log10(t1), int(P["N_time_grid_B"]))
    return np.unique(np.concatenate(([t0], A, B)))


def evolve(K0_eV, z_init, z_final, x_e, include=losses.PROCESSES, floor_eV=None, fixed_z=False,
           augmented=False, t_eval=None, **opts):
    """Integrate K(t). Returns dict with t [s], z, K [eV] (and per-process lost energy [eV], D, χ [m] if augmented)."""
    floor = (P["K_floor"] if floor_eV is None else floor_eV) * K_.e
    t_eval = time_grid(z_init, z_final) if t_eval is None else np.asarray(t_eval)
    t0, t1 = t_eval[0], t_eval[-1]
    names = list(include)

    def zt(t):
        return z_init if fixed_z else float(cosmology.redshift(min(max(t, cosmology.T_MIN_S), cosmology.T_MAX_S)))

    # [A11] [A17] continuous mean loss of a test particle
    def rhs(t, y):
        z = zt(t)
        r = losses.rates(y[0], z, x_e, names, **opts)
        dK = -sum(r.values())
        if not augmented:
            return [dK]
        v = float(kinematics.speed(max(y[0], 0.0)))
        return [dK] + [r[n] for n in names] + [v, v * (1.0 + z)]

    # [A07] stop at the floor K_floor
    def floor_event(t, y):
        return y[0] - floor
    floor_event.terminal, floor_event.direction = True, -1

    y0 = [K0_eV * K_.e] + ([0.0] * (len(names) + 2) if augmented else [])
    # per-component absolute tolerance: energies in J use atol (D05); distances in m use atol_distance
    atol = [P["atol"]] + ([P["atol"]] * len(names) + [P["atol_distance"]] * 2 if augmented else [])
    sol = solve_ivp(rhs, (t0, t1), y0, method="Radau", t_eval=t_eval, events=floor_event,
                    rtol=P["rtol"], atol=atol, dense_output=False)
    if sol.status < 0:
        raise RuntimeError(sol.message)
    n_ok = sol.y.shape[1]
    out = {"t": t_eval, "z": np.array([zt(t) for t in t_eval]), "status": sol.status,
           "t_floor": float(sol.t_events[0][0]) if sol.t_events[0].size else None}
    K = np.full(t_eval.size, floor / K_.e)
    K[:n_ok] = sol.y[0] / K_.e
    out["K"] = K
    if augmented:
        # after the floor event every component is held at its value AT the event (not at the last grid point before it)
        held = sol.y_events[0][0] if sol.t_events[0].size else sol.y[:, -1]
        for i, n in enumerate(names):
            arr = np.full(t_eval.size, held[1 + i] / K_.e)
            arr[:n_ok] = sol.y[1 + i] / K_.e
            out["lost_" + n] = arr
        # [A09] proper path length ∫v dt and comoving distance ∫v(1+z) dt
        for j, key in enumerate(("D_proper", "D_comoving")):
            arr = np.full(t_eval.size, held[1 + len(names) + j])
            arr[:n_ok] = sol.y[1 + len(names) + j]
            out[key] = arr
    return out
