"""
CR-electron transport in the neutral IGM, extending imported/cascade_yield.py's
yields() ODE (K, N_primary, N_total) with a fourth state tracking PROPER path
length r(t) = integral of v dt (meters, MKS, matching igm_losses.py's own unit
convention). This is what Phase 2 needs to build R_CR(z): a cumulative
"ionizations produced by path length <= r" curve per injected electron, not
just the total yield at thermalization that cascade_yield.yields() returns.

Per Section A.1 of the plan / master_plan_response.tex: r(t) is a path length,
a rigorous UPPER BOUND on radial displacement (gyroradius ~1e-8-1e-2 pc vs.
Mpc-scale bubble; Jana & Nath 2018), never a literal radius. Every use of this
module's r(t) output must carry that caveat.
"""
import sys
import pathlib
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "imported"))
import igm_losses as L          # noqa: E402
import cascade_yield as CY      # noqa: E402

EV = L.EV_MKS
MPC_M = 3.0857e22  # m, for convenience (1 Mpc = 3.0857e24 cm = 3.0857e22 m)


def electron_trajectory(K_ini_eV, z_i, z_f=5.5, rtol=1e-7):
    """Integrate one CR electron injected at (K_ini_eV, z_i) down to
    thermalization (or z_f). Returns a dict with dense interpolators for
    N_total(r) and N_primary(r) (cumulative ionizations produced by proper
    path length <= r, in meters), plus the final thermalization values.
    """
    t_i, t_f = L.age_s(z_i), L.age_s(z_f)

    def rhs(t, y):
        K = max(float(y[0]), 0.0)
        if K <= 0.0:
            return [0.0, 0.0, 0.0, 0.0]
        z = float(L.z_interp(t))
        tot = float(L.total_loss(z, K, float(L.H_interp(t)), L.toggles()))
        _, _, _, v = L.kinematics(K)
        rate = L.n_HI(z) * float(v) * float(CY.sigma_ion(K / EV))
        shower_mult = 1.0 + CY.mean_shower_fast(K / EV, z)
        return [-tot, rate, rate * shower_mult, float(v)]

    def ev(t, y):
        return y[0] - float(L.thermal_floor(L.z_interp(t)))
    ev.terminal, ev.direction = True, -1

    # Per-component atol: K [J] ~1e-14-1e-1, N_primary/N_total [count] ~1-1e5,
    # r [m] ~1-1e24 -- a single scalar atol badly ill-conditions the stiff
    # Radau Newton iteration across this range (verified: caused an immediate
    # "step size less than spacing between numbers" failure with atol=1e-25).
    sol = solve_ivp(rhs, (t_i, t_f), [K_ini_eV * EV, 0.0, 0.0, 0.0], events=ev,
                     method="Radau", rtol=rtol, atol=[1e-25, 1e-6, 1e-6, 1.0],
                     dense_output=False)

    # Use the solver's own adaptive step points directly (avoids a known
    # scipy dense-output edge case when the trajectory is cut short by the
    # thermalization event); Radau's adaptive stepping already resolves the
    # solution's variation, so this is not a resolution loss.
    r_dense = sol.y[3]             # proper path length, meters
    Nprim_dense = sol.y[1]
    Ntot_dense = sol.y[2]

    return dict(
        r_m=r_dense, N_primary=Nprim_dense, N_total=Ntot_dense,
        N_primary_final=float(sol.y[1][-1]), N_total_final=float(sol.y[2][-1]),
        r_final_m=float(sol.y[3][-1]), z_thermal=float(L.z_interp(sol.t[-1])),
        thermalized=bool(sol.t_events[0].size),
    )


def N_total_within_r(traj, r_m):
    """Cumulative N_total ionizations deposited by proper path length <= r_m,
    for a single electron's trajectory dict (from electron_trajectory).
    Saturates at N_total_final for r_m beyond the trajectory's final range."""
    r_m = np.atleast_1d(r_m)
    out = np.interp(r_m, traj["r_m"], traj["N_total"],
                     left=0.0, right=traj["N_total_final"])
    return out if out.size > 1 else out[0]


if __name__ == "__main__":
    print("Self-test: electron_trajectory reproduces cascade_yield.yields() totals")
    for K_ini, z_i in [(1e3, 20.0), (1e5, 10.0), (1e8, 15.0)]:
        traj = electron_trajectory(K_ini, z_i)
        ref_prim, ref_tot, ref_z = CY.yields(K_ini, z_i)
        print(f"K={K_ini:.0e} eV z_i={z_i:4.1f}  "
              f"N_primary: {traj['N_primary_final']:.4f} vs ref {ref_prim:.4f}  "
              f"N_total: {traj['N_total_final']:.4f} vs ref {ref_tot:.4f}  "
              f"z_therm: {traj['z_thermal']:.4f} vs ref {ref_z:.4f}  "
              f"r_final={traj['r_final_m']/MPC_M:.4e} proper Mpc")
