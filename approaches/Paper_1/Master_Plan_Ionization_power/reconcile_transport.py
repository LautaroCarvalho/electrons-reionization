"""
Phase 1 -- reconciliation (Master Rule 5).

Cross-checks this project's cr_transport.py (proper path length r(t), built by
extending imported/cascade_yield.py's ODE) against the independently-built
imported/stage6_range.py pipeline's cached proper path length `ell` (same
neutral IGM, x_e=1e-4, same igm_losses.py physics, but a separate integration
with no cascade-recursion bookkeeping). Agreement here means the two
codebases this session/project produced independently are consistent; any
real discrepancy is reported explicitly, not silently picked over.
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import cr_transport as T

HERE = pathlib.Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
MPC_M = T.MPC_M

Z_TEST = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0])
K_TEST = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11])


def stage6_ell_interp():
    d = np.load(HERE / "imported" / "redo_range_table.npz")
    Kg, zg, ell = d["K"], d["z"], d["ell"]
    logK = np.log10(Kg)

    def f(K, z):
        iz = np.argmin(np.abs(zg - z))
        return np.interp(np.log10(K), logK, ell[iz])
    return f, zg


def main():
    ell_interp, zg_native = stage6_ell_interp()

    my_r = np.full((Z_TEST.size, K_TEST.size), np.nan)
    ref_ell = np.full((Z_TEST.size, K_TEST.size), np.nan)
    for iz, z in enumerate(Z_TEST):
        for ik, K in enumerate(K_TEST):
            traj = T.electron_trajectory(K, z)
            my_r[iz, ik] = traj["r_final_m"] / MPC_M
            ref_ell[iz, ik] = ell_interp(K, z)
        print(f"z={z:5.1f} done", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ref_ell > 0, np.abs(my_r - ref_ell) / ref_ell, np.nan)

    print(f"\n{'z':>5} {'K (eV)':>10} {'this session (Mpc)':>20} "
          f"{'stage6 ell (Mpc)':>18} {'rel. diff':>10}")
    for iz, z in enumerate(Z_TEST):
        for ik, K in enumerate(K_TEST):
            print(f"{z:5.1f} {K:10.1e} {my_r[iz,ik]:20.4e} "
                  f"{ref_ell[iz,ik]:18.4e} {rel[iz,ik]:10.3f}")

    finite = rel[np.isfinite(rel)]
    med = np.median(finite)
    worst = np.nanmax(finite)
    print(f"\nmedian relative difference = {med:.3f}   worst = {worst:.3f}")
    if worst < 0.30:
        verdict = ("AGREE within 30% at every tested node -- consistent with "
                   "the two codebases solving the same physics with independent "
                   "numerical implementations (different ODE state variables, "
                   "solver tolerances, and K/z grids).")
    else:
        verdict = ("DISCREPANCY: worst-case relative difference exceeds 30%. "
                   "Reported explicitly per Master Rule 5 -- not silently picked "
                   "over. See printed table above for which (K,z) node(s) diverge.")
    print(verdict)

    # ---- figure -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(Z_TEST)))
    for iz, z in enumerate(Z_TEST):
        ax.loglog(K_TEST, my_r[iz], "o-", color=colors[iz], label=f"z={z:g} (this work)")
        ax.loglog(K_TEST, ref_ell[iz], "x--", color=colors[iz])
    ax.set_xlabel(r"$K_{\rm ini}$ (eV)")
    ax.set_ylabel(r"proper path length $r$ (Mpc)")
    ax.set_title("Solid+circle: cr\\_transport.py (this work)\nDash+x: stage6\\_range.py (redo pipeline)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    ax2 = axes[1]
    for iz, z in enumerate(Z_TEST):
        ax2.semilogx(K_TEST, rel[iz] * 100, "o-", color=colors[iz], label=f"z={z:g}")
    ax2.axhline(30, color="r", ls=":", label="30% tolerance")
    ax2.set_xlabel(r"$K_{\rm ini}$ (eV)")
    ax2.set_ylabel(r"relative difference (\%)")
    ax2.set_title("Phase 1 reconciliation: relative agreement")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    outpath = FIGDIR / "phase1_transport_reconciliation.png"
    fig.savefig(outpath, dpi=200)
    print(f"Figure saved to: {outpath}")
    return med, worst


if __name__ == "__main__":
    main()
