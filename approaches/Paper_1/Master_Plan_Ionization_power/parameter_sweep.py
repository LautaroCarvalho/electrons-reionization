"""
Parameter sensitivity sweep: z_inject=10 (vs. the baseline z_form=z_inject=20
already in the document), crossed with the DSA injection index
p in {1.8, 2.0, 2.2} (harder, canonical, softer than the p=2.0 fiducial).
K_min, K_max, xi_CR,e, SFR, and every other parameter are held fixed at their
params.py baseline values -- only z_inject and p are overridden, using the
params-override architecture (no baseline value in params.py is edited).

Phase 2: per-electron trajectories do not depend on p or on z_form (only on
the injection redshift z_i, which is the loop/snapshot z itself -- see
cr_modified_stromgren_radius.solve_R_CR_Mpc's docstring), so trajectories are
built ONCE per snapshot z and reused across all three p values.

Phase 3: the Y(K,z) cascade-yield table is independent of both p and the
z_inject cutoff (pure per-electron microphysics), so the already-cached
Y_table_phase3.npz is reused rather than rebuilt.

Results are saved to phase2_sweep_zinject10.npz and phase3_sweep_zinject10.npz
-- new files, tagged by z_inject=10, that do NOT overwrite the existing
phase2_results.npz / phase3_results.npz baseline (z_form=20, p=2.0) caches.
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import params as P
import stromgren_common as SC
import cr_modified_stromgren_radius as M2
import cr_modified_reionization_history as M3

HERE = pathlib.Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

Z_FORM_SWEEP = 10.0
P_VALUES = [1.8, 2.0, 2.2]
Z_ARR_SWEEP = np.array([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])


def run_phase2_sweep():
    print("=" * 78)
    print(f"Phase 2 sweep: z_inject={Z_FORM_SWEEP}, p in {P_VALUES}")
    print("=" * 78)
    R_UV = SC.R_UV_proper_Mpc(Z_ARR_SWEEP, z_form=Z_FORM_SWEEP)
    R_CR = {p: np.full_like(Z_ARR_SWEEP, np.nan) for p in P_VALUES}

    for i, z in enumerate(Z_ARR_SWEEP):
        trajs = M2.build_trajectories(z)   # built once, reused for all p
        for p in P_VALUES:
            R_CR[p][i] = M2.solve_R_CR_Mpc(z, P.XI_CR_E_FID, trajs=trajs,
                                            p=p, z_form=Z_FORM_SWEEP)
        print(f"z={z:4.1f}  R_UV={R_UV[i]:.4e} Mpc  "
              + "  ".join(f"R_CR(p={p})={R_CR[p][i]:.4e}" for p in P_VALUES),
              flush=True)

    ratio = {p: R_CR[p] / R_UV for p in P_VALUES}
    print(f"\n{'z':>5} {'R_UV(Mpc)':>11} "
          + " ".join(f"{'R_CR/R_UV,p='+str(p):>16}" for p in P_VALUES))
    for i, z in enumerate(Z_ARR_SWEEP):
        print(f"{z:5.1f} {R_UV[i]:11.4e} "
              + " ".join(f"{ratio[p][i]:16.4e}" for p in P_VALUES))

    out = HERE / "phase2_sweep_zinject10.npz"
    np.savez(out, z=Z_ARR_SWEEP, z_form=Z_FORM_SWEEP, p_values=np.array(P_VALUES),
             R_UV=R_UV, **{f"R_CR_p{p}": R_CR[p] for p in P_VALUES},
             **{f"ratio_p{p}": ratio[p] for p in P_VALUES})
    print(f"Saved: {out}")

    # figure: R_CR/R_UV(z) for the 3 p values, plus the baseline (z_form=20,
    # p=2.0) ratio curve from the already-saved phase2_results.npz for direct
    # comparison
    base = np.load(HERE / "phase2_results.npz")
    base_ratio = base["R_CR_fid"] / base["R_UV"]

    fig, ax = plt.subplots(figsize=(6.8, 5))
    colors = {1.8: "C0", 2.0: "C1", 2.2: "C2"}
    for p in P_VALUES:
        ax.plot(Z_ARR_SWEEP, ratio[p] * 100, "o-", color=colors[p],
                 label=fr"$z_{{\rm inject}}=10$, $p={p}$")
    ax.plot(base["z"], base_ratio * 100, "k^--",
             label=r"baseline: $z_{\rm inject}=20$, $p=2.0$")
    ax.set_xlabel("redshift $z$ (bubble snapshot)")
    ax.set_ylabel(r"$R_{\rm CR}/R_{\rm UV}$ (\%)")
    ax.set_title("Parameter sensitivity: injection redshift and spectral index")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "sweep_RCR_over_RUV.png", dpi=200)
    print(f"Figure saved to: {FIGDIR/'sweep_RCR_over_RUV.png'}")

    return R_UV, R_CR, ratio


def run_phase3_sweep():
    print("\n" + "=" * 78)
    print(f"Phase 3 sweep: z_inject_cutoff={Z_FORM_SWEEP}, p in {P_VALUES}")
    print("=" * 78)
    Y_path = HERE / "Y_table_phase3.npz"
    d = np.load(Y_path)
    assert np.allclose(d["z"], M3.Z_TAB) and np.allclose(d["K"], M3.K_TAB_EV), \
        "cached Y_table grid does not match cr_modified_reionization_history's " \
        "current Z_TAB/K_TAB_EV -- would silently mix incompatible grids"
    Y_table = d["Y"]
    print(f"Reusing cached cascade-yield table: {Y_path}")

    z_base, Q_base, _ = M3.run_history(0.0, Y_table, include_cr=False)
    tau_base = M3.tau_of_z(z_base, Q_base)

    results = {}
    for p in P_VALUES:
        z_p, Q_p, _ = M3.run_history(P.XI_CR_E_FID, Y_table, include_cr=True,
                                      p=p, z_inject_cutoff=Z_FORM_SWEEP)
        tau_p = M3.tau_of_z(z_p, Q_p)
        results[p] = (z_p, Q_p, tau_p)
        z_half = M3.first_cross(z_p, Q_p, 0.5)
        z_full = M3.first_cross(z_p, Q_p, 0.999)
        dtau = tau_p[0] - tau_base[0]
        print(f"p={p:4.1f}  z(Q=0.5)={z_half:.3f}  z(Q=0.999)={z_full:.3f}  "
              f"tau(z<=25)={tau_p[0]+0.020:.5f}  delta_tau={dtau:.3e}")

    out = HERE / "phase3_sweep_zinject10.npz"
    save_kwargs = dict(z=z_base, z_inject_cutoff=Z_FORM_SWEEP,
                        p_values=np.array(P_VALUES), Q_base=Q_base, tau_base=tau_base)
    for p in P_VALUES:
        z_p, Q_p, tau_p = results[p]
        save_kwargs[f"Q_p{p}"] = Q_p
        save_kwargs[f"tau_p{p}"] = tau_p
    np.savez(out, **save_kwargs)
    print(f"Saved: {out}")

    # figure: delta_tau vs p, at z_inject=10, compared to the baseline
    # (z_form=20, p=2.0) delta_tau already in the document
    base3 = np.load(HERE / "phase3_results.npz")
    base_dtau = base3["tau_fid"][0] - base3["tau_base"][0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    dtaus = [results[p][2][0] - tau_base[0] for p in P_VALUES]
    ax.plot(P_VALUES, dtaus, "o-", color="C3", label=r"$z_{\rm inject}=10$")
    ax.axhline(base_dtau, color="k", ls="--", label=r"baseline: $z_{\rm inject}=20$, $p=2.0$")
    ax.set_xlabel("DSA injection index $p$")
    ax.set_ylabel(r"$\delta\tau$ vs.\ no-CR baseline")
    ax.set_title(r"$\delta\tau$ sensitivity to $p$ at $z_{\rm inject}=10$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "sweep_deltatau_vs_p.png", dpi=200)
    print(f"Figure saved to: {FIGDIR/'sweep_deltatau_vs_p.png'}")

    return tau_base, results, base_dtau


if __name__ == "__main__":
    run_phase2_sweep()
    run_phase3_sweep()
