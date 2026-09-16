#!/usr/bin/env python3
r"""
Run the whole figure set of this folder at a given initial redshift.

WHAT ACTUALLY CHANGES WITH z, AND WHERE IT ENTERS
-------------------------------------------------
igm_losses.py / yield_comparison.py   (electron propagation -- the yield)
    n_HI(z)  = N_HI_NORM ((1+z)/21)^3     target density, sets ionization +
                                          Coulomb + excitation rates
    T_CMB(z) = T_CMB_0 (1+z)              IC target spectrum
    U_CMB(z) ~ (1+z)^4                    IC cooling rate
    B(z)     = B0 (1+z)^2                 synchrotron cooling
    H(z)     from Planck 18                adiabatic loss AND the escape time
ionization_yield.py cosmology(z)      (route-1 cross-check, the loss+IC route channel)
    n_H, T_cmb, u_cmb, H(z)
photon_vs_electron.py
    n_H(z) proper, comoving->proper conversion, and the yields above.

WHAT DOES *NOT* SCALE, AND IS THEREFORE NOT INVENTED
----------------------------------------------------
    rho_UV : Donnan+24 (JWST PRIMER) Table 3 runs z = 9, 10, 11, 12.5, 14.5;
             the z=14.5 row holds M*, alpha and beta fixed rather than fitting
             them. There is no measurement at z = 20 -- 103 Myr of cosmic time
             beyond the last row -- and extrapolating a UV luminosity function
             that far would be a made-up number.
    xi_ion : Llerena+25 reaches z ~ 9.
    f_esc, eps_CR f_e : never measured at any redshift; already scanned.

Treatment (the user's decision, QUESTION_LOG round 7):
    figures with an ABSOLUTE vertical axis get a rho_SFR ignorance BAND,
    [1e-3, 1] x the z=10 value, labelled as ignorance and not as an error bar;
    the RATIO figure needs no anchor at all and is exact.
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import sys, json
import numpy as np
import igm_config as IC
import ionization_yield as IY
import igm_losses as L
import yield_comparison as YC
import ionization_yield_igm as IYI
import photon_vs_electron as PVE

BAND = (1.0e-3, 1.0)


def environment(z):
    cos = IY.cosmology(z)
    return {
        "z": float(z),
        "n_HI_cm3_igm_losses": float(L.n_HI(z)) * 1e-6,
        "n_H_cm3_route1": float(cos["n_H_cm3"]),
        "H_z_s": float(cos["H_z_s"]),
        "T_cmb_K": float(cos["T_cmb_K"]),
        "u_cmb_erg_cm3": float(cos["u_cmb_erg_cm3"]),
        "kT_cmb_eV": float(IY.K_B_EV * cos["T_cmb_K"]),
    }


def run(z, suffix, ratio_only=False):
    print("\n" + "#" * 78)
    print(f"#  z = {z:g}   suffix = {suffix!r}   "
          f"{'RATIO-ONLY' if ratio_only else 'full set'}")
    print("#" * 78)
    env = environment(z)
    for k, v in env.items():
        print(f"   {k:24s} {v:.6g}")

    YC.set_redshift(z)
    par = IY.Params(z=z)
    cos = IY.cosmology(z)
    chan = IY.ICPhotonChannel(cos, par).build()
    got = {"environment": env}

    with IC.fig_suffix(suffix):
        if not ratio_only:
            # 1. the yield figure, igm_losses prescription
            IYI.main(IC.IGMConfig(z=z))
            # 2. the two-route comparison
            YC.main(z=z)
        # 3-5. the competition figures (fig3 is the ratio-only one)
        PVE.main(z=z, rho_band=(None if ratio_only else BAND))
        got["loss_sat"] = float(YC.N_e_loss(np.array([1e12]))[0])
        got["E_sat"] = float(np.asarray(YC.N_e_loss_ic(np.array([1e12]), cos, par))[0])
        res = json.load(open(f"{IC.name_stem('photon_vs_electron_results')}.json"))
    got["zeta_gamma_s"] = res["derived"]["zeta_gamma_s"]
    got["zeta_e_s"] = res["derived"]["zeta_e_s"]
    got["zeta_ratio"] = res["derived"]["zeta_ratio"]
    got["eps_cr_fe_crossover"] = res["derived"]["eps_cr_fe_crossover"]
    got["checks_pass"] = sum(1 for c in res["checks"] if c["state"] == "PASS")
    got["checks_total"] = len(res["checks"])
    return got


def main():
    out = {}
    out["z20_full"] = run(20.0, "_z20")
    YC.set_redshift(10.0)
    with open("redshift_suite_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for tag, d in out.items():
        print(f"\n[{tag}]  z = {d['environment']['z']:g}")
        for k in ("loss_sat", "E_sat", "zeta_gamma_s", "zeta_e_s", "zeta_ratio",
                  "eps_cr_fe_crossover"):
            print(f"   {k:22s} {d[k]:.6g}")
        print(f"   checks                 {d['checks_pass']}/{d['checks_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
