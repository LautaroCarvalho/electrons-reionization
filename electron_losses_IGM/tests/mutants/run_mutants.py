"""Mutation run (C23): each mutant plants one realistic error in igm_losses and runs the suite; a mutant must be CAUGHT.

Run: python3 electron_losses_IGM/tests/mutants/run_mutants.py [name-substring]   (from the project root; ~15 min all)
The two slow excitation-table tests are deselected: no mutant touches the excitation table.
Control: with no mutant (name 'control') every test must pass.
"""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parents[1]                     # electron_losses_IGM/
Q = [f"tests/test_{x}.py" for x in ("bremsstrahlung", "constants", "cosmology_medium", "gaunt_ff", "integrate",
                                    "inverse_compton", "thresholds")]
ALL = ["tests/"]
MUTANTS = [   # name, module, old text, new text, test files
    ("control", "", "", "", Q),
    ("IC b: factor 4 -> 2", "inverse_compton", "return 4.0 * KIN.gamma(K_J)", "return 2.0 * KIN.gamma(K_J)", Q),
    ("IC b: T_CMB(z) -> T_CMB0", "inverse_compton", "F_kn(b_parameter(K_J, medium.T_CMB))", "F_kn(b_parameter(K_J, K.T_CMB0))", Q),
    ("adiabatic: p2c2/E -> E", "losses", "float(KIN.p2c2(K)) / float(KIN.total_energy(K))", "float(KIN.total_energy(K))", Q),
    ("distance: v -> c", "integrate", "[v, v * (1.0 + z)]", "[K_.c, K_.c * (1.0 + z)]", Q),
    ("thresholds: swap x_hi/x_lo", "thresholds", "x_hi, x_lo = photon_percentile(1.0 - tol), photon_percentile(tol)",
     "x_lo, x_hi = photon_percentile(1.0 - tol), photon_percentile(tol)", Q),
    ("thresholds: number -> energy norm.", "thresholds", "_TOTAL = 2.0 * float(mp.zeta(3))", "_TOTAL = float(mp.pi ** 4 / 15)", Q),
    ("brems: swap n_HI <-> n_p", "bremsstrahlung", "(n_HI * phi_neutral(T_eV) + n_p * phi_ep(T_eV))",
     "(n_p * phi_neutral(T_eV) + n_HI * phi_ep(T_eV))", Q),
    ("medium: T_CMB0 x 1.01", "medium", "return K.T_CMB0 * (1.0 + self.z)", "return 1.01 * K.T_CMB0 * (1.0 + self.z)", Q),
    ("ionization: forgets eV->J", "ionization", "return sigma(T) * (B_EV + mean_secondary_energy(T)) * K.e",
     "return sigma(T) * (B_EV + mean_secondary_energy(T))", ALL),
    ("excitation: n_HI -> n_e", "excitation", "return medium.n_HI * float(KIN.speed(K_J)) * stopping_cross_section",
     "return medium.n_e * float(KIN.speed(K_J)) * stopping_cross_section", ALL),
    ("constants: mu0 = 4 pi 1e-7", "constants", 'mu0 = _P["mu0"]', "mu0 = 4e-7 * math.pi", Q),
    ("brems: stale K&L table (x 1.001)", "bremsstrahlung", 'np.array(d["phi_rad"])', 'np.array(d["phi_rad"]) * 1.001', Q),
    ("thresholds: CDF integrand x^3", "thresholds", "y * y / math.expm1(y) if y > 0", "y ** 3 / math.expm1(y) / 2.701178 if y > 0", Q),
    ("coulomb: prefactor x 4pi", "coulomb", "n_e * K.e ** 4 / (4.0 * math.pi * K.eps0 ** 2", "n_e * K.e ** 4 / (1.0 * K.eps0 ** 2", ALL),
]


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, mod, old, new, files in MUTANTS:
        if sel not in name:
            continue
        env = dict(os.environ, MUT_MOD=mod, MUT_OLD=old, MUT_NEW=new,
                   PYTHONPATH=os.pathsep.join([str(HERE), str(PKG / "src")]))
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "mutplugin", "-p", "no:cacheprovider", "-W", "ignore", "--color=no",
               "--deselect", "tests/test_collisions.py::test_table_matches_direct_below_3keV",
               "--deselect", "tests/test_collisions.py::test_born_be_vs_stone_table"] + files
        r = subprocess.run(cmd, cwd=PKG, env=env, capture_output=True, text=True)
        summary = ([l for l in r.stdout.splitlines() if " passed" in l or " failed" in l] or [r.stdout[-200:]])[-1]
        failed = [l.split(" ")[1] for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))]
        verdict = ("PASS (control)" if not r.returncode else "CONTROL FAILED") if name == "control" else \
                  ("CAUGHT" if r.returncode else "NOT CAUGHT")
        print(f"{name:36s} {verdict:12s} | {summary} | {failed[:4]}", flush=True)


if __name__ == "__main__":
    main()
