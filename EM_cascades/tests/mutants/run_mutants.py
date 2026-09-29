"""Mutation run for EM_cascades (C23): each mutant plants one realistic error in em_cascades; the suite must catch it.

Run: python3 EM_cascades/tests/mutants/run_mutants.py [name-substring]   (from the project root; ~15 min)
The slow photon-convergence test is deselected except for the mutants that need it.
"""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parents[1]                       # EM_cascades/
MUTANTS = [   # name, module, old, new, needs slow test
    ("control", "", "", "", False),
    ("fractions: trapezoid 0.5 -> 0.6", "fractions", "np.cumsum(0.5 * (phi[p][1:]", "np.cumsum(0.6 * (phi[p][1:]", False),
    ("ic_spectrum: 2 pi -> pi", "ic_spectrum", "return 2.0 * math.pi * R0 ** 2", "return 1.0 * math.pi * R0 ** 2", False),
    ("ic_spectrum: Planck x/(e^x-1) -> x^2/(e^x-1)", "ic_spectrum", "return x / math.expm1(x) * kernel(q, G)",
     "return x * x / math.expm1(x) * kernel(q, G)", False),
    ("secondary: exponent p -> p/2", "secondary", "return 1.0 / (1.0 + (np.asarray(eps_eV, float) / EPS_BAR) ** P_EXP)",
     "return 1.0 / (1.0 + (np.asarray(eps_eV, float) / EPS_BAR) ** (P_EXP / 2))", False),
    ("yields: drop the primary's own ionization", "yields", "fC[k] = w * (1.0 + mC)", "fC[k] = w * mC", False),
    ("yields: photoelectron keeps E_gamma", "yields", "pe = (eg_J - K.R_H) / K.e", "pe = eg_J / K.e", False),
    ("photoionization: Mb -> 1e-24 m^2", "photoionization", "MB = 1.0e-22", "MB = 1.0e-24", False),
    ("photoionization: n_HI -> n_e in window", "photoionization", "math.log(m.n_HI * sigma(math.exp(lnE))",
     "math.log(m.n_e * sigma(math.exp(lnE))", False),
]


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, mod, old, new, slow in MUTANTS:
        if sel not in name:
            continue
        env = dict(os.environ, MUT_MOD=mod, MUT_OLD=old, MUT_NEW=new, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join([str(HERE), str(PKG / "src")]))
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "mutplugin_cas", "-p", "no:cacheprovider", "-W", "ignore",
               "--color=no", "tests"] + ([] if slow else ["--deselect", "tests/test_cascade.py::test_photon_grid_convergence"])
        r = subprocess.run(cmd, cwd=PKG, env=env, capture_output=True, text=True)
        summary = ([l for l in r.stdout.splitlines() if " passed" in l or " failed" in l] or [r.stdout[-200:]])[-1]
        failed = [l.split(" ")[1] for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))]
        verdict = ("PASS (control)" if not r.returncode else "CONTROL FAILED") if name == "control" else \
                  ("CAUGHT" if r.returncode else "NOT CAUGHT")
        print(f"{name:46s} {verdict:12s} | {summary} | {failed[:3]}", flush=True)


if __name__ == "__main__":
    main()
