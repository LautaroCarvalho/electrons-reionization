"""Physical constants: primary values read once from provenance/parameters.yaml, derived ones computed here.

Purpose
    Single place where constants enter the code. No module defines a number: primaries come from the
    parameters registry (with their sources: SI 2019 Brochure, CODATA 2022, PDG 2024), and every derived
    constant below is a formula of those primaries (listed under `derived_in_code` in parameters.yaml).

Units
    SI throughout. Energies in J unless the name ends in _eV.

Checks (tests/test_constants.py)
    α, r_e, σ_T, a_0 and Ry∞ are compared with the CODATA 2022 recommended values printed in
    references/Mohr (2024) CODATA 2022.pdf (Table XXXIII), which are independent of this derivation.
"""

import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
PARAMETERS_FILE = ROOT / "provenance" / "parameters.yaml"


def _load_primaries(path=PARAMETERS_FILE):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"]
    return {k: v["value"] for k, v in data.items()}


_P = _load_primaries()

# ---- primaries [SI2019 §2.2 Table 1; CODATA2022 Table XXXIII; PDG2024 Table 2.1] -----------------------
c = _P["c"]                 # m s^-1, exact
h = _P["h"]                 # J s, exact
e = _P["e"]                 # C, exact
k_B = _P["k_B"]             # J K^-1, exact (E01)
m_e = _P["m_e"]             # kg
m_p = _P["m_p"]             # kg
mu0 = _P["mu0"]             # N A^-2
year = _P["year"]           # s, Julian year (E03)
parsec = _P["parsec"]       # m
T_CMB0 = _P["T_CMB0"]       # K (astropy Planck18 value, decision 2026-09-27)

# ---- derived (formulas in parameters.yaml: derived_in_code) ---------------------------------------------
eps0 = 1.0 / (mu0 * c ** 2)                              # F m^-1
hbar = h / (2.0 * math.pi)                               # J s
m_e_c2 = m_e * c ** 2                                    # J
m_e_c2_eV = m_e_c2 / e                                   # eV
alpha = e ** 2 / (2.0 * eps0 * h * c)                    # fine-structure constant
r_e = e ** 2 / (4.0 * math.pi * eps0 * m_e_c2)           # m, classical electron radius
sigma_T = 8.0 * math.pi / 3.0 * r_e ** 2                 # m^2
a0 = hbar / (m_e * c * alpha)                            # m, Bohr radius (infinite nuclear mass)
Ry_inf = alpha ** 2 * m_e_c2 / 2.0                       # J
Ry_inf_eV = Ry_inf / e                                   # eV
# [A12] reduced-mass Rydberg: ionization threshold
R_H = Ry_inf * m_p / (m_e + m_p)                         # J, hydrogen with reduced mass (A12)
R_H_eV = R_H / e
a_rad = 8.0 * math.pi ** 5 * k_B ** 4 / (15.0 * h ** 3 * c ** 3)   # J m^-3 K^-4 (E04)
U_CMB0 = a_rad * T_CMB0 ** 4                             # J m^-3
alpha_re2 = alpha * r_e ** 2                             # m^2, bremsstrahlung unit (B&G 1970)
