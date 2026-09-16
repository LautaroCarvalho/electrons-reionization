"""
emis_engine_A.py -- ENGINE A: this project's own transport cascade, with the
                    H I / He I split derived here.

ELECTRON PRIMARIES
------------------
The TOTAL number of ionizations produced by an electron of kinetic energy K
injected at redshift z_i, counting every ionization down the cascade until
z = 5.5, is taken from this project's existing table
``Notebooks/redo/redo_cascade_table.npz`` (column Y + Ype), produced by
``stage2_cascade.py``.  That calculation carries the full transport: Coulomb
losses (Gould 1972), collisional excitation (Stone & Kim 2002), RBEB
collisional ionization (Kim et al. 2000), inverse Compton and bremsstrahlung
(Blumenthal & Gould 1970), and the re-absorption of the electron's own
radiated photons.  It is HYDROGEN-ONLY.

The species split is derived here, not assumed.  In the continuous-slowing-
down approximation the number of ionizations of species i produced while the
electron degrades from E_0 to threshold is

    N_i(E_0) = Int_{B_i}^{E_0}  n_i sigma_i(E) / Lambda(E)  dE                (A1)

with Lambda(E) the total energy-loss rate per unit path.  Lambda is common to
both species, so the SPLIT

    f_H(E_0) = N_HI / (N_HI + N_HeI)                                          (A2)

is a ratio of two integrals over the same Lambda and is insensitive to its
absolute normalisation.  The project's table supplies the total; Eq. (A2)
supplies the split:

    N_HI  = N_total * f_H ,        N_HeI = N_total * (1 - f_H).               (A3)

CROSS-SECTIONS FOR THE SPLIT
----------------------------
Electron-impact ionization from the Binary-Encounter-Bethe (BEB) model,
Kim & Rudd (1994), Phys. Rev. A 50, 3954, Eq. (9) -- the same model already
used for hydrogen in this project's ``igm_losses.py``:

    sigma(T) = S/(t+u+1) [ (ln t)/2 (1 - 1/t^2) + 1 - 1/t - (ln t)/(t+1) ]
    t = T/B,  u = U/B,  S = 4 pi a_0^2 N (R/B)^2

B is the binding energy, N the number of electrons in the orbital, U the mean
orbital kinetic energy and R the Rydberg.  B and N are taken from NIST; U is
NOT taken on faith from a table -- it is DERIVED here from the virial theorem
for a Coulomb-bound system, <T> = -E_total:

    H I  : E_total = -13.598 eV                  -> U = 13.598 eV
    He I : E_total = -(24.587 + 54.418) eV       -> U = 79.005/2 = 39.50 eV

The helium value reproduces the 39.51 eV tabulated by Kim & Rudd (1994) to
0.03 per cent, which is an independent check that the derivation is the right
one (see emis_verify.py, block C).

PHOTON PRIMARIES
----------------
A photon of energy E is absorbed by species j with branching ratio
n_j sigma_j(E) / sum_k n_k sigma_k(E) (Verner et al. 1996 cross-sections,
emis_common.sigma_pi), producing ONE ionization of species j plus a
photoelectron of energy E - E_th,j whose own cascade is then counted with the
electron machinery above.  Compton scattering is included through this
project's ``redo_photon_yield_table.npz`` where that table is used.

    N_j^gamma(E) = P_abs,j(E) [ 1 + N_j^e(E - E_th,j) ]
                 + sum_{k != j} P_abs,k(E) N_j^e(E - E_th,k)                  (A4)

HELIUM IS FULLY NEUTRAL by instruction, so n_HeI = n_He throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from emis_common import (E_TH, N_HE_OVER_N_H, sigma_pi)

HERE = Path(__file__).resolve().parent
REDO = HERE.parent.parent / "Notebooks" / "redo"
CASCADE_NPZ = REDO / "redo_cascade_table.npz"
PHOTON_NPZ = REDO / "redo_photon_yield_table.npz"

A0_CM = 5.29177210903e-9        # Bohr radius [cm], CODATA 2018
RYDBERG_EV = 13.605693122994    # Rydberg energy [eV], CODATA 2018

# --- BEB parameters ---------------------------------------------------------
# B from NIST (emis_common.E_TH); N exact; U from the virial theorem.
BEB = {
    "HI":  {"B": E_TH["HI"],  "N": 1, "U": E_TH["HI"]},
    "HeI": {"B": E_TH["HeI"], "N": 2, "U": (E_TH["HeI"] + E_TH["HeII"]) / 2.0},
}


def sigma_beb(T_eV, species):
    """BEB electron-impact ionization cross-section [cm^2].
    Kim & Rudd (1994) Eq. (9)."""
    p = BEB[species]
    B, N, U = p["B"], p["N"], p["U"]
    T = np.atleast_1d(np.asarray(T_eV, float))
    out = np.zeros_like(T)
    m = T > B
    if m.any():
        t = T[m] / B
        u = U / B
        S = 4.0 * np.pi * A0_CM ** 2 * N * (RYDBERG_EV / B) ** 2
        out[m] = (S / (t + u + 1.0)) * (0.5 * np.log(t) * (1.0 - 1.0 / t ** 2)
                                        + 1.0 - 1.0 / t
                                        - np.log(t) / (t + 1.0))
    out = np.maximum(out, 0.0)
    return out if np.ndim(T_eV) else out[0]


# --- A2: the H I / He I split ----------------------------------------------
_ESPLIT = np.geomspace(E_TH["HI"] * (1 + 1e-9), 1.0e9, 4000)


def _split_tables():
    """Cumulative CSDA integrals of Eq. (A1) with a common Lambda.

    Lambda(E) is built from the BEB ionization stopping power of the mixture,
    Lambda(E) = sum_i n_i B_i sigma_i(E) (energy lost per ionization ~ B_i is
    the leading term).  Only the SHAPE of Lambda enters the ratio, and the
    ratio is checked for sensitivity to that shape in emis_verify.py.
    """
    E = _ESPLIT
    s_H = sigma_beb(E, "HI")
    s_He = sigma_beb(E, "HeI")
    n_H, n_He = 1.0, N_HE_OVER_N_H
    lam = n_H * E_TH["HI"] * s_H + n_He * E_TH["HeI"] * s_He
    lam = np.maximum(lam, 1e-300)
    iH = cumulative_trapezoid(n_H * s_H / lam, E, initial=0.0)
    iHe = cumulative_trapezoid(n_He * s_He / lam, E, initial=0.0)
    return E, iH, iHe


_E_S, _I_H, _I_HE = _split_tables()


def f_species(E_eV, species):
    """Fraction of the cascade's ionizations that are of `species`.  Eq. (A2)."""
    E = np.clip(np.asarray(E_eV, float), _E_S[0], _E_S[-1])
    a = np.interp(E, _E_S, _I_H)
    b = np.interp(E, _E_S, _I_HE)
    tot = np.maximum(a + b, 1e-300)
    return (a if species == "HI" else b) / tot


# --- A3: electron yields ----------------------------------------------------
_d = np.load(CASCADE_NPZ)
_N0 = int(_d["nlow"])
K_GRID = _d["K"][_N0:]
Z_GRID_E = _d["z"]
_Y_TOT = np.maximum(_d["Y"][:, _N0:] + _d["Ype"][:, _N0:], 0.0)


def nion_electron_total(K_eV, z):
    """Total ionizations per primary electron, this project's cascade."""
    iz = int(np.argmin(np.abs(Z_GRID_E - z)))
    Y = np.maximum(_Y_TOT[iz], 1e-300)
    return np.exp(np.interp(np.log(np.asarray(K_eV, float)),
                            np.log(K_GRID), np.log(Y)))


def nion_electron(K_eV, z, species):
    """Ionizations of `species` per primary electron.  Eq. (A3)."""
    return nion_electron_total(K_eV, z) * f_species(K_eV, species)


# --- A4: photon yields ------------------------------------------------------
def p_absorb(E_eV, species):
    """Branching ratio of a bound-free absorption onto `species`.
    Helium fully neutral."""
    sH = sigma_pi(E_eV, "HI")
    sHe = N_HE_OVER_N_H * sigma_pi(E_eV, "HeI")
    tot = sH + sHe
    num = sH if species == "HI" else sHe
    return np.where(tot > 0.0, num / np.maximum(tot, 1e-300), 0.0)


def nion_photon(E_eV, z, species):
    """Ionizations of `species` per primary photon, per Eq. (A4).

    The first term is the photoionization event itself; both terms add the
    photoelectron's own cascade.  This is the LOCAL yield -- it assumes the
    photon is absorbed.  The transport-weighted version, which includes the
    probability that the photon leaves the reionization window without
    interacting, comes from redo_photon_yield_table.npz in emis_sources.py.
    """
    E = np.asarray(E_eV, float)
    out = np.zeros_like(E)
    for k in ("HI", "HeI"):
        Pk = p_absorb(E, k)
        Ke = np.maximum(E - E_TH[k], 0.0)
        casc = np.where(Ke > 0.0, nion_electron(np.maximum(Ke, 1e-3), z, species), 0.0)
        out = out + Pk * (casc + (1.0 if k == species else 0.0))
    return np.where(E >= E_TH[species], out, 0.0)


if __name__ == "__main__":
    print("BEB parameters (B, N from NIST; U from the virial theorem)")
    for k, p in BEB.items():
        print("   %-5s B = %8.4f eV   N = %d   U = %8.4f eV" % (k, p["B"], p["N"], p["U"]))
    print("   Kim & Rudd (1994) tabulate U(He I) = 39.51 eV -> derived %.4f eV,"
          " %.3f %% apart" % (BEB["HeI"]["U"], 100 * abs(BEB["HeI"]["U"] / 39.51 - 1)))
    print("\nBEB cross-sections [cm^2] and the resulting split")
    print("   %-10s %-13s %-13s %-10s %-10s" % ("E [eV]", "sig_BEB(H I)",
                                                "sig_BEB(He I)", "f_HI", "f_HeI"))
    for E in (30.0, 50.0, 1e2, 1e3, 1e4, 1e6, 1e8):
        print("   %-10.4g %-13.4e %-13.4e %-10.5f %-10.5f"
              % (E, sigma_beb(E, "HI"), sigma_beb(E, "HeI"),
                 f_species(E, "HI"), f_species(E, "HeI")))
    print("\nEngine A electron yields (z = 10)")
    print("   %-10s %-14s %-14s %-14s" % ("K [eV]", "N_tot", "N(H I)", "N(He I)"))
    for K in (1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10):
        print("   %-10.3g %-14.5g %-14.5g %-14.5g"
              % (K, nion_electron_total(K, 10.0),
                 nion_electron(K, 10.0, "HI"), nion_electron(K, 10.0, "HeI")))
    print("\nEngine A photon yields, local (z = 10)")
    print("   %-10s %-10s %-10s %-14s %-14s"
          % ("E [eV]", "P_abs H I", "P_abs He I", "N_gam(H I)", "N_gam(He I)"))
    for E in (14.0, 30.0, 60.0, 1e2, 3e2, 1e3, 1e4):
        print("   %-10.4g %-10.5f %-10.5f %-14.5g %-14.5g"
              % (E, p_absorb(E, "HI"), p_absorb(E, "HeI"),
                 nion_photon(E, 10.0, "HI"), nion_photon(E, 10.0, "HeI")))
