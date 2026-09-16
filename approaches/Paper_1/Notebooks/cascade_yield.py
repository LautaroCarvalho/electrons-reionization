"""
cascade_yield.py -- ionization yield of a cosmic-ray electron INCLUDING the
secondary shower, with an energy-dependent secondary spectrum.

Why this exists
---------------
The single-particle ODE of ``igm_losses.py`` counts only the ionizing
collisions of the *primary* electron.  Every such collision ejects a secondary
of energy eps, and whenever eps > 13.6 eV that secondary ionizes in turn.  The
number of ionizations produced by an injected electron is therefore NOT
K_ini / (a constant): both the yield per generation and the secondary spectrum
that feeds the next generation depend on energy.

What it does
------------
1. Reconstructs the BEB singly-differential cross-section dsigma/dw from the
   dipole coefficients already stored in ``igm_losses._DIPOLE_FIT``
   (Kim & Rudd 1994; Kim, Santos & Parente 2000).  Integrating it reproduces
   the module's own ``coll_ionisation_cross_section`` to 0.01% below 300 eV
   and 2.6% at 10 keV, so the differential and total cross-sections are the
   same physics.  This gives a genuinely K-dependent secondary spectrum
   p(eps | K), replacing the fixed-shape fit used inside ``_build_loss_tables``.

2. Solves the cascade equation in the continuous-slowing-down approximation,

       Y(K,z) = int [sigma_ion/Lambda_tot] (1 + <Y(eps,z)>_{p(eps|K')}) dK' ,

   on a (K, z) grid.  Valid for the secondaries because an electron below
   ~1e5 eV thermalizes in Delta z << 1.

3. Integrates the *primary* along its real cosmological trajectory (full
   seven-channel ODE, so inverse Compton and the expansion are included) with
   an extra state

       dN_tot/dt = n_HI v sigma_ion(K) [ 1 + <Y(eps, z)>_{p(eps|K)} ] ,

   so the shower is attached at the redshift where each collision happens.

Run:  python3 cascade_yield.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, interp1d

import igm_losses as L

B = L.THRESHOLD_EV_ION       # 13.6057 eV
E_LYA = L.THRESHOLD_EV_EXC   # 10.204 eV -- below this nothing inelastic happens
EV = L.EV_MKS

# The BEB differential cross-section now lives in igm_losses (switch
# FIX_SECONDARY_SPECTRUM); re-export it so the cascade and the loss model use
# one and the same secondary spectrum.
secondary_pdf = L.secondary_pdf
beb_sdcs = L.beb_sdcs


def mean_secondary_beb(K_eV):
    return L._mean_secondary_energy_beb(K_eV)


Z_NODES = np.array([5.5, 7.0, 8.5, 10.0, 12.0, 14.0, 17.0, 20.0])
K_NODES = np.logspace(np.log10(E_LYA), 6.0, 700)
CACHE = Path(__file__).resolve().parent / ".cascade_yield_cache.npz"
_NI = L._BEB_NI


# =====================================================================
# 2.  Stopping powers of the seven channels, per unit H column density
# =====================================================================
def stopping(z, Kg):
    """Lambda_i(K) [eV m^2] at redshift z, plus sigma_ion [m^2]."""
    H = float(L.Planck18.H(z).to(L.u.s ** -1).value)
    nH = L.n_HI(z)
    _, _, _, v = L.kinematics(Kg * EV)
    st = L.loss_rates(z, Kg * EV, H, L.toggles())
    sig = np.array([L.coll_ionisation_cross_section(k * EV) for k in Kg])
    Lam_ion = L.loss_interp_ion(Kg)
    Lam_exc = L.loss_interp_exc(Kg)
    Lam_oth = (st[0] + st[1] + st[2] + st[3] + st[6]) / (nH * v) / EV
    return sig, Lam_ion + Lam_exc + Lam_oth


# =====================================================================
# 3.  Cascade equation, solved by upward marching in K
# =====================================================================
def solve_Y(z, Kg=K_NODES):
    """Total ionizations Y(K) produced by an electron of energy K and all its
    progeny, in the continuous-slowing-down approximation at redshift z."""
    sig, Lam_tot = stopping(z, Kg)
    r = np.where(Lam_tot > 0, sig / Lam_tot, 0.0)     # ionizations per eV lost
    Y = np.zeros_like(Kg)
    pdf_cache = {}
    for i in range(1, len(Kg)):
        dK = Kg[i] - Kg[i - 1]
        K_mid = 0.5 * (Kg[i] + Kg[i - 1])
        if i not in pdf_cache:
            pdf_cache[i] = secondary_pdf(K_mid)
        e, p = pdf_cache[i]
        if e is None:
            Ys = 0.0
        else:
            Yi = np.interp(e, Kg[:i], Y[:i], left=0.0)
            Yi[e < B] = 0.0                # such a secondary cannot ionize
            Ys = float(np.trapz(p * Yi, e))
        Y[i] = Y[i - 1] + dK * 0.5 * (r[i] + r[i - 1]) * (1.0 + Ys)
        if Kg[i] < B:
            Y[i] = 0.0
    return Y


def build_table(force=False):
    """Y(K, z) on the (K_NODES, Z_NODES) grid, cached on disk."""
    if CACHE.exists() and not force:
        d = np.load(CACHE)
        if d["K"].shape == K_NODES.shape and np.allclose(d["K"], K_NODES) \
           and np.allclose(d["z"], Z_NODES):
            return d["Y"]
    Y = np.array([solve_Y(z) for z in Z_NODES])       # (n_z, n_K)
    np.savez(CACHE, K=K_NODES, z=Z_NODES, Y=Y)
    return Y


_Y_TABLE = None


def _interp():
    global _Y_TABLE
    if _Y_TABLE is None:
        Y = build_table()
        _Y_TABLE = RegularGridInterpolator(
            (Z_NODES, np.log(K_NODES)), Y,
            bounds_error=False, fill_value=None)
    return _Y_TABLE


def mean_shower_yield(K_eV, z):
    """<Y(eps, z)> averaged over the secondary spectrum of a primary at K."""
    e, p = secondary_pdf(K_eV)
    if e is None:
        return 0.0
    m = e >= B
    if not m.any():
        return 0.0
    zc = np.clip(z, Z_NODES[0], Z_NODES[-1])
    lk = np.clip(np.log(e[m]), np.log(K_NODES[0]), np.log(K_NODES[-1]))
    Yv = _interp()(np.column_stack([np.full(lk.size, zc), lk]))
    return float(np.trapz(p[m] * np.maximum(Yv, 0.0), e[m]))


# --- tabulate <Y_sec>(K, z) once so the ODE right-hand side stays cheap ---
_MS_K = np.logspace(np.log10(B * 1.02), 13.0, 260)
_MS = None


def _mean_shower_table():
    global _MS
    if _MS is None:
        grid = np.array([[mean_shower_yield(k, z) for k in _MS_K]
                         for z in Z_NODES])
        _MS = RegularGridInterpolator((Z_NODES, np.log(_MS_K)), grid,
                                      bounds_error=False, fill_value=None)
    return _MS


def mean_shower_fast(K_eV, z):
    f = _mean_shower_table()
    zc = float(np.clip(z, Z_NODES[0], Z_NODES[-1]))
    lk = float(np.clip(np.log(max(K_eV, B * 1.02)),
                       np.log(_MS_K[0]), np.log(_MS_K[-1])))
    return max(float(f([[zc, lk]])[0]), 0.0)


# =====================================================================
# 4.  Primary trajectory with the shower attached
# =====================================================================
_SIG_TAB = np.array([L.coll_ionisation_cross_section(k * EV)
                     for k in L.K_LOSS_GRID_EV])
_LG = np.log(np.clip(_SIG_TAB, 1e-300, None))
_SIG_F = interp1d(np.log(L.K_LOSS_GRID_EV), _LG, kind="linear",
                  bounds_error=False, fill_value=(-np.inf, _LG[-1]))


def sigma_ion(K_eV):
    K_eV = np.asarray(K_eV, dtype=float)
    return np.where(K_eV > B,
                    np.exp(_SIG_F(np.log(np.clip(K_eV, 1e-300, None)))), 0.0)


def yields(K_ini_eV, z_i, z_f=5.5):
    """(N_primary, N_total, z_thermal) for one injected electron."""
    t_i, t_f = L.age_s(z_i), L.age_s(z_f)

    def rhs(t, y):
        K = max(float(y[0]), 0.0)
        if K <= 0.0:
            return [0.0, 0.0, 0.0]
        z = float(L.z_interp(t))
        tot = float(L.total_loss(z, K, float(L.H_interp(t)), L.toggles()))
        _, _, _, v = L.kinematics(K)
        rate = L.n_HI(z) * float(v) * float(sigma_ion(K / EV))
        return [-tot, rate, rate * (1.0 + mean_shower_fast(K / EV, z))]

    def ev(t, y):
        return y[0] - float(L.thermal_floor(L.z_interp(t)))
    ev.terminal, ev.direction = True, -1

    sol = solve_ivp(rhs, (t_i, t_f), [K_ini_eV * EV, 0.0, 0.0], events=ev,
                    method="Radau", rtol=1e-7, atol=1e-25, dense_output=True)
    return (float(sol.y[1][-1]), float(sol.y[2][-1]),
            float(L.z_interp(sol.t[-1])))


# =====================================================================
def main():
    print("Secondary spectrum: BEB SDCS from igm_losses._DIPOLE_FIT "
          f"(N_i = {_NI:.5f})\n")
    print(f"{'K [eV]':>9} {'<eps>_BEB':>10} {'<eps>_code':>11} {'ratio':>7}")
    for k in (20, 50, 100, 300, 1e3, 1e4, 1e5):
        mb = mean_secondary_beb(k); mc = L._mean_secondary_energy(k)
        print(f"{k:9.4g} {mb:10.3f} {mc:11.3f} {mb/max(mc,1e-30):7.3f}")

    build_table()
    print("\nYield along the real trajectory (primary vs primary+shower):")
    print(f"{'z_i':>5} {'K_ini [eV]':>11} {'N_prim':>10} {'N_total':>10} "
          f"{'mult':>6} {'W_tot [eV]':>11} {'z_therm':>8}")
    for z_i in (20.0, 10.0):
        for K0 in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12):
            n1, nt, zt = yields(K0, z_i)
            print(f"{z_i:5.1f} {K0:11.1e} {n1:10.4g} {nt:10.4g} "
                  f"{nt/max(n1,1e-30):6.3f} {K0/max(nt,1e-30):11.4g} {zt:8.3f}")


if __name__ == "__main__":
    main()
