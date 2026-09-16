# Copied from Notebooks/photon_cascade.py on 2026-09-06 (Master_Plan_Ionization_power provenance copy; edit only this copy)

"""Ionization cascade including productive inverse-Compton photons.

The electron loss trajectory is the seven-channel model in ``igm_losses``.
In addition to electron-impact ionizations, the augmented system counts CMB
photons inverse-Compton scattered into an absorbed band.  Each photon in that
band makes one H photoionization and launches a photoelectron with
K = E_gamma - 13.6057 eV.  The latter is propagated with the same low-energy
electron cascade used for collisional secondaries.

Two band definitions are tabulated:

* ``fixed``: 13.6057 eV <= E_gamma <= 1 keV;
* ``mfp``: the upper edge is where the photon mean free path equals c/H(z),
  using the opacity model already used by ``make_paper_figures.py``.

The output separates electron-impact ionizations (``Ye``) from the additional
photon-seeded branch (``Yph``).  Run with ``python photon_cascade.py``.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

import igm_losses as L
from cascade_yield import secondary_pdf, sigma_ion, build_table, K_NODES, Z_NODES


EV = L.EV_MKS
B = L.THRESHOLD_EV_ION
E0_EV = L.E0 / EV
OUT = Path(__file__).resolve().parent / "photon_cascade_table.npz"

K_GRID = np.concatenate([
    np.geomspace(B * 1.02, 1.0e3, 22),
    np.geomspace(1.3e3, 1.0e13, 42),
])
Z_GRID = np.array([5.5, 7.0, 8.5, 10.0, 12.0, 14.0, 17.0, 20.0])
Z_FINAL = 5.5

# Fixed quadrature for the Planck seed spectrum, x = epsilon/(kT).
_NX = 128
_gx, _gw = np.polynomial.legendre.leggauss(_NX)
_X = 0.5 * (_gx + 1.0) * 60.0
_XW = 0.5 * _gw * 60.0
_PLANCK_NUMBER_W = _XW * _X / np.expm1(_X)

# The photon-rate interpolation only needs to resolve the IC-producing range.
_KR = np.geomspace(1.0e5, 1.0e13, 180)


def ic_dnde(K_eV, z, Eg_eV):
    """IC photon production rate dN/(dt dE_gamma) [s^-1 eV^-1].

    This is the isotropic Klein--Nishina kernel of Blumenthal & Gould (1970),
    integrated over a blackbody CMB.  The kernel is used only for productive
    photons, whose parent electrons are relativistic.
    """
    K_eV = float(K_eV)
    Eg = np.atleast_1d(np.asarray(Eg_eV, dtype=float))
    gamma = 1.0 + K_eV / E0_EV
    kT = L.K_B * L.T_CMB_0 * (1.0 + float(z))
    eps = _X * kT
    Gamma = 4.0 * gamma * eps / L.E0
    EgJ = Eg[:, None] * EV
    denom = Gamma[None, :] * (gamma * L.E0 - EgJ)
    q = np.divide(EgJ, denom, out=np.full_like(denom, np.inf), where=denom > 0)
    valid = (q >= 1.0 / (4.0 * gamma**2)) & (q <= 1.0)
    Gq = Gamma[None, :] * q
    qsafe = np.clip(q, 1.0e-300, None)
    kernel = (2.0 * qsafe * np.log(qsafe)
              + (1.0 + 2.0 * qsafe) * (1.0 - qsafe)
              + (Gq**2 * (1.0 - qsafe)) / (2.0 * (1.0 + Gq)))
    kernel = np.where(valid, np.maximum(kernel, 0.0), 0.0)

    # Integral n(epsilon)/epsilon d epsilon after x=epsilon/kT.
    planck_pref = 8.0 * np.pi * kT**2 / (L.PLANCK_CONSTANT_MKS**3 * L.C_LIGHT**3)
    integ = kernel @ _PLANCK_NUMBER_W
    pref = 3.0 * L.THOMSON_CROSS_SECTION_MKS * L.C_LIGHT / (4.0 * gamma**2)
    return pref * planck_pref * integ * EV


def _local_electron_yield():
    """Interpolator for the rapidly cooling photoelectron cascade."""
    Y = build_table()
    return RegularGridInterpolator(
        (Z_NODES, np.log(K_NODES)), Y, bounds_error=False, fill_value=None
    )


def photon_rates(K_eV, z, emax_eV, yloc):
    """Return (absorbed photon rate, photon+photoelectron ionization rate)."""
    if emax_eV <= B:
        return 0.0, 0.0
    Eg = np.geomspace(B * (1.0 + 1.0e-8), float(emax_eV), 180)
    dnde = ic_dnde(K_eV, z, Eg)
    R0 = float(np.trapz(dnde, Eg))
    Ke = Eg - B
    child = np.zeros_like(Ke)
    m = Ke >= K_NODES[0]
    if np.any(m):
        zz = np.full(np.count_nonzero(m), np.clip(z, Z_NODES[0], Z_NODES[-1]))
        lk = np.log(np.clip(Ke[m], K_NODES[0], K_NODES[-1]))
        child[m] = np.maximum(yloc(np.column_stack([zz, lk])), 0.0)
    Reff = float(np.trapz(dnde * (1.0 + child), Eg))
    return R0, Reff


def _rate_interpolators(mode):
    yloc = _local_electron_yield()
    if mode == "fixed":
        emax = np.full_like(Z_GRID, 1.0e3)
    elif mode == "mfp":
        # Importing here avoids coupling the core loss module to the figure code.
        from make_paper_figures import E_free
        emax = np.array([E_free(z) for z in Z_GRID])
    else:
        raise ValueError("mode must be 'fixed' or 'mfp'")

    R0 = np.zeros((len(Z_GRID), len(_KR)))
    Re = np.zeros_like(R0)
    for iz, (z, ehi) in enumerate(zip(Z_GRID, emax)):
        for ik, K in enumerate(_KR):
            R0[iz, ik], Re[iz, ik] = photon_rates(K, z, ehi, yloc)

    floor = 1.0e-300
    f0 = RegularGridInterpolator(
        (Z_GRID, np.log(_KR)), np.log(np.clip(R0, floor, None)),
        bounds_error=False, fill_value=None,
    )
    fe = RegularGridInterpolator(
        (Z_GRID, np.log(_KR)), np.log(np.clip(Re, floor, None)),
        bounds_error=False, fill_value=None,
    )

    def rates(K_eV, z):
        if K_eV < _KR[0]:
            return 0.0, 0.0
        point = [[float(np.clip(z, Z_GRID[0], Z_GRID[-1])),
                  float(np.clip(np.log(K_eV), np.log(_KR[0]), np.log(_KR[-1])))]]
        return float(np.exp(f0(point)[0])), float(np.exp(fe(point)[0]))

    return rates, emax, R0, Re


def _secondary_interpolators(Ye, Yph, n_done):
    """Mean descendant yields over the BEB secondary-electron spectrum."""
    if n_done < 2:
        return None, None
    Kk = K_GRID[:n_done]
    ge = np.zeros((len(Z_GRID), n_done))
    gp = np.zeros_like(ge)
    for ik, K in enumerate(Kk):
        eps, pdf = secondary_pdf(K)
        if eps is None:
            continue
        m = eps >= B
        if not np.any(m):
            continue
        for iz in range(len(Z_GRID)):
            ge[iz, ik] = np.trapz(
                pdf[m] * np.interp(eps[m], Kk, Ye[iz, :n_done], left=0.0), eps[m]
            )
            gp[iz, ik] = np.trapz(
                pdf[m] * np.interp(eps[m], Kk, Yph[iz, :n_done], left=0.0), eps[m]
            )
    axes = (Z_GRID, np.log(Kk))
    return (RegularGridInterpolator(axes, ge, bounds_error=False, fill_value=None),
            RegularGridInterpolator(axes, gp, bounds_error=False, fill_value=None))


def run():
    # The direct cascade has already been solved on this exact grid.  Reusing it
    # avoids hundreds of identical stiff integrations and makes Ye an explicit
    # regression check against cascade_traj_table.npz.
    direct = np.load(Path(__file__).resolve().parent / "cascade_traj_table.npz")
    if not (np.allclose(direct["K"], K_GRID) and np.allclose(direct["z"], Z_GRID)):
        raise RuntimeError("cascade_traj_table.npz is not on the expected grid")
    N1 = direct["N1"].copy()
    Ye = direct["Y"].copy()

    rf, emax_f, R0f, Ref = _rate_interpolators("fixed")
    rm, emax_m, R0m, Rem = _rate_interpolators("mfp")
    nK, nZ = len(K_GRID), len(Z_GRID)
    Ngf = np.zeros((nZ, nK)); Ngm = np.zeros_like(Ngf)
    Ypf = np.zeros_like(Ngf); Ypm = np.zeros_like(Ngf)
    spf = spm = None

    for ik, K0 in enumerate(K_GRID):
        # Electrons below the photon-rate grid cannot up-scatter a CMB photon
        # into the productive band; their photon branch is identically zero.
        if K0 >= _KR[0]:
            for iz, z_i in enumerate(Z_GRID):
                t_i, t_f = L.age_s(z_i), L.age_s(Z_FINAL)

                def rhs(t, y):
                    KJ = max(float(y[0]), 0.0)
                    if KJ <= 0.0:
                        return [0.0] * 5
                    z = float(L.z_interp(t)); KeV = KJ / EV
                    tot = float(L.total_loss(z, KJ, float(L.H_interp(t)), L.toggles()))
                    _, _, _, v = L.kinematics(KJ)
                    rc = L.n_HI(z) * float(v) * float(sigma_ion(KeV))
                    r0f, reff = rf(KeV, z); r0m, remf = rm(KeV, z)
                    sf = sm = 0.0
                    if spf is not None:
                        lk = float(np.clip(np.log(max(KeV, K_GRID[0])),
                                           np.log(K_GRID[0]), np.log(K_GRID[ik - 1])))
                        pt = [[float(np.clip(z, Z_GRID[0], Z_GRID[-1])), lk]]
                        sf = max(float(spf(pt)[0]), 0.0)
                        sm = max(float(spm(pt)[0]), 0.0)
                    return [-tot, r0f, r0m, rc * sf + reff, rc * sm + remf]

                def ev(t, y):
                    return y[0] - float(L.thermal_floor(L.z_interp(t)))
                ev.terminal, ev.direction = True, -1
                sol = solve_ivp(rhs, (t_i, t_f), [K0 * EV, 0.0, 0.0, 0.0, 0.0],
                                events=ev, method="Radau", rtol=2.0e-6, atol=1.0e-25)
                Ngf[iz, ik], Ngm[iz, ik], Ypf[iz, ik], Ypm[iz, ik] = sol.y[1:, -1]

        # Only the photon component is recursive here; Ye is the independently
        # computed direct cascade and is left untouched.
        _, spf = _secondary_interpolators(np.zeros_like(Ypf), Ypf, ik + 1)
        _, spm = _secondary_interpolators(np.zeros_like(Ypm), Ypm, ik + 1)
        if ik % 4 == 0 or ik == nK - 1:
            print(f"K={K0:10.4g} eV  z10: Ye={Ye[3,ik]:9.4g} "
                  f"Yph1={Ypf[3,ik]:9.4g} Yphm={Ypm[3,ik]:9.4g}  "
                  f"z20: Ye={Ye[7,ik]:9.4g} Yph1={Ypf[7,ik]:9.4g} "
                  f"Yphm={Ypm[7,ik]:9.4g}", flush=True)

    payload = dict(K=K_GRID, z=Z_GRID, N1=N1, Ye=Ye,
                   Ngamma_fixed=Ngf, Yph_fixed=Ypf, emax_fixed=emax_f,
                   Ngamma_mfp=Ngm, Yph_mfp=Ypm, emax_mfp=emax_m,
                   rate_K=_KR, R0_fixed=R0f, Reff_fixed=Ref,
                   R0_mfp=R0m, Reff_mfp=Rem)
    np.savez(OUT, **payload)
    print("saved", OUT)


if __name__ == "__main__":
    run()
