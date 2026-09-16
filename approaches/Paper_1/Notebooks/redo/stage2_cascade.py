"""
Stage 2: independent solution of the full cascade.

For every injection energy K0 on a grid and every injection redshift z_i, one
stiff ODE is integrated from z_i to z=5.5 (or to the thermal floor) carrying
seventeen states:

  y0  K            kinetic energy of the primary
  y1  N1           ionizations by the primary alone
  y2  Y            ionizations by the primary + every generation of the shower
  y3  Hq           Coulomb heat deposited by the primary + the whole shower
  y4  Xq           excitation energy radiated by the primary + the whole shower
  y5  Ng_f         IC photons emitted into 13.6 eV - 1 keV (primary + shower)
  y6  Yph_f        ionizations seeded by those photons (photoionization +
                   the photoelectron's own cascade), primary + shower
  y7  Hph_f        Coulomb heat deposited by those photoelectron cascades
  y8,9,10          the same three with the redshift-dependent mfp ceiling
  y11,12,13        the same three with COSMOLOGICAL PHOTON TRANSPORT, with the
                   absorption allowed only inside the EoR window z > 5.5
  y14,15,16        the same three with the transport followed down to Z_ABS_MIN

The shower is closed by marching upward in K: a secondary always has
eps <= (K-B)/2 < K, so <Y(eps,z)>, <H(eps,z)>, <X(eps,z)>, <Ng(eps,z)>,
<Yph(eps,z)>, <Hph(eps,z)> over the BEB secondary spectrum are already known
when the primary at K is integrated.

Energy conservation is checked afterwards: K0 must equal
  Hq + Xq + B*Y + (IC + synchrotron + bremsstrahlung + adiabatic, all
  generations) + residual thermal energy.

--------------------------------------------------------------------------
THE TRANSPORT VARIANTS  (new; addresses Sect. 6.5(ii) of
manuscript/code_anatomy_ionization_yield.pdf)
--------------------------------------------------------------------------
The ``f`` and ``m`` variants apply a hard ceiling: every IC photon between
13.6 eV and E_max makes exactly one photoionization, every photon above E_max
is discarded.  That discards photons that are in fact absorbed, because a
photon redshifts while sigma_pi ~ E^-3 grows: n_HI sigma_pi is nearly constant
in proper units while the horizon c/H grows, so a photon that is transparent
at emission becomes opaque as it ages.

The ``e`` and ``t`` variants replace the ceiling by the exact first-interaction
probability along the light path (``redo_common.absorption_kernel``).  There is
then no band edge at all: each emitted photon is credited with

    P(photoionized)  ionizations of its own, plus the cascade of a
                     photoelectron of energy E(z_abs) - B_H released at the
                     redshift z_abs where the absorption actually happens.

Compton scattering terminates the photon, which is conservative.  The two
variants differ only in how far the photon is followed:

  ``e``  absorption restricted to z > Z_FINAL = 5.5, i.e. only ionizations
         delivered inside the same window as everything else in the paper;
  ``t``  absorption followed down to Z_ABS_MIN, so the gap between the two
         measures how much of the yield is delivered after the EoR window
         closes.  Below z ~ 5.5 the fiducial neutral-IGM density law is an
         extrapolation, which is why Z_ABS_MIN is not pushed to zero.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

import redo_common as R
import igm_losses as L

EV = R.EV
B = R.B_H
OUT = Path(__file__).resolve().parent / "redo_cascade_table.npz"

# Same K grid as cascade_traj.py so the tables can be compared element by
# element, plus nine extra nodes below the ionization threshold that are needed
# for the heat of sub-threshold photoelectrons and secondaries.
K_LOW = np.geomspace(0.30, B * 0.995, 9)
K_MAIN = np.concatenate([np.geomspace(B * 1.02, 1e3, 22),
                         np.geomspace(1.3e3, 1e13, 42)])
K_GRID = np.concatenate([K_LOW, K_MAIN])
NLOW = len(K_LOW)
Z_GRID = np.array([5.5, 7.0, 8.5, 10.0, 12.0, 14.0, 17.0, 20.0])
Z_FINAL = 5.5
Z_CHILD_MIN = 7.0        # lowest redshift row that is usable as a child table

# ---------------------------------------------------------------- photon side
_KR = np.geomspace(1.0e5, 1.0e13, 150)          # parent energies that can emit

# Hard-band variants: the original grid, kept unchanged so that the ``f`` and
# ``m`` columns remain bit-for-bit comparable with the earlier table.
_EG = np.geomspace(B * (1 + 1e-9), 2.0e3, 160)  # absorbed-photon energies [eV]
EMAX_FIX = np.full(len(Z_GRID), 1.0e3)
EMAX_MFP = np.array([R.e_free(z) for z in Z_GRID])

# Transport variants: no ceiling, so the grid has to run up to where the
# absorption probability is negligible.  P(photoionized) < 1e-3 above 3e4 eV
# at every redshift in the grid, so that is a safe top.
EG_MAX_TR = 3.0e4
_EG_TR = np.geomspace(B * (1 + 1e-9), EG_MAX_TR, 260)
N_PATH = 128                                    # nodes along the light path
Z_ABS_MIN = 3.0                                 # floor of the ``t`` variant
TR_TAGS = ("e", "t")
TR_ZMIN = {"e": Z_FINAL, "t": Z_ABS_MIN}

# The transported emission rates need a finer redshift grid than the injection
# grid.  The ``e`` variant vanishes as z_emit -> Z_FINAL, simply because a
# photon emitted at the end of the window has no path left inside it, and
# log-interpolating that collapse across the 5.5 -> 7.0 gap of Z_GRID would
# corrupt every trajectory that spends time below z ~ 7.  Z_GRID itself is left
# untouched so that the ``f`` and ``m`` columns stay element-by-element
# comparable with cascade_traj_table.npz.
Z_TR = np.array([5.5, 5.75, 6.0, 6.4, 7.0, 7.7, 8.5, 9.2, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.5, 17.0, 18.5, 20.0])


def photon_spectra(Eg, zgrid):
    """S[z, K, Eg] = dN/(dt dE_gamma) [s^-1 eV^-1]; computed once."""
    S = np.zeros((len(zgrid), len(_KR), len(Eg)))
    for iz, z in enumerate(zgrid):
        for ik, K in enumerate(_KR):
            S[iz, ik] = R.ic_dnde(K, z, Eg)
    return S


def transport_kernels():
    """Pre-compute the light-path absorption kernels; they never change.

    Returns ``KER[tag] = [(w, Kq, Zq), ...]`` indexed by emission redshift,
    with ``w`` of shape (nE, n_path) and ``Kq``, ``Zq`` the flattened query
    points at which the child tables have to be evaluated.  Only the child
    tables change as the march proceeds, so everything here is built once.
    """
    KER = {}
    for tag in TR_TAGS:
        rows = []
        for z in Z_TR:
            w, Ke, zp = R.absorption_kernel(z, _EG_TR, TR_ZMIN[tag], N_PATH)
            Zq = np.broadcast_to(zp[None, :], Ke.shape)
            rows.append((w, Ke.ravel(), Zq.ravel()))
        KER[tag] = rows
    return KER


def band_moments(S, child_Y, child_H):
    """(R0, Reff, Rheat) for the fixed and the mfp ceiling."""
    out = {}
    for tag, emax in (("f", EMAX_FIX), ("m", EMAX_MFP)):
        R0 = np.zeros((len(Z_GRID), len(_KR)))
        Re = np.zeros_like(R0)
        Rh = np.zeros_like(R0)
        for iz, z in enumerate(Z_GRID):
            m = _EG <= emax[iz]
            Eg = _EG[m]
            Ke = np.maximum(Eg - B, 0.0)
            yv = child_Y(z, Ke)
            hv = child_H(z, Ke)
            s = S[iz][:, m]
            R0[iz] = np.trapz(s, Eg, axis=1)
            Re[iz] = np.trapz(s * (1.0 + yv[None, :]), Eg, axis=1)
            Rh[iz] = np.trapz(s * hv[None, :], Eg, axis=1)
        out[tag] = (R0, Re, Rh)
    return out


def transport_moments(S_tr, KER, child_Y, child_H):
    """(R0, Reff, Rheat) for the transported variants.

    Per emitted photon of energy Eg the kernel returns

        P0   = int dP                                (absorbed at all)
        Geff = int dP [1 + Y_e(E(z)-B, z)]           (ionizations)
        Gh   = int dP  H_q(E(z)-B, z)                (heat)

    which are then folded with the emission spectrum over Eg.  Note that the
    photoelectron is evaluated at the REDSHIFTED energy and at the redshift of
    the absorption, not at the emission redshift: that is the whole point of
    the exercise.
    """
    out = {}
    for tag in TR_TAGS:
        R0 = np.zeros((len(Z_TR), len(_KR)))
        Re = np.zeros_like(R0)
        Rh = np.zeros_like(R0)
        for iz in range(len(Z_TR)):
            w, Kq, Zq = KER[tag][iz]
            if w.shape[1] < 2:                  # z_emit == z_min: nothing left
                continue
            yv = child_Y(Zq, Kq).reshape(w.shape)
            hv = child_H(Zq, Kq).reshape(w.shape)
            P0 = w.sum(axis=1)
            Geff = (w * (1.0 + yv)).sum(axis=1)
            Gh = (w * hv).sum(axis=1)
            s = S_tr[iz]
            R0[iz] = np.trapz(s * P0[None, :], _EG_TR, axis=1)
            Re[iz] = np.trapz(s * Geff[None, :], _EG_TR, axis=1)
            Rh[iz] = np.trapz(s * Gh[None, :], _EG_TR, axis=1)
        out[tag] = (R0, Re, Rh)
    return out


KEYS = ("N1", "Y", "Hq", "Xq",
        "Ngf", "Ypf", "Hpf",
        "Ngm", "Ypm", "Hpm",
        "Nge", "Ype", "Hpe",
        "Ngt", "Ypt", "Hpt")
RKEYS = ("R0f", "Ref", "Rhf", "R0m", "Rem", "Rhm",
         "R0e", "Ree", "Rhe", "R0t", "Ret", "Rht")


def _log_interp_stack(tabs, zgrid):
    """One vector-valued interpolator for a set of photon rates (log space)."""
    v = np.log(np.clip(np.stack(tabs, axis=-1), 1e-300, None))
    return RegularGridInterpolator((zgrid, np.log(_KR)), v,
                                   bounds_error=False, fill_value=None)


# ---------------------------------------------------------------- child sides
def child_interp(vals, n_done):
    """Interpolator for a quantity known on K_GRID[:n_done] (child's own K).

    ``z`` may be a scalar (the hard-band callers) or an array broadcastable to
    ``K`` (the transport caller, which needs the child evaluated at the
    redshift of absorption, one value per path node).
    """
    if n_done < 2:
        return lambda z, K: np.zeros_like(np.atleast_1d(K), dtype=float)
    Kk = K_GRID[:n_done]
    # The z = Z_FINAL row of the table is structurally zero: a trajectory
    # injected at Z_FINAL is integrated over a zero-length interval, so every
    # accumulator ends at 0.  That row is meaningless as a CHILD table, and
    # interpolating towards it would drag the yield of every photoelectron
    # released near the end of the window down to nothing.  It is therefore
    # dropped here and the lookup is clipped to z >= Z_CHILD_MIN.  The cost is
    # nil: over 7 <= z <= 20 the child yield of a sub-keV electron varies by
    # 0.04 % (Y(1 keV) = 28.28 at every node), because such an electron
    # thermalizes in Delta z << 1 and its yield is a ratio of atomic rates.
    keep = Z_GRID >= Z_CHILD_MIN
    zc_grid = Z_GRID[keep]
    f = RegularGridInterpolator((zc_grid, np.log(Kk)), vals[keep][:, :n_done],
                                bounds_error=False, fill_value=None)

    def g(z, K):
        K = np.atleast_1d(np.asarray(K, float))
        lk = np.clip(np.log(np.clip(K, Kk[0], None)), np.log(Kk[0]), np.log(Kk[-1]))
        zz = np.clip(np.broadcast_to(np.asarray(z, float), lk.shape),
                     zc_grid[0], zc_grid[-1])
        return np.maximum(f(np.column_stack([zz.ravel(), lk.ravel()])), 0.0)
    return g


def fill_shower_row(cache, res, ik, pdfs):
    """<Q(eps,z)> for a parent at K_GRID[ik], for all sixteen accumulators.

    A secondary always has eps < K_GRID[ik], so the row is final once row ik of
    every accumulator is known: it never has to be recomputed.
    """
    eps, pdf = pdfs[ik]
    if eps is None:
        return
    Kk = K_GRID[:ik + 1]
    for iz in range(len(Z_GRID)):
        for j, k in enumerate(KEYS):
            cache[iz, ik, j] = np.trapz(
                pdf * np.interp(eps, Kk, res[k][iz, :ik + 1], left=0.0), eps)


def shower_interp_all(cache, n_done):
    if n_done < 2:
        return None
    return RegularGridInterpolator(
        (Z_GRID, np.log(K_GRID[:n_done])), cache[:, :n_done, :],
        bounds_error=False, fill_value=None)


def run():
    t_start = time.time()
    nK, nZ = len(K_GRID), len(Z_GRID)
    res = {k: np.zeros((nZ, nK)) for k in KEYS}
    sec_cache = np.zeros((nZ, nK, len(KEYS)))
    esc = np.zeros((nZ, nK))          # escaping channels of the primary only

    print("tabulating IC photon spectra ...", flush=True)
    S = photon_spectra(_EG, Z_GRID)
    S_tr = photon_spectra(_EG_TR, Z_TR)
    print(f"   done in {time.time()-t_start:.1f} s", flush=True)

    print("tabulating light-path absorption kernels ...", flush=True)
    KER = transport_kernels()
    for tag in TR_TAGS:
        p20 = KER[tag][-1][0].sum(axis=1)
        i1 = int(np.argmin(np.abs(_EG_TR - 1.0e3)))
        i3 = int(np.argmin(np.abs(_EG_TR - 3.0e3)))
        print(f"   tag '{tag}' (z_min={TR_ZMIN[tag]:.1f}): at z_i=20, "
              f"P_abs(1 keV)={p20[i1]:.3f}  P_abs(3 keV)={p20[i3]:.3f}",
              flush=True)
    print(f"   done in {time.time()-t_start:.1f} s", flush=True)

    # BEB secondary spectra of every grid energy, computed once
    pdfs = []
    for K in K_GRID:
        e, p = R.secondary_pdf(K) if K > B else (None, None)
        if e is not None:
            m = e > 0
            pdfs.append((e[m], p[m]))
        else:
            pdfs.append((None, None))

    ms = None                         # vector interpolator of <Q(eps,z)>
    rates = None                      # hard-band photon rates, on Z_GRID
    rates_tr = None                   # transported photon rates, on Z_TR
    ZERO_S = np.zeros(len(KEYS))
    ZERO_R = np.zeros(len(RKEYS))
    NY = len(KEYS) + 1                # ODE states

    for ik, K0 in enumerate(K_GRID):
        for iz, z_i in enumerate(Z_GRID):
            t_i, t_f = L.age_s(z_i), L.age_s(Z_FINAL)

            def rhs(t, y):
                KJ = max(float(y[0]), 0.0)
                if KJ <= 0.0:
                    return [0.0] * NY
                z = float(L.z_interp(t))
                KeV = KJ / EV
                lr = L.loss_rates(z, KJ, float(L.H_interp(t)), L.toggles())
                lr = np.asarray(lr, float).ravel()
                tot = float(lr.sum())
                _, _, _, v = L.kinematics(KJ)
                rc = L.n_HI(z) * float(v) * float(R.sigma_ion(KeV))

                zc = float(np.clip(z, Z_GRID[0], Z_GRID[-1]))
                if ms is not None and ik:
                    lk = float(np.clip(np.log(max(KeV, K_GRID[0])),
                                       np.log(K_GRID[0]), np.log(K_GRID[ik - 1])))
                    sec = np.maximum(np.ravel(ms(((zc, lk),))), 0.0)
                else:
                    sec = ZERO_S

                if rates is not None and KeV >= _KR[0]:
                    lk = float(np.clip(np.log(KeV), np.log(_KR[0]), np.log(_KR[-1])))
                    ztr = float(np.clip(z, Z_TR[0], Z_TR[-1]))
                    ph = np.concatenate([
                        np.exp(np.ravel(rates(((zc, lk),)))),
                        np.exp(np.ravel(rates_tr(((ztr, lk),))))])
                else:
                    ph = ZERO_R

                return [
                    -tot,
                    rc,
                    rc * (1.0 + sec[1]),
                    lr[3] / EV + rc * sec[2],
                    lr[4] / EV + rc * sec[3],
                    ph[0] + rc * sec[4],
                    ph[1] + rc * sec[5],
                    ph[2] + rc * sec[6],
                    ph[3] + rc * sec[7],
                    ph[4] + rc * sec[8],
                    ph[5] + rc * sec[9],
                    ph[6] + rc * sec[10],
                    ph[7] + rc * sec[11],
                    ph[8] + rc * sec[12],
                    ph[9] + rc * sec[13],
                    ph[10] + rc * sec[14],
                    ph[11] + rc * sec[15],
                ]

            def ev(t, y):
                return y[0] - float(L.thermal_floor(L.z_interp(t)))
            ev.terminal, ev.direction = True, -1

            sol = solve_ivp(rhs, (t_i, t_f), [K0 * EV] + [0.0] * len(KEYS),
                            events=ev, method="Radau", rtol=1e-6, atol=1e-25)
            for j, k in enumerate(KEYS):
                res[k][iz, ik] = sol.y[j + 1, -1]

        # --- refresh the recursion closures with the row just completed ------
        fill_shower_row(sec_cache, res, ik, pdfs)
        ms = shower_interp_all(sec_cache, ik + 1)
        # The hard-band branch only ever needs child yields below ~2 keV; the
        # transport branch needs them up to EG_MAX_TR, because a photon that
        # high can still be absorbed after it has redshifted.  Both stop
        # changing once the marching has passed that energy.
        if K_GRID[ik] < 1.2 * EG_MAX_TR or rates is None:
            cY = child_interp(res["Y"], ik + 1)
            cH = child_interp(res["Hq"], ik + 1)
            mb = band_moments(S, cY, cH)
            mt = transport_moments(S_tr, KER, cY, cH)
            rates = _log_interp_stack(
                [mb["f"][0], mb["f"][1], mb["f"][2],
                 mb["m"][0], mb["m"][1], mb["m"][2]], Z_GRID)
            rates_tr = _log_interp_stack(
                [mt["e"][0], mt["e"][1], mt["e"][2],
                 mt["t"][0], mt["t"][1], mt["t"][2]], Z_TR)

        if ik % 4 == 0 or ik == nK - 1:
            i10, i20 = 3, 7
            print(f"[{time.time()-t_start:7.1f}s] K={K0:10.4g} eV | "
                  f"z10: Y={res['Y'][i10,ik]:10.4g} Yf={res['Ypf'][i10,ik]:10.4g} "
                  f"Ye={res['Ype'][i10,ik]:10.4g} Yt={res['Ypt'][i10,ik]:10.4g} | "
                  f"z20: Y={res['Y'][i20,ik]:10.4g} "
                  f"Yf={res['Ypf'][i20,ik]:10.4g} Ye={res['Ype'][i20,ik]:10.4g} "
                  f"Yt={res['Ypt'][i20,ik]:10.4g}", flush=True)
            np.savez(OUT, K=K_GRID, z=Z_GRID, nlow=NLOW, done=ik + 1,
                     emax_fixed=EMAX_FIX, emax_mfp=EMAX_MFP,
                     z_abs_min=Z_ABS_MIN, eg_max_tr=EG_MAX_TR, z_tr=Z_TR, **res)

    np.savez(OUT, K=K_GRID, z=Z_GRID, nlow=NLOW, done=nK,
             emax_fixed=EMAX_FIX, emax_mfp=EMAX_MFP,
             z_abs_min=Z_ABS_MIN, eg_max_tr=EG_MAX_TR, z_tr=Z_TR, **res)
    print("saved", OUT, f"in {time.time()-t_start:.1f} s")


if __name__ == "__main__":
    run()
