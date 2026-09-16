"""
Stage 8: average number of ionizations per PRIMARY PHOTON, all generations
included, on the same footing as the per-primary-electron numbers already in
``redo_cascade_table.npz``.

--------------------------------------------------------------------------
WHAT IS COMPUTED
--------------------------------------------------------------------------
For a photon of energy E injected at redshift z_i into the fiducial neutral
IGM, ``N_gamma(E, z_i)`` is the expectation value of the total number of
H I ionizations produced by that photon and by every particle below it in the
shower, counted only while the shower is inside the redshift window of the
calculation.

The photon is followed along its own light path.  Two things happen to it:

  (i)  it redshifts, E(z) = E (1+z)/(1+z_i);
  (ii) it is removed by the first interaction, whose probability density in
       proper time is  dP = (d tau_j / dt) exp(-tau_tot) dt  for channel j.

Only two channels are included, per the scope fixed for this run:

  PHOTOIONIZATION of H(1s)   sigma_pi from Karzas & Latter (1961), as coded in
                             ``redo_common.sigma_pi``.  One ionization, plus a
                             photoelectron of kinetic energy E(z) - B_H which
                             runs the full electron cascade.

  COMPTON SCATTERING         Klein & Nishina (1929) differential cross-section
                             in the free-electron (impulse) approximation.  The
                             recoil electron takes T = E(z)(1-s), the scattered
                             photon continues with E' = E(z) s, and BOTH are
                             followed: the electron through the electron
                             cascade, the photon through this same equation at
                             the lower energy E'.  On a bound electron (a
                             fraction 1/(1+x_e) of the targets) an event with
                             T >= B_H is itself one ionization and the ejected
                             electron starts at T - B_H; an event with T < B_H
                             ionizes nothing.

so that

  N(E,z_i) = sum_path w_pi [ 1 + Y_e(E(z)-B, z) ]
           + sum_path w_C  < f_b Theta(T-B)[1 + Y_e(T-B,z)]
                             + f_f Y_e(T,z) + N(E(z) s, z) >_s          (*)

with Y_e(K,z) = Y + Yp taken from ``redo_cascade_table.npz`` (electron-impact
ionizations plus the ionizations seeded by the electron's own inverse-Compton
photons), f_b = 1/(1+x_e), f_f = x_e/(1+x_e).

Equation (*) is a Volterra equation in E: every argument on the right is
<= E, with equality only in the limit (z -> z_i, s -> 1).  It is solved by
marching UPWARD in E exactly as stage2_cascade.py marches upward in K.  The
only coupling to the energy node being solved comes from the top interpolation
cell; that part is kept on the left-hand side, which turns each energy node
into one small linear system in the redshift rows (n_z x n_z, solved exactly).
No iteration and no relaxation is used, so the near-elastic Compton limit
(s -> 1 at E << m_e c^2), where the whole Compton weight sits inside one grid
cell, is resummed exactly rather than being lost or being amplified.

--------------------------------------------------------------------------
REGIME OF VALIDITY  (important -- read before using the high-energy end)
--------------------------------------------------------------------------
Pair production is NOT included, by choice for this run.  Two channels are
therefore missing above ~1 MeV:

  * Bethe-Heitler pair production in the field of the H nucleus and of its
    electron.  Its cross-section [Heitler 1954, Sect. 26; PDG "Passage of
    particles through matter", complete-screening limit] overtakes Klein-
    Nishina at a few hundred MeV for Z = 1.  ``stage8_verify.py`` prints the
    exact crossing energy from the same formulae.
  * gamma-gamma pair production on the CMB [Breit & Wheeler 1934; Gould &
    Schreder 1967], whose threshold against the CMB peak is
    (m_e c^2)^2 / (k T_CMB(z)) ~ 5e13 eV at z = 20 but which already bites in
    the Wien tail near 1e13 eV.

Both channels return energy to electrons that would otherwise free-stream out
of the window, so ABOVE THE CROSSING ENERGY PRINTED BY stage8_verify.py THE
CURVE COMPUTED HERE IS A LOWER BOUND, not an estimate.  Below it the two
channels included are the complete list for a hydrogen IGM.

Two further approximations, both inherited from the rest of the project:

  * Compton scattering is treated in the free-electron impulse approximation.
    Below ~2 keV the scattering off a bound H electron is largely coherent
    (Rayleigh), i.e. the photon does NOT lose T.  Using the incoherent form
    there is conservative for the photon's survival, and it is irrelevant for
    the answer because photoionization outweighs Compton by a factor >~20
    below 2 keV (printed by stage8_verify.py).
  * the electron table itself discards the electron's own IC photons above
    3e4 eV (``EG_MAX_TR`` in stage2_cascade.py) on the grounds that their
    absorption probability inside the window is < 1e-3.  That remains true for
    the photoionization channel but ignores the same pair-production channels
    listed above, so Y_e at K >~ 1e12 eV carries the same lower-bound caveat.

--------------------------------------------------------------------------
WINDOW VARIANTS
--------------------------------------------------------------------------
``e``  the photon is followed only down to Z_FINAL = 5.5, i.e. only
       ionizations delivered inside the EoR window of the paper are counted;
``t``  the photon is followed down to Z_ABS_MIN = 3.0, so the gap between the
       two measures how much of the yield is delivered after the window
       closes.  Below z ~ 5.5 the fiducial neutral-IGM density law is an
       extrapolation, which is why Z_ABS_MIN is not pushed to zero.

Output: ``redo_photon_yield_table.npz``.
"""
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import redo_common as R
import igm_losses as L

EV = R.EV
B = R.B_H                                   # 13.6057 eV
E0_EV = R.E0_EV                             # 510998.95 eV
X_E = L.ION_FRACTION                        # 1e-4
F_BOUND = 1.0 / (1.0 + X_E)
F_FREE = X_E / (1.0 + X_E)

HERE = Path(__file__).resolve().parent
CASCADE = HERE / "redo_cascade_table.npz"
OUT = HERE / "redo_photon_yield_table.npz"

# ---------------------------------------------------------------- grids
# Photon energies.  Six extra nodes hug the threshold, where N jumps from 0 to
# 1 over 13.6057 -> 13.61 eV; above that a uniform logarithmic grid with
# d ln E = 0.23 up to the top of the electron grid.
E_LOW = np.geomspace(B * 1.0002, B * 1.5, 6)
E_MAIN = np.geomspace(B * 1.6, 1.0e13, 114)
E_GRID = np.concatenate([E_LOW, E_MAIN])
LNE = np.log(E_GRID)

# Redshift rows.  The grid has to reach Z_ABS_MIN because the ``t`` variant
# queries the table at the redshift of the scattering, not of the injection.
Z_GRID = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.75, 6.0, 6.4, 7.0, 7.7,
                   8.5, 9.2, 10.0, 11.0, 12.0, 13.0, 14.0, 15.5, 17.0, 18.5,
                   20.0])
NZ = len(Z_GRID)

Z_FINAL = 5.5
Z_ABS_MIN = 3.0
TAGS = ("e", "t")
TR_ZMIN = {"e": Z_FINAL, "t": Z_ABS_MIN}

N_PATH = 192            # nodes along the light path
N_S = 64                # Gauss-Legendre nodes of the Klein-Nishina spectrum
Z_CHILD_MIN = 7.0       # same clip as stage2_cascade.child_interp

_GX, _GW = np.polynomial.legendre.leggauss(N_S)


# =====================================================================
# 1.  Klein-Nishina differential cross-section
# =====================================================================
def kn_dsigma_ds(E_eV, s):
    """d sigma / d s  [m^2] with s = E'/E the scattered-energy fraction.

    Klein & Nishina (1929); in the form of Rybicki & Lightman (1979) Eq. (7.5)
    transformed from cos(theta) to s through
        s = 1 / [1 + a (1 - cos theta)],  a = E / m_e c^2,
        d cos theta = ds / (a s^2),
    which turns  d sigma / d Omega = (r_e^2/2) s^2 (s + 1/s - sin^2 theta)
    into
        d sigma / ds = (pi r_e^2 / a) [ s + 1/s - 1 + cos^2 theta ].
    Support: 1/(1+2a) <= s <= 1.
    """
    a = np.asarray(E_eV, float) * EV / L.E0
    s = np.asarray(s, float)
    cth = 1.0 - (1.0 / s - 1.0) / a
    val = (np.pi * L.ELECTRON_RADIUS_MKS ** 2 / a) * (s + 1.0 / s - 1.0 + cth ** 2)
    return np.where((s >= 1.0 / (1.0 + 2.0 * a)) & (s <= 1.0), val, 0.0)


def kn_spectrum(E_eV):
    """(s, p) with p the NORMALISED probability weights of the recoil split.

    The quadrature is Gauss-Legendre in u = ln s, so that the 1/s tail of the
    Klein-Nishina spectrum -- which spans s_min = 1/(1+2a) ~ 1/(2a), i.e. seven
    decades at E = 5 TeV -- is integrated on a flat integrand.  In the opposite
    limit a << 1 the interval shrinks as 2a and the rule stays exact.
    ``p.sum(axis=-1)`` is 1 by construction; ``stage8_verify.py`` checks that
    the same weights without the normalisation reproduce ``redo_common.
    sigma_kn`` to better than 1e-10 relative over the whole grid.
    """
    E = np.atleast_1d(np.asarray(E_eV, float))
    a = E * EV / L.E0
    u_min = np.log(1.0 / (1.0 + 2.0 * a))                 # < 0
    u = 0.5 * u_min[..., None] * (1.0 - _GX)              # in [u_min, 0]
    w = -0.5 * u_min[..., None] * _GW                     # positive
    s = np.exp(u)
    dens = kn_dsigma_ds(E[..., None], s) * s              # ds = s du
    tot = np.sum(dens * w, axis=-1, keepdims=True)
    return s, dens * w / np.maximum(tot, 1e-300)


def kn_sigma_quadrature(E_eV):
    """Total Klein-Nishina cross-section from the same quadrature [m^2]."""
    E = np.atleast_1d(np.asarray(E_eV, float))
    a = E * EV / L.E0
    u_min = np.log(1.0 / (1.0 + 2.0 * a))
    u = 0.5 * u_min[..., None] * (1.0 - _GX)
    w = -0.5 * u_min[..., None] * _GW
    s = np.exp(u)
    return np.sum(kn_dsigma_ds(E[..., None], s) * s * w, axis=-1)


# =====================================================================
# 2.  Light-path kernel, split by channel
# =====================================================================
def path_kernel(z_emit, E_eV, z_min, n_path=N_PATH):
    """First-interaction weights along the light path, per channel.

    Same optical-depth quadrature as ``redo_common.absorption_kernel`` (the
    integrand is treated as linear in tau and exp(-tau) is integrated
    analytically over each segment, so the weights are non-negative and their
    sum is bounded by 1 - exp(-tau_max) by construction), but the branching
    between photoionization and Compton scattering is kept instead of the
    Compton channel being thrown away.

    Returns (w_pi, w_C, E_path, zp), all of length ``n_path`` except zp.
    """
    z_emit = float(z_emit)
    if z_min >= z_emit:
        z = np.array([z_emit])
        return np.zeros(1), np.zeros(1), np.array([float(E_eV)]), z

    zp = np.expm1(np.linspace(np.log1p(z_emit), np.log1p(z_min), n_path))
    tp = L.age_s(zp)
    Ep = float(E_eV) * (1.0 + zp) / (1.0 + z_emit)
    nH = L.n_HI(zp)

    k_pi = nH * R.sigma_pi(Ep) * L.C_LIGHT
    k_C = nH * (1.0 + X_E) * R.sigma_kn(Ep) * L.C_LIGHT
    k_tot = k_pi + k_C

    seg = 0.5 * (k_tot[1:] + k_tot[:-1]) * np.diff(tp)
    tau = np.concatenate([[0.0], np.cumsum(seg)])

    t1, t2 = tau[:-1], tau[1:]
    dtau = np.maximum(t2 - t1, 0.0)
    u1, u2 = np.exp(-t1), np.exp(-t2)
    small = dtau < 1.0e-8
    ratio = np.divide(u1 - u2, dtau, out=np.zeros_like(dtau), where=~small)
    a1 = np.where(small, 0.5 * u1 * dtau, u1 - ratio)
    a2 = np.where(small, 0.5 * u1 * dtau, ratio - u2)

    w = np.zeros_like(tau)
    w[:-1] += a1
    w[1:] += a2

    f_pi = np.divide(k_pi, k_tot, out=np.zeros_like(k_pi), where=k_tot > 0.0)
    return w * f_pi, w * (1.0 - f_pi), Ep, zp


# =====================================================================
# 3.  Electron cascade yield Y_e(K, z) from the stage-2 table
# =====================================================================
def electron_yield(tag):
    """Y + Yp<tag> interpolated in (z, ln K); zero below the ionization edge.

    The z = 5.5 row of the electron table is structurally zero (a trajectory
    injected at Z_FINAL is integrated over a zero-length interval) and is
    dropped here for the same reason it is dropped in
    ``stage2_cascade.child_interp``: the lookup is clipped to z >= 7, at a cost
    of 0.04 % because a sub-keV electron thermalizes in Delta z << 1 and its
    yield is a ratio of atomic rates.
    """
    d = np.load(CASCADE)
    K = d["K"]
    zt = d["z"]
    keep = zt >= Z_CHILD_MIN
    V = np.maximum(d["Y"][keep] + d["Yp" + tag][keep], 0.0)
    f = RegularGridInterpolator((zt[keep], np.log(K)), V,
                                bounds_error=False, fill_value=None)
    lo, hi = np.log(K[0]), np.log(K[-1])
    zlo, zhi = zt[keep][0], zt[keep][-1]

    def g(K_eV, z):
        k = np.asarray(K_eV, float)
        zz = np.clip(np.broadcast_to(np.asarray(z, float), k.shape), zlo, zhi)
        lk = np.clip(np.log(np.maximum(k, np.exp(lo))), lo, hi)
        out = np.maximum(f(np.stack([zz.ravel(), lk.ravel()], axis=-1)), 0.0)
        return np.where(k.ravel() > 0.0, out, 0.0).reshape(k.shape)
    return g


# =====================================================================
# 4.  Bilinear routing of the scattered photon into the (z, ln E) table
# =====================================================================
def _bilinear(Eq, zq, coef, i, table, M_row):
    """Evaluate  sum coef * N(Eq, zq)  splitting known from unknown.

    Energy nodes below ``i`` are already final and their contribution is
    RETURNED as a number; the weight that lands on node ``i`` itself is the
    diagonal of the Volterra equation and is accumulated into ``M_row``
    (length NZ), to be moved to the left-hand side by the caller.
    """
    known = 0.0
    Eq = Eq.ravel()
    zq = zq.ravel()
    cf = coef.ravel()
    m = (Eq > E_GRID[0]) & (cf > 0.0)          # below the grid: N = 0 exactly
    if not m.any():
        return known
    Eq, zq, cf = Eq[m], zq[m], cf[m]

    le = np.clip(np.log(Eq), LNE[0], LNE[i])
    je = np.clip(np.searchsorted(LNE[:i + 1], le) - 1, 0, max(i - 1, 0))
    fe = (le - LNE[je]) / (LNE[je + 1] - LNE[je])
    fe = np.clip(fe, 0.0, 1.0)

    zc = np.clip(zq, Z_GRID[0], Z_GRID[-1])
    jz = np.clip(np.searchsorted(Z_GRID, zc) - 1, 0, NZ - 2)
    fz = (zc - Z_GRID[jz]) / (Z_GRID[jz + 1] - Z_GRID[jz])
    fz = np.clip(fz, 0.0, 1.0)

    for de, we in ((0, 1.0 - fe), (1, fe)):
        ke = je + de
        for dz, wz in ((0, 1.0 - fz), (1, fz)):
            kz = jz + dz
            c = cf * we * wz
            top = ke == i
            if top.any():
                np.add.at(M_row, kz[top], c[top])
            if (~top).any():
                known += float(np.sum(c[~top] * table[kz[~top], ke[~top]]))
    return known


# =====================================================================
# 5.  The march
# =====================================================================
def run():
    t0 = time.time()
    nE = len(E_GRID)
    tables, probs = {}, {}

    for tag in TAGS:
        Ye = electron_yield(tag)
        zmin = TR_ZMIN[tag]
        N = np.zeros((NZ, nE))
        P = np.zeros((NZ, nE))          # total interaction probability, audit
        Ppi = np.zeros((NZ, nE))        # photoionization branch only

        for i, E in enumerate(E_GRID):
            A = np.zeros(NZ)
            M = np.zeros((NZ, NZ))
            for iz, z_i in enumerate(Z_GRID):
                if z_i <= zmin:
                    continue
                w_pi, w_C, Ep, zp = path_kernel(z_i, E, zmin)
                P[iz, i] = w_pi.sum() + w_C.sum()
                Ppi[iz, i] = w_pi.sum()

                # ---- photoionization branch -----------------------------
                Ke = np.maximum(Ep - B, 0.0)
                A[iz] += float(np.sum(w_pi * (1.0 + Ye(Ke, zp))))

                # ---- Compton branch -------------------------------------
                s, p = kn_spectrum(Ep)                    # (n_path, N_S)
                T = Ep[:, None] * (1.0 - s)
                zb = np.broadcast_to(zp[:, None], T.shape)
                bound = (T >= B) * (1.0 + Ye(np.maximum(T - B, 0.0), zb))
                free = Ye(T, zb)
                A[iz] += float(np.sum(w_C[:, None] * p
                                      * (F_BOUND * bound + F_FREE * free)))

                # ---- the scattered photon itself ------------------------
                # Everything that lands on an energy node already solved is
                # added to A; what lands on node i is the diagonal and goes
                # into M, i.e. onto the left-hand side.
                A[iz] += _bilinear(Ep[:, None] * s, zb, w_C[:, None] * p,
                                   i, N, M[iz])

            N[:, i] = np.linalg.solve(np.eye(NZ) - M, A)

        tables[tag] = N
        probs[tag] = (P, Ppi)
        print(f"[{time.time()-t0:7.1f}s] tag '{tag}' done", flush=True)

    np.savez(OUT, E=E_GRID, z=Z_GRID, z_final=Z_FINAL, z_abs_min=Z_ABS_MIN,
             n_path=N_PATH, n_s=N_S, x_e=X_E,
             Ne=tables["e"], Nt=tables["t"],
             Pe=probs["e"][0], Pt=probs["t"][0],
             Ppie=probs["e"][1], Ppit=probs["t"][1])
    print("saved", OUT, f"in {time.time()-t0:.1f} s")


if __name__ == "__main__":
    run()
