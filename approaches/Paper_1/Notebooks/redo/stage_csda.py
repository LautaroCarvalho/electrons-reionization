"""
Independent cross-check of the low-energy cascade by a completely different
method: continuous slowing down in a *static* medium (no ODE, no cosmology,
no trajectory), solved by upward marching in energy.

    dY/dK = sigma_ion/Lambda_tot            ionizations per eV lost
    dH/dK = Lambda_coul/Lambda_tot          heat per eV lost
    dX/dK = Lambda_exc/Lambda_tot           excitation per eV lost
    dE/dK = Lambda_esc/Lambda_tot           IC + brems + adiabatic per eV lost

each with the shower term  + (sigma_ion/Lambda_tot) <Q(eps|K)>.

Closure  B*Y + H + X + E_esc = K  is exact in this scheme, so it tests the
bookkeeping of the trajectory calculation from the outside.
"""
import numpy as np
import redo_common as R
import igm_losses as L

EV = R.EV
B = R.B_H
KG = np.geomspace(0.05, 1.0e6, 5000)         # eV


def channels(z):
    """Lambda_i [eV m^2] per unit hydrogen column, for the seven channels."""
    H = float(L.Planck18.H(z).to(L.u.s ** -1).value)
    nH = L.n_HI(z)
    _, _, _, v = L.kinematics(KG * EV)
    st = np.asarray(L.loss_rates(z, KG * EV, H, L.toggles()), float)
    lam = st / (nH * v) / EV                  # (7, nK)
    return lam


def solve(z):
    lam = channels(z)
    tot = lam.sum(axis=0)
    sig = R.sigma_ion(KG)
    r = np.where(tot > 0, sig / tot, 0.0)                      # ionizations/eV
    fh = lam[3] / tot                                          # Coulomb
    fx = lam[4] / tot                                          # excitation
    fe = (lam[0] + lam[1] + lam[2] + lam[6]) / tot             # escapes

    Y = np.zeros_like(KG); Hq = np.zeros_like(KG)
    X = np.zeros_like(KG); Es = np.zeros_like(KG)
    # an electron at the bottom of the grid is below every inelastic threshold:
    # all of its energy ends up as Coulomb heat.  Without this the cascade loses
    # KG[0] per generation endpoint, i.e. ~1 eV per ionization.
    Hq[0] = KG[0]
    for i in range(1, len(KG)):
        dK = KG[i] - KG[i - 1]
        Km = 0.5 * (KG[i] + KG[i - 1])
        rm = 0.5 * (r[i] + r[i - 1])
        e, p = R.secondary_pdf(Km) if Km > B else (None, None)
        if e is None:
            sY = sH = sX = sE = 0.0
        else:
            m = e > 0
            e, p = e[m], p[m]
            sY = np.trapz(p * np.interp(e, KG[:i], Y[:i], left=0.0), e)
            sH = np.trapz(p * np.interp(e, KG[:i], Hq[:i], left=0.0), e)
            sX = np.trapz(p * np.interp(e, KG[:i], X[:i], left=0.0), e)
            sE = np.trapz(p * np.interp(e, KG[:i], Es[:i], left=0.0), e)
        Y[i] = Y[i - 1] + dK * rm * (1.0 + sY)
        Hq[i] = Hq[i - 1] + dK * (0.5 * (fh[i] + fh[i - 1]) + rm * sH)
        X[i] = X[i - 1] + dK * (0.5 * (fx[i] + fx[i - 1]) + rm * sX)
        Es[i] = Es[i - 1] + dK * (0.5 * (fe[i] + fe[i - 1]) + rm * sE)
        if KG[i] < B:
            Y[i] = 0.0
    return Y, Hq, X, Es


if __name__ == "__main__":
    print("\nCSDA cascade in a static medium (no ODE): yield, W, deposition "
          "fractions and heat per ionization")
    for z in (20.0, 10.0):
        Y, Hq, X, Es = solve(z)
        print(f"\n  z = {z}")
        print(f"  {'K [eV]':>9} {'Y':>10} {'W=K/Y':>8} {'f_ion':>7} {'f_exc':>7} "
              f"{'f_heat':>7} {'f_esc':>7} {'sum':>7} {'Q=H/Y':>7}")
        for K0 in (20.0, 50.0, 1e2, 3e2, 1e3, 1e4, 1e5, 1e6):
            j = int(np.argmin(np.abs(KG - K0)))
            fi = B * Y[j] / KG[j]
            fx = X[j] / KG[j]; fh = Hq[j] / KG[j]; fe = Es[j] / KG[j]
            print(f"  {KG[j]:9.4g} {Y[j]:10.4g} {KG[j]/max(Y[j],1e-30):8.3f} "
                  f"{fi:7.3f} {fx:7.3f} {fh:7.3f} {fe:7.3f} "
                  f"{fi+fx+fh+fe:7.4f} {Hq[j]/max(Y[j],1e-30):7.3f}")
        j = np.argmin(np.where(KG > 30, KG / np.maximum(Y, 1e-30), np.inf))
        print(f"  minimum W = {KG[j]/Y[j]:.3f} eV at K = {KG[j]:.4g} eV")
