"""Stage 1: independent checks of the ingredients of Fig. 4."""
import numpy as np
import redo_common as R
import igm_losses as L

EV = R.EV

print("\n=== 1. CMB energy density closure (my quadrature vs module constant) ===")
for z in (0.0, 10.0, 20.0):
    mine = R.u_cmb(z)
    theirs = L.U_CMB_0_J_M3 * (1.0 + z) ** 4
    print(f"  z={z:5.1f}  u_mine={mine:.6e}  u_module={theirs:.6e}  ratio={mine/theirs:.8f}")

print("\n=== 2. IC differential spectrum: energy integral vs the loss rate used in the ODE ===")
print("    (int E dN/dE dE)  vs  L_IC = 4/3 sigma_T c U_CMB (p^2c^2/(mc^2)^2) F_KN")
print(f"  {'z':>5} {'K [eV]':>10} {'gamma':>9} {'P_spec [J/s]':>13} {'L_IC [J/s]':>13} "
      f"{'ratio':>9} {'1+3/(4g^2)':>11}")
for z in (20.0, 10.0):
    for K in (1e5, 1e6, 1e7, 1e8, 1e9, 1e11):
        H = float(L.Planck18.H(z).to(L.u.s ** -1).value)
        lic = float(L.loss_rates(z, K * EV, H, L.toggles())[2])
        p = R.ic_emitted_power(K, z)
        g = 1.0 + K / R.E0_EV
        print(f"  {z:5.1f} {K:10.3g} {g:9.3g} {p:13.6e} {lic:13.6e} "
              f"{p/lic:9.6f} {1+0.75/g**2:11.6f}")

print("\n=== 3. Photoionization cross-section sanity ===")
for E in (13.61, 16.0, 54.4, 136.06, 1360.6):
    print(f"  E={E:9.2f} eV   sigma_pi={R.sigma_pi(E)[0]*1e4:.4e} cm^2")

print("\n=== 4. Free-streaming ceiling E_gamma,hor(z) and the boundaries K3, K4 ===")


def ke_for_photon(E_ph_eV, z, x):
    th = L.K_B * L.T_CMB_0 * (1.0 + z)
    tJ = E_ph_eV * EV
    g = tJ / L.E0 + np.sqrt((tJ / L.E0) ** 2 + tJ / (x * th))
    return L.E0 * ((g ** 2 + 1.0) / (2.0 * g) - 1.0) / EV


def k_ion_ic_crossover(z):
    K = np.logspace(2, 8, 8000)
    H = float(L.Planck18.H(z).to(L.u.s ** -1).value)
    st = L.loss_rates(z, K * EV, H, L.toggles())
    d = np.log(np.clip(st[5], 1e-300, None)) - np.log(np.clip(st[2], 1e-300, None))
    i = np.where(np.diff(np.sign(d)) != 0)[0]
    return K[i[-1]] if len(i) else np.nan


published = {20.0: (7.60e4, 4.20e6, 1820, 3.25e8),
             15.0: (9.32e4, 4.88e6, 1550, 3.43e8),
             10.0: (1.23e5, 5.99e6, 1272, 3.75e8),
             8.0:  (1.43e5, 6.67e6, 1151, 3.94e8),
             5.5:  (1.82e5, 7.93e6,  983, 4.29e8)}
print(f"  {'z':>5} | {'K2 mine':>9} {'K2 pub':>9} | {'K3 mine':>9} {'K3 pub':>9} | "
      f"{'Ehor mine':>10} {'Ehor pub':>9} | {'K4 mine':>10} {'K4 pub':>9}")
for z, (k2, k3, eh, k4) in published.items():
    m2 = k_ion_ic_crossover(z)
    m3 = ke_for_photon(R.B_H, z, L.X_MAX_PLANCK)
    mh = R.e_free(z)
    m4 = ke_for_photon(mh, z, L.X_MIN_PLANCK)
    print(f"  {z:5.1f} | {m2:9.3g} {k2:9.3g} | {m3:9.3g} {k3:9.3g} | "
          f"{mh:10.1f} {eh:9.0f} | {m4:10.4g} {k4:9.3g}")
