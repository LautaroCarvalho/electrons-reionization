#!/usr/bin/env python3
"""Fiducial numbers for the LRD cosmic-ray electron report.
All algebra used here is verified symbolically in verify_algebra.py."""
import numpy as np, json, os
from astropy.cosmology import Planck18
import astropy.units as u
import astropy.constants as const

OUT = {}
def rec(k, v, unit=""):
    OUT[k] = v
    print(f"{k:<38s} = {v:12.4g} {unit}")

# ---------------------------------------------------------------- constants (cgs)
c    = 2.99792458e10
e_es = 4.803204e-10          # esu
me   = 9.1093837e-28
mec2 = me*c**2               # erg
sT   = 6.6524587e-25
h    = 6.62607015e-27
kB   = 1.380649e-16
G_N  = 6.67430e-8
Msun = 1.98892e33
erg_eV = 6.241509e11
alpha_f = 1/137.035999

print("="*74); print("0.  UNIT CHECKS"); print("="*74)
rec("Hillas prefactor e*1cm*1G/1eV", e_es*1*1*erg_eV)          # expect 299.79
rec("burn-off photon energy [MeV] eta=1", (27/8)*mec2/alpha_f*erg_eV/1e6)

# ---------------------------------------------------------------- fiducial LRD
print("\n"+"="*74); print("1.  FIDUCIAL LRD  (Kuze+2026 / Rusakov+2026 / Naidu+2026)"); print("="*74)
z_LRD   = 5.0
MBH     = 10**6.5*Msun
L_Edd   = 1.26e38*(MBH/Msun)
L_bol   = 1e45                       # observed-scale, not dust-corrected
R_env   = 1e16                       # cm, ~4 light-days
R_g     = G_N*MBH/c**2
rec("M_BH [Msun]", MBH/Msun); rec("L_Edd [erg/s]", L_Edd)
rec("log10 L_Edd", np.log10(L_Edd)); rec("R_g [cm]", R_g)
rec("R_env [light-days]", R_env/(c*86400))
U_rad = L_bol/(4*np.pi*R_env**2*c)
rec("U_rad at R_env [erg/cm3]", U_rad)
U_CMB = 4.1725e-13*(1+z_LRD)**4
rec("U_CMB(z=5) [erg/cm3]", U_CMB)
rec("U_rad/U_CMB", U_rad/U_CMB)

# ---------------------------------------------------------------- the four sites
print("\n"+"="*74); print("2.  FOUR CANDIDATE ACCELERATION SITES"); print("="*74)
sites = {}

# -- (i) jet / polar funnel  ------------------------------------------------
L_j, Gam_j, eB_j, r_dis = 10**44.6, 2.0, 0.01, 1e16
U_B_jet = eB_j*L_j/(4*np.pi*r_dis**2*Gam_j**2*c)
B_jet   = np.sqrt(8*np.pi*U_B_jet)
sites['jet'] = dict(B=B_jet, U_B=U_B_jet, U_rad=U_rad/Gam_j**2, R=r_dis,
                    L_mech=L_j, s=2.0, v_over_c=1.0)
print(f"(i)   jet funnel : B'={B_jet:.3g} G  U_B={U_B_jet:.3g}  (isotropic-equivalent;"
      f" Kuze+2026 quote 1.3e2 G for a collimated funnel)")

# -- (ii) fast wind driven into the dense envelope --------------------------
v_w   = 3.0e8                       # 3000 km/s
L_kin = 1e43                        # GLIMPSED (Korber+2026)
U_B_w = 0.01*L_kin/(4*np.pi*R_env**2*v_w)
B_w   = np.sqrt(8*np.pi*U_B_w)
c_s   = np.sqrt(5/3*kB*1e4/(0.6*1.6726e-24))
Mach  = v_w/c_s
r_c   = (5/3+1)*Mach**2/((5/3-1)*Mach**2+2)
s_w   = (r_c+2)/(r_c-1)
sites['wind'] = dict(B=B_w, U_B=U_B_w, U_rad=U_rad, R=R_env,
                     L_mech=L_kin, s=s_w, v_over_c=v_w/c)
print(f"(ii)  wind shock : B={B_w:.3g} G  U_B={U_B_w:.3g}  c_s={c_s/1e5:.1f} km/s"
      f"  M={Mach:.1f}  r={r_c:.3f}  s={s_w:.3f}  alpha={(1-s_w)/2:.3f}")

# -- (iii) accretion-disc corona -------------------------------------------
R_cor = 10*R_g
U_rad_cor = L_bol/(4*np.pi*R_cor**2*c)
B_cor = 100.0
sites['corona'] = dict(B=B_cor, U_B=B_cor**2/(8*np.pi), U_rad=U_rad_cor,
                       R=R_cor, L_mech=0.1*L_bol, s=2.0, v_over_c=0.3)
print(f"(iii) corona     : R=10 R_g={R_cor:.3g} cm  U_rad={U_rad_cor:.3g} erg/cm3"
      f"  U_rad/U_B={U_rad_cor/(B_cor**2/(8*np.pi)):.3g}")

# -- (iv) host-galaxy supernovae -------------------------------------------
B_sn, R_sn = 1e-4, 300*3.086e18      # 100 uG, R_eff < 300 pc (Roy+2026)
SFR = 30.0                           # Msun/yr, SKAO paper's probe range 5-50
U_rad_sn = (SFR*2.2e43)/(4*np.pi*R_sn**2*c)   # ~2.2e43 erg/s per Msun/yr bolometric
sites['sne'] = dict(B=B_sn, U_B=B_sn**2/(8*np.pi), U_rad=U_rad_sn+U_CMB,
                    R=R_sn, L_mech=SFR*1e42, s=2.2, v_over_c=1e4*1e5/c)
print(f"(iv)  host SNe   : B={B_sn*1e6:.0f} uG  R={R_sn/3.086e18:.0f} pc"
      f"  U_rad={U_rad_sn:.3g}  U_B={B_sn**2/(8*np.pi):.3g}")

print("\n  site      B[G]     U_B[erg/cm3] U_rad/U_B   E_max,Hillas[eV]  E_max,cool[eV]"
      "  f_sync")
for nm, S in sites.items():
    Ut = S['U_B']+S['U_rad']
    E_H = e_es*S['B']*S['R']*min(S['v_over_c'],1.0)*erg_eV
    E_c = mec2*np.sqrt(3*e_es*S['B']/(4*1.0*sT*Ut))*erg_eV     # eta=1
    f_s = S['U_B']/Ut
    S.update(E_Hillas=E_H, E_cool=E_c, f_sync=f_s)
    print(f"  {nm:<8s} {S['B']:<8.3g} {S['U_B']:<12.3g} {S['U_rad']/S['U_B']:<11.3g}"
          f" {E_H:<17.4g} {E_c:<15.4g} {f_s:.3g}")
OUT['sites'] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sites.items()}

# ------------------------------------------------- 3.  radio bound on L_e
print("\n"+"="*74); print("3.  WHAT THE RADIO NON-DETECTIONS ACTUALLY BOUND"); print("="*74)
D_L = Planck18.luminosity_distance(z_LRD).to(u.cm).value
rec("D_L(z=5) [cm]", D_L); rec("D_L(z=5) [Mpc]", D_L/3.0857e24)
S_lim, nu_obs, alpha_r = 11e-6*1e-23, 3.0e9, -0.7          # Perger+2025 VLASS 3 sigma
L_nu = 4*np.pi*D_L**2*S_lim/(1+z_LRD)**(1+alpha_r)
rec("L_nu(rest 18 GHz) [erg/s/Hz]", L_nu)
L_nu_1p4 = L_nu*(1.4e9/((1+z_LRD)*nu_obs))**alpha_r
rec("L_1.4GHz rest [erg/s/Hz]", L_nu_1p4)
rec("L_1.4GHz rest [W/Hz]", L_nu_1p4*1e-7)
# integrate nu L_nu over 0.1-100 GHz rest to get the synchrotron luminosity bound
nu1, nu2 = 1e8, 1e11
L_sync_lim = L_nu*( (nu2**(1+alpha_r)-nu1**(1+alpha_r))/(1+alpha_r) )/((1+z_LRD)*nu_obs)**alpha_r
rec("L_sync limit (0.1-100 GHz) [erg/s]", L_sync_lim)
rec("  as fraction of L_bol", L_sync_lim/L_bol)
print("\n  IC de-boosting: the electron power implied by a synchrotron bound is")
print("  L_e = L_sync * (1 + U_rad/U_B)  -- the bound WEAKENS by that factor:")
for nm in ('jet', 'wind', 'sne'):
    S = sites[nm]; boost = 1+S['U_rad']/S['U_B']
    print(f"   {nm:<6s} 1+U_rad/U_B = {boost:<10.4g} -> L_e <= {L_sync_lim*boost:.4g} erg/s"
          f"  ({L_sync_lim*boost/L_bol:.3g} L_bol)")
OUT['L_sync_lim'] = float(L_sync_lim); OUT['D_L'] = float(D_L)
OUT['L_nu_1p4'] = float(L_nu_1p4)

# ------------------------------------------- 4.  forward (energetic) estimate
print("\n"+"="*74); print("4.  FORWARD ESTIMATE OF L_e  (epsilon_e x L_mech)"); print("="*74)
for eps_e in (0.01, 0.05, 0.1):
    print(f"  eps_e={eps_e:<5g} ", end="")
    for nm in ('jet', 'wind', 'sne'):
        print(f"{nm}: {eps_e*sites[nm]['L_mech']:.3g}  ", end="")
    print("erg/s")
eps_e_fid = 0.01
L_e_fid = eps_e_fid*sites['wind']['L_mech']
rec("fiducial L_e (wind, eps_e=0.01)", L_e_fid, "erg/s")
OUT['L_e_fid'] = float(L_e_fid)

# ---------------------- 5.  normalisation K of N(gamma) = K gamma^-s
print("\n"+"="*74); print("5.  NORMALISATION K OF N(gamma)=K gamma^-s"); print("="*74)
gmin, gmax_ = 2.0, sites['wind']['E_cool']/(mec2*erg_eV)
V = 4*np.pi/3*R_env**3
t_dyn = R_env/v_w
rec("gamma_max (wind, cooling-limited)", gmax_)
rec("t_dyn = R/v_w [s]", t_dyn)
for s_ in (2.0, sites['wind']['s'], 2.2):
    if abs(s_-2) < 1e-9:
        I = np.log(gmax_/gmin)
    else:
        I = (gmax_**(2-s_)-gmin**(2-s_))/(2-s_)
    K_ = L_e_fid*t_dyn/(V*mec2*I)
    print(f"  s={s_:<6.3f} Int={I:<11.4g} K={K_:<12.4g} cm^-3   "
          f"n_e,CR={K_*(gmin**(1-s_)-gmax_**(1-s_))/(s_-1):.4g} cm^-3")
    if abs(s_-2) < 1e-9: OUT['K_s2'] = float(K_)

# ------------------------------------------------- 6.  IGM link
print("\n"+"="*74); print("6.  IGM LINK VIA THE CASCADE YIELD TABLE"); print("="*74)
d = np.load('/home/byaku/Desktop/Doctorado-Trabajo/Notebooks_vClaude/cascade_traj_table.npz')
Kt, zt, Yt = d['K'], d['z'], d['Y']
iz = int(np.argmin(np.abs(zt-7.0)))
print(f"  using the z_i = {zt[iz]} row of cascade_traj_table.npz (LRD epoch)")
lgK = np.log10(Kt); lgY = np.log10(np.where(Yt[iz] > 0, Yt[iz], 1e-300))
Kf = np.geomspace(1e2, 1e12, 4001)
Yf = 10**np.interp(np.log10(Kf), lgK, lgY)
def Wbar(p, Kmin=1e2, Kmax=1e12):
    m = (Kf >= Kmin) & (Kf <= Kmax); x, y = Kf[m], Yf[m]
    num = np.trapz(x**(1-p)*x, np.log(x))      # Int K Q dK,  Q ~ K^-p
    den = np.trapz(y*x**(-p)*x, np.log(x))     # Int Y Q dK
    return num/den
print("     p      Wbar [eV/ionization]")
for p in (1.8, 2.0, 2.2, 2.5, 3.0):
    print(f"   {p:<6.2f} {Wbar(p):.1f}")
W_fid = Wbar(2.0)
OUT['Wbar_s2_z7'] = float(W_fid)

n_H_com = 1.944e-7                       # cm^-3 comoving (manuscript fiducial)
Mpc = 3.0857e24
t_era = 800e6*3.156e7                    # z=7 -> z=4, ~800 Myr
print(f"\n  comoving n_H = {n_H_com:.3g} cm^-3 ; LRD era t = {t_era/3.156e7/1e6:.0f} Myr")
print("\n   n_LRD[cMpc^-3]  L_e[erg/s]  f_esc   eps_perH[eV]  N_ion/H     dT[K]")
Theta_lo, Theta_hi = 4.31e4, 6.89e4      # the sec 9.4 lock, K per (ionisation/H)
rows = []
for n_LRD in (1e-5, 1e-4):
    for L_e in (L_e_fid, 0.1*sites['jet']['L_mech']):
        for f_esc in (0.1, 1.0):
            u_CR = n_LRD*L_e*t_era/Mpc**3          # erg/cm3 comoving
            eps = f_esc*u_CR/n_H_com*erg_eV        # eV per H
            Nion = eps/W_fid
            dT_lo, dT_hi = Nion*Theta_lo, Nion*Theta_hi
            rows.append((n_LRD, L_e, f_esc, eps, Nion, dT_lo, dT_hi))
            print(f"   {n_LRD:<15.3g} {L_e:<11.3g} {f_esc:<7.2g} {eps:<13.4g}"
                  f" {Nion:<11.4g} {dT_lo:.3g}-{dT_hi:.3g}")
OUT['igm_rows'] = [[float(x) for x in r] for r in rows]
T_ad7 = 0.0208*(1+7.0)**2
rec("\nT_ad(z=7) [K]", T_ad7)
rec("Planck sigma(tau)", 0.007)

# escape: column depth of the envelope
print("\n"+"="*74); print("7.  CAN THE ELECTRONS ESCAPE?"); print("="*74)
for NH in (1e24, 1e25):
    grammage = NH*1.6726e-24
    dE = 2.0e6*grammage                  # ~2 MeV per g/cm2 minimum ionising
    print(f"   N_H={NH:.0g} cm^-2 -> {grammage:.3g} g/cm2 -> collisional loss"
          f" ~{dE/1e6:.2g} MeV : electrons above ~{10*dE/1e6:.1f} MeV traverse it")

# ---------------------------------- 8. the IC channel and the X-ray limits
print("\n"+"="*74); print("8.  WHERE THE ELECTRON POWER ACTUALLY EMERGES"); print("="*74)
kT_ph = 5000*kB                                  # quasi-star photosphere, Gentile+2026
eps_seed = 2.821*kT_ph*erg_eV                    # peak of a 5000 K blackbody, eV
rec("seed photon energy (5000 K peak) [eV]", eps_seed)
g_lo, g_hi = 2.0, gmax_
rec("IC photon energy at gamma=2 [eV]", 4/3*g_lo**2*eps_seed)
E_KN = mec2*erg_eV                               # KN turnover: eps_gamma ~ gamma me c^2
rec("KN-limited max IC photon [eV]", g_hi*E_KN)
ndec = np.log10(g_hi*E_KN/(4/3*g_lo**2*eps_seed))
rec("decades spanned by the IC continuum", ndec)
f_2_10 = np.log10(10e3/2e3)/ndec                 # flat nu F_nu, s=2 fast cooling
rec("fraction of L_e in the 2-10 keV band", f_2_10)
for nm, L_e in (('wind eps_e=0.01', L_e_fid), ('jet eps_e=0.1', 0.1*sites['jet']['L_mech'])):
    L_IC = L_e*sites['wind']['U_rad']/(sites['wind']['U_rad']+sites['wind']['U_B'])
    print(f"   {nm:<16s} L_e={L_e:.3g} -> L_IC={L_IC:.3g} -> L_2-10keV="
          f"{f_2_10*L_IC:.3g} erg/s")
L_X_stack = 10**41.5                              # Maiolino+2024 stacked 3 sigma
L_X_single = 10**42.5
rec("Chandra stacked limit L_2-10 [erg/s]", L_X_stack)
print(f"   fiducial (wind) is {L_X_stack/(f_2_10*L_e_fid):.3g}x below the stacked X-ray limit")
print(f"   aggressive (jet) is {f_2_10*0.1*sites['jet']['L_mech']/L_X_stack:.3g}x ABOVE it")
print("   -> but a Compton-thick envelope (N_H = 1e24-1e25 cm^-2) absorbs 2-10 keV,")
print("      so neither the radio nor the X-ray non-detection is a clean bound.")
OUT['f_2_10'] = float(f_2_10); OUT['L_X_stack'] = float(L_X_stack)

# ------------------------- 9.  cross-check against manuscript Table 3
print("\n"+"="*74); print("9.  CROSS-CHECK: Wbar REPRODUCES manuscript Table 3"); print("="*74)
ref = {1.8: (888, 804), 2.0: (107, 102), 2.2: (46.3, 45.3), 2.5: (37.8, 37.6), 3.0: (37.4, 37.4)}
def Wb(iz, p):
    lY = np.log10(np.where(Yt[iz] > 0, Yt[iz], 1e-300))
    YY = 10**np.interp(np.log10(Kf), np.log10(Kt), lY)
    return np.trapz(Kf**(1-p)*Kf, np.log(Kf))/np.trapz(YY*Kf**(-p)*Kf, np.log(Kf))
print("   p     z=20    z=10    z=7    | manuscript Table 3 (z=20 / z=10)")
worst = 0.0
for p, (r20, r10) in ref.items():
    a, b, cc = Wb(7, p), Wb(3, p), Wb(1, p)
    worst = max(worst, abs(a-r20)/r20, abs(b-r10)/r10)
    print(f"   {p:<5.1f} {a:<7.1f} {b:<7.1f} {cc:<6.1f} | {r20} / {r10}")
print(f"   max relative deviation from the published table: {worst*100:.2f} %")
assert worst < 0.005

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'lrd_numbers.json'), 'w'), indent=1)
print("\nwrote lrd_numbers.json")
