"""Internal consistency checks for stromgren_sphere_summary.tex.
Every check corresponds to an equation or number quoted in the summary."""
import sympy as sp, numpy as np

print("="*72); print("1. Analytic solution of the classical I-front equation"); print("="*72)
t, trec, RS = sp.symbols('t t_rec R_S', positive=True)
Ndot, n, aB, C = sp.symbols('Ndot n alpha_B C', positive=True)
R = sp.Function('R')
# dR^3/dt = 3 Ndot/(4 pi n) - aB C n R^3   (static, uniform, Cen&Haiman eq.3 without Hubble term)
ode = sp.Eq(sp.diff(R(t)**3, t), 3*Ndot/(4*sp.pi*n) - aB*C*n*R(t)**3)
trial = (3*Ndot/(4*sp.pi*aB*C*n**2))**sp.Rational(1,3)*(1-sp.exp(-aB*C*n*t))**sp.Rational(1,3)
res = sp.simplify(ode.lhs.subs(R(t), trial).doit() - ode.rhs.subs(R(t), trial))
print("  residual of R(t)=R_S(1-e^{-t/t_rec})^{1/3} :", res, " -> OK" if res==0 else " -> FAIL")
print("  t_rec identified as 1/(alpha_B C n)  [Shapiro+2006 eq. w = t/t_rec = t C aB nH]")

print("="*72); print("2. Numerical Stromgren radius (Galactic H II region)"); print("="*72)
aB_val = 2.59e-13          # cm^3/s, T=1e4 K, Case B  (Shapiro+2006)
pc = 3.0857e18
for Nd, nH in [(1e49,100.),(1e49,1.),(1e50,100.)]:
    Rs = (3*Nd/(4*np.pi*aB_val*nH**2))**(1/3)
    print(f"  Ndot={Nd:.0e} s^-1, nH={nH:g} cm^-3 -> R_S = {Rs/pc:8.3f} pc,  t_rec = {1/(aB_val*nH)/3.156e7/1e3:8.3f} kyr")

print("="*72); print("3. Stromgren (1939): 200 pc diameter for an O star"); print("="*72)
Rs = (3*1e49/(4*np.pi*aB_val*1.0**2))**(1/3)/pc
print(f"  Ndot=1e49 s^-1, nH=1 cm^-3 -> D = 2R_S = {2*Rs:.0f} pc  (Stromgren 1939 quotes ~200 pc)")

print("="*72); print("4. Spitzer D-type solution solves the Raga-I equation at early times"); print("="*72)
ci, RSt = sp.symbols('c_i R_St', positive=True)
RSp = RSt*(1 + sp.Rational(7,4)*ci*t/RSt)**sp.Rational(4,7)
lhs = sp.diff(RSp, t)/ci
rhs = (RSt/RSp)**sp.Rational(3,4)          # Bisbas+2015 eq.(8) with the small T_o term dropped
print("  d(R_Sp)/dt/c_i - (R_St/R_Sp)^{3/4} =", sp.simplify(lhs-rhs))

print("="*72); print("5. Stagnation radius (Bisbas+2015 eq.14) vs their quoted numbers"); print("="*72)
kB, mH = 1.380649e-16, 1.6726e-24
c_i = np.sqrt(kB*1e4/(0.5*mH))/1e5; c_o = np.sqrt(kB*1e3/(1.0*mH))/1e5
print(f"  c_i(T=1e4 K, mu=0.5) = {c_i:.2f} km/s   c_o(T=1e3 K, mu=1) = {c_o:.2f} km/s (paper: 12.85, 2.87)")
print(f"  R_stag = (c_i/c_o)^(4/3) R_St = {(c_i/c_o)**(4/3)*0.314:.2f} pc for R_St=0.314 pc (paper: 2.31 pc)")

print("="*72); print("6. Cosmological photon-counting radius vs Mesinger & Haiman (2004) scaling"); print("="*72)
# their cosmology: Omega_b=0.044, h=0.71, Y=0.24
h, Ob, Y = 0.71, 0.044, 0.24
rho_c = 1.8788e-29*h**2                      # g/cm^3
nH0 = Ob*rho_c*(1-Y)/mH                      # comoving/present-day mean H density
z, Nd, tQ, xHI = 6.28, 6.5e57, 2e7*3.156e7, 1.0
nH = nH0*(1+z)**3
Rp = (3*Nd*tQ/(4*np.pi*nH*xHI))**(1/3)/3.0857e24
print(f"  n_H(z=6.28) = {nH:.3e} cm^-3")
print(f"  R = [3 Ndot t /(4 pi n_H x_HI)]^(1/3) = {Rp:.2f} proper Mpc   (Mesinger&Haiman quote 7.7 Mpc)")
print(f"  comoving = {Rp*(1+z):.1f} cMpc ; their best fit R_S = 44 cMpc -> {44/(1+z):.2f} pMpc (abstract: 6.0+-0.2)")

print("="*72); print("7. Recombination time in the IGM vs the Hubble time at z=7"); print("="*72)
H0 = h*100/3.0857e19
Om, OL = 0.27, 0.73
for zz in [6.,8.,10.]:
    nn = nH0*(1+zz)**3
    trec = 1/(aB_val*nn*1.08)                # chi_He=1.08, C=1
    Hz = H0*np.sqrt(Om*(1+zz)**3+OL)
    print(f"  z={zz:4.1f}: t_rec(C=1) = {trec/3.156e7/1e9:6.2f} Gyr ; 1/H(z) = {1/Hz/3.156e7/1e9:5.3f} Gyr ; ratio = {trec*Hz:5.1f}")

print("="*72); print("8. Luminosity scaling of proximity zones"); print("="*72)
print(f"  naive photon-counting: R ∝ L^(1/3) = L^{1/3:.3f}")
print(f"  Onorato+ fit: R_p ∝ 10^(-0.4 M1450)/2.87 -> R_p ∝ L^{1/2.87:.3f}")

print("="*72); print("9. I-front speed regimes"); print("="*72)
print(f"  3000 km/s = {3000*1e5:.0e} cm/s (Zhu+2025 plateau) ; Zeng&Hirata equilibrium limit 1e9 cm/s = {1e9/1e5:.0f} km/s")
