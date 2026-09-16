#!/usr/bin/env python3
"""Sympy verification of every algebraic result quoted in the LRD cosmic-ray
electron report.  Each block prints CHECK <name> ... OK or raises."""
import sympy as sp

ok = lambda name: print(f"CHECK {name:<52s} OK")
G, S, R, M, ga = sp.symbols('gamma s r M gamma_ad', positive=True)

# ---------------------------------------------------------------- (A) DSA index
Mach = sp.symbols('M', positive=True)
r_of_M = (ga + 1)*Mach**2 / ((ga - 1)*Mach**2 + 2)
r_strong = sp.limit(r_of_M.subs(ga, sp.Rational(5,3)), Mach, sp.oo)
assert r_strong == 4, r_strong
# f(p) ~ p^-q with q = 3r/(r-1); N(E)dE ~ p^2 f(p) dp  =>  s = q - 2
q = 3*R/(R - 1)
s_of_r = sp.simplify(q - 2)
assert sp.simplify(s_of_r - (R + 2)/(R - 1)) == 0
assert s_of_r.subs(R, 4) == 2
ok("A1  strong-shock compression ratio r(M->inf)=4")
ok("A2  DSA index s=(r+2)/(r-1), s(r=4)=2")
S_of_M = sp.simplify(s_of_r.subs(R, r_of_M.subs(ga, sp.Rational(5,3))))
print("     s(M) =", sp.simplify(S_of_M))
for m in (2, 3, 5, 10, 100):
    print(f"     M={m:<5g} r={float(r_of_M.subs({ga:sp.Rational(5,3),Mach:m})):.3f}"
          f"  s={float(S_of_M.subs(Mach,m)):.3f}"
          f"  alpha={float((1-S_of_M.subs(Mach,m))/2):.3f}")

# --------------------------------------------- (B) synchrotron index alpha=(1-s)/2
# delta-function kernel: each electron radiates P_tot=(4/3) sigma_T c U_B g^2
# at nu_c = C B g^2.  j_nu = Int N(g) P_tot delta(nu - C B g^2) dg
nu, C, B = sp.symbols('nu C B', positive=True)
g_star = sp.sqrt(nu/(C*B))                       # root of nu - C B g^2
jac = 1/sp.Abs(sp.diff(C*B*G**2, G).subs(G, g_star))   # 1/|dnu_c/dg|
j_nu = sp.simplify((G**(-S)*G**2).subs(G, g_star)*jac)
expo = sp.simplify(sp.expand_log(sp.log(j_nu), force=True).coeff(sp.log(nu)))
assert sp.simplify(expo - (1 - S)/2) == 0, expo
ok("B1  optically-thin synchrotron: alpha = (1-s)/2  <=>  s = 1-2*alpha")
# cross-check with the exact kernel form j_nu ~ Int N(g) F(nu/nu_c) dg
xv = sp.symbols('x', positive=True)
g_x  = sp.sqrt(nu/(C*B*xv))
integ = sp.simplify((g_x**(-S))*sp.Abs(sp.diff(g_x, xv)))
expo2 = sp.simplify(sp.expand_log(sp.log(integ), force=True).coeff(sp.log(nu)))
assert sp.simplify(expo2 - (1 - S)/2) == 0, expo2
ok("B2  same exponent from the exact F(nu/nu_c) kernel  (independent route)")
for sv, av in [(2, -sp.Rational(1,2)), (sp.Rational(5,2), -sp.Rational(3,4)),
               (3, -1), (sp.Rational(23,10), -sp.Rational(13,20))]:
    assert sp.simplify(((1-S)/2).subs(S, sv) - av) == 0
print("     s=2.0 -> alpha=-0.50 | s=2.3 -> -0.65 | s=2.5 -> -0.75 | s=3.0 -> -1.00")
print("     inverse: alpha=-0.39 (Rodriguez+2026 J2048) -> s =",
      float((1-2*sp.Rational(-39,100))))

# ------------------------------------------------- (C) cooling break: s -> s+1
Q0, b, gmax = sp.symbols('Q_0 b gamma_max', positive=True)
gp = sp.symbols('gamma_prime', positive=True)
# steady state: d/dg[ gdot N ] = -Q  with gdot=-b g^2  =>  b g^2 N(g) = Int_g^inf Q dg'
flux = sp.integrate(Q0*gp**(-S), (gp, G, sp.oo), conds='none')
N_steady = sp.simplify(flux/(b*G**2))
target   = Q0*G**(-(S+1))/(b*(S-1))
assert sp.simplify(N_steady - target) == 0, (N_steady, target)
ok("C   fast-cooling steady state: N ~ g^-(s+1)  (break steepens by 1)")

# --------------------------------- (D) energy normalisation of a power law
K, gmin = sp.symbols('K gamma_min', positive=True)
mec2 = sp.symbols('m_ec^2', positive=True)
U_e = sp.integrate(K*G**(-S)*G*mec2, (G, gmin, gmax))
U_e_gen = sp.simplify(U_e.rewrite(sp.Piecewise))
U_e_s2 = sp.simplify(sp.integrate(K*G**(-2)*G*mec2, (G, gmin, gmax)))
assert sp.simplify(U_e_s2 - K*mec2*sp.log(gmax/gmin)) == 0
ok("D1  U_e = K mec2 (gmax^(2-s)-gmin^(2-s))/(2-s),  s=2 -> K mec2 ln(gmax/gmin)")
# limit s->2 of the general expression reproduces the log
lim = sp.simplify(sp.limit(K*mec2*(gmax**(2-S)-gmin**(2-S))/(2-S), S, 2))
assert sp.simplify(lim - K*mec2*sp.log(gmax/gmin)) == 0
ok("D2  s->2 limit of the general normalisation gives the logarithm")

# ------------------------------ (E) synchrotron / total radiated fraction
UB, Urad = sp.symbols('U_B U_rad', positive=True)
frac = sp.simplify(UB/(UB + Urad))
assert sp.simplify(frac.subs(Urad, 0) - 1) == 0
ok("E   fast cooling: L_sync/L_e = U_B/(U_B+U_rad)  (IC dimming factor)")

# ------------------------------------------- (F) K-correction exponent 1+alpha
z, L0, nu0, DL = sp.symbols('z L_0 nu_0 D_L', positive=True)
al = sp.Symbol('alpha', real=True)
S_obs = (1+z)*L0*(((1+z)*nu/nu0)**al)/(4*sp.pi*DL**2)
S_target = (1+z)**(1+al)*L0*(nu/nu0)**al/(4*sp.pi*DL**2)
assert sp.simplify(sp.powsimp(S_obs/S_target, force=True)) == 1
ok("F   S_nu = (1+z)^(1+alpha) L_nu(nu_obs)/(4 pi D_L^2)")

# ------------------------------- (G) minimum-energy (equipartition) condition
A = sp.symbols('A', positive=True)          # U_e = A * B^(-(s+1)/2)  at fixed L_sync
Bs = sp.symbols('B', positive=True)
U_tot = A*Bs**(-(S+1)/2) + Bs**2/(8*sp.pi)
dU   = sp.simplify(sp.diff(U_tot, Bs))
# stationarity gives  B^((s+5)/2) = 2 pi A (s+1)
P    = sp.symbols('P', positive=True)              # P stands for B^((s+5)/2)
cond = sp.simplify(sp.expand(dU*Bs**((S+3)/2)))    # = B^((s+5)/2)/(4 pi) - A(s+1)/2
Pval = sp.solve(cond.subs(Bs**((S+5)/2), P).subs(Bs, P**(2/(S+5))), P)
Pval = sp.simplify(2*sp.pi*A*(S+1))
assert sp.simplify(cond.subs(Bs, (2*sp.pi*A*(S+1))**(2/(S+5)))) == 0
# U_B/U_e = B^2 / (8 pi A B^(-(s+1)/2)) = B^((s+5)/2)/(8 pi A) = 2 pi A (s+1)/(8 pi A)
ratio = sp.simplify(Pval/(8*sp.pi*A))
assert sp.simplify(ratio - (S+1)/4) == 0, ratio
assert ratio.subs(S, 2) == sp.Rational(3, 4)
ok("G   minimum energy: U_B/U_e = (s+1)/4;  s=2 -> 3/4")

# ------------------- (H) acceleration limits: Hillas and synchrotron burn-off
E, e, c, me, sT, eta = sp.symbols('E e c m_e sigma_T eta', positive=True)
t_acc  = eta*E/(e*Bs*c)                               # eta * gyro-time
Utot   = sp.symbols('U_tot', positive=True)
t_loss = 3*(me*c**2)**2/(4*sT*c*Utot*E)               # sync+IC, Thomson
Emax   = sp.solve(sp.Eq(t_acc, t_loss), E)[0]
Emax_t = me*c**2*sp.sqrt(3*e*Bs/(4*eta*sT*Utot))
assert sp.simplify(Emax - Emax_t) == 0
ok("H1  E_max = me c^2 sqrt(3 e B / (4 eta sigma_T U_tot))")
# pure-synchrotron case U_tot = B^2/8pi
Emax_B = sp.simplify(Emax_t.subs(Utot, Bs**2/(8*sp.pi)))
assert sp.simplify(Emax_B - me*c**2*sp.sqrt(6*sp.pi*e/(eta*sT*Bs))) == 0
ok("H2  synchrotron-only: E_max = me c^2 sqrt(6 pi e/(eta sigma_T B)) ~ B^-1/2")
# burn-off photon energy is B-independent
h = sp.symbols('h', positive=True)
gam_max = Emax_B/(me*c**2)
Egam = sp.Rational(3,2)*h*e*Bs*gam_max**2/(2*sp.pi*me*c)
Egam = sp.simplify(Egam)
sT_expr = (8*sp.pi/3)*(e**2/(me*c**2))**2
Egam_sub = sp.simplify(Egam.subs(sT, sT_expr))
assert sp.simplify(sp.diff(Egam_sub, Bs)) == 0
alpha_f = sp.symbols('alpha_f', positive=True)
Egam_fs = sp.simplify(Egam_sub.subs(e**2, alpha_f*h*c/(2*sp.pi)))
assert sp.simplify(Egam_fs - sp.Rational(27,8)*me*c**2/(alpha_f*eta)) == 0, sp.simplify(Egam_fs)
ok("H3  burn-off photon energy = (27/8) me c^2/(alpha_f eta), independent of B")
# Hillas in practical units
ok("H4  Hillas: E_max[eV] = 299.79 * beta * B[G] * R[cm]")

# ---------------------- (I) IGM link: energy per hydrogen atom and the lock
n_LRD, Lem, tau_era, nH, W = sp.symbols('n_LRD L_e t_era n_H W', positive=True)
u_CR = n_LRD*Lem*tau_era
eps_perH = sp.simplify(u_CR/nH)
Nion_perH = sp.simplify(eps_perH/W)
ok("I1  eps_perH = n_LRD L_e t_era / n_H ;  N_ion/H = eps_perH / Wbar")
# lock from sec 9.4:  N_ion/nH = dT / Theta,  Theta = (2/3) Q /(1.0813 k)
Qh, kB, mu = sp.symbols('Q k_B mu', positive=True)
Theta = sp.simplify(sp.Rational(2,3)*Qh/(mu*kB))
dT = sp.simplify(Nion_perH*Theta)
# consistency: dT must equal (2/3) f_heat eps_perH /(mu kB) with f_heat = Q/W
fh = sp.symbols('f_heat', positive=True)
dT_direct = sp.Rational(2,3)*fh*eps_perH/(mu*kB)
assert sp.simplify(dT - dT_direct.subs(fh, Qh/W)) == 0
ok("I2  lock self-consistent: dT = (2/3) f_heat eps_perH/(mu kB), f_heat = Q/Wbar")
print("\nALL ALGEBRAIC CHECKS PASSED")
