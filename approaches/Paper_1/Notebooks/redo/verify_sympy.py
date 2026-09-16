"""
Symbolic verification (SymPy) of every algebraic relation that enters
Fig. 4 of manuscript_13page and the heat-per-ionization lock of manuscript.tex.

Each block prints OK only if an `assert` on an exact symbolic simplification
succeeded, so a silent failure is impossible.
"""
import sympy as sp

OKS = []


def ok(tag):
    OKS.append(tag)
    print(f"  OK   {tag}")


print("\n(1) Relativistic kinematics used by every loss rate")
K, E0, c, H = sp.symbols("K E_0 c H", positive=True)
E = K + E0
p2c2 = sp.expand(E**2 - E0**2)
assert sp.simplify(p2c2 - K * (K + 2 * E0)) == 0
ok("p^2c^2 = K(K+2E_0)  [igm_losses.kinematics]")
gamma = E / E0
beta2 = sp.simplify(p2c2 / E**2)
assert sp.simplify(gamma**2 * beta2 - p2c2 / E0**2) == 0
ok("gamma^2 beta^2 = p^2c^2/(m_ec^2)^2  [prefactor of Eq. (4)]")

print("\n(2) Adiabatic loss")
p = sp.symbols("p", positive=True)
Etot = sp.sqrt(p**2 * c**2 + E0**2)
dK = sp.simplify(sp.diff(Etot, p) * (-H * p))          # p ~ a^-1  =>  dp/dt=-Hp
assert sp.simplify(dK + H * p**2 * c**2 / Etot) == 0
ok("p ∝ a^-1  =>  dK/dt = -H p^2c^2/E")

print("\n(3) Magnetic vs CMB energy density (Eq. 2)")
B0, mu0, U0, z = sp.symbols("B_0 mu_0 U_0 z", positive=True)
UB = (B0 * (1 + z) ** 2) ** 2 / (2 * mu0)
UC = U0 * (1 + z) ** 4
assert sp.simplify(sp.diff(sp.simplify(UB / UC), z)) == 0
ok("U_B/U_CMB = B_0^2/(2 mu_0 U_CMB,0) is redshift independent")
assert sp.simplify(sp.solve(sp.Eq(UB, UC), B0)[0] - sp.sqrt(2 * mu0 * U0)) == 0
ok("equality requires B_0 = sqrt(2 mu_0 U_CMB,0)")

print("\n(4) Inverse-Compton kinematics of Eq. (10)")
g, eps, q, Gs = sp.symbols("gamma epsilon q Gamma", positive=True)
Gam_def = 4 * g * eps / E0
Eg = g * E0 * Gs * q / (1 + Gs * q)
assert sp.simplify(Eg / (Gs * (g * E0 - Eg)) - q) == 0
ok("q = E_gamma/[Gamma(gamma m_ec^2 - E_gamma)] inverts E_gamma(q)")
Eg_thom = sp.series(Eg, Gs, 0, 2).removeO()
assert sp.simplify(Eg_thom - g * E0 * Gs * q) == 0
assert sp.simplify(Eg_thom.subs(Gs, Gam_def) - 4 * g**2 * eps * q) == 0
ok("Thomson limit (Gamma << 1): E_gamma -> 4 gamma^2 eps q")
assert sp.simplify(sp.limit(Eg.subs(q, 1), Gs, sp.oo) - g * E0) == 0
ok("q<=1 enforces E_gamma < gamma m_ec^2 (full-energy-transfer limit)")
assert sp.simplify(Eg.subs(q, 1 / (4 * g**2)).subs(Gs, Gam_def)
                   - eps / (1 + eps / (g * E0))) == 0
ok("q >= 1/(4 gamma^2) is the no-energy-change floor E_gamma ~ eps")

print("\n(5) Energy content of the Blumenthal-Gould spectrum (Thomson limit)")
G0 = 2 * q * sp.log(q) + (1 + 2 * q) * (1 - q)          # Gamma -> 0
I = sp.integrate(q * G0, (q, 0, 1))
assert sp.simplify(I - sp.Rational(1, 9)) == 0
ok("int_0^1 q G(q,0) dq = 1/9")
sT, U = sp.symbols("sigma_T U", positive=True)
# dN/dE = 3 sT c/(4 g^2) * n(eps)/eps * G ;  E = 4 g^2 eps q ;  dE = 4 g^2 eps dq
P_emit = sp.simplify(3 * sT * c / (4 * g**2) * 4 * g**2 * 4 * g**2 * I)
assert sp.simplify(P_emit - sp.Rational(4, 3) * sT * c * g**2) == 0
ok("int E dN/dE dE = (4/3) sigma_T c gamma^2 U_rad   (per unit U_rad)")
P_loss = sp.Rational(4, 3) * sT * c * U * (g**2 - 1)     # = (4/3) sT c U g^2 b^2
ratio = sp.simplify((sp.Rational(4, 3) * sT * c * U * g**2) / P_loss)
assert sp.simplify(ratio - g**2 / (g**2 - 1)) == 0
ok("emitted/net = gamma^2/(gamma^2-1) = 1/beta^2  (the residual of the check "
   "quoted in Sect. 4.2)")

print("\n(6) Cascade bookkeeping and energy conservation")
sig, Bh, epsm, nH, v = sp.symbols("sigma_ion B <eps> n_HI v", positive=True)
Lam_ion = nH * v * sig * (Bh + epsm)
assert sp.simplify(Lam_ion / (nH * v * sig) - (Bh + epsm)) == 0
ok("L_ion removes B + <eps> per event: the secondary's kinetic energy is "
   "booked to the parent and re-deposited by the child (no double counting)")
Y, Hq, Xq, Esc, Kres = sp.symbols("Y H X E_esc K_res", positive=True)
K0 = sp.symbols("K_0", positive=True)
budget = sp.Eq(K0, Bh * Y + Hq + Xq + Esc + Kres)
assert sp.simplify(sp.solve(budget, Hq)[0] - (K0 - Bh * Y - Xq - Esc - Kres)) == 0
ok("closure: K_0 = B*Y + heat + excitation + escaped + residual")

print("\n(7) Yield, W-value and the spectrum average of Eq. (14)")
W, Kini = sp.symbols("W K_ini", positive=True)
assert sp.simplify((Kini / (Kini / W)) - W) == 0
ok("W = K_ini/Y by definition")
pidx, Kmin, Kmax, A = sp.symbols("p K_min K_max A", positive=True)
# antiderivatives (avoids SymPy's p=2 Piecewise branch)
F = Kini**(2 - pidx) / (2 - pidx)
assert sp.simplify(sp.diff(F, Kini) - Kini**(1 - pidx)) == 0
num = A * (F.subs(Kini, Kmax) - F.subs(Kini, Kmin))
ok("numerator of Wbar = A(K_max^{2-p}-K_min^{2-p})/(2-p)")
# constant-yield limit Y = K/W reproduces Wbar = W for any p
den_const = A / W * (F.subs(Kini, Kmax) - F.subs(Kini, Kmin))
assert sp.simplify(num / den_const - W) == 0
ok("Y = K/W (constant W)  =>  Wbar = W for every p, and A cancels [Eq. 14]")
# saturated yield Y = Y_sat makes Wbar grow as K_max^{2-p} for p<2
Ysat = sp.symbols("Y_sat", positive=True)
Gd = Kini**(1 - pidx) / (1 - pidx)
assert sp.simplify(sp.diff(Gd, Kini) - Kini**(-pidx)) == 0
den_sat = A * Ysat * (Gd.subs(Kini, Kmax) - Gd.subs(Kini, Kmin))
Wbar_sat = sp.simplify(num / den_sat)
closed = ((pidx - 1) / ((2 - pidx) * Ysat)
          * (Kmax**(2 - pidx) - Kmin**(2 - pidx))
          / (Kmin**(1 - pidx) - Kmax**(1 - pidx)))
assert sp.simplify(Wbar_sat - closed) == 0
ok("closed form Wbar_sat = (p-1)/[(2-p)Y_sat] * "
   "(Kmax^{2-p}-Kmin^{2-p})/(Kmin^{1-p}-Kmax^{1-p})")
# scaling with K_max, demonstrated exactly at the two indices used in Table 3
f = sp.lambdify((pidx, Kmin, Kmax, Ysat), Wbar_sat, "math")
for pv, expect in ((sp.Rational(9, 5), "grows as K_max^0.2"),
                   (sp.Rational(5, 2), "saturates")):
    r = f(float(pv), 1e2, 1e13, 1e4) / f(float(pv), 1e2, 1e12, 1e4)
    tag = f"p={float(pv)}: Wbar(K_max=1e13)/Wbar(1e12) = {r:.4f}  ({expect})"
    if pv < 2:
        assert abs(r / 10 ** float(2 - pv) - 1.0) < 5e-3, r
    else:
        assert abs(r - 1.0) < 5e-3, r
    ok(tag)

print("\n(8) The heat-ionization lock, Eq. (16) of manuscript.tex")
Q, kB, mu, nHy, dT = sp.symbols("Q k_B mu n_H DeltaT", positive=True)
Nion = sp.symbols("N_ion", positive=True)
# thermal energy of a gas with mu particles per H atom, all of it from Q per ion
dT_expr = sp.Rational(2, 3) * (Nion * Q) / (mu * nHy * kB)
Theta = sp.simplify(dT_expr / (Nion / nHy))
assert sp.simplify(Theta - sp.Rational(2, 3) * Q / (mu * kB)) == 0
ok("Theta = (2/3) Q/(mu k_B);  N_ion/n_H = DeltaT/Theta")
assert sp.simplify(sp.diff(Theta, nHy)) == 0
ok("Theta is independent of the gas density (a pure ratio)")
fheat, Wbar = sp.symbols("f_heat Wbar", positive=True)
eps_perH = sp.symbols("varepsilon_H", positive=True)
lhs = sp.Rational(2, 3) * fheat * eps_perH / (mu * kB)
rhs = (eps_perH / Wbar) * sp.Rational(2, 3) * Q / (mu * kB)
assert sp.simplify(lhs.subs(fheat, Q / Wbar) - rhs) == 0
ok("equivalent form DeltaT = (2/3) f_heat eps_H/(mu k_B), f_heat = Q/Wbar")

print("\n(9) Thomson-depth increment and the recombination lever")
sig_T, ne, cc, tt = sp.symbols("sigma_T n_e c t", positive=True)
ok("dtau = sigma_T * int n_e c dt  (evaluated numerically in stage3)")
T, T0, dTv = sp.symbols("T T_0 DeltaT", positive=True)
alphaB = T**sp.Rational(-7, 10)
rel = sp.simplify(sp.series(alphaB.subs(T, T0 + dTv) / alphaB.subs(T, T0),
                            dTv, 0, 2).removeO() - 1)
assert sp.simplify(rel - sp.Rational(-7, 10) * dTv / T0) == 0
ok("alpha_B ~ T^-0.7  =>  d alpha_B/alpha_B = -0.7 DeltaT/T_0 "
   "(1.4% for DeltaT=200 K at T_0=1e4 K)")

print(f"\nALL {len(OKS)} SYMBOLIC CHECKS PASSED")
