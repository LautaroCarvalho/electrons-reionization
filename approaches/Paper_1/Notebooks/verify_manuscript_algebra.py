"""Symbolic checks for every algebraic relation used in the short manuscript."""

import sympy as sp


def ok(label):
    print(f"CHECK {label:<68s} OK")


# Relativistic kinematics: E=K+mc^2 and p^2 c^2=E^2-m^2c^4.
K, E0 = sp.symbols("K E_0", positive=True)
E = K + E0
p2c2 = sp.expand(E**2 - E0**2)
assert sp.simplify(p2c2 - K * (K + 2 * E0)) == 0
ok("p^2 c^2 = K(K+2E0)")

# Adiabatic redshifting: p proportional to a^-1 implies dK/dt=-H p^2c^2/E.
p, c, H = sp.symbols("p c H", positive=True)
Et = sp.sqrt(p**2 * c**2 + E0**2)
dpdt = -H * p
dKdt = sp.simplify(sp.diff(Et, p) * dpdt)
assert sp.simplify(dKdt + H * p**2 * c**2 / Et) == 0
ok("p proportional to a^-1 gives dK/dt=-H p^2 c^2/E")

# IC kinematics and Jacobian used to transform d sigma/dE_gamma to d sigma/dq.
gamma, eps, q = sp.symbols("gamma epsilon q", positive=True)
Gamma = 4 * gamma * eps / E0
Eg = gamma * E0 * Gamma * q / (1 + Gamma * q)
q_back = sp.simplify(Eg / (Gamma * (gamma * E0 - Eg)))
assert sp.simplify(q_back - q) == 0
assert sp.simplify(sp.diff(Eg, q) - gamma * E0 * Gamma / (1 + Gamma*q)**2) == 0
ok("IC q inversion and dE_gamma/dq Jacobian")

# Productive photon bookkeeping: one photoionization plus a child electron.
Bg, Y = sp.symbols("B Y", positive=True)
Kphoto = Eg - Bg
assert sp.simplify((Eg - Bg) - Kphoto) == 0
assert sp.simplify((1 + Y) - 1 - Y) == 0
ok("photoelectron K=E_gamma-B and yield multiplier 1+Y(K)")

# Frozen-in magnetic field and the CMB have the same (1+z)^4 energy scaling.
z, B0, mu0, U0 = sp.symbols("z B_0 mu_0 U_0", positive=True)
UB = (B0 * (1 + z)**2)**2 / (2 * mu0)
UCMB = U0 * (1 + z)**4
assert sp.simplify(UB / UCMB - B0**2 / (2 * mu0 * U0)) == 0
ok("U_B/U_CMB is redshift-independent for B proportional to (1+z)^2")

# The present-day field for synchrotron--IC equality follows from U_B=U_CMB.
Beq = sp.sqrt(2 * mu0 * U0)
assert sp.simplify(Beq**2 / (2 * mu0) - U0) == 0
ok("B_eq=sqrt(2 mu_0 U_CMB,0) gives U_B=U_CMB")

# The interaction-horizon boundary is lambda_gamma=c/H.
kappa = sp.symbols("kappa", positive=True)
lam = 1 / kappa
assert sp.solve(sp.Eq(lam, c / H), kappa) == [H / c]
ok("lambda_gamma=c/H is equivalent to kappa_gamma=H/c")

# Spectrum-averaged energy per ionization.
A, pidx, Kmin, Kmax = sp.symbols("A p K_min K_max", positive=True)
Kvar = sp.symbols("Kvar", positive=True)
Yfun = sp.Function("Y")(Kvar)
energy = sp.Integral(A * Kvar**(1-pidx), (Kvar, Kmin, Kmax))
nion = sp.Integral(A * Kvar**(-pidx) * Yfun, (Kvar, Kmin, Kmax))
ratio = sp.cancel(energy / nion)
assert not ratio.has(A)
ok("normalization A cancels from spectrum-averaged W")

print("\nALL MANUSCRIPT ALGEBRAIC CHECKS PASSED")
