"""
Relative abundance of ionizing photons by energy for the agents of reionization.

For a source with specific emissivity eps_nu ∝ nu^-alpha, the PHOTON emission
rate per unit frequency is  ndot_nu = eps_nu/(h nu) ∝ nu^(-alpha-1), so the
fraction of hydrogen-ionizing photons emitted above an energy E is simply
(E/E_HI)^-alpha.  Derived and checked with sympy below.

Spectral indices, both from the corpus:
  alpha_s = 2.0 +/- 0.6   effective ionizing-source index (Gaikwad et al. 2023,
                          Table 3 caption; used for eps_912 and ndot)
  alpha   = 1.7           AGN power law (Asthana et al. 2024)
Ionization thresholds: H I 13.598 eV, He I 24.587 eV, He II 54.418 eV (NIST).
"""
import sympy as sp

nu, a, n1, n2 = sp.symbols('nu alpha nu_1 nu_2', positive=True)

# photon rate per unit frequency for eps_nu ~ nu^-alpha
ndot_nu = nu**(-a-1)
N_above = sp.integrate(ndot_nu, (nu, n1, sp.oo))          # needs alpha > 0
frac    = sp.simplify(sp.integrate(ndot_nu, (nu, n2, sp.oo)) / N_above)
print("N(>nu1)          =", sp.simplify(N_above))
print("fraction(>nu2)   =", frac, "   ->", sp.simplify(frac.rewrite(sp.Pow)))
assert sp.simplify(frac - (n2/n1)**(-a)) == 0
print("sympy check: fraction(>nu2) == (nu2/nu1)^-alpha  ->  PASS\n")

E_HI, E_HeI, E_HeII = 13.598, 24.587, 54.418
def band(alpha):
    above_HeI  = (E_HeI /E_HI)**(-alpha)
    above_HeII = (E_HeII/E_HI)**(-alpha)
    return (1-above_HeI)*100, (above_HeI-above_HeII)*100, above_HeII*100

print("Fraction of H-ionizing photons per band, for eps_nu ~ nu^-alpha")
print("  alpha    13.6-24.6 eV   24.6-54.4 eV    >54.4 eV")
for alpha,lab in [(1.4,"a_s-1s"),(2.0,"a_s    "),(2.6,"a_s+1s"),(1.7,"AGN    ")]:
    b = band(alpha)
    print("  %.1f %s  %8.1f%%    %8.1f%%    %8.1f%%" % (alpha,lab,b[0],b[1],b[2]))

# how many He II-ionizing photons per 100 H I-ionizing photons
print("\nHe II-ionizing photons per 100 H I-ionizing photons:")
for alpha in (1.4,2.0,2.6,1.7):
    print("  alpha=%.1f : %5.1f" % (alpha, 100*(E_HeII/E_HI)**(-alpha)))
