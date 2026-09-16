"""
Phase 4 -- render the parameter table (Section E of the plan) directly from
params.py, so it can never drift out of sync with the code that actually
used these values. Writes a standalone .tex fragment, \\input by
master_plan_response.tex.
"""
import pathlib
import params as P

HERE = pathlib.Path(__file__).resolve().parent

def M(s):
    """Wrap a math-mode LaTeX snippet in $...$."""
    return f"${s}$"


ROWS = [
    (r"$\xi_{\rm ion}$ slope, intercept", M(f"{P.XI_ION_LOG10_SLOPE},\\ {P.XI_ION_LOG10_INTERCEPT}"),
     "Llerena et al. 2024, arXiv:2412.01358", r"$\log_{10}\xi_{\rm ion}=$slope$\cdot z+$intercept"),
    ("SFR (fiducial galaxy)", M(f"{P.SFR_FID_MSUN_YR:.1f}\\ M_\\odot\\,{{\\rm yr}}^{{-1}}"),
     "adopted, Stromgren\\_sphere/bubble\\_radius\\_vs\\_redshift.py", "fiducial single-galaxy SFR"),
    (r"$f_{\rm esc}$", M(f"{P.F_ESC}"), "Robertson et al. 2015, arXiv:1502.02024", "escape fraction of UV ionizing photons"),
    (r"$z_{\rm form}=z_{\rm inject}$", M(f"{P.Z_FORM:.0f}"), "Kitayama et al. 2004 range (10-30), adopted", "galaxy formation / CR-injection snapshot redshift"),
    (r"$p$ (DSA index)", M(f"{P.DSA_INDEX_P:.1f}"), "Park, Caprioli \\& Spitkovsky 2015, PRL 114", "CR-electron injection power-law index"),
    (r"$K_{\min}$", M(f"{P.K_MIN_EV:.0e}\\ {{\\rm eV}}"), "user-adopted simplification (Section A.3)", "unbroken power-law extrapolation limit"),
    (r"$K_{\max}$", M(f"{P.K_MAX_EV:.0e}\\ {{\\rm eV}}"), "adopted (1 TeV)", "injection spectrum upper cutoff"),
    (r"$\varepsilon_{\rm CR}$ (total CR efficiency)", M(f"{P.EPS_CR_LOW}\\text{{--}}{P.EPS_CR_HIGH}"),
     "Blasi 2013, arXiv:1206.2363", "fraction of $E_{\\rm SN}$ into all CRs"),
    (r"$K_{e/p}$ (electron/proton number ratio)", M(f"{P.K_EP_RATIO:.0e}"), "Park et al. 2015, PRL 114 (already in manuscript.tex)", "injection-level electron fraction"),
    (r"$\xi_{\rm CR,e}=\varepsilon_{\rm CR}K_{e/p}$", M(f"{P.XI_CR_E_LOW:.1e}\\text{{--}}{P.XI_CR_E_HIGH:.1e}"),
     "derived (this work)", "fraction of $E_{\\rm SN}$ into CR electrons"),
    (r"$E_{\rm SN}$", M(f"{P.E_SN_ERG:.0e}\\ {{\\rm erg}}"), "canonical core-collapse value", "kinetic energy per SN"),
    (r"$k_{\rm CC}$", M(f"{P.K_CC_PER_MSUN}\\ M_\\odot^{{-1}}"), "Salpeter IMF 8-50 $M_\\odot$; arXiv:1206.6897", "core-collapse SNe per unit stellar mass formed"),
    (r"Cosmology $h,\Omega_m,\Omega_b h^2,Y_p$",
     M(f"{P.H_LITTLE},\\ {P.OMEGA_M},\\ {P.OMEGA_B_H2},\\ {P.Y_HE}"), "Planck 2018, arXiv:1807.06209", "flat $\\Lambda$CDM parameters"),
    (r"$\alpha_B(T)$ fit", "Hui \\& Gnedin (1997) form", "Hui \\& Gnedin 1997, ApJ 292, 27", "Case-B recombination coefficient"),
    (r"$\alpha_B(10^4\,{\rm K})$", M(f"{P.alpha_B(1e4):.3e}".replace("e-", r"\times10^{-") + "}" + r"\ {\rm cm^3\,s^{-1}}"),
     "cross-checked vs. Shapiro \\& Iliev 2006 (2.59e-13)", "fiducial photoionized-gas value"),
    ("clumping factor $C_{\\rm HII}$", M(f"{P.CLUMPING_C_HII}"), "Robertson et al. 2015", "recombination-sink enhancement"),
    ("heat/ionization", M(f"{P.HEAT_PER_IONIZATION_EV_LOW}\\text{{--}}{P.HEAT_PER_IONIZATION_EV_HIGH}\\ {{\\rm eV}}"),
     "manuscript.tex channel-resolved cascade", "cascade microphysics constant"),
    (r"$\Theta$ (lock constant)", M(f"{P.THETA_LOCK_K_LOW:.1e}\\text{{--}}{P.THETA_LOCK_K_HIGH:.1e}\\ {{\\rm K}}"),
     "manuscript.tex Eq.~(lock)", r"$N_{\rm ion}/n_{\rm H}=\Delta T/\Theta$"),
    ("particles/H atom", M(f"{P.PARTICLES_PER_H_ATOM}"), "manuscript.tex (He-corrected)", "gas heat-capacity factor"),
    (r"$x_e$ (neutral IGM cascade medium)", M(f"{P.ION_FRACTION_NEUTRAL:.0e}"), "igm\\_losses.py default", "background ionization fraction for $Y(K,z)$"),
    ("cascade floor redshift", M(f"{P.Z_THERMAL_FLOOR}"), "cascade\\_yield.py default", "integration stops at thermal floor or this $z$"),
    (r"$n_{H,0}$ (comoving)", M(f"{P.NH0_CM3:.4e}".replace("e-", r"\times10^{-") + "}" + r"\ {\rm cm^{-3}}"),
     "derived from $\\Omega_b h^2, Y_p$", "present-day mean H number density"),
]


def render():
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Every physical/numerical parameter used in this work, "
        r"rendered directly from \texttt{params.py} (never hand-retyped) "
        r"so this table cannot drift out of sync with the code. Future runs "
        r"with different values are a one-line edit to that file.}",
        r"\label{tab:params}",
        r"\small",
        r"\begin{tabular}{@{}p{4.3cm}p{3.6cm}p{4.2cm}p{4.3cm}@{}}",
        r"\toprule",
        r"Symbol & Value & Source/citation & Role \\",
        r"\midrule",
    ]
    for sym, val, src, role in ROWS:
        lines.append(f"{sym} & {val} & {src} & {role} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    out = HERE / "parameter_table.tex"
    out.write_text(render())
    print(f"Wrote {out}")
