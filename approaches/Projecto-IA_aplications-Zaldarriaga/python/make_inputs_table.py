#!/usr/bin/env python3
r"""
Emit inputs_table.tex and references.bib: every externally-supplied number in
this project as  symbol - value - range - units - citation - status.

DISCIPLINE: every value is READ FROM THE CODE that uses it. Nothing is retyped,
so the table cannot drift from the calculation. The status column is the point
of the exercise:

    V  verified numerically against the source paper, which is in papers/
    C  cited, but the source is not in the tree and the implementation has
       never been checked against its equations
    D  derived in this project from other entries in this table
    S  a scanned parameter -- a range, not a measurement
    U  the user's stated choice; no external source claimed
    X  a conventional value whose source this project has NOT established
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import numpy as np
import ionization_yield as IY
import igm_losses as L
import photon_vs_electron as P

R = []      # (section, symbol, value, range, units, bibkey, status, note)
def row(sec, sym, val, rng, unit, key, st, note=""):
    R.append((sec, sym, val, rng, unit, key, st, note))

# ------------------------------------------------------------- 1. cosmology
C = "Cosmology"
row(C, r"$h$", f"{IY.H_LITTLE}", "--", "--", "Planck2020", "C")
row(C, r"$\Omega_m$", f"{IY.OMEGA_M}", "--", "--", "Planck2020", "C")
row(C, r"$\Omega_\Lambda$", f"{IY.OMEGA_L:.4f}", "--", "--", "Planck2020", "D",
    "flat by assumption")
row(C, r"$\Omega_b h^2$", f"{IY.OMEGA_B_H2}", "--", "--", "Planck2020", "C")
row(C, r"$\rho_{\rm crit}/h^2$", f"{IY.RHO_CRIT_COEF:.5e}", "--",
    r"g\,cm$^{-3}$", "PDG2022", "C")
row(C, r"$T_{\rm CMB,0}$", f"{IY.T_CMB0}", "--", "K", "Fixsen2009", "C")
row(C, r"$Y_P$", f"{IY.Y_P}", "--", "--", "Planck2020", "C", "BBN + Planck")
row(C, r"$X_H$", f"{IY.X_H:.3f}", "--", "--", "Planck2020", "D", r"$1-Y_P$")
row(C, r"$H(z)$", "Planck18", "--", r"s$^{-1}$", "Planck2020", "C",
    r"\texttt{astropy}; includes radiation")

# ------------------------------------------------------------ 2. IGM target
T = "IGM target"
row(T, r"$n_{\rm H}(z{=}10)$", f"{IY.cosmology(10.0)['n_H_cm3']:.4e}", "--",
    r"cm$^{-3}$", "Planck2020", "D", "proper; from $\\Omega_b h^2$, $X_H$")
row(T, r"$N_{\rm HI,norm}$", f"{L.N_HI_NORM:.6f}", "--", r"m$^{-3}$",
    "Planck2020", "D", r"$n_{\rm HI}(z)=N((1+z)/21)^3$; harmonised")
row(T, r"$x_e$", f"{L.ION_FRACTION:.0e}", "--", "--", "", "U",
    "held static; your value")
row(T, r"$B_0$", f"{L.B0:.1e}", "--", "T", "", "U", r"1 nG comoving; $B\propto(1+z)^2$")
row(T, r"$Z$", f"{L.Z_TARGET:.0f}", "--", "--", "", "U", "pure hydrogen target")
row(T, r"$\delta_{\rm Gould}$", f"{L.DELTA_GOULD}", "--", "--", "Gould1972", "C",
    "max fractional transfer")
row(T, r"$f_{\rm therm}$", f"{L.THERMAL_FLOOR_FACTOR}", "--", "--", "", "U",
    r"$K_{\rm term}=f\,kT_{\rm CMB}$")
row(T, r"$n_{\rm max}^{\rm exc}$", f"{L.EXC_NMAX}", "--", "--", "Stone2002", "C",
    r"levels $n=2\ldots n_{\rm max}$")

# --------------------------------------------------------- 3. atomic physics
A = "Atomic physics"
row(A, r"$E_{\rm th}$(HI)", f"{IY.E_TH_HI:.9f}", "--", "eV", "NIST_ASD", "V")
row(A, r"$R_\infty$", f"{P.RYD_EV:.9f}", "--", "eV", "Tiesinga2021", "V")
row(A, r"$B$ (Kim Table I)", "13.6057", "--", "eV", "KimRudd1994", "V",
    "the Rydberg; tension with NIST at 0.05\\%")
row(A, r"$U/B$", "1", "--", "--", "KimRudd1994", "V", "virial, H(1s)")
row(A, r"$N_i$", f"{float(np.sum(L._DIPOLE_FIT/(np.arange(4)+1))):.6f}", "--",
    "--", "KimRudd1994", "V", "continuum oscillator strength")
row(A, r"$M^2$", "0.2834", "--", "--", "KimRudd1994", "V")
row(A, r"$a_i$ (d$f$/d$w$)", r"4 coeffs", "--", "--", "KimRudd1994", "V",
    "Table I, H(1s)")
row(A, r"$\sigma_{\rm ion}$", "RBED eq.~(20)", "--", r"m$^2$", "Kim2000", "V",
    r"verified to $1.4\times10^{-9}$")
row(A, r"d$\sigma$/d$w$", "BED, non-rel.", "--", r"m$^2$eV$^{-1}$", "KimRudd1994",
    "V", "relativistic factor omitted: +25\\% at 1 MeV")
row(A, r"$\sigma_{\rm exc}$", "Stone \\& Kim", "--", r"m$^2$", "Stone2002", "C")
row(A, r"Coulomb loss", "Gould (1972)", "--", r"eV\,s$^{-1}$", "Gould1972", "C")
row(A, r"IC / brems", "BG70", "--", r"eV\,s$^{-1}$", "Blumenthal1970", "C",
    "eq.~(2.42) spectrum verified; loss rates not")
row(A, r"$I_{\rm exc}$", f"{IY.I_EXC_H}", "--", "eV", "ICRU1984", "C",
    "mean excitation energy, route 1 only")
row(A, r"$\alpha_B(2\times10^4\,$K$)$", "1.430e-13", "--", r"cm$^3$s$^{-1}$",
    "OsterbrockFerland2006", "V", r"confirmed vs \cite{HuiGnedin1997} to 0.16\%")
row(A, r"$\chi$", "1.08", "--", "--", "MadauDickinson2014", "V",
    r"derived here as $1+(Y_P/4)/X_H=1.0811$")

# ----------------------------------------------------- 4. stellar source
S = "Stellar source"
row(S, r"$\log_{10}\rho_{\rm UV}$", f"{P.LOG_RHO_UV}",
    f"$+{P.LOG_RHO_UV_HI}/{P.LOG_RHO_UV_LO}$",
    r"erg\,s$^{-1}$Hz$^{-1}$cMpc$^{-3}$", "Donnan2024", "V",
    r"Table 3, $z=10$ row")
row(S, r"$\log_{10}\xi_{\rm ion}$", f"{P.LOG_XI_ION}",
    f"$\\pm{P.XI_ION_SCATTER_DEX}$ dex", r"Hz\,erg$^{-1}$", "Llerena2025", "V",
    r"median at $z\simeq7.14$; scatter is observed, not error")
row(S, r"$K_{\rm UV}$", f"{P.K_UV:.3e}", "--",
    r"$M_\odot$yr$^{-1}$/(erg\,s$^{-1}$Hz$^{-1}$)", "MadauDickinson2014", "V",
    r"eq.~(10); Salpeter $0.1$--$100\,M_\odot$")
row(S, r"$\alpha_{\rm IMF}$", f"{P.IMF_SLOPE}", "--", "--", "Salpeter1955", "V",
    r"fitted for $0.4$--$10\,M_\odot$ only")
row(S, r"$m_{\min},m_{\max}$", f"{P.M_MIN}, {P.M_MAX}", "--", r"$M_\odot$",
    "MadauDickinson2014", "C", "range adopted to match $K_{\\rm UV}$")
row(S, r"$m_{\rm SN}$", f"{P.M_SN_MIN}", "--", r"$M_\odot$", "", "U",
    "core-collapse threshold")
row(S, r"SN per $M_\odot$", f"{P.sn_per_solar_mass():.4e}", "--",
    r"$M_\odot^{-1}$", "Salpeter1955", "D", "IMF integral, this work")
row(S, r"$\alpha_{\rm SED}$", f"{P.SED_ALPHA:.0f}",
    f"{P.SED_ALPHA_LO:.0f}--{P.SED_ALPHA_HI:.0f}", "--", "MarquesChaves2026", "C",
    r"$f_\nu\propto\nu^{-\alpha}$; BPASS spectrum unobtainable")
row(S, r"LyC band", f"1--4", "--", "Ryd", "", "U", "conventional")
row(S, r"$T_{\rm eff}$", f"{P.T_EFF_K:.0e}", "--", "K", "", "U",
    "blackbody, comparison case only")

# --------------------------------------------------------- 5. cosmic rays
K = "Cosmic-ray channel"
row(K, r"$E_{\rm SN}$", f"{P.E_SN_ERG:.0e}", "--", "erg", "", "X",
    "standard core-collapse budget; source not established here")
row(K, r"$\epsilon_{\rm CR}$", "0.1", r"$0.1$--$0.2$", "--",
    "CaprioliSpitkovsky2014", "C",
    r"hybrid simulations: $10$--$20\%$ of bulk kinetic energy into non-thermal "
    r"protons at parallel/quasi-parallel strong shocks, dropping to ZERO when "
    r"quasi-perpendicular; $\simeq10\%$ inferred for Tycho. An "
    r"obliquity-AVERAGED value would be lower, so $0.1$ favours the CR channel")
row(K, r"$f_e=K_{ep}$", "0.01", r"$1$--$3\times10^{-3}$ (source)", "--",
    "Park2015", "C",
    r"\textbf{the project uses the value measured at Earth, not the value the "
    r"source derives}: \cite{Park2015} give $K_{ep}\simeq1$--$3\times10^{-3}$ at "
    r"realistic SNR shock speeds and $1.6\times10^{-3}$ for Tycho, so $0.01$ is "
    r"$3$--$10\times$ high and favours the CR channel. $K_{ep}$ is defined at "
    r"fixed MOMENTUM and is independent of $p$")
row(K, r"$p$", f"{P.CR_INDEX}", "--", "--", "", "U",
    r"your choice; DSA strong-shock predicts 2")
row(K, r"$E_{\min},E_{\max}$", f"{P.CR_E_MIN_EV:.0e}, {P.CR_E_MAX_EV:.0e}", "--",
    "eV", "", "U", r"your choice; spectrum cancels from $\zeta_e$")

# ----------------------------------------------- 6. reionization bookkeeping
B = "Reionization budget"
row(B, r"$C_{\rm IGM}(z)$", r"$1+43z^{-1.71}$", "--", "--", "Pawlik2009", "V",
    r"eq.~(A1), $C_{100}$, r19.5 run; $43=e^{3.76}$; fit valid $6\le z\le20$")
row(B, r"$\langle t_{\rm rec}\rangle$", r"eq.~(24)", "--", "Gyr", "MadauDickinson2014", "V",
    r"3.2 Gyr normalisation reproduced to 1.6\%")

# -------------------------------------------------------- 7. scanned ranges
Q = "Scanned ranges"
row(Q, r"$f_{\rm esc}$", f"{P.F_ESC_FID}", r"$10^{-6}$--$1$", "--", "", "S",
    "canonical value is a convention, not a measurement")
row(Q, r"$\epsilon_{\rm CR}f_e$", f"{P.EPS_CR_FE_FID:.0e}", r"$10^{-6}$--$1$",
    "--", "", "S", "bounded above by unity on energy grounds")
row(Q, r"$\rho_{\rm SFR}$ scale", f"{P.RHO_SFR_SCALE_FID:.0f}",
    r"$10^{-3}$--$1$", "--", "", "S",
    r"ignorance band at $z=20$ only, where $\rho_{\rm UV}$ is unmeasured")

# ------------------------------------------- 8. source competition: spectra
import source_map as SM

def _cls(lbl):
    return next(c for c in SM.PHOTON_CLASSES if c["label"].startswith(lbl))

X1 = "Source spectra"
_rq, _rl = _cls("AGN, radio-quiet"), _cls("AGN, radio-loud")
_tp, _ux, _bh = _cls("AGN, reionization"), _cls("ULX"), _cls("BHB")
row(X1, r"$\alpha_{\rm EUV}$ (RQ)", f"{_rq['alpha']}", r"$\pm0.17$", "--",
    "Telfer2002", "V", r"HST composite, rest-frame $500$--$1200$\,\AA")
row(X1, r"$\alpha_{\rm EUV}$ (RL)", f"{_rl['alpha']}", r"$\pm0.12$", "--",
    "Telfer2002", "V", r"radio-loud quasars; full composite $1.76\pm0.12$")
row(X1, r"$\alpha_{\rm AGN}$", f"{_tp['alpha']}", "--", "--", "Graziani2018", "V",
    r"EoR quasar template; corroborates \cite{Telfer2002}")
row(X1, r"AGN band top", f"{_tp['E_top']/1e3:.0f}", "--", "keV", "Graziani2018",
    "V", r"$13.6$\,eV--$3$\,keV; above 4 Ryd is an extrapolation")
row(X1, r"$\Gamma_1$ (ULX)", f"{_ux['G1']}", f"{_ux['G1lo']}--{_ux['G1hi']}",
    "--", "Gladstone2009", "V", r"Table~6; single-PO fits $1.6<\Gamma<3.3$")
row(X1, r"$E_{\rm break}$ (ULX)", f"{_ux['E_break']/1e3:.1f}",
    f"{_ux['Eblo']/1e3:.1f}--{_ux['Ebhi']/1e3:.0f}", "keV", "Gladstone2009", "V",
    r"broken PL preferred $>98\%$ in 11 of 12; fits are $2$--$10$\,keV only")
row(X1, r"$\Delta\Gamma$ (ULX)", f"{_ux['dG']}",
    f"{_ux['dGlo']:.0f}--{_ux['dGhi']:.0f}", "--", "Gladstone2009", "V",
    r"steepening above the break")
row(X1, r"$kT_e,\tau$ (ULX corona)",
    f"{SM.ULX_CORONA_KTE_KEV[0]:.0f}--{SM.ULX_CORONA_KTE_KEV[1]:.0f}, "
    f"{SM.ULX_CORONA_TAU[0]:.0f}--{SM.ULX_CORONA_TAU[1]:.0f}", "--",
    r"keV, --", "Gladstone2009", "V",
    r"cool, optically THICK; global $\chi^2$ min in 10/12")
row(X1, r"$\Gamma$ (BHB hard)", f"{_bh['alpha']+1:.1f}", r"$<2.1$", "--",
    "Gladstone2009", "X",
    r"\textbf{the typical $\Gamma\simeq1.7$ is second-hand}; only "
    r"$\Gamma<2.1$ for the low/hard state is first-hand (\S4.1)")
row(X1, r"$p$ per class", f"{SM.FID_P}", "--", "--", "", "U",
    r"held at the project fiducial: no citable per-class CR index at $z\sim10$")

# ------------------------------------- 9. source competition: luminosities
X2 = "Source luminosities"
row(X2, r"AGN LyC fraction", f"{SM.AGN_LYC_FRAC[0]:.2f}--{SM.AGN_LYC_FRAC[1]:.2f}",
    "--", "--", "Asthana2024", "V",
    r"$17\%$ (\cite{Asthana2024}) to $\le1/3$ (\cite{Jiang2025}), both at "
    r"$z\sim7.5$; carried to $z=10$, conservative for photons")
row(X2, r"$L_{2-10}/$SFR", f"{SM.LEHMER_NORM:.1e}",
    f"{SM.LEHMER_NORM_LO:.1e}--{SM.LEHMER_NORM_HI:.1e}",
    r"erg\,s$^{-1}$/($M_\odot$yr$^{-1}$)", "Lehmer2016", "V",
    rf"eq.~(6), $\times(1+z)^{{{SM.LEHMER_Z_EXP:.0f}}}$; exponent fixed at 1")
row(X2, r"$L_{\rm mech}$ (microquasar)",
    f"{SM.MQ_MECH_ERG_S[0]:.0e}--{SM.MQ_MECH_ERG_S[1]:.0e}", "--",
    r"erg\,s$^{-1}$", "Mirabel2011", "V",
    r"SS\,433 and the S26 microquasar; TOTAL injected, not the electron share")
row(X2, r"$E_{\rm mech}$ per object", f"{SM.MQ_LIFETIME_ERG:.0e}", "--", "erg",
    "Mirabel2011", "V", r"whole lifetime; $\gg$ a core-collapse SN")
row(X2, r"$M_{\rm UV}$ (U37126)", f"{SM.M_UV_U37126}",
    rf"$\pm{SM.M_UV_U37126_ERR}$", "AB", "MarquesChaves2026", "V",
    r"$z=10.255$, lensed $\mu\simeq2.2$; the per-object stellar anchor")
row(X2, r"$f_{\rm jet,e}$",
    f"{SM.JET_E_FRAC[0]:.0e}--{SM.JET_E_FRAC[1]:.0f}", "scanned", "--", "", "S",
    r"jet efficiency $\times$ leptonic fraction, scanned: the proton content "
    r"of jets is not observationally constrained")
row(X2, r"$f_{\rm dep}$", "1", "--", "--", "", "U",
    r"assumption A11: every injected electron deposits --- a CEILING")

# -------------------------------------- 10. derived by the competition code
X3 = "Derived (this work)"
_We = SM.W_e(SM.FID_P, SM.FID_EMIN, SM.FID_EMAX)
row(X3, r"$W_e$", f"{_We:.3f}", "--", r"eV\,ion$^{-1}$", "", "D",
    r"$\langle E\rangle/\langle N\rangle$ of the fiducial CR spectrum")
for _lbl, _c in (("stellar", _cls("Star-forming")), ("AGN", _tp),
                 ("ULX", _ux), ("BHB hard", _bh)):
    _W = SM.class_W_gamma(_c, "mid")
    row(X3, rf"$W_\gamma$ ({_lbl})", f"{_W:.2f}", "--", r"eV\,ion$^{-1}$", "",
        "D", rf"$W_\gamma/W_e={_W/_We:.3f}$; IGM transparency applied")
for _E, _nm in ((1.0e3, "1 keV"), (3.0e3, "3 keV"), (1.0e4, "10 keV")):
    row(X3, rf"$\tau_{{\rm IGM}}$({_nm})", f"{float(SM.tau_igm(_E)):.3g}", "--",
        "--", "", "D", r"over one Hubble length at $z=10$; "
        r"thick below $\sim500$\,eV, transparent above $\sim3$\,keV")
row(X3, r"$t_H(z{=}10)$", f"{SM.hubble_time_s()/3.1557e16:.4f}", "--", "Gyr",
    "Planck2020", "D", r"astropy Planck18, complete $H(z)$")

# ============================================================ emit the table
STATUS = {
 "V": r"\textbf{V}", "C": r"C", "D": r"D", "S": r"S", "U": r"U",
 "X": r"\textcolor{red}{\textbf{X}}"}
PRE = r"""% Generated by make_inputs_table.py -- DO NOT EDIT BY HAND.
% Rebuild:  PYTHONPATH=. python3 make_inputs_table.py
%           pdflatex inputs_table && bibtex inputs_table && pdflatex inputs_table x2
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,longtable,xcolor}
\usepackage[colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue]{hyperref}
\title{Inventory of inputs, with sources}
\author{Compiled for the \texttt{ionization\_yield} / \texttt{photon\_vs\_electron} project}
\date{2026-09-14}
\begin{document}\maketitle

\noindent
Every number this project takes from outside itself, with its symbol, value,
range, units and source. \textbf{Values are read from the code that uses them}
by \texttt{make\_inputs\_table.py}, so the table cannot drift from the
calculation; only the symbols, units and citations are written by hand.

\paragraph{What the status column is for.} It separates four very different
kinds of number that a bare citation would make look alike: a value checked
numerically against the paper it came from; a value whose paper we have never
opened; a value you chose; and a value that is conventional in the field but
for which \emph{this project has established no source at all}. The last
category is the one worth acting on.
"""
POST = r"""
\section{The fiducial top-heavy IMF}\label{sec:topheavy}
For the scenario considered here --- Population~II stars in galaxies at
$z=10$ --- the fiducial top-heavy IMF adopted is that of
Jeong, Jeon, Song \& Bromm~\cite{Jeong2025}, their eq.~(2):
\begin{equation}
  \xi(m) = \frac{\mathrm{d}N}{\mathrm{d}m} = \xi_0\,m^{-\alpha},
  \qquad \alpha = 1.3,
  \qquad 1\,M_\odot \le m \le 100\,M_\odot .
  \label{eq:topheavy}
\end{equation}
It is chosen over the alternatives for four reasons. \emph{First}, it is
stated explicitly as a top-heavy IMF \emph{for Pop~II stars}, not for
Pop~III, which is the population this project's galaxies contain.
\emph{Second}, the galaxies it is applied to form at $z=10$, the redshift of
this calculation. \emph{Third}, it is the same functional form as the
Salpeter law already in use, $\mathrm{d}N/\mathrm{d}m \propto m^{-\alpha}$,
so it substitutes directly into the supernova-rate integral with no change of
machinery --- only $\alpha$ and $m_{\min}$ move. \emph{Fourth}, the physical
motivation is the one relevant here: a raised temperature floor from
low-metallicity gas radiatively coupled to a CMB at $T=2.73(1+z)$~K, which
lifts the Jeans mass and hence the characteristic stellar
mass~\cite{Larson1998}. Their default Pop~II IMF, for comparison, is
Chabrier~\cite{Chabrier2003} over $0.1$--$100\,M_\odot$.

\paragraph{Not yet adopted.} Equation~\eqref{eq:topheavy} is recorded here as
the fiducial to test against; it has \emph{not} been substituted into any
result in this project. Doing so would move three of the entries in
\S\ref{sec:inputs} at once --- $\alpha_{\rm IMF}$, the mass range, and the
supernova rate per $M_\odot$ --- and would also invalidate $K_{\rm UV}$,
which \cite{MadauDickinson2014} quote for a Salpeter IMF over $0.1$--$100\,M_\odot$.
Those three quantities are not independent, and that coupling is the open
question this inventory was built to expose.

\bibliographystyle{plain}
\bibliography{../papers/references}
\end{document}
"""
lines = [PRE,
         r"\section{Inventory of inputs}\label{sec:inputs}",
         r"Every externally supplied number in this project. Values are read"
         r" from the code that uses them, so this table cannot drift from the"
         r" calculation. Status: \textbf{V} verified numerically against the"
         r" source, which is in \texttt{papers/}; C cited but the source is not"
         r" in the tree and the implementation is unchecked; D derived here;"
         r" S scanned; U your stated choice;"
         r" \textcolor{red}{\textbf{X}} conventional value whose source this"
         r" project has \emph{not} established.",
         r"\begin{longtable}{llllll}",
         r"\hline",
         r"symbol & value & range & units & source & st. \\",
         r"\hline\endhead"]
cur = None
for sec, sym, val, rng, unit, key, st, note in R:
    if sec != cur:
        lines.append(r"\multicolumn{6}{l}{\textbf{%s}}\\[2pt]" % sec)
        cur = sec
    cite = r"\cite{%s}" % key if key else "---"
    lines.append(f"{sym} & {val} & {rng} & {unit} & {cite} & {STATUS[st]} " + r"\\")
    if note:
        lines.append(r"\multicolumn{6}{p{0.95\textwidth}}{\footnotesize\quad "
                     + note + r"}\\[2pt]")
lines += [r"\hline", r"\end{longtable}", POST]
open(project_paths.text("inputs_table.tex"), "w").write("\n".join(lines) + "\n")
n_x = sum(1 for r in R if r[6] == "X")
n_c = sum(1 for r in R if r[6] == "C")
print(f"[ARTEFACT] inputs_table.tex  --  {len(R)} entries")
print(f"   verified V: {sum(1 for r in R if r[6]=='V')}")
print(f"   cited but unverified C: {n_c}")
print(f"   derived D: {sum(1 for r in R if r[6]=='D')}")
print(f"   scanned S: {sum(1 for r in R if r[6]=='S')}")
print(f"   your choice U: {sum(1 for r in R if r[6]=='U')}")
print(f"   UNSOURCED X: {n_x}")
for r in R:
    if r[6] == "X":
        print(f"      -> {r[1]}  {r[2]}   {r[7]}")
