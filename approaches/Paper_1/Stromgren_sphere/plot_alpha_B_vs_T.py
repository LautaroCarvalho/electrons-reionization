#!/usr/bin/env python3
"""
Case B hydrogen recombination coefficient alpha_B as a function of temperature,
plotted over T = 1e3 K to 1e8 K.

Fit
---
Hui & Gnedin (1997), MNRAS 292, 27, Appendix A -- their fit to the
collisional-radiative data of Ferland et al. (1992):

    alpha_B(T) = 2.753e-14 * lam^1.500 / [1 + (lam/2.740)^0.407]^2.242   cm^3 s^-1
    lam        = 2 * T_HI / T,     T_HI = 157807 K

quoted as accurate to 0.7% over 1 K <= T <= 1e9 K, so the whole plotted range
(1e3-1e8 K) lies inside the fit's stated validity. See
references/Hui1996_arXiv_astro-ph_9612232v1.pdf and Sec. 1.4 of
stromgren_sphere_summary.tex. Coefficients are transcribed from that paper;
none are invented.

Regime of validity: pure radiative recombination, low-density limit,
Maxwell-Boltzmann electron energies. Case B is itself weakly density dependent
(Storey & Sochi 2014 tabulate alpha(Ne, Te, kappa)); this curve is the
low-density limit.

Outputs (light and dark, PNG + PDF):
    alpha_B_vs_T_light.png / .pdf
    alpha_B_vs_T_dark.png  / .pdf

Run:  python3 plot_alpha_B_vs_T.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# --- the fit -----------------------------------------------------------------
try:                                     # single source of truth if available
    from alpha_HI_recombination_fits import hui_case_B
except ImportError:                      # keep this script self-contained
    T_HI_TR = 157807.0                   # K, H I ionization threshold (Hui & Gnedin 1997)

    def hui_case_B(T):
        lam = 2.0 * T_HI_TR / np.asarray(T, dtype=float)
        return 2.753e-14 * lam**1.500 / (1.0 + (lam / 2.740)**0.407)**2.242

# --- design tokens (dataviz reference palette; dark is selected, not flipped) --
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series="#2a78d6"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                  muted="#898781", grid="#2c2c2a", axis="#383835",
                  series="#3987e5"),
}

T_MIN, T_MAX = 1.0e3, 1.0e8
T_ANCHOR = 1.0e4                          # the canonical nebular reference point


def make_figure(theme_name):
    c = THEMES[theme_name]
    T = np.logspace(np.log10(T_MIN), np.log10(T_MAX), 2000)
    a = hui_case_B(T)

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    # recessive hairline grid: major only, solid (never dashed)
    ax.grid(True, which="major", color=c["grid"], linewidth=0.8,
            linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    # the single series: 2px line, round cap/join
    ax.loglog(T, a, color=c["series"], linewidth=2.0,
              solid_capstyle="round", solid_joinstyle="round", zorder=3)

    # one selective direct label at the canonical anchor, with a 2px surface ring
    a0 = float(hui_case_B(T_ANCHOR))
    ax.plot([T_ANCHOR], [a0], marker="o", markersize=7,
            color=c["series"], markeredgecolor=c["surface"],
            markeredgewidth=2, zorder=4)
    exponent = int(np.floor(np.log10(a0)))
    mantissa = a0 / 10.0**exponent
    label = (f"$T = 10^4$ K\n"
             f"$\\alpha_B = {mantissa:.2f}\\times10^{{{exponent}}}$"
             f" cm$^3$ s$^{{-1}}$")
    ax.annotate(label,
                xy=(T_ANCHOR, a0), xytext=(1.15e3, 2.6e-15),
                color=c["ink2"], fontsize=9.5, linespacing=1.5,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=c["axis"], linewidth=0.9,
                                shrinkA=0, shrinkB=4))

    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(1e-18, 3e-12)

    ax.set_xlabel("Temperature  $T$  [K]", color=c["ink2"], fontsize=11, labelpad=9)
    ax.set_ylabel(r"$\alpha_B(T)$  [cm$^3$ s$^{-1}$]", color=c["ink2"],
                  fontsize=11, labelpad=9)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", colors=c["muted"], labelsize=10,
                   length=4, width=0.8)
    ax.tick_params(which="minor", colors=c["muted"], length=2, width=0.6)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(0.8)

    # title names the single series, so no legend box is needed
    ax.set_title("Case B hydrogen recombination coefficient",
                 color=c["ink"], fontsize=14, pad=26, loc="left")
    ax.text(0.0, 1.045,
            "Hui & Gnedin (1997) fit to Ferland et al. (1992); "
            "quoted accuracy 0.7% over 1 K – 10$^9$ K",
            transform=ax.transAxes, color=c["ink2"], fontsize=9.5,
            ha="left", va="bottom")
    fig.text(0.013, 0.015,
             "Low-density limit, Maxwell–Boltzmann electron energies.",
             color=c["muted"], fontsize=8.5, ha="left", va="bottom")

    fig.subplots_adjust(left=0.115, right=0.975, top=0.855, bottom=0.135)
    return fig


if __name__ == "__main__":
    for theme in ("light", "dark"):
        fig = make_figure(theme)
        for ext in ("png", "pdf"):
            out = f"alpha_B_vs_T_{theme}.{ext}"
            fig.savefig(out, facecolor=fig.get_facecolor())
            print("wrote", out)
        plt.close(fig)

    # tabulated values, so every plotted number is auditable
    print("\nalpha_B [cm^3 s^-1] at decade points")
    for T in np.logspace(3, 8, 6):
        print(f"  T = {T:9.1e} K   alpha_B = {float(hui_case_B(T)):.4e}")
    d = 1e-5
    print("\nlocal slope d ln(alpha_B) / d ln(T)")
    for T in (1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        sl = (np.log(hui_case_B(T*(1+d))) - np.log(hui_case_B(T*(1-d)))) / np.log((1+d)/(1-d))
        print(f"  T = {T:9.1e} K   slope = {float(sl):+.4f}")
