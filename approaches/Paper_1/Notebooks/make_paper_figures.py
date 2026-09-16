"""
make_paper_figures.py -- English-labelled publication figures for the manuscript
"The electronic component of cosmic rays in the intergalactic medium during the
epoch of reionization".

Every curve is produced by the physics of ``igm_losses.py`` (the same module that
drives the marimo notebooks ``perdidas_energia_z10.py`` and ``z_vs_delta_z.py``);
these are English-labelled variations of the notebook figures plus two derived
panels (ionization yield and effective W-value) built from the same
``coll_ionisation_cross_section`` used inside ``loss_rates``.

Output: manuscript/figures/*.pdf
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import igm_losses as L

OUT = Path(__file__).parent / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- English names for the seven channels (same fixed indices as the module) ---
MECH_EN = [
    "Adiabatic expansion", "Synchrotron", "Inverse Compton",
    "Coulomb (plasma)", "Collisional excitation", "Collisional ionization",
    "Bremsstrahlung",
]

# --- the four kinetic-energy bins of the conceptual flow chart ---
BIN_EDGES_EV = [10.204, 1.0e5, 3.0e8]
BIN_LABELS = ["very cold", "cold", "warm", "hot"]

Z_HI, Z_LO = 20.0, 5.5
MPC = L.MEGAPARSEC_MKS


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def bin_bands(ax, ymin, ymax, annotate=True):
    """Shade the four flow-chart energy bins on a log-K axis."""
    edges = [1.0] + BIN_EDGES_EV + [1e14]
    for k, x in enumerate(BIN_EDGES_EV):
        ax.axvline(x, color="0.35", ls=":", lw=1.2, zorder=0)
    if annotate:
        for k, lab in enumerate(BIN_LABELS):
            xc = np.sqrt(edges[k] * edges[k + 1])
            ax.text(xc, 0.965, lab, ha="center", va="top", fontsize=9.5,
                    color="0.25", transform=ax.get_xaxis_transform())



# =====================================================================
# Bin boundaries K_1..K_4(z) and the photon opacity used for K_4
# =====================================================================
SIG_PI_0 = 6.30e-18 * 1e-4          # m^2, H(1s) at threshold
E_ION = L.THRESHOLD_EV_ION          # 13.6057 eV


def sigma_pi(E_eV):
    """H(1s) photoionization cross-section (Karzas & Latter 1961)."""
    x = np.atleast_1d(np.asarray(E_eV, float)) / E_ION
    out = np.zeros_like(x)
    m = x > 1.0
    eps = np.sqrt(x[m] - 1.0)
    out[m] = (SIG_PI_0 * x[m] ** -4 * np.exp(4.0 - 4.0 * np.arctan(eps) / eps)
              / (1.0 - np.exp(-2.0 * np.pi / eps)))
    return out


def sigma_kn(E_eV):
    """Klein-Nishina total cross-section per electron."""
    a = np.atleast_1d(np.asarray(E_eV, float)) * L.EV_MKS / L.E0
    st = L.THOMSON_CROSS_SECTION_MKS
    return st * 0.75 * (((1 + a) / a ** 3)
                        * ((2 * a * (1 + a)) / (1 + 2 * a) - np.log(1 + 2 * a))
                        + np.log(1 + 2 * a) / (2 * a) - (1 + 3 * a) / (1 + 2 * a) ** 2)


def photon_mfp(E_eV, z):
    """Proper mean free path [Mpc] of a photon of energy E_eV in the IGM at z."""
    nH = L.n_HI(z)
    kappa = nH * sigma_pi(E_eV) + (nH + L.ION_FRACTION * nH) * sigma_kn(E_eV)
    return 1.0 / kappa / MPC


def hubble_radius(z):
    return L.C_LIGHT / float(L.Planck18.H(z).to(L.u.s ** -1).value) / MPC


def E_free(z):
    """Photon energy at which the proper mfp equals the Hubble radius c/H(z)."""
    f = lambda lE: photon_mfp(10.0 ** lE, z)[0] / hubble_radius(z) - 1.0
    return 10.0 ** brentq(f, np.log10(20.0), 5.0)


def ke_for_photon(E_ph_eV, z, x):
    """K_e that up-scatters a CMB photon of energy x*kT_CMB(z) to E_ph_eV."""
    th = L.K_B * L.T_CMB_0 * (1.0 + np.asarray(z, float))
    tJ = np.asarray(E_ph_eV, float) * L.EV_MKS
    g = tJ / L.E0 + np.sqrt((tJ / L.E0) ** 2 + tJ / (x * th))
    return L.E0 * ((g ** 2 + 1.0) / (2.0 * g) - 1.0) / L.EV_MKS


def k_ion_ic_crossover(z):
    """K_e where the collisional-ionization and inverse-Compton rates cross."""
    K = np.logspace(2, 8, 4000)
    H = float(L.Planck18.H(z).to(L.u.s ** -1).value)
    st = L.loss_rates(z, K * L.EV_MKS, H, L.toggles())
    d = np.log(np.clip(st[5], 1e-300, None)) - np.log(np.clip(st[2], 1e-300, None))
    i = np.where(np.diff(np.sign(d)) != 0)[0]
    return K[i[-1]] if len(i) else np.nan


def boundary_curves(zz):
    """K_1..K_4 of the four-bin partition, as functions of z."""
    zz = np.asarray(zz, float)
    K1 = np.full_like(zz, E_ION)
    K2 = np.array([k_ion_ic_crossover(z) for z in zz])
    K3 = ke_for_photon(E_ION, zz, L.X_MAX_PLANCK)
    K4 = np.array([ke_for_photon(E_free(z), z, L.X_MIN_PLANCK) for z in zz])
    return K1, K2, K3, K4


# =====================================================================
# Fig. 7 -- photon opacity of the IGM and the resulting bin boundaries
# =====================================================================
def fig_thresholds():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    E = np.logspace(np.log10(13.61), 5, 1200)
    cols = {20.0: ("#1f77b4", (8, 6)), 10.0: ("#2ca02c", (8, -6)),
            5.5: ("#d62728", (8, -20))}
    for z, (c, off) in cols.items():
        axes[0].loglog(E, photon_mfp(E, z), color=c, lw=2.0, label=rf"$z={z}$")
        axes[0].axhline(hubble_radius(z), color=c, ls=":", lw=1.5)
        Ef = E_free(z)
        axes[0].plot([Ef], [hubble_radius(z)], "o", color=c, ms=7, zorder=5)
        axes[0].annotate(rf"${Ef/1e3:.2f}$ keV", (Ef, hubble_radius(z)),
                         textcoords="offset points", xytext=off,
                         fontsize=10, color=c)
    axes[0].set(xlabel=r"Photon energy $E_\gamma$ [eV]",
                ylabel=r"Proper mean free path [Mpc]",
                xlim=(13.6, 1e5), ylim=(1e-6, 1e6))
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].text(0.97, 0.05, r"dotted: $c/H(z)$", transform=axes[0].transAxes,
                 ha="right", fontsize=10, color="0.3")

    zz = np.linspace(Z_LO, Z_HI, 60)
    K1, K2, K3, K4 = boundary_curves(zz)
    axes[1].semilogy(zz, K1, "k-", lw=2.0, label=r"$K_1$ (ionization threshold)")
    axes[1].semilogy(zz, K2, "k--", lw=2.0, label=r"$K_2$ (coll. ion. $=$ IC)")
    axes[1].semilogy(zz, K3, "k-.", lw=2.0, label=r"$K_3$ ($E_\gamma = 13.6$ eV)")
    axes[1].semilogy(zz, K4, "k:", lw=2.4, label=r"$K_4$ (free-streaming)")
    for y, lab in ((2.5, "very cold"), (2e3, "cold"),
                   (4e7, "warm"), (3e10, "hot")):
        axes[1].text(Z_HI - 0.5, y, lab, fontsize=11, color="0.15", va="center")
    axes[1].set(xlabel=r"Redshift $z$",
                ylabel=r"Kinetic energy $K_{\mathrm{e}}$ [eV]",
                xlim=(Z_HI, Z_LO), ylim=(1e0, 1e13))
    axes[1].legend(loc="lower right", fontsize=9.5, framealpha=0.94)
    for ax in axes:
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls="--", alpha=0.1)
    fig.tight_layout()
    save(fig, "fig07_thresholds")


def print_thresholds():
    print("\n=== bin boundaries ===")
    print(f"{'z':>5} {'K1 [eV]':>10} {'K2 [eV]':>12} {'K3 [eV]':>12} "
          f"{'E_free [eV]':>12} {'K4 [eV]':>12} {'c/H [Mpc]':>11}")
    for z in (20.0, 15.0, 10.0, 8.0, 5.5):
        K1, K2, K3, K4 = boundary_curves([z])
        print(f"{z:5.1f} {K1[0]:10.4g} {K2[0]:12.4g} {K3[0]:12.4g} "
              f"{E_free(z):12.4g} {K4[0]:12.4g} {hubble_radius(z):11.4g}")


# =====================================================================
# Fig. 1 -- loss rates of the seven channels vs kinetic energy
# =====================================================================
def fig_loss_rates():
    K_eV = np.logspace(0.2, 14, 900)
    K_J = K_eV * L.EV_MKS
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    for ax, z in zip(axes, (20.0, 10.0)):
        H = float(L.Planck18.H(z).to(L.u.s ** -1).value)
        stack = L.loss_rates(z, K_J, H, L.toggles())
        for i in range(7):
            ax.loglog(K_eV, stack[i] / L.EV_MKS, color=L.MECH_COLORS[i],
                      lw=1.8, label=MECH_EN[i])
        ax.loglog(K_eV, stack.sum(axis=0) / L.EV_MKS, color="k", lw=2.2,
                  ls="--", label="Total")
        ax.set_xlim(K_eV[0], K_eV[-1])
        ax.set_ylim(1e-16, 1e4)
        ax.set_xlabel(r"Kinetic energy $K_{\mathrm{e}}$ [eV]")
        ax.set_title(rf"$z = {z:.0f}$")
        bin_bands(ax, 1e-16, 1e4)
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls="--", alpha=0.1)
    axes[0].set_ylabel(r"$|\mathrm{d}K_{\mathrm{e}}/\mathrm{d}t|$ [eV s$^{-1}$]")
    axes[1].legend(loc="lower right", fontsize=9.5, ncol=2, framealpha=0.92)
    fig.tight_layout()
    save(fig, "fig01_loss_rates")


# =====================================================================
# Fig. 2 -- dominant-mechanism map in the (z, K_e) plane, 5.5 <= z <= 20
# =====================================================================
def fig_phase_space():
    Zm, Km, dom, _ = L.phase_space_map(420, 620, 0, 14, "all",
                                       z_lo=Z_LO, z_hi=Z_HI)
    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    ax.pcolormesh(Zm, Km, dom, cmap=L.MECH_CMAP, vmin=-0.5,
                  vmax=L.N_MECH_MAP - 0.5, shading="auto", alpha=0.40)

    zz = np.linspace(Z_LO, Z_HI, 240)
    b1, b2, b3, b4 = boundary_curves(zz)
    ax.plot(zz, b1, "k-", lw=2.0, label=r"$K_1$: ionization threshold")
    ax.plot(zz, b2, "k--", lw=2.0,
            label=r"$K_2$: collisional ionization $=$ inverse Compton")
    ax.plot(zz, b3, "k-.", lw=2.0,
            label=r"$K_3$: up-scattered photon $=13.6$ eV")
    ax.plot(zz, b4, "k:", lw=2.4,
            label=r"$K_4$: up-scattered photon free-streams")

    for y, lab in ((3.0, "very cold"), (2e3, "cold"),
                   (3e6, "warm"), (5e10, "hot")):
        ax.text(Z_HI - 0.4, y, lab, fontsize=11, color="0.15", va="center")

    ax.set(yscale="log", xlim=(Z_HI, Z_LO), ylim=(1e0, 1e14),
           xlabel=r"Redshift $z$",
           ylabel=r"Kinetic energy $K_{\mathrm{e}}$ [eV]")
    patches = [mpatches.Patch(color=L.MECH_COLORS[i], label=MECH_EN[i], alpha=0.6)
               for i in np.unique(dom)]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=patches + h, loc="lower right", fontsize=9.5, ncol=2,
              framealpha=0.94)
    ax.grid(True, which="major", ls="-", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig02_phase_space")


# =====================================================================
# Fig. 3 -- cooling histories K_e(z) for injection at z = 20 and z = 10
# =====================================================================
def fig_cooling():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    energies = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12]
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(energies)))
    for ax, z_i in zip(axes, (20.0, 10.0)):
        t_i = L.age_s(z_i)
        t_eval = np.logspace(np.log10(t_i), np.log10(L.age_s(Z_LO)), 6000)
        for i, K0 in enumerate(energies):
            ta, za, Ka = L.trajectory(K0, "all", t_eval=t_eval)
            dz = z_i - za
            m = dz > 0
            ax.plot(dz[m], Ka[m], color=cols[i], lw=1.8,
                    label=rf"$10^{{{int(np.log10(K0))}}}$ eV")
        ax.set(xscale="log", yscale="log", xlim=(1e-4, z_i - Z_LO),
               ylim=(1e-1, 3e12),
               xlabel=r"$\Delta z = z_{\mathrm{i}} - z$")
        ax.set_title(rf"injection at $z_{{\mathrm{{i}}}} = {z_i:.0f}$")
        for y in BIN_EDGES_EV:
            ax.axhline(y, color="0.35", lw=1.0, ls=":")
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls="--", alpha=0.1)
    axes[0].set_ylabel(r"Kinetic energy $K_{\mathrm{e}}$ [eV]")
    axes[1].legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig03_cooling_histories")


# =====================================================================
# Fig. 4 -- fractional energy budget per channel vs injection energy
# =====================================================================
def _budget(K0_eV, z_i):
    t_i, t_f = L.age_s(z_i), L.age_s(Z_LO)
    sol = L.integrate(K0_eV, np.array([t_i, t_f]), tog=L.toggles(),
                      accumulate=True, rtol=1e-8, atol=1e-25,
                      t_span=(t_i, t_f))
    y_end = sol.y[:, -1]
    frac = y_end[1:] / (K0_eV * L.EV_MKS)
    return frac, float(L.z_interp(sol.t[-1]))


def fig_budget():
    K0s = np.logspace(2, 13, 34)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    for ax, z_i in zip(axes, (20.0, 10.0)):
        fr = np.array([_budget(K, z_i)[0] for K in K0s])
        for i in range(7):
            ax.loglog(K0s, np.clip(fr[:, i], 1e-12, None),
                      color=L.MECH_COLORS[i], lw=1.9, label=MECH_EN[i])
        ax.set_xlim(K0s[0], K0s[-1])
        ax.set_ylim(1e-8, 2.0)
        ax.axhline(1.0, color="k", ls=":", lw=1.2)
        ax.set_xlabel(r"Injection energy $K_{\mathrm{ini}}$ [eV]")
        ax.set_title(rf"injection at $z_{{\mathrm{{i}}}} = {z_i:.0f}$")
        bin_bands(ax, 1e-8, 2.0, annotate=False)
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls="--", alpha=0.1)
    axes[0].set_ylabel(r"$\int L_i\,\mathrm{d}t \,/\, K_{\mathrm{ini}}$")
    axes[1].legend(loc="lower left", fontsize=9.5, ncol=2, framealpha=0.92)
    fig.tight_layout()
    save(fig, "fig04_energy_budget")


# =====================================================================
# Fig. 5 -- ionization yield: primary alone vs full secondary shower
# =====================================================================
_CASC = Path(__file__).parent / "cascade_traj_table.npz"


def _cascade():
    d = np.load(_CASC)
    return d["K"], d["z"], d["N1"], d["Y"]


def fig_ionization_yield():
    K, zs, N1, Y = _cascade()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    styles = {20.0: ("#1f77b4", "-"), 10.0: ("#d62728", "--")}

    for z_i, (c, ls) in styles.items():
        iz = int(np.argmin(np.abs(zs - z_i)))
        m = Y[iz] > 0
        axes[0].loglog(K[m], Y[iz][m], color=c, ls=ls, lw=2.2,
                       label=rf"$Y$, $z_{{\mathrm{{i}}}} = {z_i:.0f}$")
        axes[0].loglog(K[m], N1[iz][m], color=c, ls=":", lw=1.6, alpha=0.85,
                       label=rf"$N_1$ (primary only), $z_{{\mathrm{{i}}}} = {z_i:.0f}$")
        axes[1].loglog(K[m], K[m] / Y[iz][m], color=c, ls=ls, lw=2.2,
                       label=rf"$z_{{\mathrm{{i}}}} = {z_i:.0f}$")
        j = int(np.argmin(K[m] / Y[iz][m]))
        Wmin = (K[m] / Y[iz][m])[j]
        axes[1].plot([K[m][j]], [Wmin], "o", color=c, ms=7, zorder=5)
        axes[1].annotate(rf"${Wmin:.1f}$ eV", (K[m][j], Wmin),
                         textcoords="offset points",
                         xytext=(6, -16 if z_i == 10.0 else 8),
                         fontsize=10, color=c)

    axes[1].axhline(35.0, color="0.35", ls="-.", lw=1.5,
                    label=r"$35$ eV (neutral gas)")
    axes[0].set(xlabel=r"Injection energy $K_{\mathrm{ini}}$ [eV]",
                ylabel=r"Ionizations per injected electron",
                xlim=(1e1, 1e13), ylim=(1e-1, 1e6))
    axes[1].set(xlabel=r"Injection energy $K_{\mathrm{ini}}$ [eV]",
                ylabel=r"$W(K_{\mathrm{ini}}) = K_{\mathrm{ini}}/Y$ [eV]",
                xlim=(1e1, 1e13), ylim=(1e1, 1e10))
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.92)
    axes[1].legend(loc="upper left", fontsize=10, framealpha=0.92)
    for ax in axes:
        bin_bands(ax, *ax.get_ylim(), annotate=False)
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls="--", alpha=0.1)
    fig.tight_layout()
    save(fig, "fig05_ionization_yield")


# =====================================================================
# Fig. 6 -- kinetic energy vs comoving path length (radius of the shell)
# =====================================================================
def fig_distance():
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    energies = [1e3, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12]
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(energies)))
    t_eval = np.logspace(np.log10(L.age_s(Z_HI)), np.log10(L.age_s(Z_LO)), 6000)
    for i, K0 in enumerate(energies):
        ta, za, Ka = L.trajectory(K0, "all", t_eval=t_eval)
        _, _, _, v = L.kinematics(Ka * L.EV_MKS)
        chi = cumulative_trapezoid(v * (1.0 + za), ta, initial=0.0) / MPC
        ax.loglog(np.clip(chi, 1e-8, None), Ka, color=cols[i], lw=1.9,
                  label=rf"$K_{{\mathrm{{ini}}}} = 10^{{{int(np.log10(K0))}}}$ eV")
    ta_l = t_eval
    chi_light = cumulative_trapezoid(
        L.C_LIGHT * (1.0 + L.z_interp(ta_l)), ta_l, initial=0.0) / MPC
    ax.axvline(chi_light[-1], color="k", ls=":", lw=1.6,
               label=r"light travel, $z=20\rightarrow5.5$")
    for y in BIN_EDGES_EV:
        ax.axhline(y, color="0.35", lw=1.0, ls=":")
    ax.set(xlabel=r"Comoving path length $\chi$ [Mpc]",
           ylabel=r"Kinetic energy $K_{\mathrm{e}}$ [eV]",
           xlim=(1e-6, 3e3), ylim=(1e-2, 3e12))
    ax.legend(loc="lower left", fontsize=9.5, ncol=2, framealpha=0.92)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls="--", alpha=0.1)
    fig.tight_layout()
    save(fig, "fig06_energy_vs_distance")


# =====================================================================
# Numerical table dumped to stdout for the text
# =====================================================================
def print_table():
    """Energy budget per channel and the cascade ionization yield."""
    print("\n K_ini[eV]   z_i    f_IC     f_ion    f_exc    f_coul   f_ad"
          "     z_therm")
    for z_i in (20.0, 10.0):
        for K0 in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12):
            fr, zt = _budget(K0, z_i)
            print(f"{K0:9.1e} {z_i:5.0f} {fr[2]:8.3g} {fr[5]:8.3g} "
                  f"{fr[4]:8.3g} {fr[3]:8.3g} {fr[0]:8.3g} {zt:10.3f}")

    if not _CASC.exists():
        print("\n(cascade_traj_table.npz not found -- run cascade_traj.py)")
        return
    K, zs, N1, Y = _cascade()
    print("\n K_ini[eV]   z_i    N_primary       Y_cascade    mult     W=K/Y [eV]")
    for z_i in (20.0, 10.0):
        iz = int(np.argmin(np.abs(zs - z_i)))
        for K0 in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12):
            j = int(np.argmin(np.abs(np.log(K) - np.log(K0))))
            n1, yy = N1[iz, j], Y[iz, j]
            print(f"{K[j]:9.3e} {z_i:5.0f} {n1:13.5g} {yy:14.5g} "
                  f"{yy/max(n1,1e-30):7.3f} {K[j]/max(yy,1e-30):13.5g}")


if __name__ == "__main__":
    print("Generating manuscript figures ...")
    import sys
    which = sys.argv[1:] or ["all"]
    if "all" in which or "1" in which: fig_loss_rates()
    if "all" in which or "2" in which: fig_phase_space()
    if "all" in which or "3" in which: fig_cooling()
    if "all" in which or "4" in which: fig_budget()
    if "all" in which or "5" in which: fig_ionization_yield()
    if "all" in which or "6" in which: fig_distance()
    if "all" in which or "7" in which: fig_thresholds()
    print_thresholds()
    if "all" in which or "T" in which: print_table()
