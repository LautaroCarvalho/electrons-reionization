import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="z vs delta z - intervalos cosmologicos y enfriamiento",
)


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # $z$ vs $\Delta z$: intervalos cosmológicos y enfriamiento del electrón

        Dos bloques:

        **A. Correspondencia $\Delta z \leftrightarrow \Delta t$** (Figs. 01–05).
        Dos criterios distintos para definir el paso de integración en $z$:

        | Criterio | Definición | Figuras |
        |---|---|---|
        | Temporal | $\Delta t$ fijo (3 Myr) | 01, 03 |
        | Densidad | $\dfrac{(1+z)^3}{(1+z+\Delta z)^3} = 0.95$ (la densidad no cae más de un 5 %) | 02, 04, 05 |

        **B. Enfriamiento del electrón según el redshift de inyección** (Figs. 06–10).
        La misma trayectoria $K_e$ vista contra cinco abscisas distintas: redshift,
        edad del universo, y tres nociones de distancia que **no** son la misma cosa.

        La física vive en `igm_losses.py`, compartida con `perdidas_energia_z10.py`.
        """
    )
    return (mo,)


@app.cell
def _():
    # =====================================================================
    #  Física compartida: constantes, cosmología, kernel Klein-Nishina,
    #  tablas colisionales, loss_rates() e integradores con caché.
    # =====================================================================
    import igm_losses as ig
    from igm_losses import (
        C_LIGHT, EV_MKS, MEGAPARSEC_MKS, YEAR_MKS,
        KE_99_ARRAY, KE_CUTOFF_ARRAY, TOTAL_PLANCK_AREA, X_MAX_PLANCK, X_MIN_PLANCK,
        ELECTRON_REST_ENERGY_MKS, K_B, T_CMB_0, Z_ARRAY_INTERP, T_ARRAY_INTERP,
        Planck18, age_s, compute_threshold_curves, cumulative_trapezoid, grid_minor,
        interp1d, kinematics, np, plt, trajectory, u, z_interp,
    )
    import matplotlib.lines as mlines

    print(f"igm_losses cargado - FIX_KN_KERNEL={ig.FIX_KN_KERNEL}, "
          f"EXC_NMAX={ig.EXC_NMAX}, brems en el total={ig.INCLUDE_BREMS_IN_TOTAL}")
    return (
        C_LIGHT,
        EV_MKS,
        KE_99_ARRAY,
        KE_CUTOFF_ARRAY,
        K_B,
        MEGAPARSEC_MKS,
        Planck18,
        T_ARRAY_INTERP,
        T_CMB_0,
        X_MAX_PLANCK,
        X_MIN_PLANCK,
        YEAR_MKS,
        Z_ARRAY_INTERP,
        age_s,
        compute_threshold_curves,
        cumulative_trapezoid,
        ig,
        interp1d,
        kinematics,
        mlines,
        np,
        plt,
        trajectory,
        u,
        z_interp,
    )


@app.cell
def _(YEAR_MKS):
    # =====================================================================
    #  Configuración propia de este notebook
    # =====================================================================

    # --- Bloque A: criterio de paso en redshift ---
    DENSITY_RATIO = 0.95      # (1+z)^3 / (1+z+dz)^3
    DT_FIXED_MYR = 3.0        # intervalo temporal fijo de la Fig. 01
    Z_ORIGIN = 30.0           # origen temporal de las Figs. 03 y 05

    # --- Bloque B: inyección de electrones ---
    Z_END_PLOT = 2.0                          # hasta dónde se integra cada trayectoria
    Z_REF_LINE = 5.5                          # línea roja de referencia
    Z_REF_BG = 10.0                           # origen de las curvas de umbral de fondo
    INITIAL_ENERGIES_EV = [1e12, 1e4]
    N_T_TRAJ = 5000                           # nodos por trayectoria

    # z de inyección y su color. La Fig. 06 omite z=30, tal como el notebook original.
    Z_COLORS = {
        30.0: "#000000",
        20.0: "#8c564b",
        10.0: "#1f77b4",
        9.0: "#ff7f0e",
        8.0: "#2ca02c",
        7.0: "#d62728",
        6.0: "#9467bd",
    }
    Z_LIST_FULL = [30.0, 20.0, 10.0, 9.0, 8.0, 7.0, 6.0]
    Z_LIST_FIG06 = [20.0, 10.0, 9.0, 8.0, 7.0, 6.0]

    SEC_TO_MYR = 1.0 / (YEAR_MKS * 1e6)
    return (
        DENSITY_RATIO,
        DT_FIXED_MYR,
        INITIAL_ENERGIES_EV,
        N_T_TRAJ,
        SEC_TO_MYR,
        Z_COLORS,
        Z_END_PLOT,
        Z_LIST_FIG06,
        Z_LIST_FULL,
        Z_ORIGIN,
        Z_REF_BG,
        Z_REF_LINE,
    )


@app.cell
def _(DENSITY_RATIO, Planck18, interp1d, np, u):
    # =====================================================================
    #  Bloque A: relación Delta z <-> Delta t
    # =====================================================================
    def delta_z_from_density(z, target_ratio=None):
        """
        Delta z tal que (1+z)^3 / (1+z+Dz)^3 = target_ratio.
        Solución analítica: Dz = (1+z) [ (1/ratio)^(1/3) - 1 ].
        """
        r = DENSITY_RATIO if target_ratio is None else target_ratio
        return (np.asarray(z, dtype=float) + 1.0) * ((1.0 / r) ** (1.0 / 3.0) - 1.0)


    def delta_z_label(target_ratio=None):
        """Etiqueta LaTeX con el factor REALMENTE usado (el original decía 10^{2/3}-1)."""
        r = DENSITY_RATIO if target_ratio is None else target_ratio
        f = (1.0 / r) ** (1.0 / 3.0) - 1.0
        return rf"$\Delta z = (1+z)\,({f:.5f})$"


    # Inversión exacta edad -> z, sobre una grilla que llega hasta z=1500.
    _z_fine = np.insert(np.geomspace(1e-5, 1500.0, 50000), 0, 0.0)
    _age_fine = Planck18.age(_z_fine).to(u.Myr).value
    age_to_z = interp1d(_age_fine[::-1], _z_fine[::-1], kind="cubic",
                        bounds_error=False, fill_value=np.nan)


    def delta_z_from_time(z, dt_myr):
        """Delta z que corresponde a retroceder dt_myr desde cada z (cálculo exacto)."""
        z = np.asarray(z, dtype=float)
        age_now = Planck18.age(z).to(u.Myr).value
        return age_to_z(age_now - dt_myr) - z


    def delta_t_from_delta_z(z, dz):
        """Intervalo temporal en Myr entre z y z+dz."""
        return (Planck18.age(z) - Planck18.age(z + dz)).to(u.Myr).value


    def add_redshift_top_axis(ax, z_array, t_array, z_ticks):
        """Eje superior con marcas de redshift sobre un eje x de tiempo."""
        ax2 = ax.twiny()
        t_ticks = np.interp(z_ticks[::-1], z_array[::-1], t_array[::-1])[::-1]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels([f"{z:g}" for z in z_ticks])
        ax2.set_xlabel(r"Corrimiento al rojo $z$", fontsize=12, labelpad=10)
        ax2.grid(False)
        return ax2

    return (
        add_redshift_top_axis,
        delta_t_from_delta_z,
        delta_z_from_density,
        delta_z_from_time,
        delta_z_label,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 01 — $\Delta z$ para un $\Delta t$ constante de 3 Myr
    Cálculo exacto por inversión de la edad cósmica: $\Delta z(z) = z\big(t(z)-\Delta t\big) - z$.
    """)
    return


@app.cell
def _(DT_FIXED_MYR, delta_z_from_time, np, plt):
    # ---------------------------------------------------------------- FIG 01
    fig_01, _ax = plt.subplots(figsize=(9, 6))

    _z = np.linspace(0.0, 30.0, 1000)
    _ax.plot(_z, delta_z_from_time(_z, DT_FIXED_MYR),
             label="Cálculo exacto (Interpolación)", color="black", lw=2)

    _ax.set_xlabel(r"Redshift $z$", fontsize=13)
    _ax.set_ylabel(r"Intervalo $\Delta z$", fontsize=13)
    _ax.set_title(rf"Correspondencia de $\Delta z$ para "
                  rf"$\Delta t = {DT_FIXED_MYR}$ Myr en función de $z$", fontsize=14)
    _ax.legend(fontsize=11)
    _ax.grid(True, linestyle=":", alpha=0.7, color="grey")
    fig_01.tight_layout()
    fig_01
    return (fig_01,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 02 — $\Delta z$ para una caída de densidad del 5 %
    $\Delta z = (1+z)\big[(1/0.95)^{1/3}-1\big]$, lineal en $1+z$.

    *La leyenda original decía $\Delta z=(z+1)(10^{2/3}-1)$: ese factor vale 3.64 y el
    implementado 0.0172, un factor 211 de diferencia. Sobraba de una versión con otro
    `target_ratio`; acá la etiqueta se genera a partir del valor real.*
    """)
    return


@app.cell
def _(DENSITY_RATIO, delta_z_from_density, delta_z_label, np, plt):
    # ---------------------------------------------------------------- FIG 02
    fig_02, _ax = plt.subplots(figsize=(8, 6))

    _z = np.linspace(5.5, 30, 1000)
    _dz = delta_z_from_density(_z)
    _ax.plot(_z, _dz, color="navy", lw=2, label=delta_z_label())

    _ax.set_xlabel(r"Corrimiento al rojo $z$", fontsize=14)
    _ax.set_ylabel(r"Intervalo $\Delta z$", fontsize=14)
    _ax.set_title(rf"Relación para $\frac{{(z+1)^3}}{{(z+\Delta z+1)^3}} = {DENSITY_RATIO}$",
                  fontsize=16, pad=15)
    _ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    _ax.set_xlim(30, 5.4)
    _ax.set_ylim(0, _dz.max())
    _ax.legend(fontsize=12, loc="lower left")
    fig_02.tight_layout()
    fig_02
    return (fig_02,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 03 — $\Delta z$ frente al tiempo transcurrido desde $z=30$
    """)
    return


@app.cell
def _(
    Planck18,
    YEAR_MKS,
    Z_ORIGIN,
    add_redshift_top_axis,
    age_s,
    delta_z_from_density,
    np,
    plt,
    u,
):
    # ---------------------------------------------------------------- FIG 03
    fig_03, _ax = plt.subplots(figsize=(8, 6))

    _z = np.linspace(Z_ORIGIN, 5.5, 1000)
    _elapsed = Planck18.age(_z).to(u.Myr).value - age_s(Z_ORIGIN) / YEAR_MKS / 1e6
    _dz = delta_z_from_density(_z)

    _ax.plot(_elapsed, _dz, color="darkred", lw=2, label=r"$\Delta z$ (Planck 2018)")
    _ax.set_xlabel(rf"Tiempo transcurrido desde $z={Z_ORIGIN:g}$ [Myr]", fontsize=14)
    _ax.set_ylabel(r"Intervalo $\Delta z$", fontsize=14)
    _ax.set_title(rf"Intervalo $\Delta z$ en función del tiempo cósmico "
                  rf"($z_i = {Z_ORIGIN:g}$)", fontsize=15, pad=15)
    _ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    _ax.set_xlim(0, _elapsed.max())
    _ax.legend(fontsize=12, loc="upper right")

    add_redshift_top_axis(_ax, _z, _elapsed, np.array([30, 20, 15, 10, 7, 5.5]))
    fig_03.tight_layout()
    fig_03
    return (fig_03,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 04 — $\Delta t$ asociado a la caída de densidad del 5 %
    """)
    return


@app.cell
def _(DENSITY_RATIO, delta_t_from_delta_z, delta_z_from_density, np, plt):
    # ---------------------------------------------------------------- FIG 04
    fig_04, _ax = plt.subplots(figsize=(8, 6))

    _z = np.linspace(5.5, 30, 1000)
    _dt = delta_t_from_delta_z(_z, delta_z_from_density(_z))

    _ax.plot(_z, _dt, color="teal", lw=2, label=r"$\Delta t$ (Planck 2018)")
    _ax.set_xlabel(r"Corrimiento al rojo $z$", fontsize=14)
    _ax.set_ylabel(r"Intervalo $\Delta t$ [Myr]", fontsize=14)
    _ax.set_title(rf"Intervalo temporal para "
                  rf"$\frac{{(z+1)^3}}{{(z+\Delta z+1)^3}} = {DENSITY_RATIO}$",
                  fontsize=16, pad=15)
    _ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    _ax.set_xlim(_z.max(), 5.5)
    _ax.set_ylim(0, _dt.max() * 1.05)
    _ax.legend(fontsize=12, loc="upper right")
    fig_04.tight_layout()
    fig_04
    return (fig_04,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 05 — $\Delta z$ y $\Delta t$ juntos, frente al tiempo transcurrido
    """)
    return


@app.cell
def _(
    DENSITY_RATIO,
    Planck18,
    YEAR_MKS,
    Z_ORIGIN,
    add_redshift_top_axis,
    age_s,
    delta_t_from_delta_z,
    delta_z_from_density,
    np,
    plt,
    u,
):
    # ---------------------------------------------------------------- FIG 05
    fig_05, _a1 = plt.subplots(figsize=(9, 6))

    _z = np.linspace(Z_ORIGIN, 5.5, 500)
    _dz = delta_z_from_density(_z)
    _elapsed = Planck18.age(_z).to(u.Myr).value - age_s(Z_ORIGIN) / YEAR_MKS / 1e6
    _dt = delta_t_from_delta_z(_z, _dz)

    _c1, _c2 = "darkred", "navy"
    _a1.set_xlabel(rf"Tiempo transcurrido desde $z={Z_ORIGIN:g}$ [Myr]", fontsize=14)
    _a1.set_ylabel(r"Intervalo $\Delta z$", fontsize=14, color=_c1)
    _l1 = _a1.plot(_elapsed, _dz, color=_c1, lw=2, label=r"$\Delta z$")
    _a1.tick_params(axis="y", labelcolor=_c1)
    _a1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    _a1.set_xlim(0, _elapsed.max())

    _a2 = _a1.twinx()
    _a2.set_ylabel(r"Intervalo $\Delta t$ asociado [Myr]", fontsize=14, color=_c2)
    _l2 = _a2.plot(_elapsed, _dt, color=_c2, lw=2, ls="-.", label=r"$\Delta t$ [Myr]")
    _a2.tick_params(axis="y", labelcolor=_c2)
    _a2.grid(False)

    _lines = _l1 + _l2
    _a1.legend(_lines, [l.get_label() for l in _lines], loc="center left", fontsize=12)

    add_redshift_top_axis(_a1, _z, _elapsed, np.array([30, 20, 15, 10, 7, 5.5]))
    _a1.set_title(rf"Evolución de $\Delta z$ y $\Delta t$ para "
                  rf"$\frac{{(z+1)^3}}{{(z+\Delta z+1)^3}} = {DENSITY_RATIO}$",
                  fontsize=15, pad=20)
    fig_05.tight_layout()
    fig_05
    return (fig_05,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Umbrales cinemáticos de dispersión Compton sobre el CMB

    Reproduce las dos celdas de salida numérica del notebook original (la segunda era un
    superconjunto de la primera, así que quedan unificadas). Las raíces del integrando de
    Planck $x^2/(e^x-1)$ son constantes; sólo escalan con $kT_{\rm CMB}(z)$.
    """)
    return


@app.cell
def _(
    EV_MKS,
    K_B,
    T_CMB_0,
    X_MAX_PLANCK,
    X_MIN_PLANCK,
    compute_threshold_curves,
    np,
):
    # ---- Umbrales cinemáticos: reporte numérico ----
    def report_thresholds(z, min_sim_eV=10.2, target_upscatter_eV=1e3,
                          losing_tolerance=1e-2):
        thermal = K_B * T_CMB_0 * (1.0 + z)
        ke_cut, ke_99 = compute_threshold_curves(np.array([z]), min_sim_eV,
                                                 target_upscatter_eV)
        print(f"--- Parámetros del CMB (z={z:.6f}) ---")
        print(f"Energía térmica del CMB       = {thermal:e} J")
        print(f"\n--- Corte inferior (param_values[6], tolerancia "
              f"{losing_tolerance:.0%}) ---")
        print(f"X_max distribución BB         = {X_MAX_PLANCK:e}")
        print(f"MaxPhotonEnergy BB            = {X_MAX_PLANCK * thermal / EV_MKS:e} eV")
        print(f"Corte de energía cinética     = {float(ke_cut[0]):e} eV")
        print(f"\n--- Blanco de 99 % de upscatter (>= {target_upscatter_eV:.1e} eV) ---")
        print(f"X_min distribución BB (1 %)   = {X_MIN_PLANCK:e}")
        print(f"MinPhotonEnergy (99 % excede) = {X_MIN_PLANCK * thermal / EV_MKS:e} eV")
        print(f"Energía cinética requerida    = {float(ke_99[0]):e} eV\n")
        return float(ke_cut[0]), float(ke_99[0])


    KE_CUT_Z10, KE_99_Z10 = report_thresholds(10.0)
    return


@app.cell
def _(
    C_LIGHT,
    EV_MKS,
    INITIAL_ENERGIES_EV,
    KE_99_ARRAY,
    KE_CUTOFF_ARRAY,
    MEGAPARSEC_MKS,
    N_T_TRAJ,
    SEC_TO_MYR,
    T_ARRAY_INTERP,
    Z_ARRAY_INTERP,
    Z_COLORS,
    Z_END_PLOT,
    Z_REF_BG,
    Z_REF_LINE,
    age_s,
    compute_threshold_curves,
    cumulative_trapezoid,
    kinematics,
    mlines,
    np,
    plt,
    trajectory,
    z_interp,
):
    # =====================================================================
    #  Bloque B: panel reutilizable de trayectorias por redshift de inyección
    #  Las cinco figuras 06-10 comparten las MISMAS integraciones: la caché de
    #  igm_losses.trajectory() las calcula una sola vez por (K, toggles, malla).
    # =====================================================================
    XMODES = {
        "z":        r"Redshift $z$",
        "age":      r"Edad del Universo [Myr]",
        "path":     r"Distance [Mpc]",
        "comoving": r"Comoving Distance [Mpc]",
        "proper":   r"Proper Distance [Mpc]",
    }


    def _x_of_trajectory(xmode, t_a, z_a, K_eV):
        """Abscisa de una trayectoria según el modo pedido."""
        if xmode == "z":
            return z_a
        if xmode == "age":
            return t_a * SEC_TO_MYR
        _, _, _, v = kinematics(K_eV * EV_MKS)
        if xmode == "path":                       # longitud de camino propia recorrida
            return cumulative_trapezoid(v, t_a, initial=0.0) / MEGAPARSEC_MKS
        com = cumulative_trapezoid(v * (1.0 + z_a), t_a, initial=0.0)
        if xmode == "comoving":                   # distancia comóvil recorrida
            return com / MEGAPARSEC_MKS
        return com / (1.0 + z_a) / MEGAPARSEC_MKS  # separación propia actual


    def _background_curves(xmode):
        """
        (x, ke_cutoff, ke_99, x_ref) para las curvas de umbral y la línea de z=5.5.

        Para los modos de distancia se usa una malla temporal CRECIENTE desde
        t(Z_REF_BG): el original enmascaraba T_ARRAY_INTERP, que está ordenado de
        forma decreciente, y cumulative_trapezoid devolvía distancias NEGATIVAS
        (invisibles bajo `set_xlim(left=0)` en la Fig. 10).
        """
        if xmode == "z":
            return Z_ARRAY_INTERP, KE_CUTOFF_ARRAY, KE_99_ARRAY, Z_REF_LINE
        if xmode == "age":
            return (T_ARRAY_INTERP * SEC_TO_MYR, KE_CUTOFF_ARRAY, KE_99_ARRAY,
                    age_s(Z_REF_LINE) * SEC_TO_MYR)

        t0, t1 = age_s(Z_REF_BG), age_s(Z_END_PLOT)
        t_bg = np.linspace(t0, t1, 4000)
        z_bg = z_interp(t_bg)
        ke_c, ke_9 = compute_threshold_curves(z_bg)

        if xmode == "path":                       # la luz recorre c * (t - t0)
            x = C_LIGHT * (t_bg - t0) / MEGAPARSEC_MKS
            x_ref = C_LIGHT * (age_s(Z_REF_LINE) - t0) / MEGAPARSEC_MKS
            return x, ke_c, ke_9, x_ref

        com = cumulative_trapezoid(C_LIGHT * (1.0 + z_bg), t_bg, initial=0.0)
        if xmode == "comoving":
            x = com / MEGAPARSEC_MKS
        else:
            x = com / (1.0 + z_bg) / MEGAPARSEC_MKS
        x_ref = float(np.interp(age_s(Z_REF_LINE), t_bg, x))
        return x, ke_c, ke_9, x_ref


    def draw_zi_panel(ax, tog_key, title, xmode, z_list, label_dx=0.1, label_ha="right"):
        """Traza las trayectorias de todos los z_i y las curvas de fondo en un panel."""
        t_end = age_s(Z_END_PLOT)

        for z_start in z_list:
            t_eval = np.logspace(np.log10(age_s(z_start)), np.log10(t_end), N_T_TRAJ)
            for K_ini_eV in INITIAL_ENERGIES_EV:
                t_a, z_a, K_a = trajectory(K_ini_eV, tog_key, t_eval)
                ax.plot(_x_of_trajectory(xmode, t_a, z_a, K_a), K_a,
                        color=Z_COLORS[z_start], lw=1.5)
                if z_start == z_list[0]:
                    x0 = _x_of_trajectory(xmode, t_a, z_a, K_a)[0]
                    dx = x0 * label_dx if xmode == "age" else label_dx
                    ax.text(x0 + dx if xmode != "age" else x0 * (1 - label_dx),
                            K_ini_eV, rf"$10^{{{int(np.log10(K_ini_eV))}}}$ eV",
                            fontsize=11, va="center", ha=label_ha, color="black")

        x_bg, ke_c, ke_9, x_ref = _background_curves(xmode)
        l_cut, = ax.plot(x_bg, ke_c, color="black", ls="--", alpha=0.8,
                         label=r"Fotoionización secundaria")
        l_99, = ax.plot(x_bg, ke_9, color="black", ls="-.", alpha=0.8,
                        label=r"99 \% upscatter a 1 keV" if plt.rcParams["text.usetex"]
                        else "99% upscatter a 1 keV")
        l_ref = ax.axvline(x_ref, color="red", ls="--", alpha=0.5,
                           label=rf"$z_{{\mathrm{{ref}}}}={Z_REF_LINE}$")

        ax.set_yscale("log")
        ax.set_ylim(1e0, 3e13)
        ax.set_title(title, fontsize=15, pad=15)
        ax.set_ylabel(r"Energía cinética $K_{\mathrm{e}}$ [eV]", fontsize=14)
        ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")

        handles = [mlines.Line2D([0], [0], color=Z_COLORS[z], lw=2, label=rf"$z_i = {z:g}$")
                   for z in z_list] + [l_cut, l_99, l_ref]
        ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=True,
                  edgecolor="black", facecolor="white", ncol=2)
        return ax


    def zi_figure(xmode, z_list, tog_top="all", tog_bot="all",
                  title_bot=None, **panel_kw):
        """Figura de dos paneles (baseline arriba, variante abajo)."""
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 14.0), sharex=True)
        draw_zi_panel(axes[0], tog_top, "Todos los mecanismos de pérdida activos",
                      xmode, z_list, **panel_kw)
        draw_zi_panel(axes[1], tog_bot,
                      title_bot or "Todos los mecanismos activos",
                      xmode, z_list, **panel_kw)
        axes[1].set_xlabel(XMODES[xmode], fontsize=14)
        return fig, axes

    return (zi_figure,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 06 — $K_e$ frente al redshift, según el $z$ de inyección
    Panel inferior sin Compton inverso. Omite $z_i=30$, como el notebook original.

    *El original construía `z_interp` sobre una grilla que llegaba sólo a $z=15$ y luego
    extrapolaba: en $z_i=20$ el error era $-0.14$ y en $z_i=30$ llegaba a $-2.8$. La grilla
    compartida de `igm_losses` llega a $z=35$ y las trayectorias arrancan en el $z$ exacto.*
    """)
    return


@app.cell
def _(Z_END_PLOT, Z_LIST_FIG06, zi_figure):
    # ---------------------------------------------------------------- FIG 06
    fig_06, _axes = zi_figure("z", Z_LIST_FIG06, tog_top="all", tog_bot="no_ic",
                              title_bot="Sin considerar: Compton")
    for _ax in _axes:
        _ax.set_xlim(max(Z_LIST_FIG06) + 0.8, Z_END_PLOT)
    fig_06
    return (fig_06,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 07 — $K_e$ frente a la edad del universo
    *El original fijaba `xlim = (age(35), 500)` Myr: de los 7 valores de $z_i$ sólo entraban
    3 ($z_i=30,20,10$); las trayectorias de $z_i=9,8,7,6$ arrancan entre 544 y 929 Myr y
    quedaban fuera del gráfico, aunque seguían apareciendo en la leyenda. El límite ahora
    llega a 1200 Myr. Además $z_i=30$ y $z_i=10$ compartían el color `#1f77b4`.*
    """)
    return


@app.cell
def _(SEC_TO_MYR, Z_LIST_FULL, age_s, zi_figure):
    # ---------------------------------------------------------------- FIG 07
    fig_07, _axes = zi_figure("age", Z_LIST_FULL, tog_top="all", tog_bot="no_ic",
                              title_bot="Sin considerar: Compton", label_dx=0.15)
    for _ax in _axes:
        _ax.set_xscale("log")
        _ax.set_xlim(age_s(35.0) * SEC_TO_MYR, 1200.0)
    fig_07
    return (fig_07,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 08 — $K_e$ frente a la **longitud de camino propia** recorrida
    $s(t)=\int_{t_i}^{t} v\,dt'$: el largo del recorrido del electrón medido en el marco
    propio. Las curvas negras de fondo son la distancia que recorrería la luz, $c\,(t-t_{10})$.

    Los dos paneles son idénticos: en el notebook original `cooling_toggles_custom` tenía
    los seis mecanismos en `True`, igual que `cooling_toggles_baseline`.
    """)
    return


@app.cell
def _(Z_LIST_FULL, zi_figure):
    # ---------------------------------------------------------------- FIG 08
    fig_08, _axes = zi_figure("path", Z_LIST_FULL, label_dx=0.5, label_ha="left")
    for _ax in _axes:
        _ax.set_xlim(-10.0, 600.0)
    fig_08
    return (fig_08,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 09 — $K_e$ frente a la **distancia comóvil** recorrida
    $\chi(t)=\int v\,(1+z)\,dt'$: el mismo recorrido medido en coordenadas comóviles.
    Los dos paneles son idénticos, como en el original.
    """)
    return


@app.cell
def _(Z_LIST_FULL, zi_figure):
    # ---------------------------------------------------------------- FIG 09
    fig_09, _axes = zi_figure("comoving", Z_LIST_FULL, label_dx=0.5, label_ha="left")
    for _ax in _axes:
        _ax.set_xlim(-100.0, 2000.0)
    fig_09
    return (fig_09,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 10 — $K_e$ frente a la **separación propia** respecto al origen
    $D_p(t)=a(t)\int v/a\,dt' = \chi(t)/(1+z)$: la distancia propia *actual* entre el
    electrón y su punto de inyección. No es lo mismo que la Fig. 08 — el camino recorrido
    se comprime al expandirse el universo por detrás del electrón.

    *En el original las curvas de umbral de fondo y la línea roja de $z_{\rm ref}$ salían
    **negativas** ($-445$ y $-221$ Mpc) porque `cumulative_trapezoid` se aplicaba sobre
    `t_array_interp` enmascarado, que está ordenado de mayor a menor tiempo; con
    `set_xlim(left=0)` quedaban invisibles.*
    """)
    return


@app.cell
def _(Z_LIST_FULL, zi_figure):
    # ---------------------------------------------------------------- FIG 10
    fig_10, _axes = zi_figure("proper", Z_LIST_FULL, label_dx=0.5, label_ha="left")
    for _ax in _axes:
        _ax.set_xlim(left=0)
    fig_10
    return (fig_10,)


@app.cell
def _(
    INITIAL_ENERGIES_EV,
    Z_LIST_FULL,
    fig_01,
    fig_02,
    fig_03,
    fig_04,
    fig_05,
    fig_06,
    fig_07,
    fig_08,
    fig_09,
    fig_10,
    ig,
):
    # =====================================================================
    #  Verificación: inventario de figuras y ejes
    # =====================================================================
    FIGURES = {
        "01 Delta z (Delta t = 3 Myr)": fig_01,
        "02 Delta z (densidad 0.95)": fig_02,
        "03 Delta z vs tiempo (z=30)": fig_03,
        "04 Delta t (densidad 0.95)": fig_04,
        "05 Delta z y Delta t juntos": fig_05,
        "06 K vs redshift": fig_06,
        "07 K vs edad del universo": fig_07,
        "08 K vs camino propio": fig_08,
        "09 K vs distancia comóvil": fig_09,
        "10 K vs separación propia": fig_10,
    }

    _rows, _tot = [], 0
    for _name, _f in FIGURES.items():
        _na = len(_f.axes)
        _nl = sum(len(a.lines) for a in _f.axes)
        _tot += _na
        _rows.append(f"  {_name:<32s} ejes={_na}  lineas={_nl:>3d}")

    print(f"FIGURAS: {len(FIGURES)}   EJES TOTALES: {_tot}\n" + "\n".join(_rows))
    print(f"\nIntegraciones de trayectoria cacheadas: {len(ig._TRAJ_CACHE)} "
          f"(el original repetía {len(Z_LIST_FULL) * len(INITIAL_ENERGIES_EV) * 2 * 5} "
          f"a lo largo de las 5 figuras)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Diagnóstico: errores detectados en `z_vs_delta_z.ipynb`

    ### Numéricos
    * **Extrapolación de `z_interp` fuera de rango.** Las Figs. 06 y 07 corrieron con
      los interpoladores de la celda 8 (`z_max_interp = 15`), pero inyectan electrones
      en $z_i = 20$ y $30$. El spline cúbico extrapolaba: error $-0.14$ en $z=20$ y
      $-2.8$ en $z=30$. La grilla compartida llega a $z=35$.
    * **Distancias negativas en la Fig. 10.** `cumulative_trapezoid` se aplicaba sobre
      `t_array_interp` enmascarado, que está ordenado de mayor a menor $t$. Las curvas
      de umbral quedaban entre $-445$ y $0$ Mpc y la línea de $z_{\rm ref}=5.5$ en
      $-221$ Mpc; `set_xlim(left=0)` las ocultaba por completo.
    * **Kernel Klein–Nishina** con el factor $q$ faltante, **excitación truncada a
      $n=2,3$**, **sin bremsstrahlung** y **piso térmico $kT$**: los mismos cuatro
      problemas que en `perdidas_energia_z10.py`, corregidos vía los interruptores de
      `igm_losses.py`.

    ### De graficación
    * **Fig. 07:** `xlim` cortado en 500 Myr dejaba fuera 4 de las 7 trayectorias
      ($z_i = 9, 8, 7, 6$ arrancan entre 544 y 929 Myr) mientras la leyenda seguía
      listando los 7 colores. Ahora llega a 1200 Myr.
    * **Fig. 07:** $z_i=30$ y $z_i=10$ compartían el color `#1f77b4`. Las celdas
      posteriores ya lo habían corregido a negro; se unifica.
    * **Fig. 02:** la leyenda anunciaba $\Delta z=(z+1)(10^{2/3}-1)$, factor 3.6416,
      cuando lo implementado es $(1/0.95)^{1/3}-1 = 0.01724$ — un factor 211. La
      etiqueta ahora se genera desde el valor real.
    * **Figs. 06, 08, 09, 10:** la segunda curva de umbral se pasaba a la leyenda sin
      `label`, produciendo una entrada `_childN`. Ahora está etiquetada en las cinco.
    * **Fig. 06:** el label decía `Fotoionización secundaria)` con un paréntesis de más.
    * **Figs. 08, 09, 10:** `cooling_toggles_custom` idéntico a `baseline`, de modo que
      los dos paneles coinciden y el título del inferior dice "Todos los mecanismos
      activos". Se conserva tal cual, documentado.
    * **Fig. 10:** usaba `np.trapz`, renombrado a `trapezoid` en NumPy 2.

    ### Redundancias eliminadas
    | Elemento | Original | Ahora |
    |---|---|---|
    | Bloques de constantes MKS | 6 | 1 (en `igm_losses`) |
    | `compute_F_KN` (2-D `nquad` × 50) | 5 veces | 1, con cuadratura de Gauss–Legendre vectorizada |
    | Tablas colisionales | 5 veces | 1, cacheada en `.igm_losses_cache.npz` |
    | `Planck18.age` sobre 10 000 nodos | 5 veces | 1 |
    | `calculate_delta_z` | 4 definiciones | 1 |
    | `get_cooling_rate_evaluator` | 5 definiciones | 1 (`loss_rates`) |
    | `draw_phase_panel` | 5 definiciones casi iguales | 1 (`draw_zi_panel` + `zi_figure`) |
    | Integraciones de trayectoria | 140 | 28 (caché por energía, toggles y malla) |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
