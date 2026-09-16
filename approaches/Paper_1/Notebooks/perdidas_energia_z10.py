import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="Perdidas de energia - electron IGM (z=10 -> 5.5)",
)


@app.cell
def _():
    # =====================================================================
    #  Toda la física vive en igm_losses.py, compartido con z_vs_delta_z.py:
    #  constantes, cosmología, kernel Klein-Nishina, tablas colisionales,
    #  loss_rates() e integradores. Editá los interruptores de corrección
    #  ahí y marimo recalcula los dos notebooks.
    # =====================================================================
    import igm_losses as ig
    from igm_losses import (ALL_ON, B0, COSMIC_T_YEARS, COSMIC_T_YEARS_EL, C_LIGHT, E0, 
        ELECTRON_CHARGE_MKS, EV_MKS, FIX_BREMS_EE, FIX_KN_KERNEL, H_interp, 
        ION_FRACTION, KE_99_ARRAY, KE_CUTOFF_ARRAY, ListedColormap, MECH_CMAP, 
        MECH_COLORS, MECH_KEYS, MECH_NAMES, MEGAPARSEC_MKS, N_MECH_MAP, PARSEC_MKS, 
        PHASE_ENERGIES_EV, PREFACTOR_BREMS, Planck18, THOMSON_CROSS_SECTION_MKS, 
        T_ARRAY_INTERP, T_EVAL, T_EVAL_DENSE, T_EVAL_EL, T_FINAL, T_INIT, U_CMB_0_J_M3, 
        VACUUM_PERMEABILITY, YEAR_MKS, Z_EVAL, Z_EVAL_EL, Z_FINAL, Z_INIT, 
        add_threshold_curves, colored_trajectory, cumulative_trapezoid, energy_label, 
        f_kn_kernel_grid, grid_minor, integrate, kinematics, loss_rates, 
        make_evt_cached, make_rhs, make_time_grid, mech_legend, n_HI, np, 
        phase_space_map, plt, sample_with_floor, solve_ivp, thermal_floor, ticker, 
        toggles, total_loss, trajectory, u, z_at_value, z_interp
    )

    print(f"igm_losses cargado - {ig.N_MECH_MAP} mecanismos, "
          f"FIX_KN_KERNEL={ig.FIX_KN_KERNEL}, EXC_NMAX={ig.EXC_NMAX}")
    return (
        ALL_ON,
        B0,
        COSMIC_T_YEARS,
        COSMIC_T_YEARS_EL,
        C_LIGHT,
        E0,
        ELECTRON_CHARGE_MKS,
        EV_MKS,
        FIX_BREMS_EE,
        FIX_KN_KERNEL,
        H_interp,
        ION_FRACTION,
        KE_99_ARRAY,
        KE_CUTOFF_ARRAY,
        ListedColormap,
        MECH_CMAP,
        MECH_COLORS,
        MECH_KEYS,
        MECH_NAMES,
        MEGAPARSEC_MKS,
        N_MECH_MAP,
        PARSEC_MKS,
        PHASE_ENERGIES_EV,
        PREFACTOR_BREMS,
        Planck18,
        THOMSON_CROSS_SECTION_MKS,
        T_ARRAY_INTERP,
        T_EVAL,
        T_EVAL_DENSE,
        T_EVAL_EL,
        T_FINAL,
        T_INIT,
        U_CMB_0_J_M3,
        VACUUM_PERMEABILITY,
        YEAR_MKS,
        Z_EVAL,
        Z_EVAL_EL,
        Z_FINAL,
        Z_INIT,
        add_threshold_curves,
        colored_trajectory,
        cumulative_trapezoid,
        energy_label,
        f_kn_kernel_grid,
        grid_minor,
        integrate,
        kinematics,
        loss_rates,
        make_evt_cached,
        make_rhs,
        make_time_grid,
        mech_legend,
        n_HI,
        np,
        phase_space_map,
        plt,
        sample_with_floor,
        solve_ivp,
        thermal_floor,
        ticker,
        toggles,
        total_loss,
        trajectory,
        u,
        z_at_value,
        z_interp,
    )


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Pérdidas de energía de un electrón en el medio intergaláctico

        Evolución de la energía cinética $K_e$ de un electrón inyectado en $z_{\rm ini}=10$
        y propagado hasta $z_{\rm fin}=5.5$, considerando:

        | # | Mecanismo | Referencia |
        |---|-----------|------------|
        | 0 | Expansión adiabática | $\dot K = -H(z)\,p^2c^2/E$ |
        | 1 | Sincrotrón | $B(z)=B_0(1+z)^2$ |
        | 2 | Compton inverso (CMB, Klein–Nishina) | Blumenthal & Gould (1970) |
        | 3 | Coulomb (plasma) | Gould (1972) |
        | 4 | Excitación colisional H(1s)→H(np) | Stone & Kim (2002) |
        | 5 | Ionización colisional H(1s)→H⁺+e⁻ | Kim et al. (2000) (RBEB) |
        | 6 | Bremsstrahlung | Blumenthal & Gould (1970) |

        **Estructura:** constantes → configuración → cosmología → tablas precalculadas →
        modelo de pérdidas unificado → **una figura por celda** (25 figuras).
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 01 — Giroradio $r_g(t)$, enfriamiento sincrotrón puro
    Modelo de ángulo de paso fijo ($\sin\theta = 0.5$, tal como en el notebook original).
    """)
    return


@app.cell
def _(
    B0,
    C_LIGHT,
    E0,
    ELECTRON_CHARGE_MKS,
    EV_MKS,
    PARSEC_MKS,
    THOMSON_CROSS_SECTION_MKS,
    T_FINAL,
    T_INIT,
    VACUUM_PERMEABILITY,
    YEAR_MKS,
    Z_FINAL,
    Z_INIT,
    energy_label,
    grid_minor,
    make_time_grid,
    np,
    plt,
    solve_ivp,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 01
    fig_01, _ax = plt.subplots(figsize=(8, 5))

    _sin_theta = 0.5   # NOTA: el comentario original decia "90 grados" (=> 1.0)
    _t, _ty, _z = make_time_grid(n=500, mode="elapsed")
    _B = B0 * (1 + _z) ** 2


    def _rhs01(t, K):
        _Kv = max(K[0], 0.0)
        _UB = (B0 * (1 + float(z_interp(t))) ** 2) ** 2 / (2 * VACUUM_PERMEABILITY)
        _C = 2.0 * THOMSON_CROSS_SECTION_MKS * C_LIGHT * _UB * _sin_theta**2
        return [-_C * ((_Kv / E0 + 1.0) ** 2 - 1.0)]


    for _K0 in np.array([1e6, 1e8, 1e10, 1e12, 1e14]):
        _s = solve_ivp(_rhs01, (_t[0], _t[-1]), [_K0 * EV_MKS], t_eval=_t,
                       method="Radau", rtol=1e-8, atol=1e-20)
        _K = np.maximum(_s.y[0], 0.0)
        _p = np.sqrt(_K * (_K + 2 * E0)) / C_LIGHT
        _rg = _p * _sin_theta / (ELECTRON_CHARGE_MKS * _B) / PARSEC_MKS
        _ax.plot(_ty, _rg, label=energy_label(_K0), lw=2)

    _ax.set(xscale="log", yscale="log",
            xlabel=f"Tiempo cósmico [yr] ($z={Z_INIT}$ - $z={Z_FINAL}$)",
            ylabel="Giroradio $r_g$ (pc)")
    _ax.set_xlim(T_INIT / YEAR_MKS, T_FINAL / YEAR_MKS)
    _ax.grid(True, which="major", ls="-", alpha=0.5)
    grid_minor(_ax)
    _ax.legend(loc="upper right", fontsize=13)
    fig_01.tight_layout()
    fig_01
    return (fig_01,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 02 — Distancia recorrida, $\vec v_{\rm dir}=[1,0,0]$
    Envolvente 3D entre $|r_g - r_g^{\rm ini}|$ y $r_g + r_g^{\rm ini}$ más la deriva helicoidal.
    """)
    return


@app.cell
def _(
    B0,
    COSMIC_T_YEARS_EL,
    C_LIGHT,
    E0,
    ELECTRON_CHARGE_MKS,
    EV_MKS,
    PARSEC_MKS,
    THOMSON_CROSS_SECTION_MKS,
    T_EVAL_EL,
    T_FINAL,
    T_INIT,
    VACUUM_PERMEABILITY,
    YEAR_MKS,
    Z_EVAL_EL,
    Z_INIT,
    cumulative_trapezoid,
    energy_label,
    grid_minor,
    np,
    plt,
    solve_ivp,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 02
    fig_02, _ax = plt.subplots(figsize=(8.0, 5.0))

    _v_dir = np.array([1.0, 0.0, 0.0])
    _vn = np.linalg.norm(_v_dir)
    _sin_a = np.sqrt(_v_dir[0] ** 2 + _v_dir[2] ** 2) / _vn
    _cos_a = _v_dir[1] / _vn


    def _rhs02(t, K):
        _Kv = max(K[0], 0.0)
        _p2 = _Kv**2 + 2 * _Kv * E0
        _UB = (B0 * (1 + float(z_interp(t))) ** 2) ** 2 / (2 * VACUUM_PERMEABILITY)
        _C = (4 / 3) * THOMSON_CROSS_SECTION_MKS * C_LIGHT * _UB
        return [-(_C / E0**2) * _p2 * (1.5 * _sin_a**2)]


    _B_ev = B0 * (1 + Z_EVAL_EL) ** 2
    _B_ini = B0 * (1 + Z_INIT) ** 2

    for _K0 in np.array([1e4, 1e6, 1e8, 1e10, 1e12, 1e14]):
        _KJ = _K0 * EV_MKS
        _s = solve_ivp(_rhs02, (T_EVAL_EL[0], T_EVAL_EL[-1]), [_KJ], t_eval=T_EVAL_EL,
                       method="Radau", rtol=1e-8, atol=1e-20)
        _K = np.maximum(_s.y[0], 0.0)
        _E = _K + E0
        _p0 = np.sqrt(_KJ**2 + 2 * _KJ * E0) / C_LIGHT
        _p = np.sqrt(_K**2 + 2 * _K * E0) / C_LIGHT
        _rg0 = _p0 * _sin_a / (ELECTRON_CHARGE_MKS * _B_ini) / PARSEC_MKS
        _rg = _p * _sin_a / (ELECTRON_CHARGE_MKS * _B_ev) / PARSEC_MKS
        _vpar = _p * _cos_a * C_LIGHT**2 / _E
        _y = cumulative_trapezoid(_vpar, T_EVAL_EL, initial=0) / PARSEC_MKS
        _dmax = np.sqrt((_rg + _rg0) ** 2 + _y**2)
        _dmin = np.maximum(np.sqrt(np.abs(_rg - _rg0) ** 2 + _y**2), 1e-10)
        _ln, = _ax.plot(COSMIC_T_YEARS_EL, _dmax, label=energy_label(_K0), lw=1.5)
        _ax.fill_between(COSMIC_T_YEARS_EL, _dmin, _dmax, color=_ln.get_color(), alpha=0.15)

    _ax.set(xscale="log", yscale="log",
            xlabel="Tiempo cósmico [yr]", ylabel="Distancia recorrida [pc]")
    _ax.set_xlim(T_INIT / YEAR_MKS, T_FINAL / YEAR_MKS)
    grid_minor(_ax)
    _ax.legend(loc="upper left", fontsize=12, ncol=2, frameon=False)
    fig_02.tight_layout()
    fig_02
    return (fig_02,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 03 — Distancia recorrida, promedios isótropos del ángulo de paso
    $\langle\sin\alpha\rangle=\pi/4$ (giroradio), $\langle|\cos\alpha|\rangle=1/2$ (deriva).
    """)
    return


@app.cell
def _(
    B0,
    COSMIC_T_YEARS_EL,
    C_LIGHT,
    E0,
    ELECTRON_CHARGE_MKS,
    EV_MKS,
    PARSEC_MKS,
    THOMSON_CROSS_SECTION_MKS,
    T_EVAL_EL,
    T_FINAL,
    T_INIT,
    VACUUM_PERMEABILITY,
    YEAR_MKS,
    Z_EVAL_EL,
    Z_INIT,
    cumulative_trapezoid,
    energy_label,
    grid_minor,
    np,
    plt,
    solve_ivp,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 03
    fig_03, _ax = plt.subplots(figsize=(8.0, 5.0))

    _mean_sin = np.pi / 4.0      # <sin(alpha)>  isotropo
    _mean_cos = 0.5              # <|cos(alpha)|> isotropo
    # <sin^2(alpha)> = 2/3 ya esta incorporado en el prefactor 4/3 sigma_T c U_B


    def _rhs03(t, K):
        _Kv = max(K[0], 0.0)
        _p2 = _Kv**2 + 2 * _Kv * E0
        _UB = (B0 * (1 + float(z_interp(t))) ** 2) ** 2 / (2 * VACUUM_PERMEABILITY)
        return [-((4 / 3) * THOMSON_CROSS_SECTION_MKS * C_LIGHT * _UB / E0**2) * _p2]


    _B_ev = B0 * (1 + Z_EVAL_EL) ** 2
    _B_ini = B0 * (1 + Z_INIT) ** 2

    for _K0 in np.array([1e4, 1e6, 1e8, 1e10, 1e12, 1e14]):
        _KJ = _K0 * EV_MKS
        _s = solve_ivp(_rhs03, (T_EVAL_EL[0], T_EVAL_EL[-1]), [_KJ], t_eval=T_EVAL_EL,
                       method="Radau", rtol=1e-8, atol=1e-20)
        _K = np.maximum(_s.y[0], 0.0)
        _E = _K + E0
        _p0 = np.sqrt(_KJ**2 + 2 * _KJ * E0) / C_LIGHT
        _p = np.sqrt(_K**2 + 2 * _K * E0) / C_LIGHT
        _rg0 = _p0 * _mean_sin / (ELECTRON_CHARGE_MKS * _B_ini) / PARSEC_MKS
        _rg = _p * _mean_sin / (ELECTRON_CHARGE_MKS * _B_ev) / PARSEC_MKS
        _vpar = _p * _mean_cos * C_LIGHT**2 / _E
        _y = cumulative_trapezoid(_vpar, T_EVAL_EL, initial=0) / PARSEC_MKS
        _dmax = np.sqrt((_rg + _rg0) ** 2 + _y**2)
        _dmin = np.maximum(np.sqrt(np.abs(_rg - _rg0) ** 2 + _y**2), 1e-10)
        _ln, = _ax.plot(COSMIC_T_YEARS_EL, _dmax, label=energy_label(_K0), lw=1.5)
        _ax.fill_between(COSMIC_T_YEARS_EL, _dmin, _dmax, color=_ln.get_color(), alpha=0.15)

    _ax.set(xscale="log", yscale="log",
            xlabel="Tiempo cósmico [yr]", ylabel="Distancia recorrida [pc]")
    _ax.set_xlim(T_INIT / YEAR_MKS, T_FINAL / YEAR_MKS)
    _ax.set_ylim(1e4, 1e9)
    grid_minor(_ax)
    _ax.legend(loc="upper left", fontsize=12, ncol=2, frameon=False)
    fig_03.tight_layout()
    fig_03
    return (fig_03,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 04 — Propagación libre: distancia máxima y tiempo de vuelo
    Sin pérdidas radiativas; el electrón viaja en línea recta a $v(\gamma)$ constante.
    """)
    return


@app.cell
def _(
    C_LIGHT,
    E0,
    EV_MKS,
    PARSEC_MKS,
    T_FINAL,
    T_INIT,
    YEAR_MKS,
    grid_minor,
    np,
    plt,
    solve_ivp,
):
    # ---------------------------------------------------------------- FIG 04
    _R_TARGET_PC = 1e7
    _R_TARGET_M = _R_TARGET_PC * PARSEC_MKS
    _K_scan = np.logspace(2, 14, 60)


    def _rhs04(t, y):
        _g = max(y[0] / E0, 1.0)
        return [0.0, C_LIGHT * np.sqrt(1.0 - 1.0 / _g**2)]


    def _hit(t, y):
        return y[1] - _R_TARGET_M
    _hit.terminal = False

    _dmax_pc, _t_target = [], []
    for _K0 in _K_scan:
        _s = solve_ivp(_rhs04, (T_INIT, T_FINAL), [E0 + _K0 * EV_MKS, 0.0], method="Radau",
                       events=_hit, rtol=1e-6, atol=[1e-20, 1e-3])
        _dmax_pc.append(_s.y[1][-1] / PARSEC_MKS)
        _t_target.append((_s.t_events[0][0] - T_INIT) / YEAR_MKS
                         if _s.t_events[0].size else np.nan)

    fig_04, (_a1, _a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    _a1.plot(_K_scan, _dmax_pc, lw=2, color="darkblue", marker="o", ms=4, ls="-",
             label="Propagación libre")
    _a1.set(yscale="log", ylabel=r"Max Distance Reached [pc]", ylim=(1e7, 2e8))
    grid_minor(_a1)
    _a1.legend(loc="lower right")

    _a2.plot(_K_scan, _t_target, lw=2, color="darkred", marker="o", ms=4, ls="-")
    _a2.set(xscale="log", yscale="log", ylim=(3e6, 5e8),
            xlabel=r"Initial Kinetic Energy $K_{\mathrm{e,ini}}$ [eV]",
            ylabel=f"Time to reach $R={_R_TARGET_PC:.0e}$ pc [yr]")
    grid_minor(_a2)
    fig_04.tight_layout()
    fig_04
    return (fig_04,)


@app.cell
def _(
    EV_MKS,
    T_EVAL,
    YEAR_MKS,
    Z_FINAL,
    Z_INIT,
    energy_label,
    grid_minor,
    integrate,
    np,
    plt,
    sample_with_floor,
    ticker,
    z_interp,
):
    # ---- utilidades comunes a las figuras 05-13 (K normalizada, 2 paneles) ----
    def normalized_panels(K_list_eV, tog, figsize=(7.0, 8.5), sharey=True,
                          t_eval=None, t_years=None, z_eval=None,
                          hlines=((0.9, "k"), (0.5, "r")), rtol=1e-8, atol=1e-20,
                          stop_at_thermal=False, ylim=None, z_lo=None, z_hi=None,
                          locmaj=False, legend_loc="lower left", frameon=True):
        """
        Panel superior K(t)/K_ini vs tiempo cósmico; inferior vs redshift.
        Devuelve (fig, ax1, ax2).
        """
        t_eval = T_EVAL if t_eval is None else t_eval
        t_years = (t_eval / YEAR_MKS) if t_years is None else t_years
        z_eval = z_interp(t_eval) if z_eval is None else z_eval
        z_lo = Z_INIT if z_lo is None else z_lo
        z_hi = Z_FINAL if z_hi is None else z_hi

        fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharey=sharey)
        for K0 in K_list_eV:
            sol = integrate(K0, t_eval, tog=tog, rtol=rtol, atol=atol,
                            stop_at_thermal=stop_at_thermal)
            K, _ = sample_with_floor(sol, t_eval, z_eval)
            norm = np.clip(K / EV_MKS, 1e-30, None) / K0
            a1.plot(t_years, norm, label=energy_label(K0), lw=1.5)
            a2.plot(z_eval, norm, label=energy_label(K0), lw=1.5)

        lm = ticker.LogLocator(base=10.0, numticks=15)
        a1.set(xscale="log", yscale="log", xlabel="Tiempo cósmico [yr]",
               ylabel=r"Energía cinética normalizada $K(t)/K_{\mathrm{ini}}$")
        a1.set_xlim(t_years[0], t_years[-1])
        a2.set(yscale="log", xlabel=r"\textit{Redshift} $z$" if plt.rcParams["text.usetex"]
               else r"$Redshift$ $z$",
               ylabel=r"Energía cinética normalizada $K(z)/K_{\mathrm{ini}}$")
        a2.invert_xaxis()
        a2.set_xlim(z_lo, z_hi)
        if ylim:
            a1.set_ylim(*ylim)
        if locmaj:
            a1.yaxis.set_major_locator(lm)
            a2.yaxis.set_major_locator(lm)
        for lvl, col in hlines:
            lbl = {0.9: r"10 \% Loss", 0.5: r"50 \% Loss", 0.18: r"82 \% Loss"}.get(lvl)
            a1.axhline(lvl, color=col, ls=":", lw=1.5, alpha=0.7, label=lbl)
            a2.axhline(lvl, color=col, ls=":", lw=1.5, alpha=0.7)
        grid_minor(a1)
        grid_minor(a2)
        a1.legend(loc=legend_loc, fontsize=12, ncol=2, frameon=frameon)
        fig.tight_layout()
        return fig, a1, a2

    return (normalized_panels,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 05 — Sincrotrón puro, $K(t)/K_{\rm ini}$ y $K(z)/K_{\rm ini}$
    """)
    return


@app.cell
def _(MECH_KEYS, normalized_panels, np):
    # ---------------------------------------------------------------- FIG 05
    fig_05, _a1, _a2 = normalized_panels(
        np.array([1e4, 1e6, 1e8, 1e10, 1e11, 1e12, 1e13]),
        tog={**{k: False for k in MECH_KEYS}, "synchrotron": True},
        ylim=(1e-1, 1.5), frameon=False,
    )
    fig_05
    return (fig_05,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 06 — Expansión adiabática pura, inyección en $z_{\rm ini}=30$
    Única figura del notebook que usa una ventana cosmológica distinta.
    """)
    return


@app.cell
def _(MECH_KEYS, Z_FINAL, make_time_grid, normalized_panels, np):
    # ---------------------------------------------------------------- FIG 06
    _t6, _ty6, _z6 = make_time_grid(z_i=30.0, z_f=Z_FINAL, mode="elapsed")

    fig_06, _a1, _a2 = normalized_panels(
        np.array([1e3, 1e5, 1e6, 1e7, 1e8, 1e11, 1e13]),
        tog={**{k: False for k in MECH_KEYS}, "adiabatic": True},
        t_eval=_t6, t_years=_ty6, z_eval=_z6,
        hlines=((0.9, "k"), (0.5, "r"), (0.18, "g")),
        ylim=(1e-2, 1.5), z_lo=30.0, frameon=False, legend_loc="lower left",
    )
    _a1.legend_.remove()
    _a2.legend(loc="lower left", fontsize=12, ncol=2, frameon=False)
    fig_06
    return (fig_06,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 07 — Compton inverso sobre el CMB (Klein–Nishina exacto)
    """)
    return


@app.cell
def _(MECH_KEYS, normalized_panels, np):
    # ---------------------------------------------------------------- FIG 07
    fig_07, _a1, _a2 = normalized_panels(
        np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e9, 1e11]),
        tog={**{k: False for k in MECH_KEYS}, "compton": True},
        atol=1e-25, stop_at_thermal=True, frameon=False,
    )
    fig_07
    return (fig_07,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 08 — Compton inverso: $z$ fijo (UTOPIA) vs $z$ dinámico
    La línea verde marca el $\Delta z$ correspondiente a $3\times10^7$ yr desde $z=10$.
    """)
    return


@app.cell
def _(
    EV_MKS,
    H_interp,
    MECH_KEYS,
    Planck18,
    T_EVAL,
    Z_EVAL,
    Z_FINAL,
    Z_INIT,
    energy_label,
    grid_minor,
    np,
    plt,
    solve_ivp,
    thermal_floor,
    total_loss,
    u,
    z_at_value,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 08
    _delta_t = 3e7 * u.yr
    DELTA_Z_3E7 = float(
        (Z_INIT - z_at_value(Planck18.age, Planck18.age(Z_INIT) + _delta_t)).to_value(
            u.dimensionless_unscaled
        )
    )
    print(f"Delta z sobre {_delta_t.to_value(u.yr):.2e} yr desde z={Z_INIT}: "
          f"Dz = {DELTA_Z_3E7:.4f}")

    _tog_ic = {**{k: False for k in MECH_KEYS}, "compton": True}


    def _rhs_ic(z_fixed=None):
        def rhs(t, K):
            _Kv = max(float(K[0]), 0.0)
            _z = Z_INIT if z_fixed is not None else float(z_interp(t))
            return [-float(total_loss(_z, _Kv, float(H_interp(t)), _tog_ic))]
        return rhs


    def _evt_ic(z_fixed=None):
        def ev(t, K):
            _z = Z_INIT if z_fixed is not None else float(z_interp(t))
            return K[0] - float(thermal_floor(_z))
        ev.terminal = True
        ev.direction = -1
        return ev


    fig_08, _axes = plt.subplots(2, 1, figsize=(8.0, 10.5), sharex=True, sharey=True)
    _afix, _adyn = _axes

    for _K0 in np.array([1e3, 1e5, 1e6, 1e7, 1e9, 1e11, 1e12]):
        for _ax, _zf in ((_afix, True), (_adyn, None)):
            _s = solve_ivp(_rhs_ic(_zf), (T_EVAL[0], T_EVAL[-1]), [_K0 * EV_MKS],
                           t_eval=T_EVAL, events=_evt_ic(_zf), method="Radau",
                           rtol=1e-8, atol=1e-25)
            _K = np.interp(T_EVAL, _s.t, np.maximum(_s.y[0], 0.0))
            _ax.plot(Z_EVAL, (_K / EV_MKS) / _K0, label=energy_label(_K0), lw=1.5)

    for _ax in (_afix, _adyn):
        _first = _ax is _afix
        _ax.set(yscale="log",
                ylabel=r"Energía cinética normalizada $K(z)/K_{\mathrm{ini}}$")
        _ax.axhline(0.9, color="k", ls=":", lw=1.5, alpha=0.7,
                    label=r"10 \% Loss" if _first else None)
        _ax.axhline(0.5, color="r", ls=":", lw=1.5, alpha=0.7,
                    label=r"50 \% Loss" if _first else None)
        _ax.axvline(Z_INIT - DELTA_Z_3E7, color="g", ls="--", lw=1.5, alpha=0.7,
                    label=r"$\sim$ 3.2 Myr" if _first else None)
        grid_minor(_ax)

    _adyn.set_xlabel(r"\textit{Redshift} $z$" if plt.rcParams["text.usetex"] else "Redshift $z$")
    _afix.invert_xaxis()
    _afix.set_xlim(Z_INIT, Z_FINAL)
    _afix.set_title(r"Enfriamiento por IC vs CMB ($z$ fijo - UTOPIA)", loc="left", fontsize=14)
    _adyn.set_title(r"Enfriamiento por IC vs CMB ($z$ dinámico)", loc="left", fontsize=14)
    _afix.legend(loc="lower left", fontsize=12, ncol=2, frameon=True)
    fig_08.tight_layout()
    fig_08
    return (fig_08,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 09 — Coulomb puro sobre el plasma ($\chi_e=10^{-4}$)
    """)
    return


@app.cell
def _(MECH_KEYS, normalized_panels, np):
    # ---------------------------------------------------------------- FIG 09
    fig_09, _a1, _a2 = normalized_panels(
        np.array([1e2, 1e3, 10 ** 3.5, 1e4, 1e5, 1e6]),
        tog={**{k: False for k in MECH_KEYS}, "coulomb": True},
        stop_at_thermal=True,
    )
    fig_09
    return (fig_09,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 10 — Excitación colisional $\mathrm{H}(1s)\rightarrow\mathrm{H}(np)$
    """)
    return


@app.cell
def _(
    COSMIC_T_YEARS_EL,
    MECH_KEYS,
    T_EVAL_EL,
    Z_EVAL_EL,
    normalized_panels,
    np,
):
    # ---------------------------------------------------------------- FIG 10
    fig_10, _a1, _a2 = normalized_panels(
        np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e9]),
        tog={**{k: False for k in MECH_KEYS}, "excitation": True},
        t_eval=T_EVAL_EL, t_years=COSMIC_T_YEARS_EL, z_eval=Z_EVAL_EL,
        ylim=(0.9e-4, 1.5), locmaj=True, stop_at_thermal=True,
    )
    fig_10
    return (fig_10,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 11 — Ionización colisional $\mathrm{H}(1s)\rightarrow \mathrm{H}^+ + e^-$
    """)
    return


@app.cell
def _(
    COSMIC_T_YEARS_EL,
    MECH_KEYS,
    T_EVAL_EL,
    Z_EVAL_EL,
    normalized_panels,
    np,
):
    # ---------------------------------------------------------------- FIG 11
    fig_11, _a1, _a2 = normalized_panels(
        np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e9]),
        tog={**{k: False for k in MECH_KEYS}, "ionization": True},
        t_eval=T_EVAL_EL, t_years=COSMIC_T_YEARS_EL, z_eval=Z_EVAL_EL,
        ylim=(0.9e-4, 1.5), locmaj=True, stop_at_thermal=True,
    )
    fig_11
    return (fig_11,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 12 — Bremsstrahlung sobre H neutro + plasma
    Canales: $e$–HI (apantallado), $e$–HII (sin apantallar) y $e$–$e$.
    """)
    return


@app.cell
def _(
    COSMIC_T_YEARS_EL,
    C_LIGHT,
    E0,
    EV_MKS,
    FIX_BREMS_EE,
    ION_FRACTION,
    PREFACTOR_BREMS,
    T_EVAL_EL,
    T_FINAL,
    T_INIT,
    YEAR_MKS,
    Z_EVAL_EL,
    Z_FINAL,
    Z_INIT,
    energy_label,
    grid_minor,
    n_HI,
    np,
    plt,
    solve_ivp,
    ticker,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 12
    def _rhs_brems(t, K):
        _Kv = max(float(K[0]), 0.0)
        _E = _Kv + E0
        _g = _E / E0
        _p2 = _Kv**2 + 2 * _Kv * E0
        _v = C_LIGHT * np.sqrt(_p2) / _E
        _z = float(z_interp(t))
        _nH = n_HI(_z)
        _ne = ION_FRACTION * _nH
        _phi_ion = max(np.log(2.0 * _g) - 1.0 / 3.0, 0.0)
        _phi_neu = (np.log(183.0) + 1.0 / 18.0) if _g >= 137.0 else _phi_ion
        # FIX_BREMS_EE: el canal e-e original contaba (n_HI + n_e) electrones,
        # duplicando el blanco HI que ya esta en el factor de forma neutro.
        _n_ee = _ne if FIX_BREMS_EE else (_nH + _ne)
        _rate = _nH * _phi_neu + _ne * _phi_ion + _n_ee * _phi_ion
        return [-PREFACTOR_BREMS * _v * _E * _rate]


    fig_12, (_a1, _a2) = plt.subplots(2, 1, figsize=(7.0, 8.5), sharey=True)
    for _K0 in np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]):
        _s = solve_ivp(_rhs_brems, (T_EVAL_EL[0], T_EVAL_EL[-1]), [_K0 * EV_MKS],
                       t_eval=T_EVAL_EL, method="Radau", rtol=1e-8, atol=1e-20)
        _n = np.maximum(_s.y[0], 0.0) / EV_MKS / _K0
        _a1.plot(COSMIC_T_YEARS_EL, _n, label=energy_label(_K0), lw=1.5)
        _a2.plot(Z_EVAL_EL, _n, label=energy_label(_K0), lw=1.5)

    _lm = ticker.LogLocator(base=10.0, numticks=15)
    _a1.set(xscale="log", yscale="log", xlabel="Tiempo cósmico [yr]",
            ylabel=r"Energía cinética normalizada $K(t)/K_{\mathrm{ini}}$",
            ylim=(0.89, 1.01))
    _a1.set_xlim(T_INIT / YEAR_MKS, T_FINAL / YEAR_MKS)
    _a1.yaxis.set_major_locator(_lm)
    _a2.set(yscale="log",
            xlabel=r"\textit{Redshift} $z$" if plt.rcParams["text.usetex"] else "Redshift $z$",
            ylabel=r"Energía cinética normalizada $K(z)/K_{\mathrm{ini}}$")
    _a2.yaxis.set_major_locator(_lm)
    _a2.invert_xaxis()
    _a2.set_xlim(Z_INIT, Z_FINAL)
    for _ax in (_a1, _a2):
        _ax.axhline(0.9, color="k", ls=":", lw=1.5, alpha=0.7,
                    label=r"10 \% Loss" if _ax is _a1 else None)
        _ax.axhline(0.5, color="r", ls=":", lw=1.5, alpha=0.7,
                    label=r"50 \% Loss" if _ax is _a1 else None)
        grid_minor(_ax)
    _a1.legend(loc="lower left", fontsize=12, ncol=2, frameon=True)
    fig_12.tight_layout()
    fig_12
    return (fig_12,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 13 — Enfriamiento TOTAL (todos los mecanismos activos)
    """)
    return


@app.cell
def _(normalized_panels, np, toggles):
    # ---------------------------------------------------------------- FIG 13
    fig_13, _a1, _a2 = normalized_panels(
        np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]),
        tog=toggles(), atol=1e-25, stop_at_thermal=True,
        ylim=(1e-10, 1.5), locmaj=True, legend_loc="upper right",
    )
    _a1.legend_.remove()
    _a1.legend(loc="upper right", fontsize=12, ncol=1, frameon=True)
    fig_13
    return (fig_13,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 14 — Impacto de sincrotrón y adiabático sobre la **energía remanente**
    $K_{\rm tot}/K_{\rm sin\,sinc}$ y $K_{\rm tot}/K_{\rm sin\,ad}$ frente al tiempo cósmico.

    > **Nota sobre $B_0$.** Con $B_0 = 1$ nG comóvil, $U_B/U_{\rm CMB} \approx 10^{-7}$ a
    > $z=10$ (ver la celda de diagnóstico al final), de modo que el sincrotrón es
    > $\sim$7 órdenes de magnitud menor que Compton inverso **en todo el rango de
    > energías**. El cociente $\equiv 1$ de estos paneles es por tanto un resultado
    > físico, no un artefacto numérico: para que el sincrotrón compita con el CMB haría
    > falta $B_0 \gtrsim 3.24\ \mu$G comóvil (equipartición $U_B = U_{\rm CMB}$).
    """)
    return


@app.cell
def _(
    COSMIC_T_YEARS,
    T_EVAL,
    T_FINAL,
    T_INIT,
    YEAR_MKS,
    Z_EVAL,
    energy_label,
    grid_minor,
    integrate,
    np,
    plt,
    sample_with_floor,
    toggles,
):
    # ---------------------------------------------------------------- FIG 14
    fig_14, (_a1, _a2) = plt.subplots(2, 1, figsize=(8.0, 10.0), sharex=True)

    for _K0 in np.array([1e4, 1e6, 1e8, 1e10, 1e11, 1e12, 1e13]):
        _Ks = {}
        for _tag, _tg in (("tot", toggles()),
                          ("nos", toggles(synchrotron=False)),
                          ("noa", toggles(adiabatic=False))):
            _s = integrate(_K0, T_EVAL, tog=_tg, atol=1e-25, stop_at_thermal=True)
            _Ks[_tag], _ = sample_with_floor(_s, T_EVAL, Z_EVAL)
        _lbl = energy_label(_K0)
        _a1.plot(COSMIC_T_YEARS, _Ks["tot"] / _Ks["nos"], label=_lbl, lw=1.5)
        _a2.plot(COSMIC_T_YEARS, _Ks["tot"] / _Ks["noa"], label=_lbl, lw=1.5)

    _a1.set(xscale="log", yscale="log",
            ylabel=r"Energy Ratio $K_{\mathrm{tot}}(t) / K_{\mathrm{no\_synch}}(t)$")
    grid_minor(_a1)
    _a1.legend(loc="lower left", fontsize=10, ncol=2, frameon=True)
    _a2.set(xscale="log", yscale="log", xlabel="Cosmic Time [yr]",
            ylabel=r"Energy Ratio $K_{\mathrm{tot}}(t) / K_{\mathrm{no\_ad}}(t)$")
    _a2.set_xlim(T_INIT / YEAR_MKS, T_FINAL / YEAR_MKS)
    grid_minor(_a2)
    fig_14.tight_layout()
    fig_14
    return (fig_14,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 15 — Fracción de la **tasa** de pérdida no debida a sincrotrón / adiabático
    $(L_{\rm tot}-L_i)/L_{\rm tot}$ evaluado sobre la trayectoria real.
    El panel superior queda pegado a 1 por la misma razón que la Fig. 14 (ver nota de $B_0$).
    """)
    return


@app.cell
def _(
    H_interp,
    T_EVAL,
    Z_EVAL,
    Z_FINAL,
    Z_INIT,
    energy_label,
    integrate,
    loss_rates,
    np,
    plt,
    toggles,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 15
    fig_15, (_a1, _a2) = plt.subplots(2, 1, figsize=(8.0, 10.0), sharex=True)

    for _K0 in np.array([1e4, 1e5, 1e6, 1e7, 1e9, 1e11, 1e13]):
        _s = integrate(_K0, T_EVAL, tog=toggles(), rtol=1e-6, atol=1e-25,
                       stop_at_thermal=True)
        _act = T_EVAL <= _s.t[-1]
        _ta = T_EVAL[_act]
        _Ka = _s.sol(_ta)[0]
        _L = loss_rates(z_interp(_ta), _Ka, H_interp(_ta), toggles())
        _Ltot = _L.sum(axis=0)
        _lbl = energy_label(_K0)
        _a1.plot(Z_EVAL[_act], np.maximum(_Ltot - _L[1], 1e-100) / _Ltot, label=_lbl, lw=1.5)
        _a2.plot(Z_EVAL[_act], np.maximum(_Ltot - _L[0], 1e-100) / _Ltot, label=_lbl, lw=1.5)

    for _ax, _lab in ((_a1, r"$(L_{\mathrm{tot}} - L_{\mathrm{sinc}}) / L_{\mathrm{tot}}$"),
                      (_a2, r"$(L_{\mathrm{tot}} - L_{\mathrm{ad}}) / L_{\mathrm{tot}}$")):
        _ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.5)
        _ax.set(xscale="linear", yscale="linear", ylabel=_lab)
        _ax.grid(True, which="minor", ls="--", alpha=0.1)
        _ax.grid(True, which="major", ls="-", alpha=0.3)
    _a1.legend(loc="lower right", fontsize=12, ncol=2, frameon=True)
    _a2.set_xlabel("Redshift $z$")
    _a2.set_xlim(Z_INIT + 0.1, Z_FINAL)
    fig_15.tight_layout()
    fig_15
    return (fig_15,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 16 — Espacio de fases del mecanismo dominante + trayectorias
    """)
    return


@app.cell
def _(
    MECH_CMAP,
    N_MECH_MAP,
    PHASE_ENERGIES_EV,
    Z_FINAL,
    Z_INIT,
    add_threshold_curves,
    energy_label,
    mech_legend,
    np,
    phase_space_map,
    plt,
    trajectory,
):
    # ---------------------------------------------------------------- FIG 16
    _Zm, _Km, _dom, _ = phase_space_map(1000, 600, 0, 14, "all")

    fig_16, _ax = plt.subplots(figsize=(10.0, 7.0))
    _ax.pcolormesh(_Zm, _Km, _dom, cmap=MECH_CMAP, vmin=-0.5, vmax=N_MECH_MAP - 0.5,
                   shading="auto", alpha=0.35)

    _tcol = plt.cm.inferno(np.linspace(0, 1, len(PHASE_ENERGIES_EV)))
    for _i, _K0 in enumerate(PHASE_ENERGIES_EV):
        _, _za, _Ka = trajectory(_K0, "all")
        _ax.plot(_za, _Ka, color=_tcol[_i], lw=1.5, ls="--",
                 label=energy_label(_K0, r"K_{\mathrm{ini}}"))

    _ax.set(yscale="log", xlim=(Z_INIT + 0.1, Z_FINAL), ylim=(1e0, 1e12),
            xlabel=r"Redshift $z$", ylabel=r"Energía cinética $K_{\mathrm{e}}$ [eV]")
    add_threshold_curves(_ax)
    mech_legend(_ax, np.unique(_dom), alpha=0.6, loc="upper right", fontsize=12, ncol=2)
    _ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")
    fig_16.tight_layout()
    fig_16
    return (fig_16,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 17 — Trayectorias coloreadas por el mecanismo dominante instantáneo
    """)
    return


@app.cell
def _(
    EV_MKS,
    H_interp,
    N_MECH_MAP,
    PHASE_ENERGIES_EV,
    Z_FINAL,
    Z_INIT,
    add_threshold_curves,
    colored_trajectory,
    energy_label,
    loss_rates,
    mech_legend,
    np,
    plt,
    toggles,
    trajectory,
):
    # ---------------------------------------------------------------- FIG 17
    fig_17, _ax = plt.subplots(figsize=(10.0, 7.0))

    for _K0 in PHASE_ENERGIES_EV:
        _ta, _za, _Ka = trajectory(_K0, "all")
        _L = loss_rates(_za, _Ka * EV_MKS, H_interp(_ta), toggles())
        _dom = np.argmax(_L[:N_MECH_MAP], axis=0)
        colored_trajectory(_ax, _za, _Ka, _dom, lw=2.5)
        _ax.text(_za[0] + 0.3, _Ka[0], energy_label(_K0, "").replace("$ = ", "$"),
                 fontsize=11, va="center", ha="left", color="black")

    _ax.set(yscale="log", xlim=(Z_INIT + 0.5, Z_FINAL), ylim=(1e0, 1e14),
            xlabel=r"Redshift $z$", ylabel=r"Energía cinética $K_{\mathrm{e}}$ [eV]")
    add_threshold_curves(_ax)
    mech_legend(_ax, range(N_MECH_MAP), alpha=1.0, loc="upper right", fontsize=11, ncol=2)
    _ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")
    fig_17.tight_layout()
    fig_17
    return (fig_17,)


@app.cell
def _(
    C_LIGHT,
    EV_MKS,
    H_interp,
    KE_99_ARRAY,
    KE_CUTOFF_ARRAY,
    MECH_KEYS,
    MEGAPARSEC_MKS,
    N_MECH_MAP,
    PHASE_ENERGIES_EV,
    T_ARRAY_INTERP,
    T_FINAL,
    T_INIT,
    Z_FINAL,
    Z_INIT,
    add_threshold_curves,
    colored_trajectory,
    cumulative_trapezoid,
    energy_label,
    kinematics,
    loss_rates,
    mech_legend,
    np,
    toggles,
    trajectory,
):
    # --- panel reutilizable para las figuras 18 y 19 ---
    def draw_phase_panel(ax, tog_key, title, xmode="z", xmax_mpc=60.0,
                         text_dx=0.05, text_ha="right", lw=1.5):
        tog = toggles() if tog_key == "all" else toggles(adiabatic=False)
        for K0 in PHASE_ENERGIES_EV:
            t_a, z_a, K_a = trajectory(K0, tog_key)
            L = loss_rates(z_a, K_a * EV_MKS, H_interp(t_a), tog)
            dom = np.argmax(L[:N_MECH_MAP], axis=0)
            if xmode == "z":
                xvals = z_a
                x0, dx, ha = z_a[0], text_dx, text_ha
            else:
                _, _, _, v = kinematics(K_a * EV_MKS)
                xvals = cumulative_trapezoid(v, t_a, initial=0.0) / MEGAPARSEC_MKS
                x0, dx, ha = xvals[0], 2.0, "left"
            colored_trajectory(ax, xvals, K_a, dom, lw=lw)
            ax.text(x0 + dx, K_a[0], energy_label(K0, "").replace("$ = ", "$"),
                    fontsize=11, va="center", ha=ha, color="black")

        ax.set(yscale="log", ylim=(1e0, 3e13))
        ax.set_title(title, fontsize=15, pad=15)
        ax.set_ylabel(r"Energía cinética $K_{\mathrm{e}}$ [eV]", fontsize=14)
        if xmode == "z":
            ax.set_xlim(Z_INIT + 0.5, Z_FINAL)
            add_threshold_curves(ax)
        else:
            ax.set_xlim(-1, xmax_mpc)
            d_bg = C_LIGHT * (T_ARRAY_INTERP - T_INIT) / MEGAPARSEC_MKS
            m = (T_ARRAY_INTERP >= T_INIT) & (T_ARRAY_INTERP <= T_FINAL)
            ax.plot(d_bg[m], KE_CUTOFF_ARRAY[m], color="black", ls="--", alpha=0.8,
                    label=r"Fotoionizacion secundaria")
            ax.plot(d_bg[m], KE_99_ARRAY[m], color="black", ls="--", alpha=0.8)
        ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")
        idx = [i for i, k in enumerate(MECH_KEYS[:N_MECH_MAP]) if tog[k]]
        mech_legend(ax, idx, alpha=0.8, loc="upper right", fontsize=10, ncol=2)
        return ax

    return (draw_phase_panel,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 18 — Comparación: todos los mecanismos vs sin expansión adiabática ($K$ vs $z$)
    """)
    return


@app.cell
def _(MECH_NAMES, draw_phase_panel, plt):
    # ---------------------------------------------------------------- FIG 18
    fig_18, _axes = plt.subplots(2, 1, figsize=(10.0, 14.0), sharex=True)
    draw_phase_panel(_axes[0], "all", "Todos los mecanismos de pérdida activos")
    draw_phase_panel(_axes[1], "no_ad", f"Sin considerar: {MECH_NAMES[0]}")
    _axes[1].set_xlabel(r"Redshift $z$", fontsize=14)
    fig_18.tight_layout()
    fig_18
    return (fig_18,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 19 — Energía cinética frente a **distancia recorrida** [Mpc]
    Ambos paneles con todos los mecanismos activos (como en el notebook original).
    """)
    return


@app.cell
def _(draw_phase_panel, plt):
    # ---------------------------------------------------------------- FIG 19
    fig_19, _axes = plt.subplots(2, 1, figsize=(10.0, 14.0), sharex=True)
    draw_phase_panel(_axes[0], "all", "Todos los mecanismos de pérdida activos", xmode="d")
    draw_phase_panel(_axes[1], "all", "Todos los mecanismos activos", xmode="d")
    _axes[1].set_xlabel(r"Distance [Mpc]", fontsize=14)
    fig_19.tight_layout()
    fig_19
    return (fig_19,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 20 — Espacio de fases **sin** expansión adiabática
    """)
    return


@app.cell
def _(
    MECH_CMAP,
    N_MECH_MAP,
    PHASE_ENERGIES_EV,
    Z_FINAL,
    Z_INIT,
    add_threshold_curves,
    energy_label,
    mech_legend,
    np,
    phase_space_map,
    plt,
    trajectory,
):
    # ---------------------------------------------------------------- FIG 20
    _Zm, _Km, _dom, _ = phase_space_map(400, 600, 0, 14, "no_ad")

    fig_20, _ax = plt.subplots(figsize=(10.0, 7.0))
    _ax.pcolormesh(_Zm, _Km, _dom, cmap=MECH_CMAP, vmin=-0.5, vmax=N_MECH_MAP - 0.5,
                   shading="auto", alpha=0.35)

    _tcol = plt.cm.inferno(np.linspace(0, 1, len(PHASE_ENERGIES_EV)))
    for _i, _K0 in enumerate(PHASE_ENERGIES_EV):
        _, _za, _Ka = trajectory(_K0, "no_ad")
        _ax.plot(_za, _Ka, color=_tcol[_i], lw=1.5, ls="--",
                 label=energy_label(_K0, r"K_{\mathrm{ini}}"))

    _ax.set(yscale="log", xlim=(Z_INIT + 0.1, Z_FINAL), ylim=(1e0, 1e12),
            xlabel=r"Redshift $z$", ylabel=r"Energía cinética $K_{\mathrm{e}}$ [eV]")
    add_threshold_curves(_ax)
    mech_legend(_ax, np.unique(_dom), alpha=0.6, loc="upper right", fontsize=12, ncol=2)
    _ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")
    fig_20.tight_layout()
    fig_20
    return (fig_20,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 21 — Pérdida **acumulada** por mecanismo (EDO aumentada de 8 estados)
    Fracción de la energía inicial disipada por cada canal, $\int L_i\,dt / K_{\rm ini}$.
    """)
    return


@app.cell
def _(
    EV_MKS,
    MECH_COLORS,
    MECH_NAMES,
    N_MECH_MAP,
    T_EVAL_DENSE,
    Z_INIT,
    make_evt_cached,
    make_rhs,
    np,
    plt,
    solve_ivp,
    toggles,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 21
    TARGET_ENERGIES_EV = [1e12, 1e9, 1e7, 1e5]

    fig_21, _axes = plt.subplots(4, 1, figsize=(8.0, 16.0), sharex=True)
    for _idx, _K0 in enumerate(TARGET_ENERGIES_EV):
        _ax = _axes[_idx]
        _s = solve_ivp(make_rhs(toggles(), accumulate=True),
                       (T_EVAL_DENSE[0], T_EVAL_DENSE[-1]),
                       [_K0 * EV_MKS] + [0.0] * 7, t_eval=T_EVAL_DENSE,
                       events=make_evt_cached(), method="Radau", rtol=1e-6, atol=1e-25)
        _za = z_interp(_s.t)
        _acc = _s.y[1:] / EV_MKS
        for _j in range(N_MECH_MAP):
            if np.max(_acc[_j]) > 1e-5:
                _ax.plot(_za, _acc[_j] / _K0, color=MECH_COLORS[_j], lw=2.0,
                         label=MECH_NAMES[_j])
        _ax.set_title(rf"Initial Kinetic Energy: $10^{{{int(np.log10(_K0))}}}$ eV", fontsize=14)
        _ax.set(yscale="log")
        _ax.set_ylabel(r"Accumulated loss $/\,K_{\mathrm{ini}}$", fontsize=14)
        _ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")
        _ax.grid(True, which="minor", ls=":", alpha=0.15, color="grey")
        _ax.legend(loc="lower right", fontsize=11, frameon=True,
                   edgecolor="black", facecolor="white", ncol=2)

    _axes[-1].set_xlabel(r"Redshift $z$", fontsize=14)
    _axes[-1].set_xlim(Z_INIT, 7.0)
    fig_21.tight_layout()
    fig_21
    return TARGET_ENERGIES_EV, fig_21


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 22 — Pérdida **por paso** de la malla, $\Delta E$ entre nodos consecutivos
    Nota: depende del espaciado de `T_EVAL_DENSE`, no es una cantidad física intrínseca.
    """)
    return


@app.cell
def _(
    EV_MKS,
    MECH_COLORS,
    MECH_NAMES,
    N_MECH_MAP,
    TARGET_ENERGIES_EV,
    T_EVAL_DENSE,
    Z_INIT,
    make_evt_cached,
    make_rhs,
    np,
    plt,
    solve_ivp,
    toggles,
    z_interp,
):
    # ---------------------------------------------------------------- FIG 22
    fig_22, _axes = plt.subplots(4, 1, figsize=(8.0, 16.0), sharex=True)
    for _idx, _K0 in enumerate(TARGET_ENERGIES_EV):
        _ax = _axes[_idx]
        _s = solve_ivp(make_rhs(toggles(), accumulate=True),
                       (T_EVAL_DENSE[0], T_EVAL_DENSE[-1]),
                       [_K0 * EV_MKS] + [0.0] * 7, t_eval=T_EVAL_DENSE,
                       events=make_evt_cached(), method="Radau", rtol=1e-6, atol=1e-25)
        _za = z_interp(_s.t)
        _step = np.diff(_s.y[1:] / EV_MKS, axis=1)
        for _j in range(N_MECH_MAP):
            if np.max(_step[_j]) > 1e-10:
                _ax.plot(_za[1:], np.maximum(_step[_j], 1e-50), color=MECH_COLORS[_j],
                         lw=2.0, label=MECH_NAMES[_j])
        _ax.set_title(rf"Initial Kinetic Energy: $10^{{{int(np.log10(_K0))}}}$ eV", fontsize=14)
        _ax.set(yscale="log")
        _ax.set_ylabel(r"Energy Loss per Step $\Delta E$ [eV]", fontsize=14)
        _ax.set_ylim(bottom=1e-6, top=float(np.max(_step)) * 10)
        _ax.grid(True, which="major", ls="-", alpha=0.3, color="grey")
        _ax.grid(True, which="minor", ls=":", alpha=0.15, color="grey")
        _ax.legend(loc="upper right", fontsize=11, frameon=True,
                   edgecolor="black", facecolor="white", ncol=2)

    _axes[-1].set_xlabel(r"Redshift $z$", fontsize=14)
    _axes[-1].set_xlim(Z_INIT, 7.0)
    fig_22.tight_layout()
    fig_22
    return (fig_22,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 23 — Cociente de pérdidas **acumuladas** vs redshift
    $L_{\rm tot}/L_{\rm sin\,sinc}$ y $L_{\rm tot}/L_{\rm sin\,ad}$ con $L = K_{\rm ini}-K(t)$.
    El panel del sincrotrón queda en 1 por la misma razón que la Fig. 14 (ver nota de $B_0$).
    *(Esta figura fallaba en el notebook original: `PREFACTOR_BREMS` no definido y
    `z_eval` de 1000 puntos contra curvas de 10000.)*
    """)
    return


@app.cell
def _(
    EV_MKS,
    T_EVAL,
    Z_EVAL,
    energy_label,
    grid_minor,
    integrate,
    np,
    plt,
    sample_with_floor,
    toggles,
):
    # ---------------------------------------------------------------- FIG 23
    _EPS = 1e-30
    fig_23, (_a1, _a2) = plt.subplots(2, 1, figsize=(9.0, 10.0), sharex=True)

    for _K0 in np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]):
        _KJ = _K0 * EV_MKS
        _res = {}
        for _tag, _tg in (("all", toggles()),
                          ("nos", toggles(synchrotron=False)),
                          ("noa", toggles(adiabatic=False))):
            _s = integrate(_K0, T_EVAL, tog=_tg, atol=1e-25, stop_at_thermal=True)
            _res[_tag], _ = sample_with_floor(_s, T_EVAL, Z_EVAL)
        _l_all = _KJ - _res["all"]
        _lbl = energy_label(_K0, r"K_{\mathrm{e, ini}}")
        _a1.plot(Z_EVAL, _l_all / (_KJ - _res["nos"] + _EPS), label=_lbl, lw=1.5)
        _a2.plot(Z_EVAL, _l_all / (_KJ - _res["noa"] + _EPS), label=_lbl, lw=1.5)

    for _ax in (_a1, _a2):
        _ax.set_yscale("log")
        _ax.invert_xaxis()
        grid_minor(_ax)
        _ax.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.8,
                    label="Ratio = 1 (No Contribution)")
        _ax.legend(loc="upper right", fontsize=12, ncol=2, frameon=False)

    _a1.set_ylabel(r"Loss Ratio: $L_{\mathrm{Total}}/L_{\mathrm{No\_Synchrotron}}$", fontsize=14)
    _a2.set_ylabel(r"Loss Ratio: $L_{\mathrm{Total}}/L_{\mathrm{No\_Adiabatic}}$", fontsize=14)
    _a2.set_xlabel(r"\textit{Redshift} $z$" if plt.rcParams["text.usetex"] else "Redshift $z$",
                   fontsize=14)
    fig_23.tight_layout()
    fig_23
    return (fig_23,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 24 — Mecanismo dominante en el plano $(z,\,K_e)$, 7 mecanismos
    Mapa categórico de $\arg\max_i L_i$ sobre una ventana ampliada ($5.5 \le z \le 15$).
    Las curvas de nivel negras dan $\Gamma_{\rm ad}=L_{\rm ad}/L_{\rm tot}$.

    *La versión original graficaba $\Gamma_{\rm ad}$ con un contorno "Dominance Horizon"
    en $0.5$, pero nunca llegó a ejecutarse (`K_B_MKS` y `PREFACTOR_BREMS` sin definir,
    y a `rate_ion` le faltaba el factor `EV_MKS`). Con la física completa
    $\Gamma_{\rm ad}$ **no supera 0.26** en ninguna parte de esta ventana: la expansión
    adiabática nunca domina, de modo que ese contorno estaba vacío y las tres cajas de
    texto describían regiones inexistentes.*
    """)
    return


@app.cell
def _(
    ALL_ON,
    EV_MKS,
    ListedColormap,
    MECH_COLORS,
    Planck18,
    loss_rates,
    mech_legend,
    np,
    plt,
    u,
):
    # ---------------------------------------------------------------- FIG 24
    _z1d = np.linspace(5.5, 15.0, 400)
    _k1d = np.logspace(0, 14, 500)
    _Zg, _Kg = np.meshgrid(_z1d, _k1d)
    _Hg = np.broadcast_to(Planck18.H(_z1d).to(u.s**-1).value, _Zg.shape)

    _L = loss_rates(_Zg, _Kg * EV_MKS, _Hg, dict(ALL_ON))    # los 7 mecanismos
    _dom = np.argmax(_L, axis=0)
    _frac_ad = _L[0] / np.where(_L.sum(axis=0) == 0, 1e-30, _L.sum(axis=0))

    fig_24, _ax = plt.subplots(figsize=(9.5, 7.0))
    _ax.pcolormesh(_Zg, _Kg, _dom, cmap=ListedColormap(MECH_COLORS),
                   vmin=-0.5, vmax=6.5, shading="auto", alpha=0.45)

    # Curvas de nivel de la contribucion adiabatica (el nivel 0.5 no existe)
    _lv = [l for l in (0.05, 0.10, 0.20, 0.25) if l < _frac_ad.max()]
    _ct = _ax.contour(_Zg, _Kg, _frac_ad, levels=_lv, colors="black",
                      linewidths=1.6, linestyles="--")
    _ax.clabel(_ct, inline=True, fontsize=11,
               fmt=lambda v: rf"$\Gamma_{{\mathrm{{ad}}}}={v:.2f}$")

    _ax.set(yscale="log")
    _ax.invert_xaxis()
    _ax.set_xlabel(r"\textit{Redshift} $z$" if plt.rcParams["text.usetex"] else "Redshift $z$",
                   fontsize=14)
    _ax.set_ylabel(r"Energía cinética $K_e$ [eV]", fontsize=14)
    _ax.set_title(rf"$\max_{{z,K}}\ \Gamma_{{\mathrm{{ad}}}} = {_frac_ad.max():.3f}$"
                  r"  $\Rightarrow$ la expansión adiabática nunca domina",
                  loc="left", fontsize=13)
    mech_legend(_ax, np.unique(_dom), alpha=0.7, loc="lower left", fontsize=11, ncol=2)
    _ax.grid(True, which="major", ls="-", alpha=0.25, color="grey")
    fig_24.tight_layout()
    fig_24
    return (fig_24,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Figura 25 — Impacto de la expansión adiabática, $K_{\rm con}/K_{\rm sin}$
    Modelo de la celda original 42: IC + bremsstrahlung + Coulomb + excitación +
    ionización (**sin** sincrotrón), con y sin término adiabático.
    *(Esta figura tampoco llegó a ejecutarse en el notebook original.)*
    """)
    return


@app.cell
def _(
    COSMIC_T_YEARS,
    EV_MKS,
    MECH_KEYS,
    T_EVAL,
    T_FINAL,
    T_INIT,
    YEAR_MKS,
    Z_EVAL,
    Z_FINAL,
    Z_INIT,
    energy_label,
    grid_minor,
    integrate,
    np,
    plt,
    sample_with_floor,
):
    # ---------------------------------------------------------------- FIG 25
    _tog_42 = {k: True for k in MECH_KEYS}
    _tog_42["synchrotron"] = False          # el modelo original no incluia sincrotron
    _tog_42_noad = dict(_tog_42, adiabatic=False)

    fig_25, (_a1, _a2) = plt.subplots(2, 1, figsize=(7.0, 8.5), sharey=True)
    _cols = plt.cm.inferno(np.linspace(0.1, 0.85, 8))

    for _i, _K0 in enumerate(np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])):
        _r = {}
        for _tag, _tg in (("with", _tog_42), ("without", _tog_42_noad)):
            _s = integrate(_K0, T_EVAL, tog=_tg, atol=1e-20, stop_at_thermal=True)
            _K, _ = sample_with_floor(_s, T_EVAL, Z_EVAL)
            _r[_tag] = np.clip(_K / EV_MKS, 1e-20, None)
        _ratio = _r["with"] / _r["without"]
        _a1.plot(COSMIC_T_YEARS, _ratio, lw=1.5, ls="-", color=_cols[_i],
                 label=energy_label(_K0, r"K_{\mathrm{e, ini}}"))
        _a2.plot(Z_EVAL, _ratio, lw=1.5, ls="-", color=_cols[_i])

    _a1.set(xscale="log", yscale="log", xlabel="Tiempo cósmico [yr]",
            ylabel=r"Energy Ratio $K_{\mathrm{with}}/K_{\mathrm{without}}$")
    _a1.set_xlim(T_INIT / YEAR_MKS, T_FINAL / YEAR_MKS)
    _a1.axhline(1.0, color="k", ls=":", lw=1.5, alpha=0.7)
    grid_minor(_a1)
    _a1.legend(loc="lower left", fontsize=12, ncol=2, frameon=False)

    _a2.set(yscale="log", ylabel=r"Energy Ratio $K_{\mathrm{with}}/K_{\mathrm{without}}$")
    _a2.set_xlabel(r"\textit{Redshift} $z$" if plt.rcParams["text.usetex"] else "Redshift $z$")
    _a2.invert_xaxis()
    _a2.set_xlim(Z_INIT, Z_FINAL)
    _a2.axhline(1.0, color="k", ls=":", lw=1.5, alpha=0.7)
    grid_minor(_a2)
    fig_25.tight_layout()
    fig_25
    return (fig_25,)


@app.cell
def _(
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
    fig_11,
    fig_12,
    fig_13,
    fig_14,
    fig_15,
    fig_16,
    fig_17,
    fig_18,
    fig_19,
    fig_20,
    fig_21,
    fig_22,
    fig_23,
    fig_24,
    fig_25,
):
    # =====================================================================
    #  Verificación: inventario de figuras y ejes
    # =====================================================================
    FIGURES = {
        "01 Giroradio (sincrotrón)": fig_01,
        "02 Distancia v_dir=[1,0,0]": fig_02,
        "03 Distancia isótropa": fig_03,
        "04 Propagación libre": fig_04,
        "05 Sincrotrón K/K_ini": fig_05,
        "06 Adiabático (z=30)": fig_06,
        "07 Compton inverso": fig_07,
        "08 IC z fijo vs dinámico": fig_08,
        "09 Coulomb": fig_09,
        "10 Excitación colisional": fig_10,
        "11 Ionización colisional": fig_11,
        "12 Bremsstrahlung": fig_12,
        "13 Enfriamiento total": fig_13,
        "14 Ratio energía (sinc/ad)": fig_14,
        "15 Ratio tasas (sinc/ad)": fig_15,
        "16 Espacio de fases + trayect.": fig_16,
        "17 Trayectorias coloreadas": fig_17,
        "18 Comparación con/sin ad (z)": fig_18,
        "19 K vs distancia [Mpc]": fig_19,
        "20 Espacio de fases sin ad": fig_20,
        "21 Pérdidas acumuladas": fig_21,
        "22 Pérdidas por paso": fig_22,
        "23 Ratio pérdidas acumuladas": fig_23,
        "24 Mapa mecanismo dominante": fig_24,
        "25 Impacto adiabático (K ratio)": fig_25,
    }

    _rows = []
    _tot_ax = 0
    for _name, _f in FIGURES.items():
        _na = len(_f.axes)
        _nl = sum(len(a.lines) for a in _f.axes)
        _nc = sum(len(a.collections) for a in _f.axes)
        _tot_ax += _na
        _rows.append(f"  {_name:<34s} ejes={_na}  lineas={_nl:>3d}  colecciones={_nc:>3d}")

    print(f"FIGURAS: {len(FIGURES)}   EJES TOTALES: {_tot_ax}\n")
    print("\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Diagnóstico: errores detectados en el notebook original

    Los interruptores de la **celda 2** están activados con las correcciones
    aplicadas. Poné cualquiera en su valor original y marimo recalcula todo el
    notebook solo (reactividad).

    | Id | Interruptor | Original | Ahora | Qué corrige |
    |----|-------------|----------|-------|-------------|
    | E1 | `FIX_KN_KERNEL` | `False` | `True` | factor $q$ faltante en el kernel Klein–Nishina: el original decaía como $b^{-1}$ en vez de $\ln b / b^{2}$ |
    | E2 | `FIX_BREMS_EE` | `False` | `True` | doble conteo del blanco HI en el canal $e$–$e$ de la Fig. 12 (factor $\sim 2$) |
    | E3 | `INCLUDE_BREMS_IN_TOTAL` | `False` | `True` | el bremsstrahlung faltaba en el modelo "total" (Figs 13, 16–24) |
    | E4 | `THERMAL_FLOOR_FACTOR` | `1.0` | `1.5` | piso térmico $\langle K\rangle = \tfrac32 k T_{\rm CMB}(z)$ en vez de $kT$ |
    | E5 | `EXC_NMAX` | 3 en Figs 13–22 | `10` | niveles $n$ de excitación colisional, unificados en todo el notebook ($\sim 10\,\%$) |

    ### Errores adicionales corregidos sin interruptor

    * `PREFACTOR_BREMS` y `K_B_MKS` **sin definir** (celdas 40 y 41 del `.ipynb`):
      la Fig. 23 lanzaba `NameError` y la Fig. 24 nunca llegó a ejecutarse.
    * En la Fig. 23 original, `z_eval` tenía 1000 puntos y las curvas 10 000
      $\Rightarrow$ `ValueError` en `ax.plot`.
    * A `rate_ion` le faltaba el factor `EV_MKS` en las celdas 41 y 42 (la tabla
      está en m²·eV y la tasa en J/s): la ionización quedaba $1.6\times10^{-19}$
      veces más chica.
    * El bremsstrahlung se calculaba pero **no se sumaba** a `total_dKdt` en la
      celda 40.
    * Interpolación de las tablas colisionales: `interp1d(kind="cubic")` sobre una
      grilla log-espaciada pero evaluada linealmente en $K$ $\Rightarrow$ ahora es
      log–log, monótona y estrictamente no negativa (desaparecen los `max(0, ...)`
      defensivos que había desperdigados).
    * Discontinuidad del $+27\,\%$ en $\sigma_{\rm exc}$ al pasar de la rama
      `asympt` a la `rel` en $T = 10^{5}$ eV (heredada de Stone & Kim; queda
      documentada, no alterada).
    * La Fig. 04 abría una leyenda sin artistas etiquetados (`UserWarning`).
    """)
    return


@app.cell
def _(
    ALL_ON,
    B0,
    EV_MKS,
    FIX_KN_KERNEL,
    H_interp,
    MECH_NAMES,
    T_INIT,
    U_CMB_0_J_M3,
    VACUUM_PERMEABILITY,
    Z_INIT,
    f_kn_kernel_grid,
    loss_rates,
    np,
):
    # ---- Diagnóstico numérico de las correcciones aplicadas ----
    print("[E1] Kernel Klein-Nishina  F_KN(b):")
    _b = np.array([1e-2, 1e-1, 1.0, 10.0, 1e2])
    _f_orig = f_kn_kernel_grid(_b, with_q_weight=False)
    _f_fix = f_kn_kernel_grid(_b, with_q_weight=True)
    print(f"{'b':>9} {'original':>12} {'corregido':>12} {'orig/corr':>10}")
    for _bi, _fo, _ff in zip(_b, _f_orig, _f_fix):
        print(f"{_bi:9.1e} {_fo:12.4e} {_ff:12.4e} {_fo / _ff:10.2f}")
    print("     b = 4 gamma k T_CMB(z) / m c^2 ;  a z=10, b=1 <-> K_e ~ 2.5e13 eV")
    print(f"     activo: FIX_KN_KERNEL={FIX_KN_KERNEL}\n")

    # ---- Densidades de energía: por qué el sincrotrón es irrelevante ----
    print("Densidades de energía y campo equivalente al CMB:")
    for _z in (0.0, 5.5, 10.0):
        _UB = (B0 * (1 + _z) ** 2) ** 2 / (2 * VACUUM_PERMEABILITY)
        _UC = U_CMB_0_J_M3 * (1 + _z) ** 4
        _Beq = np.sqrt(2 * VACUUM_PERMEABILITY * _UC)
        print(f"  z={_z:5.1f}   U_B={_UB:9.3e}  U_CMB={_UC:9.3e}  "
              f"U_B/U_CMB={_UB / _UC:9.2e}   B_eq={_Beq * 1e4 * 1e6:8.1f} uG")
    print(f"  -> B0 de equipartición (comóvil) = "
          f"{np.sqrt(2 * VACUUM_PERMEABILITY * U_CMB_0_J_M3) * 1e4 * 1e6:.2f} uG "
          f"(el notebook usa {B0 * 1e4 * 1e9:.1f} nG)\n")

    # ---- Desglose de tasas a z = z_init ----
    print(f"Tasas de pérdida |dK/dt| [J/s] a z={Z_INIT} (7 mecanismos):")
    _Hz = float(H_interp(T_INIT))
    print(f"{'K [eV]':>9} " + " ".join(f"{m[:9]:>10}" for m in MECH_NAMES)
          + f"{'  dominante':>22}")
    for _K in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14):
        _L = loss_rates(Z_INIT, _K * EV_MKS, _Hz, dict(ALL_ON)).ravel()
        print(f"{_K:9.0e} " + " ".join(f"{v:10.2e}" for v in _L)
              + f"{MECH_NAMES[int(np.argmax(_L))]:>22}")
    return


if __name__ == "__main__":
    app.run()
