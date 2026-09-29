"""Collisional ionization of H(1s) by electron impact: RBED cross section and mean energy lost (A14; E15–E17).

Cross section  [Kim2000 Eq. 20, p. 052710-4] (relativistic binary-encounter-dipole, NOT RBEB; E16):
    σ = 2π r_e² N / (b′(β_t² + β_u² + β_b²)) × { D(t)[ln(β_t²/(1−β_t²)) − β_t² − ln 2b′]
          + (2 − N_i/N)[1 − 1/t − (ln t/(t+1))(1+2t′)/(1+t′/2)² + (b′²/(1+t′/2)²)(t−1)/2] }
    (4π a0² α⁴ = 4π r_e²), t = T/B, t′ = T/mc², b′ = B/mc², u′ = U/mc², β_x² = 1 − 1/(1+x′)².
Oscillator strength  [KimRudd1994 Table I, H 1s]: df/dw = Σ_j c_j y^j, y = 1/(w+1) (w = W/B), j = 2…5.
    Then, derived here (checked in tests against the printed N_i = 0.4343 and M² = 0.2834):
    D(t) = Σ_j c_j [1 − (2/(t+1))^j]/j   [Kim2000 Eq. 6],   N_i = Σ_j c_j/(j−1)   [Kim2000 Eq. 4].
    B = U = 13.6057 eV (Table I; user decision 2026-09-28), N = 1.
Energy lost per ionization  W̄ = B + ⟨ε⟩, secondary spectrum p(ε) ∝ 1/[1 + (ε/ε̄)^2.1], ε ∈ [0, (T−B)/2]
    [FurlanettoStoever2010 Eq. 2, p. 1871; ε̄ = 8 eV for H I]. Closed form (D07, derived):
    ∫_0^X dε/(1+(ε/ε̄)^p) = X ₂F₁(1, 1/p; 1+1/p; −(X/ε̄)^p),  ∫_0^X ε dε/(…) = (X²/2) ₂F₁(1, 2/p; 1+2/p; −(X/ε̄)^p)
    (agrees with mpmath quadrature to 2e-16 for X = 1 … 1e12 eV; replaces the notebook quadrature, E15).
Validity   Kim et al. warn that for T ≫ mc² the density effect must be included and the asymptotic Bethe form
           may be more reliable (p. 052710-5, -6): not included here (flagged in A14).
"""

import math
import pathlib

import numpy as np
import yaml
from scipy.special import hyp2f1

from . import constants as K
from . import kinematics as KIN

ROOT = pathlib.Path(__file__).resolve().parents[3]
KR = yaml.safe_load((ROOT / "provenance" / "data" / "kim_rudd1994_H_1s.yaml").read_text(encoding="utf-8"))
COEF = {2: KR["coefficients"]["b"], 3: KR["coefficients"]["c"], 4: KR["coefficients"]["d"], 5: KR["coefficients"]["e"]}
B_EV = K._P["B_ion_RBED"]
U_EV = KR["U_eV"]
N_EL = KR["N"]
EPS_BAR = K._P["secondary_E_bar"]
P_EXP = K._P["secondary_exponent"]
N_I = sum(c / (j - 1) for j, c in COEF.items())


def D(t):
    return sum(c * (1.0 - (2.0 / (t + 1.0)) ** j) / j for j, c in COEF.items())


def sigma(T):
    """RBED ionization cross section [m²] for kinetic energy T [eV]; 0 for T ≤ B."""
    if T <= B_EV:
        return 0.0
    mc2 = K.m_e_c2_eV
    t, tp, bp, up = T / B_EV, T / mc2, B_EV / mc2, U_EV / mc2
    b2 = lambda xp: 1.0 - 1.0 / (1.0 + xp) ** 2
    bt2, bb2, bu2 = b2(tp), b2(bp), b2(up)
    # ln(β_t²/(1−β_t²)) = ln(γ²β²) = ln(t′(t′+2)) exactly (avoids 1 − β² → 0 in floating point at T ≫ mc²)
    dipole = D(t) * (math.log(tp * (tp + 2.0)) - bt2 - math.log(2.0 * bp))
    binary = (2.0 - N_I / N_EL) * (1.0 - 1.0 / t - math.log(t) / (t + 1.0) * (1.0 + 2.0 * tp) / (1.0 + tp / 2.0) ** 2
                                   + bp ** 2 / (1.0 + tp / 2.0) ** 2 * (t - 1.0) / 2.0)
    return 2.0 * math.pi * K.r_e ** 2 * N_EL / (bp * (bt2 + bu2 + bb2)) * (dipole + binary)


def mean_secondary_energy(T):
    """⟨ε⟩ [eV] over [0, (T−B)/2] (closed form)."""
    X = (T - B_EV) / 2.0
    if X <= 0:
        return 0.0
    z = -(X / EPS_BAR) ** P_EXP
    return (X / 2.0) * hyp2f1(1.0, 2.0 / P_EXP, 1.0 + 2.0 / P_EXP, z) / hyp2f1(1.0, 1.0 / P_EXP, 1.0 + 1.0 / P_EXP, z)


def stopping_cross_section(T):
    """S(T) = σ(T) (B + ⟨ε⟩) [m² J]."""
    return sigma(T) * (B_EV + mean_secondary_energy(T)) * K.e


# [A14] RBED (E16) with mean loss B + <ε> (Furlanetto & Stoever 2010, E17)
def loss_rate(K_J, medium):
    """−dE/dt [J s^-1] = n_HI v S(T)."""
    return medium.n_HI * float(KIN.speed(K_J)) * stopping_cross_section(float(K_J) / K.e)
