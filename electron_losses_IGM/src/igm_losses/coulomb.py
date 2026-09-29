"""Coulomb losses of a fast electron on the free electrons of the plasma (cold plasma, A05; E18).

    −dE/dx = n_e e⁴/(4π ε0² m v²) · B      [Gould1972 Eq. 1.1, p. 146; Gaussian 4π n e⁴/(m v²) with e² → e²/(4πε0)]
    ω_p² = n_e e²/(ε0 m)                   [Eq. 2.12]
    β ≫ α  (relativistic, Møller):  B = ln[(γ−1)^½ β mc² (2δ)^½/(ħ ω_p)] + ½(1 + (2γ−1)/γ²) ln(1−δ)
                                        + ½δ/(1−δ) + ¼((γ−1)/γ)² δ²                          [Eq. 5.5, p. 153]
    β ≪ α  (classical):             B = ln[2(½m) v³ δ^½ / (Γ e² ω_p)],  Γ = e^{γ_E}, e² → e²/(4πε0)  [Eq. 2.20, p. 149]
    Switch at β = α (user decision E18; Gould: the case β ≈ α "remains to be solved", p. 152, so the jump is reported).
    δ = maximum fractional energy transfer = 1/2 [Eq. 2.18; parameter delta_gould].
    −dE/dt = v · (−dE/dx). Valid for v ≫ v_thermal (all of Gould's results; A05, cold plasma).
"""

import math

from . import constants as K
from . import kinematics as KIN

DELTA = K._P["delta_gould"]
EULER_GAMMA = 0.57721566490153286      # Euler's constant (Gould Eq. 2.14 quotes 0.5772…)


def stopping_number(K_J, n_e):
    g = float(KIN.gamma(K_J))
    b2 = float(KIN.beta2(K_J))
    beta = math.sqrt(b2)
    v = K.c * beta
    wp = math.sqrt(n_e * K.e ** 2 / (K.eps0 * K.m_e))
    if beta >= K.alpha:
        return (math.log(math.sqrt(g - 1.0) * beta * K.m_e_c2 * math.sqrt(2.0 * DELTA) / (K.hbar * wp))
                + 0.5 * (1.0 + (2.0 * g - 1.0) / g ** 2) * math.log(1.0 - DELTA)
                + 0.5 * DELTA / (1.0 - DELTA) + 0.25 * ((g - 1.0) / g) ** 2 * DELTA ** 2)
    e2 = K.e ** 2 / (4.0 * math.pi * K.eps0)
    return math.log(2.0 * 0.5 * K.m_e * v ** 3 * math.sqrt(DELTA) / (math.exp(EULER_GAMMA) * e2 * wp))


# [A05] cold plasma: target electrons at rest
def loss_rate(K_J, medium):
    """−dE/dt [J s^-1]; 0 if there are no free electrons."""
    n_e = medium.n_e
    if n_e <= 0.0:
        return 0.0
    v = float(KIN.speed(K_J))
    return n_e * K.e ** 4 / (4.0 * math.pi * K.eps0 ** 2 * K.m_e * v * v) * stopping_number(K_J, n_e) * v
