"""Relativistic kinematics of the electron as a function of its kinetic energy K [J] (decision D13: state = K).

    γ = 1 + K/mc²,   p²c² = K² + 2K mc²  (no cancellation at K ≪ mc²),   β² = p²c²/E²,   v = cβ,   E = K + mc².
"""

import numpy as np

from . import constants as K_


def gamma(K):
    return 1.0 + np.asarray(K, dtype=float) / K_.m_e_c2


def p2c2(K):
    K = np.asarray(K, dtype=float)
    return K * (K + 2.0 * K_.m_e_c2)


def total_energy(K):
    return np.asarray(K, dtype=float) + K_.m_e_c2


def beta2(K):
    return p2c2(K) / total_energy(K) ** 2


def speed(K):
    return K_.c * np.sqrt(beta2(K))


def gamma2beta2(K):
    """γ²β² = p²c²/(mc²)²."""
    return p2c2(K) / K_.m_e_c2 ** 2
