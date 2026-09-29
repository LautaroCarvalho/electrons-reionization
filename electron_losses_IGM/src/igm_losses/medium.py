"""The intergalactic medium seen by the electron: gas densities, magnetic field and CMB (assumptions A01–A06).

    n_H(z)   = (1 − Y_p) Ω_b ρ_crit,0 (1+z)³ / m_p          [A02; Y_p from Planck 2018 VI Table 2 caption]
    n_HI(z)  = (1 − x_e) n_H,   n_e = n_p = x_e n_H          [A03, A04; x_e constant in z, swept in some figures]
    B(z)     = B0 (1+z)²,        U_B = B²/(2 μ0)              [A01]
    T_CMB(z) = T_CMB0 (1+z),     U_CMB = a_rad T_CMB⁴         [A06, E04]

Library: astropy Planck18 for Ω_b and ρ_crit,0 (A10). Everything else from igm_losses.constants / parameters.yaml.
Units: SI (m^-3, T, J m^-3, K).
"""

from dataclasses import dataclass

import astropy.units as u
from astropy.cosmology import Planck18

from . import constants as K

RHO_B0 = (Planck18.Ob0 * Planck18.critical_density0).to(u.kg / u.m ** 3).value   # kg m^-3
N_H0 = (1.0 - K._P["Y_p"]) * RHO_B0 / K.m_p                                        # m^-3 at z = 0


@dataclass(frozen=True)
class Medium:
    """State of the medium at one redshift for a given ionization fraction x_e."""
    z: float
    x_e: float

    @property
    def n_H(self):
        # [A02] homogeneous H density from Planck 2018 Ω_b and Y_p
        return N_H0 * (1.0 + self.z) ** 3

    @property
    def n_HI(self):
        # [A03] [A04] n_HI = (1 − x_e) n_H, x_e constant in z
        return (1.0 - self.x_e) * self.n_H

    @property
    def n_e(self):
        # [A03] n_e = x_e n_H (no He)
        return self.x_e * self.n_H

    @property
    def n_p(self):
        return self.n_e

    @property
    def B(self):
        # [A01] frozen-in uniform field, B ∝ (1+z)²
        return K._P["B0"] * (1.0 + self.z) ** 2

    @property
    def U_B(self):
        return self.B ** 2 / (2.0 * K.mu0)

    @property
    def T_CMB(self):
        return K.T_CMB0 * (1.0 + self.z)

    @property
    def U_CMB(self):
        # [A06] the CMB is the only photon field
        return K.a_rad * self.T_CMB ** 4
