"""Registry of energy-loss rates −dE/dt [J s^-1] of an electron with kinetic energy K [J] (guideline 4).

One implementation per process (the notebook re-implemented each one 2–12 times):

    adiabatic       H p²c²/E                                        [A16: dp/dt = −H p]
    synchrotron     2 σ_T c U_B γ²β² sin²θ  (fixed pitch)           [B&G 1970 Eq. 4.5; R&L Eq. 6.5b]
                    (4/3) σ_T c U_B γ²β²     (isotropic, default)    [B&G Eq. 2.21; R&L Eq. 6.7b; A08]
    ic              inverse_compton.loss_rate                        [B&G Eqs. 2.48, 2.56; A06, E05]
    coulomb         coulomb.loss_rate                                [Gould 1972 Eqs. 2.20, 5.5; A05, E18]
    excitation      excitation.loss_rate                             [Stone et al. 2002; Inokuti 1971; A13]
    ionization      ionization.loss_rate                             [Kim et al. 2000 RBED; F&S 2010; A14]
    bremsstrahlung  bremsstrahlung.loss_rate                         [BREMS CS_int; K&L 1961; B&G 1970; A15]

`rates(K, z, x_e, include, ...)` returns a dict name → rate; `total_rate` sums it. The medium at z comes from
medium.Medium (A01–A04) and H from cosmology (A10). K ≤ 0 returns zeros (the ODE may probe K < 0 in trial steps).
"""

from . import bremsstrahlung, coulomb, cosmology, excitation, inverse_compton, ionization
from . import constants as K_
from . import kinematics as KIN
from .medium import Medium

PROCESSES = ("adiabatic", "synchrotron", "ic", "coulomb", "excitation", "ionization", "bremsstrahlung")


# [A16] adiabatic loss dp/dt = −H p
def adiabatic(K, medium, **_):
    return float(cosmology.hubble(medium.z)) * float(KIN.p2c2(K)) / float(KIN.total_energy(K))


# [A08] isotropic pitch-angle average unless sin_theta is given
def synchrotron(K, medium, sin_theta=None, **_):
    """sin_theta=None → isotropic pitch-angle average (⟨sin²α⟩ = 2/3)."""
    g2b2 = float(KIN.gamma2beta2(K))
    if sin_theta is None:
        return 4.0 / 3.0 * K_.sigma_T * K_.c * medium.U_B * g2b2
    return 2.0 * K_.sigma_T * K_.c * medium.U_B * g2b2 * sin_theta ** 2


def ic(K, medium, **_):
    return float(inverse_compton.loss_rate(K, medium))


def coulomb_(K, medium, **_):
    return coulomb.loss_rate(K, medium)


def excitation_(K, medium, **_):
    return excitation.loss_rate(K, medium)


def ionization_(K, medium, **_):
    return ionization.loss_rate(K, medium)


def bremsstrahlung_(K, medium, **_):
    if float(K) / K_.e < 10.0:            # below the bremsstrahlung tables (10 eV); floor is 10.2 eV
        return 0.0
    return float(bremsstrahlung.loss_rate(K, medium.n_HI, medium.n_p))


FUNCTIONS = {"adiabatic": adiabatic, "synchrotron": synchrotron, "ic": ic, "coulomb": coulomb_,
             "excitation": excitation_, "ionization": ionization_, "bremsstrahlung": bremsstrahlung_}


# [A11] continuous mean losses; [A17] test particle, secondaries not followed
def rates(K, z, x_e, include=PROCESSES, **opts):
    """dict name → −dE/dt [J s^-1] for the processes in `include`."""
    if K <= 0.0:
        return {p: 0.0 for p in include}
    m = Medium(float(z), float(x_e))
    return {p: FUNCTIONS[p](K, m, **opts) for p in include}


def total_rate(K, z, x_e, include=PROCESSES, **opts):
    return sum(rates(K, z, x_e, include, **opts).values())
