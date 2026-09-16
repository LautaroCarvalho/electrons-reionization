"""Numerical regressions for claims shared by both manuscript drafts."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
d = np.load(ROOT / "photon_cascade_table.npz")
base = np.load(ROOT / "cascade_traj_table.npz")
K, z = d["K"], d["z"]

assert np.allclose(K, base["K"])
assert np.allclose(z, base["z"])
assert np.allclose(d["Ye"], base["Y"])
for key in ("Ye", "Yph_fixed", "Yph_mfp", "Ngamma_fixed", "Ngamma_mfp"):
    assert np.nanmin(d[key]) > -1.0e-90
# The wider band must not reduce a resolved yield.  Around the onset, values
# below one ionization carry sub-percent ODE/interpolation noise.
tol = np.maximum(1e-5, 0.01 * d["Yph_fixed"])
assert np.all(d["Yph_mfp"] + tol >= d["Yph_fixed"])


def row(zi):
    return int(np.argmin(np.abs(z - zi)))


def nearest(K0):
    return int(np.argmin(np.abs(np.log(K) - np.log(K0))))


for zi, expected_fixed, expected_mfp in (
    (20.0, 2.9e6, 3.9e6), (10.0, 4.0e6, 4.5e6)
):
    iz = row(zi)
    total_f = d["Ye"][iz] + d["Yph_fixed"][iz]
    total_m = d["Ye"][iz] + d["Yph_mfp"][iz]
    assert abs(total_f[-1] / expected_fixed - 1.0) < 0.03
    assert abs(total_m[-1] / expected_mfp - 1.0) < 0.03
    high = K > 1e6
    W2 = K[high] / total_f[high]
    assert 38.0 < W2.min() < 40.0

# Direct-cascade headline values and the low-energy W minimum.
for zi, plateau in ((20.0, 7.7e3), (10.0, 1.24e4)):
    iz = row(zi)
    assert abs(d["Ye"][iz, -1] / plateau - 1.0) < 0.03
    low = K < 1e5
    assert 35.0 < np.min(K[low] / d["Ye"][iz, low]) < 36.0

for K0, target in ((1e2, 2.65), (1e3, 28.3)):
    jj = nearest(K0)
    assert abs(d["Ye"][row(20), jj] / target - 1.0) < 0.04

# Quoted 2.6e8-eV photon-seeded yields.
j = nearest(2.6e8)
assert abs(d["Yph_fixed"][row(20), j] / 2.8e6 - 1.0) < 0.02
assert abs(d["Yph_fixed"][row(10), j] / 3.8e6 - 1.0) < 0.02

# Each productive high-energy photon gives one primary photoionization plus
# the tabulated cascade of its freed electron (about 3.6 total on average).
for zi in (20.0, 10.0):
    iz = row(zi)
    ratio = d["Yph_fixed"][iz, -1] / d["Ngamma_fixed"][iz, -1]
    assert 3.5 < ratio < 3.7


def wbar(iz, p, mode):
    Kf = np.geomspace(1e2, 1e12, 6001)
    Ynode = d["Ye"][iz].copy()
    if mode:
        Ynode += d[f"Yph_{mode}"][iz]
    Y = 10 ** np.interp(np.log10(Kf), np.log10(K),
                         np.log10(np.maximum(Ynode, 1e-300)))
    num = np.trapz(Kf ** (1-p) * Kf, np.log(Kf))
    den = np.trapz(Y * Kf ** (-p) * Kf, np.log(Kf))
    return num / den


reference = {
    20.0: {
        None: (888.0, 107.0, 46.3, 37.8, 37.4),
        "fixed": (290.0, 79.8, 44.2, 37.7, 37.4),
        "mfp": (263.0, 77.4, 44.0, 37.7, 37.4),
    },
    10.0: {
        None: (804.0, 102.0, 45.3, 37.6, 37.4),
        "fixed": (271.0, 77.1, 43.5, 37.5, 37.4),
        "mfp": (260.0, 76.2, 43.4, 37.5, 37.4),
    },
}
for zi, modes in reference.items():
    for mode, values in modes.items():
        for p, expected in zip((1.8, 2.0, 2.2, 2.5, 3.0), values):
            assert abs(wbar(row(zi), p, mode) / expected - 1.0) < 0.006

print("ALL MANUSCRIPT NUMERICAL CONSISTENCY CHECKS PASSED")
