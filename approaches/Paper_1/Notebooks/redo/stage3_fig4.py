"""
Stage 3b: every number behind Fig. 4 of manuscript_13page, and the heat lock,
recomputed from my own table (redo_cascade_table.npz) and compared with the
published values and with the project's own photon_cascade_table.npz.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
mine = np.load(HERE / "redo_cascade_table.npz")
theirs = np.load(HERE.parent / "photon_cascade_table.npz")

K, z = mine["K"], mine["z"]
NLOW = int(mine["nlow"])
Km = K[NLOW:]                       # the grid shared with cascade_traj_table
row = {zi: int(np.argmin(np.abs(z - zi))) for zi in (10.0, 20.0)}


def near(K0, grid=Km):
    return int(np.argmin(np.abs(np.log(grid) - np.log(K0))))


print("=" * 78)
print("A.  ELEMENT-BY-ELEMENT COMPARISON WITH photon_cascade_table.npz")
print("=" * 78)
assert np.allclose(K[NLOW:], theirs["K"]), "grids differ"
for tag, a, b in (("Ye  (electron impact)", mine["Y"][:, NLOW:], theirs["Ye"]),
                  ("Yph (fixed 1 keV band)", mine["Ypf"][:, NLOW:], theirs["Yph_fixed"]),
                  ("Yph (mfp ceiling)", mine["Ypm"][:, NLOW:], theirs["Yph_mfp"]),
                  ("Ngamma (fixed band)", mine["Ngf"][:, NLOW:], theirs["Ngamma_fixed"])):
    m = (b > 1e-3) & (a > 1e-3)
    r = a[m] / b[m]
    print(f"  {tag:24s}  n={m.sum():4d}  median ratio {np.median(r):7.4f}  "
          f"5-95% [{np.percentile(r,5):6.4f}, {np.percentile(r,95):6.4f}]")

print()
print("=" * 78)
print("B.  HEADLINE NUMBERS OF SECT. 5 AND THE ABSTRACT (manuscript_13page)")
print("=" * 78)
hdr = f"  {'quantity':<52} {'mine':>12} {'published':>12}"
print(hdr); print("  " + "-" * 76)


def show(label, val, pub, fmt="{:12.4g}"):
    print(f"  {label:<52} {fmt.format(val):>12} {fmt.format(pub):>12}")


for zi, plateau, ymin_pub in ((20.0, 7.7e3, None), (10.0, 1.24e4, None)):
    iz = row[zi]
    Ye = mine["Y"][iz, NLOW:]
    show(f"direct-cascade plateau Y_e(K->1e13), z_i={zi:.0f}", Ye[-1], plateau)
    W = Km / np.maximum(Ye, 1e-300)
    lo = Km < 1e5
    j = int(np.argmin(W[lo]))
    show(f"minimum W of the direct cascade, z_i={zi:.0f} [eV]", W[lo][j], 35.4)
    show(f"   ... located at K [eV]", Km[lo][j], 1.0e3)
for K0, pub in ((1e2, 2.65), (1e3, 28.3)):
    j = near(K0)
    show(f"Y_e at K={K0:.0e} eV, z_i=20", mine["Y"][row[20.0], NLOW:][j], pub)

print()
for zi, pf, pm in ((20.0, 2.9e6, 3.9e6), (10.0, 4.0e6, 4.5e6)):
    iz = row[zi]
    tf = (mine["Y"] + mine["Ypf"])[iz, NLOW:]
    tm = (mine["Y"] + mine["Ypm"])[iz, NLOW:]
    show(f"total plateau, fixed 1 keV band, z_i={zi:.0f}", tf[-1], pf)
    show(f"total plateau, mfp ceiling,      z_i={zi:.0f}", tm[-1], pm)
    show(f"   ... horizon/fixed increase [%]", 100 * (tm[-1] / tf[-1] - 1),
         {20.0: 37.0, 10.0: 13.0}[zi], "{:12.1f}")
    hi = Km > 1e6
    W2 = Km[hi] / tf[hi]
    j = int(np.argmin(W2))
    show(f"second minimum of W, z_i={zi:.0f} [eV]", W2[j], 38.85)
    show(f"   ... located at K [eV]", Km[hi][j], 7.0e7)
    j26 = near(2.6e8)
    show(f"Y_ph at K=2.6e8 eV, z_i={zi:.0f}",
         mine["Ypf"][iz, NLOW:][j26], {20.0: 2.8e6, 10.0: 3.8e6}[zi])
    show(f"ionizations per productive photon, z_i={zi:.0f}",
         mine["Ypf"][iz, NLOW:][-1] / mine["Ngf"][iz, NLOW:][-1], 3.6)

print()
print("=" * 78)
print("C.  SPECTRUM-AVERAGED W (Table 3 of manuscript_13page)")
print("=" * 78)


def wbar(iz, p, mode):
    Kf = np.geomspace(1e2, 1e12, 6001)
    Yn = mine["Y"][iz, NLOW:].copy()
    if mode:
        Yn = Yn + mine[mode][iz, NLOW:]
    Y = 10 ** np.interp(np.log10(Kf), np.log10(Km),
                        np.log10(np.maximum(Yn, 1e-300)))
    num = np.trapz(Kf ** (1 - p) * Kf, np.log(Kf))
    den = np.trapz(Y * Kf ** (-p) * Kf, np.log(Kf))
    return num / den


PUB = {20.0: {None: (888, 107, 46.3, 37.8, 37.4),
              "Ypf": (290, 79.8, 44.2, 37.7, 37.4),
              "Ypm": (263, 77.4, 44.0, 37.7, 37.4)},
       10.0: {None: (804, 102, 45.3, 37.6, 37.4),
              "Ypf": (271, 77.1, 43.5, 37.5, 37.4),
              "Ypm": (260, 76.2, 43.4, 37.5, 37.4)}}
print(f"  {'z_i':>4} {'branch':>8} " + "".join(f"{p:>16}" for p in
                                               (1.8, 2.0, 2.2, 2.5, 3.0)))
for zi in (20.0, 10.0):
    for mode, tag in ((None, "direct"), ("Ypf", "1 keV"), ("Ypm", "horizon")):
        cells = []
        for p, pub in zip((1.8, 2.0, 2.2, 2.5, 3.0), PUB[zi][mode]):
            cells.append(f"{wbar(row[zi], p, mode):8.4g}/{pub:<7.4g}")
        print(f"  {zi:4.0f} {tag:>8} " + "".join(f"{c:>16}" for c in cells))
print("       (mine / published, eV)")

print()
print("=" * 78)
print("D.  HEAT PER IONIZATION AND THE LOCK OF EQ. (16) OF manuscript.tex")
print("=" * 78)
KB_EV = 8.617333262e-5
MU = 1.0813


def theta(Q):
    return (2.0 / 3.0) * Q / (MU * KB_EV)


print(f"  {'K_ini [eV]':>11} | {'z_i=20':^34} | {'z_i=10':^34}")
print(f"  {'':>11} | {'Q_dir':>8}{'Q_tot':>9}{'closure':>9}{'W_tot':>8} | "
      f"{'Q_dir':>8}{'Q_tot':>9}{'closure':>9}{'W_tot':>8}")
for K0 in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e12):
    j = near(K0)
    line = f"  {Km[j]:11.3g} |"
    for zi in (20.0, 10.0):
        iz = row[zi]
        Y = mine["Y"][iz, NLOW:][j]
        H = mine["Hq"][iz, NLOW:][j]
        X = mine["Xq"][iz, NLOW:][j]
        Yp = mine["Ypf"][iz, NLOW:][j]
        Hp = mine["Hpf"][iz, NLOW:][j]
        B = 13.6057
        clo = (B * Y + H + X) / Km[j]
        Qd = H / max(Y, 1e-30)
        Qt = (H + Hp) / max(Y + Yp, 1e-30)
        line += (f" {Qd:8.3f}{Qt:9.3f}{clo:9.4f}"
                 f"{Km[j]/max(Y+Yp,1e-30):8.3f} |")
    print(line)
print("   Q_dir = heat/ionization, electron impact only")
print("   Q_tot = (heat + photoelectron heat)/(all ionizations), fixed band")
print("   closure = (13.6*Y + heat + excitation)/K_ini  -> 1 when nothing escapes")

print()
Qs = []
for zi in (20.0, 10.0):
    iz = row[zi]
    m = (Km >= 1e2) & (Km <= 1e6)
    q = mine["Hq"][iz, NLOW:][m] / np.maximum(mine["Y"][iz, NLOW:][m], 1e-30)
    Qs += [q.min(), q.max()]
    print(f"  z_i={zi:.0f}: Q over 1e2-1e6 eV = {q.min():.2f} - {q.max():.2f} eV")
print(f"  published range: 6.0 - 9.6 eV")
print(f"  Theta = (2/3)Q/(mu k_B) = {theta(min(Qs)):.3e} - {theta(max(Qs)):.3e} K"
      f"   (published 4.3e4 - 6.9e4 K)")
