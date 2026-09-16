#!/usr/bin/env python3
r"""
The quantitative claims the per-figure paragraphs make, computed and gated.

Every number a figure paragraph asserts has to come from somewhere. The ones
already in the manuscript's registries are reused; the ones added for the
paragraphs are computed HERE and written to provenance/figure_claims.json so
the gate can check them like any other literal.
"""
from __future__ import annotations
import project_paths  # noqa: F401  -- anchors CWD to the project root
import json
import numpy as np
import igm_losses as L
import ionization_yield as IY
import yield_comparison as YC
import photon_vs_electron as PVE

rows, prov = [], {}
def rec(cid, name, ok, detail):
    rows.append((cid, name, "PASS" if ok else "FAIL", detail))
def put(k, v):
    prov[k] = float(v); return v

YC.set_redshift(10.0)
H_z = YC.H_z_snapshot()
E = np.logspace(2, 12, 600)
fr, _ = YC.loss_branching(E, H_z)
KEYS = list(L.MECH_KEYS)
idx = {k: i for i, k in enumerate(KEYS)}
print("mechanisms:", KEYS)

# =====================================================================
# CLAIM 1  excitation is the largest IRRECOVERABLE sink
# =====================================================================
# "Irrecoverable" is the operative word. Excitation energy leaves as Lyman-series
# photons, every one of which is BELOW 13.6 eV (Ly-alpha is 10.2 eV), so it can
# never ionize hydrogen. Compton energy is larger at high E but in the loss+IC route it
# comes back as upscattered photons above 13.6 eV, so it is not a sink at all.
exc = fr[idx["excitation"]]
i_lo = 0                                    # 100 eV, the low edge of the grid
print()
print("CLAIM 1: excitation as an irrecoverable sink")
print(f"   at the low edge of the plotted grid, E = {E[i_lo]:.0f} eV:")
put("exc_frac_at_100eV", exc[i_lo])
put("exc_max_frac_in_range", exc.max())
put("exc_max_at_E_eV", E[int(np.argmax(exc))])
print(f"      excitation = {exc[i_lo]:.4f} of dK/dt")
print(f"   maximum over the plotted range = {exc.max():.4f} at "
      f"E = {E[int(np.argmax(exc))]:.0f} eV")
rising = exc[0] > exc[3]
print(f"   still RISING toward lower E at the grid edge? {rising}")
rec("F1", "excitation's share is largest at the low-energy EDGE of the grid, "
          "not at an interior maximum",
    rising and abs(exc.max() - exc[i_lo]) / exc[i_lo] < 1e-6,
    f"excitation = {exc[i_lo]:.4f} of dK/dt at E = {E[i_lo]:.0f} eV, which is "
    f"the maximum over the plotted range ({exc.max():.4f}) and is still rising "
    f"as E falls. So it must be described as 'reaches 0.353 at the 100 eV edge "
    f"of the plotted range', NOT as 'peaks at 0.353' -- the registry key "
    f"peakE_excitation_eV = 100 eV is the grid boundary, not a turnover.")

# the other channels that also cannot be recovered, at the same energy
others = {k: fr[idx[k]][i_lo] for k in KEYS if k not in ("excitation", "ionization")}
print("   competing non-ionizing channels at the same energy:")
for k, v in sorted(others.items(), key=lambda kv: -kv[1]):
    print(f"      {k:16s} {v:.3e}")
put("exc_over_next_nonionizing", exc[i_lo] / max(others.values()))
rec("F2", "excitation dominates every other non-ionizing channel there",
    exc[i_lo] > 10 * max(others.values()),
    f"at E = {E[i_lo]:.0f} eV excitation takes {exc[i_lo]:.4f} of dK/dt while "
    f"the largest other non-ionizing channel takes {max(others.values()):.3e} "
    f"-- a factor {exc[i_lo]/max(others.values()):.0f}. Excitation energy leaves "
    f"as Lyman-series photons, ALL of which lie below 13.6 eV (Ly-alpha is "
    f"10.2 eV), so it can never ionize hydrogen. Compton is larger at high E "
    f"but is NOT a sink in the loss+IC route, where that energy returns as upscattered "
    f"photons above 13.6 eV -- which is the whole difference between D and E.")

# =====================================================================
# CLAIM 2  the negligible channels, and what their smallness licenses
# =====================================================================
negl = {k: fr[idx[k]].max() for k in ("synchrotron", "coulomb", "bremsstrahlung")}
print()
print("CLAIM 2: the channels the superseded route omitted")
for k, v in negl.items():
    print(f"   {k:16s} peak share = {v:.3e}  ({100*v:.4f}%)")
    put(f"peak_{k}_pct", 100 * v)
put("max_negligible_pct", 100 * max(negl.values()))
rec("F3", "synchrotron, Coulomb and bremsstrahlung each stay below 0.25%",
    max(negl.values()) < 2.5e-3,
    f"peak shares of dK/dt anywhere in 1e2-1e12 eV: synchrotron "
    f"{100*negl['synchrotron']:.5f}%, Coulomb {100*negl['coulomb']:.4f}%, "
    f"bremsstrahlung {100*negl['bremsstrahlung']:.4f}%. The largest is "
    f"{100*max(negl.values()):.4f}%. THAT is the quantitative licence for the "
    f"superseded deposition-fit route to omit all three and still land within "
    f"10% -- a justification the earlier text asserted without one.")

# =====================================================================
# CLAIM 3  panel (a): the channels do not live at the same energies
# =====================================================================
print()
print("CLAIM 3: the energy ranges of the two channels in panel (a)")
g_lo, g_hi = IY.E_TH_HI, PVE.photon_band_max()
e_lo, e_hi = PVE.CR_E_MIN_EV, PVE.CR_E_MAX_EV
dex_g = np.log10(g_hi / g_lo)
dex_e = np.log10(e_hi / e_lo)
put("photon_band_lo_eV", g_lo); put("photon_band_hi_eV", g_hi)
put("photon_band_dex", dex_g); put("electron_band_dex", dex_e)
put("band_dex_ratio", dex_e / dex_g)
print(f"   stellar LyC band : {g_lo:.3f} - {g_hi:.3f} eV = {dex_g:.4f} dex")
print(f"   CR electron band : {e_lo:.0f} - {e_hi:.0e} eV = {dex_e:.4f} dex")
print(f"   ratio of spans   : {dex_e/dex_g:.2f}")
gap = np.log10(e_lo / g_hi)
put("band_gap_dex", gap)
print(f"   and they do not even touch: {gap:.3f} dex separates the top of the")
print(f"   photon band from the bottom of the electron band")
rec("F4", "the two channels occupy disjoint energy ranges, by a wide margin",
    e_lo > g_hi and dex_e > 10 * dex_g,
    f"the escaping stellar band is {g_lo:.3f}-{g_hi:.3f} eV ({dex_g:.3f} dex, "
    f"1-4 Ryd by construction) while CR electrons span {e_lo:.0e}-{e_hi:.0e} eV "
    f"({dex_e:.2f} dex) -- a {dex_e/dex_g:.1f}x wider range, and the two do not "
    f"overlap at all: {gap:.2f} dex of empty axis separates them. CONSEQUENCE "
    f"for reading panel (a): the two curves never compete at the same energy, "
    f"so nothing can be concluded by comparing their heights there. The "
    f"comparison only becomes meaningful after integrating over energy, which "
    f"is what zeta is.")

# =====================================================================
# CLAIM 5  how much ENERGY actually goes to adiabatic expansion
# =====================================================================
# The branching panel shows a RATE share, L_i/L_tot at one energy. "How much
# energy is lost" is a different question: it is the share integrated over the
# whole degradation of the primary,
#       f_i(K) = (1/K) int_0^K [L_i(K')/L_tot(K')] dK' ,
# i.e. the fraction of the primary's initial energy that ends up in channel i
# by the time it has slowed to rest. Computed on a linear grid so the integral
# is not dominated by its own log spacing.
print()
print("CLAIM 5: integrated ENERGY fraction per channel, f_i(K)")


def energy_fractions(K_max, z, n=4000):
    YC.set_redshift(z)
    Hz = YC.H_z_snapshot()
    Kg = np.linspace(L.THRESHOLD_EV_ION, K_max, n)
    f, _ = YC.loss_branching(Kg, Hz)
    out = {}
    for k in KEYS:
        out[k] = float(np.trapz(f[idx[k]], Kg) / (Kg[-1] - Kg[0]))
    return out


for z in (10.0, 20.0):
    print(f"   z = {z:g}")
    for K_max, tag in ((1.0e3, "1keV"), (1.0e5, "100keV"), (1.0e6, "1MeV"),
                       (1.0e9, "1GeV"), (1.0e12, "1TeV")):
        f = energy_fractions(K_max, z)
        zt = "" if z == 10.0 else "_z20"
        put(f"Efrac_adiabatic_{tag}{zt}", f["adiabatic"])
        put(f"Efrac_compton_{tag}{zt}", f["compton"])
        put(f"Efrac_excitation_{tag}{zt}", f["excitation"])
        put(f"Efrac_ionization_{tag}{zt}", f["ionization"])
        print(f"      K = {tag:7s} adiab {f['adiabatic']:.5f}  compton "
              f"{f['compton']:.5f}  exc {f['excitation']:.5f}  ion "
              f"{f['ionization']:.5f}")
YC.set_redshift(10.0)

# The integrated adiabatic share is NON-MONOTONIC: negligible at 1 keV where
# collisions own everything, negligible at 1 TeV where Compton takes 99.99%
# before anything else can act, and maximal in between. Anchoring the check on
# 1 TeV (as a first version did) compared two numbers that are both ~0.
TAGS = ("1keV", "100keV", "1MeV", "1GeV", "1TeV")
a10 = {t: prov[f"Efrac_adiabatic_{t}"] for t in TAGS}
a20 = {t: prov[f"Efrac_adiabatic_{t}_z20"] for t in TAGS}
tmax = max(a10, key=a10.get)
put("Efrac_adiabatic_max", a10[tmax])
put("Efrac_adiabatic_max_pct", 100 * a10[tmax])
put("Efrac_adiabatic_max_z20_pct", 100 * a20[tmax])
put("Efrac_adiabatic_1keV_pct", 100 * a10["1keV"])
put("Efrac_adiabatic_1TeV_pct", 100 * a10["1TeV"])
put("Efrac_compton_1TeV_pct", 100 * prov["Efrac_compton_1TeV"])
put("Efrac_adiabatic_drop_z20", a10[tmax] / a20[tmax])
print()
print(f"   integrated adiabatic share is maximal at K = {tmax}: "
      f"{100*a10[tmax]:.3f}% at z=10, {100*a20[tmax]:.3f}% at z=20")
rec("F5", "the ENERGY lost to adiabatic expansion, integrated over the cascade",
    tmax == "1MeV" and a20[tmax] < a10[tmax]
    and a10["1keV"] < 1e-3 and a10["1TeV"] < 1e-4,
    f"integrating L_adiab/L_tot over the primary's degradation, the share peaks "
    f"at K = 1 MeV: {100*a10[tmax]:.3f}% of the initial energy at z = 10, "
    f"falling to {100*a20[tmax]:.3f}% at z = 20 (a factor "
    f"{a10[tmax]/a20[tmax]:.2f}), because H(z) grows as (1+z)^1.5 while the "
    f"competing CMB energy density grows as (1+z)^4. It is NON-MONOTONIC and "
    f"negligible at both ends: {100*a10['1keV']:.4f}% at 1 keV, where "
    f"collisions own everything, and {100*a10['1TeV']:.5f}% at 1 TeV, where "
    f"Compton takes {100*prov['Efrac_compton_1TeV']:.3f}% before anything else "
    f"can act. NOTE this is NOT the 9.29% of the branching panel: that is an "
    f"instantaneous RATE share reached only near 3e5 eV, whereas this averages "
    f"over the whole cascade.")

print()
for cid, name, st, det in rows:
    print(f"   [{st}] {cid:3s} {name}\n         {det}\n")
print(f"   {sum(1 for r in rows if r[2]=='PASS')}/{len(rows)} checks pass.")
with open("provenance/figure_claims.json", "w") as fh:
    json.dump({"derived": prov,
               "checks": [{"id": c, "name": n, "state": s, "detail": d}
                          for c, n, s, d in rows]}, fh, indent=2, sort_keys=True)
print(f"[ARTEFACT] provenance/figure_claims.json  --  {len(prov)} keys")
