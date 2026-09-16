"""Verify every numeric claim in Reionization_State_of_the_Art.tex against the
text of the paper it cites.  A claim passes if ALL of its required numeric
tokens are found in the normalised text of the cited paper."""
import json, re, sys, unicodedata

KEY = json.load(open('verify_keymap.json'))
CACHE = {}

def norm(t):
    t = unicodedata.normalize('NFKD', t)
    for a, b in [('−','-'),('–','-'),('—','-'),('‐','-'),('‑','-'),
                 ('×','x'),('∗','*'),('ˆ',''),('˜',''),(' ',''),
                 ('“','"'),('”','"'),('′',"'")]:
        t = t.replace(a, b)
    t = t.replace('̸','')
    return re.sub(r'\s+', '', t)

def text(bibkey):
    if bibkey not in CACHE:
        CACHE[bibkey] = norm(open('txt/'+KEY[bibkey], encoding='utf-8', errors='ignore').read())
    return CACHE[bibkey]

def check(bibkey, tokens):
    T = text(bibkey)
    return [tok for tok in tokens if norm(tok) not in T]

CLAIMS = []
def C(sec, key, desc, *toks):
    CLAIMS.append((sec, key, desc, list(toks)))

# ---------------- Section 1.3: cross-checks (inputs) ----------------
C('1.3','Planck2018','Omega_b h^2 = 0.02237','0.02237')
C('1.3','Planck2018','h from H0 = 67.36 +- 0.54','67.36','0.54')
C('1.3','Planck2018','Y_p = 0.2454','0.2454')
C('1.3','Robertson2015','fiducial log10 xi_ion = 53.14, fesc = 0.2','53.14','fesc=0.2')

# ---------------- Section 2.1: optical depth ----------------
C('2.1','Planck2018','tau = 0.054 +- 0.007','0.054','0.007')
C('2.1','Pagano2019','tau = 0.0566 +0.0053 -0.0062','0.0566','0.0053','0.0062')
C('2.1','Pagano2019','tau = 0.059 +- 0.006 combined','0.059','0.006')
C('2.1','Pagano2019','z_re = 8.14 +- 0.61','8.14','0.61')
C('2.1','MonteroCamacho2026','tau = 0.067 +- 0.011','0.067','0.011')

# ---------------- Section 2.2: end of reionization ----------------
C('2.2','McGreer2014','xHI <= 0.06+0.05 at z=5.9','0.06','0.05','5.9')
C('2.2','McGreer2014','xHI <= 0.04+0.05 at z=5.6','0.04','5.6')
C('2.2','Davies2025','4 dark-pixel limits','0.030','0.048','0.095','0.037','0.191','0.056','0.199','0.087')
C('2.2','Davies2025','redshift bins','5.481','5.654','5.831','6.043')
C('2.2','Becker2014','110 Mpc/h trough, tau_eff > 7, z=5.5','110','5.5')
C('2.2','Becker2014','factor 3 or more xHI fluctuations','5.6','5.8')
C('2.2','Bosman2021','67 sightlines z>5.5','67')
C('2.2','Bosman2021','2.5 sigma at z=5.3; >3.5 sigma at z>=5.4','2.5','3.5','5.4')
C('2.2','Bosman2021','t = 1.1 Gyr','1.1')
C('2.2','Zhu2021','55 QSOs; F30 = 0.9,0.6,0.15','55','0.9','0.6','0.15')
C('2.2','Zhu2021','ultra-long gaps L>80 h^-1 Mpc; nine','80')
C('2.2','Gaikwad2023','reionization completes by z ~ 5.2','5.2')

# ---------------- Section 2.3: xHI table ----------------
C('2.3','Gaikwad2023','xHI(z=6.0) = 0.174 +0.093 -0.109','1.744','0.925','1.089')
C('2.3','Durovcikova2024','ATON 0.21 +0.17 -0.07 at z=6.10','0.21','0.17','0.07','6.10')
C('2.3','Durovcikova2024','ATON 0.21 +0.33 -0.07 at z=6.46','0.33','6.46')
C('2.3','Durovcikova2024','ATON 0.37 +-0.17 at z=6.87','0.37','6.87')
C('2.3','Durovcikova2024','CROC 0.57 +0.26 -0.47 at z=6.46','0.57','0.26','0.47')
C('2.3','Durovcikova2024','t_Q <~ 7 Myr','7')
C('2.3','Mason2017','xHI = 0.59 +0.11 -0.15 at z~7','0.59','0.11','0.15')
C('2.3','Davies2018','xHI(7.09) = 0.48 +- 0.26','0.48','0.26','7.09')
C('2.3','Davies2018','xHI(7.54) = 0.60 +0.20 -0.23','0.60','0.20','0.23','7.54')
C('2.3','Umeda2023','xHI 0.53 +0.18 -0.47 at z=7.12','0.53','0.18','0.47','7.12')
C('2.3','Umeda2023','xHI 0.92 +0.08 -0.10 at z=9.91','0.92','0.08','0.10','9.91')
C('2.3','Umeda2023','27 bright galaxies, M_UV < -18.5','27','18.5')
C('2.3','Fan2022','EoR midpoint 6.9 < z < 7.6','6.9','7.6')
C('2.3','Sims2025','z50 = 7.16 +0.15 -0.12; dz < 1.8','7.16','0.15','0.12','1.8')
C('2.3','Sims2025','M_min > 2.6e9 Msun; Vc > 50 km/s','2.6','50')
C('2.3','Giovinazzo2026','1428 galaxies; reionization ends z~5.8','1428','5.8')

# ---------------- Section 2.4: UV background ----------------
C('2.4','Becker2021','lmfp 9.09 +1.62 -1.28 pMpc at z=5.1','9.09','1.62','1.28','5.1')
C('2.4','Becker2021','lmfp 0.75 +0.65 -0.45 pMpc at z=6.0','0.75','0.65','0.45')
C('2.4','Gaikwad2023','Gamma12 at z=4.90','0.501','0.275','0.232')
C('2.4','Gaikwad2023','lmfp at z=4.90','50.119','33.058','19.919')
C('2.4','Gaikwad2023','Gamma12 at z=6.00','0.145','0.157','0.087')
C('2.4','Gaikwad2023','lmfp at z=6.00','8.318','7.531','4.052')
C('2.4','Gaikwad2023','ndot at z=6.00','0.701','0.357','0.191')
C('2.4','Gaikwad2023','lmfp x6, fHI x10^4 evolution','6','104')

# ---------------- Section 2.5: thermal state ----------------
C('2.5','Gaikwad2020','T0 = 11000+-1600, 10500+-2100, 12000+-2200','11000','1600','10500','2100','12000','2200')
C('2.5','Gaikwad2020','at z = 5.4, 5.6, 5.8; resolution <= 8 km/s','5.4','5.6','5.8','8')
C('2.5','Villasenor2021','400 sims; 14 bins 2.2<=z<=5.0; 5 bins 2.4<z<2.9','400','14','2.2','5.0','2.4','2.9')
C('2.5','Villasenor2021','HI by z~6.0 T0~1.3e4 K; HeII by z~3.0 T0~1.4e4 K','6.0','1.3','3.0','1.4')
C('2.5','Eide2020','xHII = 0.99998 at z=6; <T> ~ 20000 K','0.99998','20,000')

# ---------------- Section 2.6: 21-cm ----------------
C('2.6','HERA2022','Delta^2 <= 457 mK^2 at k=0.34, z=7.9','457','0.34','7.9')
C('2.6','HERA2022','Delta^2 <= 3496 mK^2 at k=0.36, z=10.4','3,496','0.36','10.4')
C('2.6','HERA2022','TS > 11.0 (35.2) K at z=7.9','11.0','35.2')
C('2.6','HERA2022','TS > 6.2 (26.4) K at z=10.4','6.2','26.4')
C('2.6','HERA2022','T_CMB = 24.3 K at z=7.9','24.3')
C('2.6','HERA2022','TK ranges','3.2','313.2','13.0','4768')
C('2.6','HERA2022','LX/SFR 68% HPD 10^39.9-10^41.6','39.9','41.6')
C('2.6','HERA2022','LX/SFR 95% 10^40.4-10^41.8; prior <10^42','40.4','41.8','42')
C('2.6','HERA2022','94 nights','94')
C('2.6','Ghara2025','ionized+heated < 0.46 (0.05); T < 44 K (4 K)','0.46','0.05','44')
C('2.6','Ghara2025','heated region < 14 (3) h^-1 Mpc; z=9.1','14','9.1')
C('2.6','Ghara2025','10 nights LOFAR; z 8.3, 9.1, 10.1','10','8.3','10.1')

# ---------------- Section 2.7: 21-cm forest ----------------
C('2.7','Soltinsky2026','J352-15 at z=5.82','5.82')
C('2.7','Soltinsky2026','sensitivity 3.62 mJy/beam per 6.1 kHz','3.62','6.1')
C('2.7','Soltinsky2026','T_HI <~ 27 K for xHI = 0.1 at z~5.6','27','0.1','5.6')

# ---------------- Section 2.8: kSZ ----------------
C('2.8','Reichardt2020','D_kSZ = 3.0 +- 1.0 muK^2','3.0','1.0')
C('2.8','Reichardt2020','dz_re = 1.0 +1.6 -0.7; < 4.1 at 95%','1.6','0.7','4.1')
C('2.8','Chaubal2026','D_kSZ = 1.98 +- 0.87 muK^2 (v2)','1.98','0.87')
C('2.8','Chaubal2026','D_tSZ = 4.98 +- 0.35 muK^2 (v2)','4.98','0.35')
C('2.8','Chaubal2026','dz50 < 4.2; dz90 < 7.0 (v2)','4.2','7.0')

# ---------------- Section 2.9: topology ----------------
C('2.9','Umeda2023','log Rb 1.67 +0.14 -0.16 to -0.69 +0.89 -0.24','1.67','0.14','0.16','0.69','0.89','0.24')

# ---------------- Section 2.10: helium ----------------
C('2.10','Kulkarni2018','AGN reionize HeII by z = 2.9','2.9')

# ---------------- Section 3.1: criteria ----------------
C('3.1','Munoz2024','canonical log xi_ion = 25.2; JWST 25.5-26.0','25.2','25.5','26.0')
C('3.1','Atek2023','log xi_ion = 25.80 +- 0.14','25.80','0.14')
C('3.1','Robertson2015','C_HII = 3 fiducial','CHII=3')
C('3.1','DAloisio2020','clumping peaks 5-20 at dt=10 Myr; relaxes to ~3 by 300 Myr','5-20','10','300')
C('3.1','DAloisio2020','Jeans mass 10^4 Msun; recombinations +50%','104','50')
C('3.1','Davies2024','global C ~ 12; ~3 photons per baryon','12','3')
C('3.1','Austin2025','1721 galaxies, 5.6<z<6.5, ~550 arcmin^2','1721','5.6','6.5','550')
C('3.1','Austin2025','xi_ion relation 25.05 +0.39 -0.34','25.05','0.39','0.34')
C('3.1','Austin2025','ndot = 50.31 +0.07 -0.06','50.31','0.07','0.06')
C('3.1','Austin2025','C_HII,rec = 6.2 +4.1 -2.1; M_UV,lim = -13.5','6.2','4.1','2.1','13.5')
C('3.1','Furlanetto2009','E < 10.2 eV all energy to heat; 1-10 keV equal split','10.2')
C('3.1','Pacucci2014','factor of three large-scale 21-cm power; k=0.2','three','0.2')
C('3.1','Dasgupta2026','stochastic LX affects k > 0.3 cMpc^-1','0.3')

# ---------------- Section 3.2: agents ----------------
C('3.2','Robertson2015','matched tau = 0.066 +- 0.012','0.066','0.012')
C('3.2','Finkelstein2019','fesc < 5%; log Mh ~ 9; M_UV > -15 dominate','5','9','15')
C('3.2','Finkelstein2019','Q_HII(z=7) = 78% +- 8%','78','8')
C('3.2','Naidu2019','fesc = 0.21 +0.06 -0.04 for M_UV < -13.5','0.21','0.06','0.04','13.5')
C('3.2','Naidu2019','fesc propto Sigma^0.4 +- 0.1','0.4','0.1')
C('3.2','Naidu2019','<5% of M_UV<-18 and log(M*)>8 give >80%','MUV<-18andlog(M?/M)>8','80')
C('3.2','Atek2023','8 galaxies, M_UV -17 to -15, 0.005 L*','17','15','0.005')
C('3.2','Giovinazzo2026','~20% of sources with fesc>10% produce ~87%','20','10','87')
C('3.2','Rosdahl2018','fesc 7-10%, about 3x higher; 99.9% ionized by z~7','7-10','about3times','99.9')
C('3.2','Rosdahl2018','L-weighted 8.5% binaries vs 2.7% singles, 3.1x','8.5','2.7','3.1times')
C('3.2','Kulkarni2013','Zcrit = 10^-4 Zsun; ~10^6 yr; >50% at z>30; <1% at z=10','10-4','106','50','30')
C('3.2','Kulkarni2013','<~10% at z=10','10')
C('3.2','Murphy2021','1.7-500 Msun; rotation 25%; overshoot 20%; IMF 20-30%','1.7-500','25','20','20-30')
C('3.2','Madau2015','tau = 0.056; H reionized by z = 5.7; He 13% of tau','0.056','5.7','13')
C('3.2','Kulkarni2018','AGN < 3% at z~6; ~10% if to M1450 = -18; 80,000 AGN','3','10','18','80,000')
C('3.2','Dayal2020','AGN 10-25% cumulative by z=4; M* < 10^9','10-25','109')
C('3.2','Asthana2024','17% emissivity; 1% haloes; 10% haloes; alpha = -1.7; 10 Myr','17','1','10','1.7')
C('3.2','Singha2025','Ndot = 3.77 +1.08 -0.95 e51; 31-75%; Gamma 0.5-2 e-12','3.77','1.08','0.95','31-75','0.5-2')
C('3.2','Singha2025','fesc,gal 0.03-0.20; eta = 0.10 +- 0.02','0.03-0.20','0.10','0.02')
C('3.2','Madau2016','electron fraction never exceeds 1%; peak 1-2 keV; z<10','1','10')
C('3.2','Eide2020','BHs increase local temperature by ~10^4 K','104')
C('3.2','Sazonov2015','E < 30 MeV; heats 10-100 K by z ~ 15','30','10-100','15')
C('3.2','Tueros2014','primordial magnetic field 10^-17 G','10-17')
C('3.2','Reis2021','O(100) K; floor -165 mK at z=15-19','100','165','15-19')
C('3.2','Reis2021','-178 mK at z=19; -216 at z=15; -264 at z=10','178','216','264')
C('3.2','Reis2021','power attenuated 6.6x at z=9, k=0.1; amplified 2-5','6.6','0.1','2-5')
C('3.2','Furlanetto2003','strip ~10% of gas from minihalos','10')
C('3.2','Xu2021','absorption reduced ~15% at z = 17','15','17')
C('3.2','Fialkov2019','SFE >= 2.8%; Mh ~ 10^9 at z=17; 1.9x CMB at 78 MHz; 0.1%','2.8','109','1.9','78','0.1')
C('3.2','Reis2020','power spectrum enhanced up to two orders of magnitude at z~17','17')
C('3.2','Sharma2018','cooling time 3 orders shorter; 10^3 times smaller; 78 MHz','78','103')
C('3.2','Kannan2021','THESAN L = 95.5 cMpc; DM 3.1e6, gas 5.8e5 Msun','95.5','3.1','5.8')

if __name__ == '__main__':
    fails = []
    for sec, key, desc, toks in CLAIMS:
        miss = check(key, toks)
        if miss:
            fails.append((sec, key, desc, miss))
    print(f"claims checked : {len(CLAIMS)}")
    print(f"fully verified : {len(CLAIMS)-len(fails)}")
    print(f"needs review   : {len(fails)}\n")
    for sec, key, desc, miss in fails:
        print(f"[{sec:4s}] {key:20s} {desc}")
        print(f"         MISSING TOKENS: {miss}")
