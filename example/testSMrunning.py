import numpy as np
import matplotlib.pyplot as plt
from smeftewpt.running.smrge import SMRGE


# SM input values at MZ = 91.187 GeV (PDG 2022, in MS-bar scheme)
MZ = 91.187  # GeV
alpha_em = 1 / 127.95
sin2_thetaW = 0.2313
alpha_s = 0.1179

e    = np.sqrt(4 * np.pi * alpha_em)
gY0  = e / np.sqrt(1 - sin2_thetaW)    # g_Y = e / cos(θ_W)
g20  = e / np.sqrt(sin2_thetaW)        # g2  = e / sin(θ_W)
g30  = np.sqrt(4 * np.pi * alpha_s)    # g3

# Yukawa couplings: yt = mt / v_field, where v_field = v_EW/sqrt(2) = 174.10 GeV
v = 246.22                                   # VEV 
yt0  = 162.5  / v / np.sqrt(2)               # top MS-bar mass ~162.5 GeV at MZ
yb0  = 2.9    / v / np.sqrt(2)               # bottom MS-bar mass ~2.9 GeV evolved to MZ
ytau0 = 1.777 / v / np.sqrt(2)               # tau pole mass ≈ MS-bar at this scale

# Higgs quartic
mh = 125.25   # GeV
lam0 = mh**2 / (2 * v**2)

print(f"Input at MZ = {MZ} GeV:")
print(f"  g_Y = {gY0:.4f},  g2 = {g20:.4f},  g3 = {g30:.4f}")
print(f"  yt  = {yt0:.4f},  yb = {yb0:.4f},  ytau = {ytau0:.4f}")
print(f"  lambda = {lam0:.4f}")

rge = SMRGE(MZ, g_Y=gY0, g2=g20, g3=g30, yt=yt0, yb=yb0, ytau=ytau0, lam=lam0)

# Running from MZ up to high energies 
mu_values = np.logspace(np.log10(1.0001 * MZ), 17, 500)

# Solve once to the highest scale (cached after this)
gY_run   = [rge.get_gY(mu)   for mu in mu_values]
g2_run   = [rge.get_g2(mu)   for mu in mu_values]
g3_run   = [rge.get_g3(mu)   for mu in mu_values]
yt_run   = [rge.get_yt(mu)   for mu in mu_values]
yb_run   = [rge.get_yb(mu)   for mu in mu_values]
ytau_run = [rge.get_ytau(mu) for mu in mu_values]
lam_run  = [rge.get_lam(mu)  for mu in mu_values]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# gague couplings
ax = axes[0]
ax.plot(np.log10(mu_values), gY_run,   label=r'$g_Y$')
ax.plot(np.log10(mu_values), g2_run,   label=r'$g_2$')
ax.plot(np.log10(mu_values), g3_run,   label=r'$g_3$')
ax.set_xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$')
ax.set_ylabel('Coupling')
ax.set_title('Gauge couplings')
ax.legend()
ax.grid(True, alpha=0.3)

# Yukawas
ax = axes[1]
ax.plot(np.log10(mu_values), yt_run,   label=r'$y_t$')
ax.plot(np.log10(mu_values), yb_run,   label=r'$y_b$')
ax.plot(np.log10(mu_values), ytau_run, label=r'$y_\tau$')
ax.set_xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$')
ax.set_ylabel('Coupling')
ax.set_title('Yukawa couplings')
ax.legend()
ax.grid(True, alpha=0.3)

# Higgs quartic
ax = axes[2]
ax.plot(np.log10(mu_values), lam_run, color='C3', label=r'$\lambda$')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$')
ax.set_ylabel('Coupling')
ax.set_title('Higgs quartic')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
