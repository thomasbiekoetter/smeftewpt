import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from smeftewpt.vacstruct.phases import get_Tcrit
from smeftewpt.potentials.fourdim import FourDim


mpl.use('pgf')
pgf_dc = {
    'text.usetex': True,
    'pgf.rcfonts': False
}
mpl.rcParams.update(pgf_dc)


CtH = np.linspace(-3.0, 0.0, 4)
CH = np.linspace(-4.5, -1.0, 25)

Tcrit = np.zeros(shape=(len(CtH), len(CH)))

for i, ct in enumerate(tqdm(CtH, desc=" outer", position=0)):
    for j, ch in enumerate(tqdm(CH, desc=" inner loop", position=1, leave=False)):
        pot = FourDim(ch, ct)
        T = get_Tcrit(pot, dT=0.2, verbose=False)
        Tcrit[i, j] = T

fig, ax = plt.subplots(
    figsize=(4.0, 3.0),
    constrained_layout=True)

for i, c in enumerate(CtH):
    ax.plot(
        CH,
        Tcrit[i, :],
        label=r"$C_{tH} / \Lambda^2 = " + str(c) + r"~[\mathrm{TeV}^{-2}]$")

ax.set_ylabel(r"$T_c~\mathrm{[GeV]}$")
majorLocator = MultipleLocator(20)
minorLocator = MultipleLocator(2)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(
    axis='y',
    direction='in',
    which='both',
    right=True)

ax.tick_params(
    axis='y',
    direction='in',
    which='both',
    right=True)
ax.set_xlim(CH[0], CH[-1])
ax.set_xlabel(r"$C_H / \Lambda^2~[\mathrm{TeV}^{-2}]$")
majorLocator = MultipleLocator(0.5)
minorLocator = MultipleLocator(0.05)
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(
    axis='x',
    direction='in',
    which='both',
    top=True)

ax.legend(
    frameon=False)

plt.savefig("Tcrit.pdf")
