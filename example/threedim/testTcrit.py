import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from smeftewpt.vacstruct.phases import get_Tcrit
from smeftewpt.potentials.threedim import ThreeDim


mpl.use('pgf')
pgf_dc = {
    'text.usetex': True,
    'pgf.rcfonts': False
}
mpl.rcParams.update(pgf_dc)


CH = np.linspace(-4.5, -1.0, 45)
Tmin = 30.0
dT = 0.2
Tcrit = np.zeros(shape=(len(CH), ))

for i, ch in enumerate(tqdm(CH)):
    pot = ThreeDim(ch, LambdaUV=1e3)
    T = get_Tcrit(pot, Tmin=Tmin, dT=dT, verbose=True)
    Tcrit[i] = T

fig, ax = plt.subplots(
    figsize=(4.0, 3.0),
    constrained_layout=True)

y = np.where(Tcrit < 0, np.nan, Tcrit)
y = np.where(y <= Tmin + dT, np.nan, y)
ax.plot(CH, y)

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

plt.savefig("Tcrit.pdf")
