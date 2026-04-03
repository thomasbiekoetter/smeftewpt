import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from smeftewpt.potentials.fourdim import FourDim


mpl.use('pgf')
pgf_dc = {
    'text.usetex': True,
    'pgf.rcfonts': False
}
mpl.rcParams.update(pgf_dc)


pot = FourDim(-3.0, 0.0)


def V(x, T=0.0):
    x = np.asarray(x)
    return np.vectorize(lambda xi: pot.Veff(xi, T))(x)


fig, ax = plt.subplots(
    figsize=(4.0, 3.0),
    constrained_layout=True)

xm = 320.0
y = np.zeros(shape=(10, 1000))
x = np.linspace(-xm, xm, 1000)
for i in range(0, 5):
    T = 30 + i * 30
    y[i, :] = V(x, T=T) / pot.vev ** 4
    ax.plot(x / pot.vev, y[i, :], color="C0")
    ax.text(
        x[200] / pot.vev,
        y[i, 200],
        r"$T = " + str(T) + r"~\mathrm{GeV}$",
        color="C0")
ax.axvline(-1, color="black", lw=1, ls=":")
ax.axvline(1, color="black", lw=1, ls=":")
ax.set_ylabel(r"$V_{\mathrm{eff}} / v^4$")
ax.set_ylim(-0.014, 0.04)
majorLocator = MultipleLocator(0.01)
minorLocator = MultipleLocator(0.001)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_minor_locator(minorLocator)
ax.tick_params(
    axis='y',
    direction='in',
    which='both',
    right=True)
ax.set_xlim(-xm / pot.vev, xm / pot.vev)
ax.set_xlabel(r"$h / v \quad (v = 246~\mathrm{GeV})$")
majorLocator = MultipleLocator(0.5)
minorLocator = MultipleLocator(0.05)
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_minor_locator(minorLocator)
ax.tick_params(
    axis='x',
    direction='in',
    which='both',
    top=True)
ax.text(
    -0.5, 0.036,
    r"$C_H = -4 /\mathrm{TeV}$, $C_{tH} = 0$")


plt.savefig("plot.pdf")
