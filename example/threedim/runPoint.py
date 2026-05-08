import numpy as np
import matplotlib.pyplot as plt
from smeftewpt.potentials.threedim import ThreeDim


pot = ThreeDim(-3.0)

def V(x, T=60.0):
    x = np.asarray(x)
    return np.vectorize(lambda xi: pot.Veff(xi, T))(x)
xm = 320.0
x = np.linspace(-xm, xm, 1000)
