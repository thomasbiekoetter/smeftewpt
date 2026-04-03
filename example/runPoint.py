from smeftewpt.potentials.fourdim import FourDim


pot = FourDim(-4.0, 0.0)
print(pot.Veff(2 * pot.vev))
