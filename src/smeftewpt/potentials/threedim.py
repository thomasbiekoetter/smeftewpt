import numpy as np
from smeftewpt.input import Gfermi
from smeftewpt.input import Mh
from smeftewpt.input import Mz
from smeftewpt.input import AlphaEWatMZ
from smeftewpt.input import Mt
from smeftewpt.input import LambdaUV


class ThreeDim:

    def __init__(self, CH, LambdaUV):

        self.LambdaUV = LambdaUV
        self.CH = CH
        self.Ckin = 0.0
        self.CtH = 0.0
        self.calc_internal_paras()


    def calc_internal_paras(self):

        self.vev = 1.0 / (2.0 ** 0.25 * np.sqrt(Gfermi))

        self.yt = np.sqrt(2.0) * Mt / self.vev +  \
            self.CtH * self.vev ** 2 / 2.0

        self.elec = np.sqrt(4.0 * np.pi * AlphaEWatMZ)

        self.sinweak = np.sqrt((1.0 - np.sqrt(1.0 -  \
            (4.0 * np.pi * AlphaEWatMZ) /  \
            (np.sqrt(2.0) * Gfermi * Mz ** 2))) / 2.0)

        self.cosweak = np.sqrt(1.0 - self.sinweak ** 2)

        self.mw = Mz * self.cosweak

        self.g1 = self.elec / self.cosweak

        self.g2 = self.elec / self.sinweak

        self.mz = Mz

        self.mh = Mh

        self.mt = Mt

        self.musq =  Mh ** 2 / 2. + (3 * self.CH * self.vev ** 4) / (4. * self.LambdaUV ** 2)

        self.lam = Mh ** 2 / (2. * self.vev ** 2) + (3 * self.CH * v ** 2) /  \
            (2. * self.LambdaUV ** 2)


    def make_hard_matching(self, T):
        # To be checked
        cth = self.lam / 4.0 + 3.0 * self.g2**2  / 16.0  \
            + self.g1**2 / 48.0 + self.t**2 / 12.0
        self.musq3D = -self.musq + cth * T ** 2
        self.lam3D = self.lam * T
        self.CH3D = self.CH * T


    def Veff(self, h, T):
        self.make_hard_matching(T)
        y = self.V0(h) + self.V1(h)
        return y


    def Vtree(self, h):
        # Prefactors to be checked
        y = 0.5 * self.musq3D * h ** 2 +  \
            0.25 * self.lam3D * h ** 4 -  \
            (1.0 / 6.0) * 0.75 * self.CH3D * h ** 6
        return y


    def V1(self, h):
        y = 0.0
        return y
