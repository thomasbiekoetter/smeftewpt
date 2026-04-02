import numpy as np
from smeftewpt.input import Gfermi
from smeftewpt.input import Mh
from smeftewpt.input import Mz
from smeftewpt.input import AlphaEWatMZ
from smeftewpt.input import Mt
from smeftewpt.input import LambdaUV
from smeftewpt.util import is_equal
from scipy.optimize import minimize


class FourDim:

    def __init__(self, CH, CtH):

        self.CH = CH / LambdaUV ** 2
        self.CtH = CtH / LambdaUV ** 2
        self.Ckin = 0.0
        self.calc_internal_paras()
        self.check_tree_min()


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

        self.musq = (-9.0 * self.CH * self.vev ** 4 +  \
            6.0 * self.mh ** 2 * (-1.0 + 2.0 * self.Ckin * self.vev ** 2)) /  \
            (4.0 * (3.0 - 12.0 * self.Ckin * self.vev ** 2 +  \
            8.0 * self.Ckin ** 2 * self.vev ** 4))

        self.lam = (3.0 * self.mh ** 2 -  \
            4.0 * self.Ckin * self.mh ** 2 * self.vev ** 2 +  \
            9.0 * self.CH * self.vev ** 4 -  \
            6.0 * self.CH * self.Ckin * self.vev ** 6) /  \
            (6.0 * self.vev ** 2 - 24.0 * self.Ckin * self.vev ** 4 +  \
            16.0 * self.Ckin ** 2 * self.vev ** 6)


    def Vtree(self, h):

        y = 0.5 * self.musq * h ** 2 +  \
            0.25 * (self.lam - 0.75 * self.Ckin * self.musq) * h ** 4 -  \
            (1.0 / 6.0) *(0.75 * self.CH + 2.0 * self.Ckin * self.lam) * h ** 6

        return y


    def check_tree_min(self):
        def f(x): return self.Vtree(x)
        res = minimize(f, 1e3)
        if not is_equal(res.x, self.vev):
            print("Problem with Vtree minimum.")
            print(res.x, self.vev)
            quit
