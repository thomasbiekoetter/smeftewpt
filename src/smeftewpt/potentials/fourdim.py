import numpy as np
from smeftewpt.input import Gfermi
from smeftewpt.input import Mh
from smeftewpt.input import Mz
from smeftewpt.input import AlphaEWatMZ
from smeftewpt.input import Mt
from smeftewpt.input import LambdaUV
from smeftewpt.util import is_equal
from smeftewpt.util import logsave
from scipy.optimize import minimize
from ableiter.first import First
from ableiter.second import Second
from pythermalfunctions.jspline import Jb_spline as Jb
from pythermalfunctions.jspline import Jf_spline as Jf


class FourDim:

    def __init__(self, CH, CtH):

        self.CH = CH / LambdaUV ** 2
        self.CtH = CtH / LambdaUV ** 2
        self.Ckin = 0.0
        self.calc_internal_paras()
        self.check_tree_min()
        self.msq0CW = self.calc_msq(self.vev)
        self.nCW = np.array([1.0, 3.0, 6.0, 3.0, -12.0])
        self.check_zerotemp_masses()
        self.check_onshellnes()


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
            quit()


    def VCW(self, h):

        msq = self.calc_msq(h)

        y = 0.0
        # Neglect GB contributions for now
        # Insignificant according to arXiv: 1409.0005
        for i in [0, 2, 3, 4]:
            y += self.nCW[i] * (msq[i] ** 2 * (  \
                logsave(msq[i], self.msq0CW[i]) -  \
                1.5) + 2.0 * msq[i] * self.msq0CW[i]) / (64.0 * np.pi ** 2)

        return y


    def calc_msq(self, h, T=None):

        mhsq = -5.0 * h ** 4 * ((3.0 * self.CH) / 4.0 +  \
            2.0 * self.Ckin * self.lam) + self.musq +  \
            3.0 * h ** 2 * (self.lam - (4.0 * self.Ckin * self.musq) / 3.0)

        mgsq = -1.0 * h ** 4 * ((3.0 * self.CH) / 4.0 +  \
            2.0 * self.Ckin * self.lam) + self.musq +  \
            1.0 * h ** 2 * (self.lam - (4.0 * self.Ckin * self.musq) / 3.0)

        mwsq = 0.25 * self.g2 ** 2 * h ** 2

        mzsq = 0.25 * (self.g1 ** 2 + self.g2 ** 2) * h ** 2

        mtsq = (h / np.sqrt(2.0) * (self.yt - h ** 2 * self.CtH / 2.0)) ** 2

        return np.array([mhsq, mgsq, mwsq, mzsq, mtsq])


    def VT(self, h, T):

        if is_equal(T, 0.0):
            return 0.0

        msq = self.calc_msq(h)
        Tsq = T ** 2

        y = 0.0
        # Neglect GB contributions for now
        # Insignificant according to arXiv: 1409.0005
        for i in [0, 2, 3, 4]:
            if i != 4:
                J = Jb(msq[i] / Tsq)
            else:
                J = -Jf(msq[i] / Tsq) # Different sign compared to 2012.03953
                                      # because different definition of Jf
            y += self.nCW[i] * J
        y *= Tsq ** 2 / (2.0 * np.pi ** 2)

        return y


    def Veff(self, h, T=0.0, remove_cc=True):
        if is_equal(T, 0.0):
            y = self.Vtree(h) + self.VCW(h)
            if remove_cc:
                y -= self.Vtree(0) + self.VCW(0)
        else:
            y = self.Vtree(h) + self.VCW(h) + self.VT(h, T)
            if remove_cc:
                y -= self.Vtree(0) + self.VCW(0) + self.VT(0, T)
        return y


    def check_zerotemp_masses(self):

        ms = np.sqrt(np.abs(self.msq0CW))

        if not is_equal(ms[0], self.mh):
            print("Mass of mH in EW minimum wrong.")
            print(ms[0], self.mh)
            quit()

        if not is_equal(ms[1], 0.0):
            print("Mass of mG in EW minimum wrong.")
            print(ms[1], 0.0)
            quit()

        if not is_equal(ms[2], self.mw):
            print("Mass of mW in EW minimum wrong.")
            print(ms[2], 0.0)
            quit()

        if not is_equal(ms[3], self.mz):
            print("Mass of mZ in EW minimum wrong.")
            print(ms[3], 0.0)
            quit()

        if not is_equal(ms[4], self.mt):
            print("Mass of mt in EW minimum wrong.")
            print(ms[4], 0.0)
            quit()


    def check_onshellnes(self):

        # Add 2nd dimension to x because ableiter only works with arrays
        x = np.array([self.vev, 0.0])

        # Check tadpole condition
        def f(x): return self.VCW(x[0])
        ab1 = First(f, 2)
        def df(x): return ab1.df_dxi(x, 0)
        if not is_equal(df(x), 0.0):
            print("dVCW / dh does not vanish in EW minimum.")
            print(df(x - 1.0), df(x), df(x + 1.0))
            quit()

        # Check curvature, i.e. mH
        def f(x): return self.Veff(x[0], T=0.0)
        ab2 = Second(f, 2)
        def d2f(x): return ab2.d2f_dxidxj(x, 0, 0)
        if not is_equal(np.sqrt(d2f(x)), self.mh):
            print("d2Veff / dh2 at T = 0 is not equal to mH^2.")
            print(np.sqrt(d2f(x)), self.mh)
            quit()
