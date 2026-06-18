import numpy as np
from smeftewpt.input import Gfermi
from smeftewpt.input import Mh
from smeftewpt.input import Mz
from smeftewpt.input import AlphaEWatMZ
from smeftewpt.input import AlphaSatMZ
from smeftewpt.input import Mt_MZ
from smeftewpt.input import Mb_MZ
from smeftewpt.input import Ml
from smeftewpt.input import LambdaUV
from smeftewpt.running.smrge import SMRGE


class ThreeDim:

    def __init__(self, CH, LambdaUV, mu_match_fac=1.0):

        self.LambdaUV = LambdaUV
        self.mu_match_fac = mu_match_fac
        self.CH = CH / LambdaUV ** 2
        self.Ckin = 0.0
        self.CtH = 0.0
        self.calc_internal_paras_atMZ()
        self.make_running_4D()


    def calc_internal_paras_atMZ(self):

        self.vev = 1.0 / (2.0 ** 0.25 * np.sqrt(Gfermi))

        self.yt_MZ = np.sqrt(2.0) * Mt_MZ / self.vev +  \
            self.CtH * self.vev ** 2 / 2.0

        self.yb_MZ = np.sqrt(2.0) * Mb_MZ / self.vev

        self.yl_MZ = np.sqrt(2.0) * Ml / self.vev

        self.elec = np.sqrt(4.0 * np.pi * AlphaEWatMZ)

        self.sinweak = np.sqrt((1.0 - np.sqrt(1.0 -  \
            (4.0 * np.pi * AlphaEWatMZ) /  \
            (np.sqrt(2.0) * Gfermi * Mz ** 2))) / 2.0)

        self.cosweak = np.sqrt(1.0 - self.sinweak ** 2)

        self.mw = Mz * self.cosweak

        self.g1_MZ = self.elec / self.cosweak

        self.g2_MZ = self.elec / self.sinweak

        self.g3_MZ  = np.sqrt(4 * np.pi * AlphaSatMZ)

        self.input_scale = Mz

#       self.mz = Mz

#       self.mh = Mh

#       self.mt = Mt

        # We treat here the SMEFT Wilsons at mu = MZ (neglect their running). Consistent?

        self.musq_MZ = Mh ** 2 / 2. + (3 * self.CH * self.vev ** 4) / 4.0

        self.lam_MZ = Mh ** 2 / (2. * self.vev ** 2) + (3 * self.CH * self.vev ** 2) / 2.0


    def make_running_4D(self, scale_max=1e3):
        self.rge = SMRGE(
            self.input_scale, g_Y=self.g1_MZ, g2=self.g2_MZ, g3=self.g3_MZ,
            yt=self.yt_MZ, yb=self.yb_MZ, ytau=self.yl_MZ,
            lam=self.lam_MZ, m2=self.musq_MZ)


    def make_hard_matching(self, T, mode="O(g^4)"):
        # First get 4D parameters at scale mu = mu_match_fac * pi * T
        mu = self.mu_match_fac * np.pi * T
        yt = self.rge.get_yt(mu)
        lam = self.rge.get_lam(mu)
        musq = self.rge.get_m2(mu)
        g1 = self.rge.get_gY(mu)
        g2 = self.rge.get_g2(mu)
        # SMEFT coefficients (no running yet)
        CH = self.CH
        # To be checked
        if mode == "O(g^4)":
            Tsq = T ** 2
            # CH term not contained in Eq. (52) of 2503.20016
            self.msq3D = -musq + Tsq * (g1 ** 2 + 3 * g2 ** 2 +
                8 * lam - 4 * CH * Tsq + 4 * yt ** 2) / 16.0
            # Compared to Eq. (52) of 2503.20016 I miss terms ~ g1, g2
            self.lam3D = lam * T - CH * T ** 3
            self.C3D = -CH * Tsq
        elif mode == "LO_Chala":
            # Eq. 52 of 2503.20016
            Tsq = T ** 2
            self.msq3D = -musq + Tsq * (g1 ** 2 + 3 * g2 ** 2 +
                8 * lam + 4 * yt ** 2) / 16.0
            self.lam3D = lam * T - CH * T ** 3 + T * (
                g1 ** 4 + 2 * g1 ** 2 * g2 ** 2 + 3 * g2 ** 4) / (128.0 * np.pi ** 2)
            self.C3D = -CH * Tsq


    def Veff(self, h, T, mode="O(g^4)"):
        self.make_hard_matching(T, mode=mode)
        h = h / np.sqrt(T)
        y = self.V0(h, T)
        y += self.V1(h)
        y += self.V2(h)
        return y * T


    def V0(self, h, T):
        # Prefactors such that consistent with Eq. (52) of 2503.20016
        # The prefactor for the h^6 term is inconsistent with the
        # definition of the phi^6 term in Eq. 7 of 2503.20016,
        # but if I don't switch its sign, then the potential is not BfB
        # for negative CH. But CH has to be negative because otherwise
        # the 4D potential is not BfB, see definition of the phi^6
        # term in Eq. 2 of 2503.20016
        y = self.msq3D * h ** 2 +  \
            self.lam3D * h ** 4 +  \
            self.C3D * h ** 6
        y = y
        return y


    def V1(self, h):
        # To be implemented if necessary
        y = 0.0
        return y


    def V2(self, h):
        # To be implemented if necessary
        y = 0.0
        return y
