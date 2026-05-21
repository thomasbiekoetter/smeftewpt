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
        self.CH = CH
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

        self.musq_MZ =  Mh ** 2 / 2. + (3 * self.CH * self.vev ** 4) / (4. * self.LambdaUV ** 2)

        self.lam_MZ = Mh ** 2 / (2. * self.vev ** 2) + (3 * self.CH * self.vev ** 2) /  \
            (2. * self.LambdaUV ** 2)


    def make_running_4D(self, scale_max=1e3):
        self.rge = SMRGE(
            self.input_scale, g_Y=self.g1_MZ, g2=self.g2_MZ, g3=self.g3_MZ,
            yt=self.yt_MZ, yb=self.yb_MZ, ytau=self.yl_MZ,
            lam=self.lam_MZ, m2=self.musq_MZ)


    def make_hard_matching(self, T):
        # First get 4D parameters at scale mu = mu_match_fac * pi * T
        mu = self.mu_match_fac * np.pi * T
        yt = self.rge.get_yt(mu)
        lam = self.rge.get_lam(mu)
        musq = self.rge.get_m2(mu)
        # SMEFT coefficients (no running yet)
        CH = self.CH
        # To be checked
        msq3D = -musq # + ...
#       cth = self.lam / 4.0 + 3.0 * self.g2**2  / 16.0  \
#           + self.g1**2 / 48.0 + self.t**2 / 12.0
#       self.musq3D = -self.musq + cth * T ** 2
#       self.lam3D = self.lam * T
#       self.CH3D = self.CH * T


    def Veff(self, h, T):
        self.make_hard_matching(T)
#       y = self.V0(h) + self.V1(h)
#       return y


#   def Vtree(self, h):
#       # Prefactors to be checked
#       y = 0.5 * self.musq3D * h ** 2 +  \
#           0.25 * self.lam3D * h ** 4 -  \
#           (1.0 / 6.0) * 0.75 * self.CH3D * h ** 6
#       return y


#   def V1(self, h):
#       y = 0.0
#       return y
