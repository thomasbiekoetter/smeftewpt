import numpy as np
from scipy.integrate import solve_ivp


class SMRGE:
    """One-loop SM RGEs for all marginal couplings and the Higgs mass parameter.

    Convention: 16π² dc/d(ln μ) = β(c), MS-bar scheme.
    g_Y is the SM hypercharge coupling (not GUT-normalized).
    Yukawa: third-generation only (yt, yb, ytau).
    Higgs quartic: defined via V ⊃ λ(H†H)².
    Higgs mass: defined via V ⊃ m²(H†H); EWSB requires m² < 0.

    Parameters
    ----------
    mu0 : float
        Reference RG scale in GeV at which inputs are given.
    g_Y, g2, g3, yt, yb, ytau, lam, m2 : float or None
        Coupling values at mu0.  None → treated as 0 in the ODE.
    """

    COUPLINGS = ['g_Y', 'g2', 'g3', 'yt', 'yb', 'ytau', 'lam', 'm2']

    def __init__(self, mu0, *, g_Y=None, g2=None, g3=None,
                 yt=None, yb=None, ytau=None, lam=None, m2=None):
        self.mu0 = mu0
        vals = [g_Y, g2, g3, yt, yb, ytau, lam, m2]
        self._y0 = np.array([v if v is not None else 0.0 for v in vals])
        self._sol = None       # cached OdeSolution (dense_output)
        self._sol_t_end = None

    @staticmethod
    def _beta(t, y):
        """RHS of the ODE system: dy/dt = β(y) / (16π²).

        Yukawa beta functions follow from the Machacek-Vaughn (1984) formula:
          16π² β(Y_u)/Y_u = 3Tr(Y_u†Y_u) + 3Tr(Y_d†Y_d) + Tr(Y_e†Y_e)
                            + 3/2 Y_u†Y_u - 3/2 Y_d†Y_d - gauge

        In the single-generation diagonal approximation this gives a ytau²
        term (coefficient +1, Nc=1 for leptons) in β(yt), β(yb) and β(ytau),
        and 3yt² + 3yb² (Nc=3 for quarks) in β(ytau).
        """
        g_Y, g2, g3, yt, yb, ytau, lam, m2 = y
        loop = 1.0 / (16.0 * np.pi**2)

        g_Y2 = g_Y**2;  g22 = g2**2;  g32 = g3**2
        yt2  = yt**2;   yb2 = yb**2;  ytau2 = ytau**2

        # Gauge couplings  [Arason et al., PRD 46, 3945 (1992)]
        b_gY  = ( 41/6) * g_Y**3
        b_g2  = (-19/6) * g2**3
        b_g3  =  -7     * g3**3

        # Yukawa couplings  [Machacek & Vaughn, NPB 236 (1984); Arason et al.]
        # ytau² enters β(yt) and β(yb) via Tr(Y_e†Y_e) in γ(H); coefficient
        # is 1 (not 3) because leptons carry no colour.
        b_yt   = yt   * ( 9/2*yt2  + 3/2*yb2 + ytau2  - 8*g32 - 9/4*g22 - 17/12*g_Y2)
        b_yb   = yb   * ( 9/2*yb2  + 3/2*yt2 + ytau2  - 8*g32 - 9/4*g22 -  5/12*g_Y2)
        b_ytau = ytau * ( 5/2*ytau2 + 3*yt2  + 3*yb2  -          9/4*g22 - 15/4 *g_Y2)

        # Higgs quartic  [Espinosa & Quirós, NPB 384 (1992)]
        b_lam = (24*lam**2
                 - 3*lam*(3*g22 + g_Y2)
                 + 3/8*(2*g22**2 + (g22 + g_Y2)**2)
                 + 12*lam*yt2   - 6*yt**4
                 + 12*lam*yb2   - 6*yb**4
                 +  4*lam*ytau2 - 2*ytau**4)

        # Higgs mass parameter  [Espinosa & Quirós; Ford, Jack & Jones, NPB 387 (1992)]
        b_m2 = m2 * (12*lam + 6*yt2 + 6*yb2 + 2*ytau2 - 9/2*g22 - 3/2*g_Y2)

        return loop * np.array([b_gY, b_g2, b_g3, b_yt, b_yb, b_ytau, b_lam, b_m2])

    def _ensure_solved(self, mu):
        t_end = np.log(mu)
        if self._sol is None or t_end > self._sol_t_end:
            sol = solve_ivp(
                self._beta,
                t_span=[np.log(self.mu0), t_end],
                y0=self._y0,
                method='RK45',
                dense_output=True,
                rtol=1e-9,
                atol=1e-11,
            )
            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")
            self._sol = sol.sol
            self._sol_t_end = t_end

    def _get(self, mu, idx):
        if mu < self.mu0:
            raise ValueError(
                f"Scale mu = {mu:.4g} GeV must be >= mu0 = {self.mu0:.4g} GeV"
            )
        self._ensure_solved(mu)
        if self._sol is None or not callable(self._sol):
            raise RuntimeError("ODE solution not available")
        else:
            return float(self._sol(np.log(mu))[idx])

    def get_gY(self, mu):    """Return g_Y at scale mu (GeV)."""; return self._get(mu, 0)
    def get_g2(self, mu):    """Return g2 at scale mu (GeV)."""; return self._get(mu, 1)
    def get_g3(self, mu):    """Return g3 at scale mu (GeV)."""; return self._get(mu, 2)
    def get_yt(self, mu):    """Return yt at scale mu (GeV)."""; return self._get(mu, 3)
    def get_yb(self, mu):    """Return yb at scale mu (GeV)."""; return self._get(mu, 4)
    def get_ytau(self, mu):  """Return ytau at scale mu (GeV)."""; return self._get(mu, 5)
    def get_lam(self, mu):   """Return lambda at scale mu (GeV)."""; return self._get(mu, 6)
    def get_m2(self, mu):    """Return m² at scale mu (GeV); negative for EWSB."""; return self._get(mu, 7)

