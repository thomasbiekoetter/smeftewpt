import numpy as np
from scipy.optimize import minimize
from smeftewpt.util import is_equal


def get_phases(pot, Tmin=20.0, Tmax=150.0, dT=0.5):

    def V(x, T):
        return pot.Veff(x, T)

    def dV(x, T):
        return pot.dVeff(x, T)

    def d2V(x, T):
        return pot.d2Veff(x, T)

    xmin = 1.0
    T = Tmax
    x_high = []
    T_high = []
    V_high = []
    while (xmin < 1e1):
        if T < Tmin:
            break
        def f(x): return V(x[0], T)
        res = minimize(f, xmin)
        xmin = float(res.x[0])
        if abs(xmin) < 1e1:
            x_high.append(xmin)
            T_high.append(T)
            V_high.append(V(xmin, T))
        T -= dT

    xmin = pot.vev
    T = Tmin
    x_low = []
    T_low = []
    V_low = []
    while (xmin > 1e1):
        if T > Tmax:
            break
        def f(x): return V(x[0], T)
        res = minimize(f, xmin)
        xmin = float(res.x[0])
        if abs(xmin) > 1e1:
            x_low.append(xmin)
            T_low.append(T)
            V_low.append(V(xmin, T))
        T += dT

    return T_high, x_high, V_high, T_low, x_low, V_low


def get_Tcrit(pot, T_buffer=2.0, Tmin=20.0, Tmax=150.0, dT=0.5, verbose=True):

    T_high, x_high, V_high, T_low, x_low, V_low =  \
        get_phases(pot, Tmin=Tmin, Tmax=Tmax, dT=dT)

    def V(x, T):
        return pot.Veff(x, T)

    # Check T overlaop
    T_low_max = np.max(T_low)
    T_high_min = np.min(T_high)
    if abs(T_low_max - T_high_min) < T_buffer:
        if verbose:
            print("Phases barely overlap. Return mean of T-range.")
        return 0.5 * (T_low_max + T_high_min)

    T_low = np.array(T_low)
    V_low = np.array(V_low)
    T_low_lap = T_low[(T_low >= T_high_min - 1e-4) & (T_low <= T_low_max + 1e-4)]
    V_low_lap = V_low[(T_low >= T_high_min - 1e-4) & (T_low <= T_low_max + 1e-4)]

    T_high = np.array(T_high)
    V_high = np.array(V_high)
    T_high_lap = T_high[(T_high >= T_high_min - 1e-4) & (T_high <= T_low_max + 1e-4)]
    V_high_lap = V_high[(T_high >= T_high_min - 1e-4) & (T_high <= T_low_max + 1e-4)]
    T_high_lap = np.flip(T_high_lap)
    V_high_lap = np.flip(V_high_lap)

    deltaV = V_low_lap - V_high_lap
    for T, v in zip(T_low_lap, deltaV):
        if v > 0.0:
            return T

    if verbose:
        print("Failed to determine Tcrit.")
        print(T_high_min, T_low_max)

    return -1.0


# def get_Tnuc():
# 
#     ...
# 
#     Tstart = T_low_max - T_buffer
# 
#     for T, x, V in zip(T_low, x_low, V_low):
#         if is_equal(T, Tstart, eps=dT):
#             xstart_low = x
#             break
# 
#     for T, x, V in zip(T_high, x_high, V_high):
#         if is_equal(T, Tstart, eps=dT):
#             xstart_high = x
#             break
# 
#     print(xstart_high, xstart_low)
