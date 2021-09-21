'''
Implementation of fourth order Runge-Kutta

The scritp was in 2017 translated by Sebastian G. Winther-Larsen from a matlab script originally written by Arnt Inge Vistnes
'''

# Importing ODEs to solve
from .functions import *

def rk4r(xn, vn, tn, delta_t, params):

    ffa = lambda y, v, t, param: params["func"](y, v, t, param)

    half_delta_t = 0.5*delta_t
    t_p_half = tn + half_delta_t

    x1 = xn
    v1 = vn
    a1 = ffa(x1, v1, tn, params)

    x2 = x1 + v1*half_delta_t
    v2 = v1 + a1*half_delta_t
    a2 = ffa(x2, v2, t_p_half, params)

    x3 = x1 + v2*half_delta_t
    v3 = v1 + a2*half_delta_t
    a3 = ffa(x3, v3, t_p_half, params)

    tp = tn + delta_t
    x4 = x1 + v3*delta_t
    v4 = v1 + a3*delta_t
    a4 = ffa(x4, v4, tp, params)

    # Returning approximate values at end of interval
    delta_t6 = delta_t/6.0
    xp = xn + delta_t6*(v1 + 2.0*(v2 + v3) + v4)
    vp = vn + delta_t6*(a1 + 2.0*(a2 + a3) + a4)
    tp = tn + delta_t

    return xp, vp, tp
