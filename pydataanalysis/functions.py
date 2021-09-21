'''
Returns the left hand side of ODE dv/dt.
Used in fourth order Runge-Kutta for numerical
solution of two coupled ODEs.

Supposed to model a forced harmonic oscillation where the
external force only works a given time.

Here one can add other functions as well and change the function that is called in the params dictionary in main program.

The scritp was in 2017 translated by Sebastian G. Winther-Larsen from a matlab script originally written by Arnt Inge Vistnes
'''

import numpy as np

def forced_oscillation(y, v, t, params):

    if t < params["end"]:
        dvdt = damped_oscillation(y, v, t, params) + params["C"]*np.cos(params["D"]*t)
    else:
        dvdt = damped_oscillation(y, v, t, params)

    return dvdt

def damped_oscillation(y, v, t, params):

    dvdt = - params["A"]*v - params["B"]*y

    return dvdt
