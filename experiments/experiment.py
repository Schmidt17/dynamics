import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt

from plot import phase_portrait, time_series


def rhs(y, t):
    dydt = np.zeros_like(y)

    dydt[0] = y[1]
    dydt[1] = -(k + kappa) * y[0] + kappa * sigmoid(y[2], 0.1, 0.1 * y[0])
    dydt[2] = y[3]
    dydt[3] = -(k + kappa) * y[2] + kappa * sigmoid(y[0], 0.1, 0.1 * y[2])

    return dydt


def sigmoid(x, s, offset=0):
    return 1. / (1. + np.exp(-(x - offset) / s))


k = 4
kappa = 4

tmax = 50

y0 = np.array([0, 0, 1, 0])
ts = np.arange(0, tmax, 0.05)

y_sol = sci.odeint(rhs, y0, ts)

_ = phase_portrait(y_sol, (0, 2))
_ = time_series(ts, y_sol, (0, 2))

plt.show()
