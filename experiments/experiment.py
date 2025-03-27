import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sci
from plot import phase_portrait, time_series


def rhs(y, t, k, tau):
    dydt = np.zeros_like(y)

    # simple input oscillator
    dydt[0] = y[1]
    dydt[1] = -k * y[0]

    # oect 1 gate
    dydt[2] = (y[0] - y[2]) / tau
    out1 = 1 - sigmoid(y[2], 0.05, 0)

    # oect 2 gate
    dydt[3] = (out1 - y[3]) / tau

    return dydt


def sigmoid(x, s, offset=0):
    return 1.0 / (1.0 + np.exp(-(x - offset) / s))


k = 1
tau = 0.1

tmax = 20

y0 = np.array([1, 0, 0, 0])
ts = np.arange(0, tmax, 0.05)

y_sol = sci.odeint(rhs, y0, ts, args=(k, tau))

out1 = 1 - sigmoid(y_sol[:, 2], 0.05, 0)
out2 = 1 - sigmoid(y_sol[:, 3], 0.05, 0)
y = np.hstack((y_sol, out1.reshape(-1, 1), out2.reshape(-1, 1)))

_ = phase_portrait(y, (0, 4))
_ = phase_portrait(y, (4, 5))
_ = time_series(ts, y)

plt.show()
