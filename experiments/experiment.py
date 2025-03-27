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
    out1 = sigmoid(y[2], 0.1, 0)

    # oect 2 gate
    dydt[3] = (out1 - y[3]) / tau

    return dydt


def sigmoid(x, s, offset=0):
    return 1.0 / (1.0 + np.exp(-(x - offset) / s))


k = 4
tau = 2

tmax = 20

y0 = np.array([1, 0, 0, 0])
ts = np.arange(0, tmax, 0.05)

y_sol = sci.odeint(rhs, y0, ts, args=(k, tau))
out2 = sigmoid(y_sol[:, 3], 0.1, 0)
y = np.hstack((y_sol, out2.reshape((-1, 1))))

_ = phase_portrait(y, (0, 4))
_ = time_series(ts, y, (0, 2, 3, 4))

plt.show()
