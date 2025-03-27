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

    # oect 2 gate
    dydt[3] = (output(y[2]) - y[3]) / tau

    return dydt


def sigmoid(x, s, offset=0):
    return 1.0 / (1.0 + np.exp(-(x - offset) / s))


def output(gate, s=0.05, offset=0):
    return 1.0 - sigmoid(gate, s, offset)


k = 1
tau = 0.1

tmax = 20

y0 = np.array([1, 0, 0, 0])
ts = np.arange(0, tmax, 0.05)

y_sol = sci.odeint(rhs, y0, ts, args=(k, tau))

out1 = output(y_sol[:, 2])
out2 = output(y_sol[:, 3])

# add output series to y-array, for convenient plotting
y = np.hstack((y_sol, out1.reshape(-1, 1), out2.reshape(-1, 1)))

_ = phase_portrait(y, (0, 4))
_ = phase_portrait(y, (0, 5))
_ = phase_portrait(y, (4, 5))
_ = time_series(ts, y)

plt.show()
