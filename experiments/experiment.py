import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sci
import yaml
from plot import phase_portrait, time_series


def rhs(y, t, k, tau, s, off_level):
    dydt = np.zeros_like(y)

    # simple input oscillator
    dydt[0] = y[1]
    dydt[1] = -k * y[0]

    # oect 1 gate
    dydt[2] = (y[0] - y[2]) / tau

    # oect 2 gate
    dydt[3] = (output(y[2], s=s, off_level=off_level) - y[3]) / tau

    return dydt


def sigmoid(x, s, offset=0):
    return 1.0 / (1.0 + np.exp(-(x - offset) / s))


def output(gate, s, offset=0, off_level=0):
    return 1.0 - (1.0 - off_level) * sigmoid(gate, s, offset)


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["main"]

k = params["system"]["k"]
tau = params["system"]["tau"]
s = params["system"]["s"]
off_level = params["system"]["off_level"]

tmax = params["general"]["tmax"]
dt = params["general"]["dt"]

y0 = np.array(params["initial"]["y0"])
ts = np.arange(0, tmax, dt)

y_sol = sci.odeint(rhs, y0, ts, args=(k, tau, s, off_level))

out1 = output(y_sol[:, 2], s=s, off_level=off_level)
out2 = output(y_sol[:, 3], s=s, off_level=off_level)

# add output series to y-array, for convenient plotting
y = np.hstack((y_sol, out1.reshape(-1, 1), out2.reshape(-1, 1)))

_ = phase_portrait(y, (0, 4))
_ = phase_portrait(y, (0, 5))
_ = phase_portrait(y, (4, 5))
_ = time_series(ts, y)

plt.show()
