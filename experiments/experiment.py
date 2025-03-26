import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


def rhs(y, t):
    dydt = np.zeros_like(y)

    dydt[0] = y[1]
    dydt[1] = -(k + kappa) * y[0] + kappa * y[2]
    dydt[2] = y[3]
    dydt[3] = -(k + kappa) * y[2] + kappa * y[0]

    return dydt


def phase_portrait(y, inds=(0, 1), ax=None):
    if len(inds) != 2:
        raise ValueError(f"`inds` must have length 2, got {len(inds)}")

    if y.ndim < 2 or y.shape[1] < 2:
        raise ValueError(f"y has to be of shape (n, >1), but got {y.shape}")

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(y[:, inds[0]], y[:, inds[1]])

    ax.set_xlabel(f'$y_{inds[0]}$')
    ax.set_ylabel(f'$y_{inds[1]}$')

    return plt.gcf()


def time_series(t, y, inds=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if y.ndim == 1:
        ax.plot(t, y)
    elif y.ndim > 1:
        if inds is None:
            inds = np.arange(y.shape[1])

        for i in inds:
            ax.plot(t, y[:, i], label=f'$y_{i}$')

        plt.legend(loc="upper right")

    return plt.gcf()


k = 4
kappa = 0.5

tmax = 20

y0 = np.array([0, 0, 1, 0])
ts = np.arange(0, tmax, 0.05)

y_sol = sci.odeint(rhs, y0, ts)

_ = phase_portrait(y_sol, (0, 2))
_ = time_series(ts, y_sol, (0, 2))

plt.show()
