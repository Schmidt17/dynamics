import numpy as np
import matplotlib.pyplot as plt


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
