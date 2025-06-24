import numpy as np


def roulette_wheel(data, s=2.):
    r_min, r_max = np.min(data), np.max(data)
    p = np.exp(-s * (data - r_max) / (r_max - r_min))
    i = p.argsort()
    p[i] /= np.cumsum(p[i])
    return p / np.sum(p)
