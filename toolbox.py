import cmath
import math
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from mpmath import findroot, erfc
import matplotlib.pyplot as plt

def toPeriodic(x, L, discrete=False):
    out = (x < 0)
    x[out] = x[out] % L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] % L
    return x


def phaseSpace(xp, vp, wp, L, Q):
    pixel=200
    if len(wp) == 1:
        wp = np.ones(len(xp))
    X = L
    V = 10
    M = np.zeros([pixel, pixel])
    x = X / pixel
    v = 2 * V / pixel
    for i in range(len(xp)):
        if np.abs(vp[i]) < 10 and M[int((vp[i] + V) // v), int(xp[i] // x) % pixel] < 600:
            M[int((vp[i] + V) // v), int(xp[i] // x) % pixel] = M[int((vp[i] + V) // v), int(xp[i] // x) % pixel] + wp[i]
    plt.imshow(M, vmin=0, vmax=np.max(M), cmap='plasma', interpolation="nearest")
    plt.colorbar()
    plt.axis('off')


def showDistribution(xp, L, resolution):
    x = np.linspace(0, L, resolution + 1)
    y = np.zeros(resolution)
    length = L / resolution
    for item in xp:
        y[int(item // length)] = y[int(item // length)] + 1
    plt.plot(x[0:resolution], y)  # Phase space at a given time
    plt.show()


def showChargeDensity(xp, Q, dx, NG, L, rho_back, resolution):
    x = np.linspace(0, L, resolution)
    y = np.ones(resolution) * rho_back * L / resolution
    for i in range(resolution):
        for item in xp:
            if item - 0.5 * dx < x[i] < item + 0.5 * dx:
                y[i] = y[i] + Q * NG / resolution
            elif x[i] < item - L + 0.5 * dx or x[i] > item + L - 0.5 * dx:
                y[i] = y[i] + Q * NG / resolution
    plt.plot(x, y)  # Phase space at a given time
    plt.show()


def findDecayRate(k):
    gamma = float(np.imag(findroot(
        lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
    return gamma
