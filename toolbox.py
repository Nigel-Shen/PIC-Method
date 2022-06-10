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


def phaseSpace(g, fraz, vp, Q):
    g = g[:, np.abs(vp) < 10]
    fraz = fraz[:, np.abs(vp) < 10]
    vp = vp[np.abs(vp) < 10]
    col = (vp + 10) // (20 / 128)
    col = col.astype(int)

    mat = sparse.csr_matrix((- fraz[0] * Q, (col, g[0]))) + sparse.csr_matrix((- fraz[1] * Q, (col, g[1])))
    mat = mat.todense()
    print(mat.ndim, np.zeros([128 - mat.shape[0], mat.shape[1]]).ndim)
    mat = np.append(mat, np.zeros([128 - mat.shape[0], mat.shape[1]]), axis=0)
    plt.imshow(mat, vmin=0, vmax=np.max(mat), cmap='plasma', interpolation="nearest")
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


def fieldSolve(rho, L):
    rhoHat = np.fft.rfft(rho)
    phiHat = np.append([0], rhoHat[1:] * (L / (2 * np.pi * np.arange(1, rhoHat.size))) ** 2)
    phi = np.fft.irfft(phiHat)
    return phi
