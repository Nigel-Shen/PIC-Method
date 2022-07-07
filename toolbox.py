import cmath
import math
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from mpmath import findroot, erfc
import matplotlib.pyplot as plt


def toPeriodic(x, L, discrete=False):
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x


def phaseSpace(g, fraz, vp, Q, VT):
    g = g[:, np.abs(vp) < 10 * VT]
    fraz = fraz[:, np.abs(vp) < 10 * VT]
    vp = vp[np.abs(vp) < 10 * VT]
    col = (vp + 10 * VT) // (20 *VT / 128)
    col = col.astype(int)

    mat = sparse.csr_matrix((- fraz[0] * Q, (col, g[0]))) + sparse.csr_matrix((- fraz[1] * Q, (col, g[1]))) +sparse.csr_matrix((- fraz[2] * Q, (col, g[2])))
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


def findDecayRate(k):
    gamma = float(np.imag(findroot(
        lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
    return gamma


def fieldSolve(rho, L, hat=False):
    if hat:
        rhoHat = rho
    else:
        rhoHat = np.fft.fft(rho)     
    Ka = np.arange(1, rhoHat.size // 2)
    Kb = Ka[::-1]
    K = np.append(np.append(Ka, [rhoHat.size // 2]), - Kb)
    phiHat = np.append([0], rhoHat[1:] * (L / (2 * np.pi * K)) ** 2)
    EHat = np.append([0], rhoHat[1:] * L / (2j * np.pi * K))
    EHat[rhoHat.size // 2] = 0
    if hat:
        return phiHat, EHat
    else:
        phi = np.real(np.fft.ifft(phiHat))
        E = np.real(np.fft.ifft(EHat))
        return phi, E
