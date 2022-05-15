import cmath
import math
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from mpmath import findroot, erfc
import matplotlib.pyplot as plt


def phaseSpace(xp, vp, wp, L, pixel=200):
    X = L
    V = 10 * (1 + 1/pixel)
    M = np.zeros([pixel, pixel])
    x = X / pixel
    v = 2 * V / pixel
    for i in range(len(xp)):
        if np.abs(vp[i]) < 10:
            M[int((vp[i] + V) // v), int(xp[i] // x) % pixel] = M[int((vp[i] + V) // v), int(xp[i] // x) % pixel] + wp[i]
    plt.imshow(M, vmin=0, vmax=np.max(M), cmap='plasma', interpolation="nearest", extent=[0, X, -V, V])
    plt.colorbar()


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


L = 10 * np.pi  # Length of the container
DT = .005  # Length of a time step
NT = 1600  # number of time steps
NG = 128  # Number of Grid points
N = 200000  # Number of simulation particles
WP = 1  # omega p
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
XP1 = 0.05  # Magnitude of perturbation in x
mode = 1  # Mode of the sin wave in perturbation
Q = WP ** 2 * L / (QM * N * lambdaD)  # Charge of a particle
rho_back = - Q * N / L  # background rho
dx = L / NG  # cell length
k = lambdaD * mode * 2 * np.pi / L

i = 0
xp = np.random.rand(N) * L
vp = np.zeros(N)
while i < N:
    v = np.random.rand(1, 2) * np.array([8, 1]) - np.array([4,0])
    if v[0, 1] < 2 * v[0, 0] ** 2 * np.exp(- v[0, 0] ** 2):
        vp[i] = v[0, 0]
        i = i + 1
wp = np.ones(N)
f0 = 2 * vp ** 2 * np.exp(- vp ** 2) / (L * np.sqrt(np.pi))
# Perturbation
p = np.linspace(0, N - 1, N).astype(int)
un = np.ones(NG - 1)
Poisson = sparse.spdiags(np.array([un, -2 * un, un]), [-1, 0, 1], NG - 1, NG - 1,
                         format='csc')  # Matrix to represent Poisson
# Energy
Ek = []  # Kinetic Energy
Ep = []  # Potential Energy
Et = []  # Total Energy
momentum = []
PhiMax = []
# showDistribution(xp, Q, NG, L, 64)
picnum=0
plt.rcParams['figure.dpi'] = 300
for it in range(NT):
    print(it)
    if it % 100 == 0 and picnum < 16:
        picnum = picnum + 1
        plt.subplot(4, 4, picnum)
        phaseSpace(xp, vp, wp, L)
        plt.title('$t$=%s' % str(np.round(it * DT,4)))

    xp = xp + vp * DT / 2
    # Apply bc on the particle position, periodic
    out = (xp < 0)
    xp[out] = xp[out] + L
    out = (xp >= L)
    xp[out] = xp[out] - L

    # projection p->g
    g1 = np.floor(xp / dx - 0.5).astype(int)  # which grid point to project onto
    g = np.array([g1, g1 + 1])  # used to determine bc
    fraz1 = (1 - abs(xp / dx - g1 - 0.5)) * wp
    fraz = np.array([fraz1, wp - fraz1])

    # apply bc on the projection
    out = (g < 0)
    g[out] = g[out] + NG
    out = (g > NG - 1)
    g[out] = g[out] - NG
    mat = sparse.csr_matrix((fraz[0], (p, g[0]))) + sparse.csr_matrix((fraz[1], (p, g[1])))  # interpolation
    rho = np.asarray((Q / dx) * mat.sum(0) + rho_back * np.ones([1, NG]))

    # computing fields
    Phi = sparse.linalg.splu(Poisson).solve(-rho[0, 0:NG - 1] * dx ** 2)
    Phi = np.append(Phi, [0])
    Eg = np.transpose([np.append(Phi[NG - 1], Phi[0:NG - 1]) - np.append(Phi[1:NG], Phi[0])]) / (2 * dx)
    # projection p -> q and update of vp

    if it == 0: # leap frog
        vp = vp + np.transpose(mat * Eg)[0] * QM * DT / 2
    xp = xp + vp * DT / 2
    vp = vp + np.transpose(mat * Eg)[0] * QM * DT
    E = - Q * np.sin(np.pi * it * DT) * np.cos(2 * np.pi * xp / L) / (2 * np.pi)
    S1 = - (2 * np.sqrt(np.pi) / L) * vp ** 2 * np.exp(-vp ** 2) * np.cos(np.pi * it * DT) * np.sin(2 * np.pi * xp / L)
    S2 = - (4 * np.sqrt(np.pi) * vp ** 3 / L ** 2) * np.exp(- vp ** 2) * np.sin(np.pi * it * DT) * np.cos(2 * np.pi * xp / L)
    S3 = - (QM * E / L) * (1 - np.sin(np.pi * it * DT) * np.sin(2 * np.pi * xp / L)) * (4 * np.exp(- vp ** 2) / np.sqrt(np.pi)) * (vp - vp ** 3)
    wp = wp + DT * (S1 + S2 + S3) / f0
    # energies
    kinetic = sum(Q * wp * vp * vp * 0.5 / QM)
    potential = sum(mat * Phi * Q * wp)

    Ek.append(kinetic)
    Ep.append(potential)
    Et.append(kinetic + potential)
    momentum.append(sum(Q * wp * vp / QM))
    PhiMax.append(np.max(np.abs(Phi)))

plt.show()
plt.plot(np.linspace(1, NT * DT, NT), Et, label='Total Energy')
plt.plot(np.linspace(1, NT * DT, NT), Ek, label='Kinetic Energy')
plt.plot(np.linspace(1, NT * DT, NT), Ep, label='Potential Energy')  # Total Energy at a given time
plt.legend()
plt.show()
plt.plot(np.linspace(1, NT * DT, NT), momentum, label='Momentum')
plt.legend()
plt.show()
