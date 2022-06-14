import cmath
import math
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from mpmath import findroot, erfc
import matplotlib.pyplot as plt
from toolbox import *
from true import *


def findsource(xp, vp, L, it, DT):
    E = Q * N * np.sin(np.pi * it * DT / 4) * np.cos(2 * np.pi * xp / L) / (4 * np.pi)
    S1 = - 0.5 * (np.sqrt(np.pi / 2) / L) * np.exp(- vp ** 2 / 2) * np.cos(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L) / 4
    S2 = - 0.5 * (np.sqrt(2 * np.pi) * vp / L ** 2) * np.exp(- vp ** 2 / 2) * np.sin(np.pi * it * DT / 4) * np.cos(
        2 * np.pi * xp / L)
    S3 = - (QM * E / L) * (1 - 0.5 * np.sin(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L)) * (np.exp(- vp ** 2 / 2) * vp / np.sqrt(2 * np.pi))
    return (S1 + S2 + S3) * Q * N


L = 2 * np.pi # Length of the container
DT = .02  # Length of a time step
NT = 400  # number of time steps
NG = 256  # Number of Grid points
N = 200000  # Number of simulation particles
WP = 1  # omega p
DT = DT * WP
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
L = L / lambdaD
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
dx = L / NG  # cell length
xp = np.linspace(0, L, N, endpoint=False)
vp = np.random.randn(N)
# while i < N:
#     v = np.random.rand(1, 2) * np.array([8, 1]) - np.array([4, 0])
#     if v[0, 1] < 2 * v[0, 0] ** 2 * np.exp(- v[0, 0] ** 2):
#         vp[i] = v[0, 0]
#         i = i + 1
wp = np.ones(N)
f0 = N * Q * np.exp(- vp ** 2 / 2) * np.sqrt(1 / (2 * np.pi)) / L
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
picnum = 0
plt.rcParams['figure.dpi'] = 300
for it in range(NT):
    print(it)
    if it % 25 == 1 and picnum < 16:
        picnum = picnum + 1
        plt.subplot(4, 4, picnum)
        phaseSpace(g, fraz, vp, Q)
        plt.title('$t$=%s' % str(np.round(it * DT, 4)))

    # Apply bc on the particle position, periodic

    xp = toPeriodic(xp,L)

    # projection p->g
    g1 = np.floor(xp / dx - 0.5).astype(int)  # which grid point to project onto
    g = np.array([g1, g1 + 1])  # used to determine bc
    fraz1 = (1 - abs(xp / dx - g1 - 0.5)) * wp
    fraz = np.array([fraz1, wp - fraz1])

    # apply bc on the projection
    g = toPeriodic(g, NG, True)
    mat = sparse.csr_matrix((fraz[0], (p, g[0]))) + sparse.csr_matrix((fraz[1], (p, g[1])))  # interpolation
    rho = np.asarray((Q / dx) * mat.sum(0) - (Q * sum(wp) / L) * np.ones([1, NG]))[0]

    # computing fields
    # Phi = sparse.linalg.splu(Poisson).solve(-rho[0, 0:NG - 1] * dx ** 2)
    # Phi = np.append(Phi, [0])
    Phi = fieldSolve(rho, L)
    Eg = np.transpose([np.append(Phi[NG - 1], Phi[0:NG - 1]) - np.append(Phi[1:NG], Phi[0])]) / (2 * dx)
    # projection p -> q and update of vp
    #
    # if it % 25 ==0 and picnum < 16:
    #     picnum = picnum + 1
    #     plt.subplot(4, 4, picnum)
    #     plt.plot(np.linspace(0,NG-1,NG), Eg.transpose()[0], label='%s' %N)
    #     plt.title('$t$=%s' % str(np.round(it * DT, 4)))
    #     #plt.show()

    if it == 0:
        vp = vp + np.transpose(mat * Eg)[0] * QM * DT / (2 * wp)
    else:
        vp = vp + np.transpose(mat * Eg)[0] * QM * DT / wp
    xp = xp + vp * DT / 2
    wp = wp + DT * findsource(xp, vp, L, it + 0.5, DT) / f0
    xp = xp + vp * DT / 2

    # energies
    kinetic = sum(Q * wp * vp * vp * 0.5 / QM)
    # potential = sum(mat * Phi * Q * wp / 2)

    Ek.append(kinetic)
    #Ep.append(potential)
    #Et.append(kinetic + potential)
    momentum.append(sum(Q * wp * vp / QM))
    #PhiMax.append(np.max(np.abs(Phi)))
plt.legend()
plt.show()
# plt.plot(np.linspace(1, NT * DT, NT), Et, label='Total Energy')
# plt.plot(np.linspace(1, NT * DT, NT), Ek, label='Kinetic Energy')
# plt.plot(np.linspace(1, NT * DT, NT), Ep, label='Potential Energy')  # Total Energy at a given time
# plt.legend()
# plt.show()
# plt.plot(np.linspace(1, NT * DT, NT), momentum, label='Momentum')
# plt.legend()
# plt.show()
