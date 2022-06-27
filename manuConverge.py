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
    S1 = - 0.5 * (np.sqrt(np.pi * VT / 2) / L) * np.exp(- vp ** 2 / (2 * VT)) * np.cos(np.pi * it * DT / 4) * np.sin(
        2 * np.pi * xp / L) / 4
    S2 = - 0.5 * (np.sqrt(2 * np.pi * VT) * vp / L ** 2) * np.exp(- vp ** 2 / (2 * VT)) * np.sin(
        np.pi * it * DT / 4) * np.cos(
        2 * np.pi * xp / L)
    S3 = - (QM * E / L) * (1 - 0.5 * np.sin(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L)) * (
                np.exp(- vp ** 2 / (2 * VT)) * vp / np.sqrt(2 * np.pi))
    return S1 + S2 + S3


dE = []
dPhi = []
dRho = []
for h in [1 / 2, 3 / 4, 1, 4 / 3, 2]:
    L = 1  # Length of the container
    DT = .02 * h  # Length of a time step
    NT = int(2 / (0.02 * h))  # number of time steps
    NG = int(32 / h ** 2) * 2  # Number of Grid points, needs to be even
    N = int(200000 * h ** (-4))  # Number of simulation particles
    WP = 1  # omega p
    QM = -1  # charge per mass
    VT = 1  # Thermal Velocity
    lambdaD = VT / WP
    mode = 1  # Mode of the sin wave in perturbation
    Q = WP ** 2 * L / (QM * N)  # Charge of a particle
    rho_back = - Q * N / L  # background rho
    dx = L / NG  # cell length

    i = 0
    xp = np.linspace(0, 1, N, endpoint=False) * L
    vp = np.random.randn(N) * VT
    # while i < N:
    #     v = np.random.rand(1, 2) * np.array([8, 1]) - np.array([4, 0])
    #     if v[0, 1] < 2 * v[0, 0] ** 2 * np.exp(- v[0, 0] ** 2):
    #         vp[i] = v[0, 0]
    #         i = i + 1
    wp = np.ones(N)
    f0 = np.exp(- vp ** 2 / (2 * VT)) * np.sqrt(VT) / (L * np.sqrt(2 * np.pi))
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

        # Apply bc on the particle position, periodic
        xp = toPeriodic(xp, L)

        # projection p->g
        g1 = np.floor(xp / dx - 0.5).astype(int)  # which grid point to project onto
        g = np.array([g1, g1 + 1])  # used to determine bc
        fraz1 = (1 - abs(xp / dx - g1 - 0.5)) * wp
        fraz = np.array([fraz1, wp - fraz1])

        # apply bc on the projection
        g = toPeriodic(g, NG, True)
        mat = sparse.csr_matrix((fraz[0], (p, g[0]))) + sparse.csr_matrix((fraz[1], (p, g[1])))  # interpolation
        rho = np.asarray((Q / dx) * mat.sum(0) - (Q * sum(wp) / L) * np.ones([1, NG]))[0]

        # if it % 25 == 0 and picnum < 16:
        #     picnum = picnum + 1
        #     plt.subplot(4, 4, picnum)
        #     phaseSpace(g, fraz, vp, Q/dx, VT)
        #     plt.title('$t$=%s' % str(np.round(it * DT, 4)))
        #     plt.savefig('strange.png')

        # computing fields
        # Phi = sparse.linalg.splu(Poisson).solve(-rho[0, 0:NG - 1] * dx ** 2)
        # Phi = np.append(Phi, [0])
        Phi = fieldSolve(rho, L)
        Eg = np.transpose([np.append(Phi[NG - 1], Phi[0:NG - 1]) - np.append(Phi[1:NG], Phi[0])]) / (2 * dx)
        # projection p -> q and update of vp

        # if it % 25 == 0 and picnum < 16:
        #     picnum = picnum + 1
        #     plt.subplot(4, 4, picnum)
        #     plt.plot(np.linspace(0, NG - 1, NG), Eg.transpose()[0]- trueField(it * DT, NG, N, Q) * Q * N / (4 * np.pi), label='%s' % N)
        #     plt.title('$t$=%s' % str(np.round(it * DT, 4)))
        #     plt.savefig('field.png')

            # plt.show()
        if it == 0:
            vp = vp + np.transpose(mat * Eg)[0] * QM * DT / (2 * wp)
        else:
            vp = vp + np.transpose(mat * Eg)[0] * QM * DT / wp
        xp = xp + vp * DT / 2
        wp = wp + DT * findsource(xp, vp, L, it + 0.5, DT) / f0
        xp = xp + vp * DT / 2

        print(it, max(Eg))
        # energies
        # kinetic = sum(Q * wp * vp * vp * 0.5 / QM)
        # potential = sum(mat * Phi * Q * wp / 2)

        # Ek.append(kinetic)
        # Ep.append(potential)
        # Et.append(kinetic + potential)
        # momentum.append(sum(Q * wp * vp / QM))
        # PhiMax.append(np.max(np.abs(Phi)))
    # plt.plot(np.linspace(1, NT * DT, NT), Et, label='Total Energy')
    # plt.plot(np.linspace(1, NT * DT, NT), Ek, label='Kinetic Energy')
    # plt.plot(np.linspace(1, NT * DT, NT), Ep, label='Potential Energy')  # Total Energy at a given time
    # plt.legend()
    # plt.show()
    # plt.plot(np.linspace(1, NT * DT, NT), momentum, label='Momentum')
    # plt.legend()
    # plt.show()

    dE.append(np.sqrt(np.sum((Eg.transpose()[0] - trueField(it * DT, NG, N, Q)) ** 2) / NG))
    dPhi.append(np.sqrt(np.sum((Phi - truePhi(it * DT, NG, N, Q, L)) ** 2) / NG))
    dRho.append(np.sqrt(np.sum((rho + (Q * sum(wp) / L) - trueRho(it * DT, NG, N, Q, L)) ** 2) / NG))
plt.cla()
plt.plot([1 / 2, 3 / 4, 1, 4 / 3, 2], dE, label='Electric Field Convergence')
plt.plot(np.linspace(1 / 2, 2, 20), 1.1 * 10 ** (-3) * np.linspace(1 / 2, 2, 20) ** 2, label='Order 2 Reference')
plt.xlabel('h(Generalized Discretization Parameter)')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('dE.png')
plt.cla()
plt.plot([1 / 2, 3 / 4, 1, 4 / 3, 2], dPhi, label='Potential Field Convergence')
plt.plot(np.linspace(1 / 2, 2, 20), 1.1 * 10 ** (-3) * np.linspace(1 / 2, 2, 20) ** 2, label='Order 2 Reference')
plt.xlabel('h(Generalized Discretization Parameter)')
plt.ylabel('error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('dPhi.png')
# plt.cla()
# plt.plot([1/2, 3/4, 1, 4/3, 2], dRho, label='Charge Density Convergence')
# plt.plot(np.linspace(1 / 2, 2, 20), 1.1 * 10 ** (-3) * np.linspace(1 / 2, 2, 20) ** 2, label='Order 2 Reference')
# plt.xlabel('h(Generalized Discretization Parameter)')
# plt.ylabel('error')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('dRho.png')
# plt.cla()
