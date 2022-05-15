import cmath
import math
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from mpmath import findroot, erfc
import matplotlib.pyplot as plt


def phaseSpace(xp, vp, pixel=200):
    X = np.max(np.abs(xp)) * (1 + 1 / pixel)
    V = np.max(np.abs(vp)) * (1 + 1 / pixel)
    M = np.zeros([pixel, pixel])
    x = X / pixel
    v = 2 * V / pixel
    for i in range(len(xp)):
        M[int((vp[i] + V) // v), int(xp[i] // x)] = M[int((vp[i] + V) // v), int(xp[i] // x)] + 1
    plt.imshow(M, vmin=0, vmax=np.max(M), cmap='plasma', interpolation="nearest", extent=[0, X, -V, V])
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

l = [1/50,1/20,1/10,1/5,5,10]
for m in range(len(l)):
    L = 5 * np.pi  # Length of the container
    DT = .2  # Length of a time step
    NT = 400  # number of time steps
    NG = 64  # Number of Grid points
    N = int(1000000 * l[m])  # Number of simulation particles
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
    period = 2 * np.pi / float(np.real(findroot(
        lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                    math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
    i = 0
    xp = np.zeros(N)
    while i < N:
        x = np.random.rand(1, 2) * np.array([L, N / L * (1 + XP1)])
        if x[0, 1] < N / L * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L * mode)):
            xp[i] = x[0, 0]
            i = i + 1
    vp = VT * np.random.randn(N)
    p = np.linspace(0, N - 1, N).astype(int)
    un = np.ones(NG - 1)
    Poisson = sparse.spdiags(np.array([un, -2 * un, un]), [-1, 0, 1], NG - 1, NG - 1,
                             format='csc')  # Matrix to represent Poisson
    # Energy
    plt.rcParams['figure.dpi'] = 300
    Ek = []  # Kinetic Energy
    Ep = []  # Potential Energy
    E = []  # Total Energy
    momentum = []
    PhiMax = []
    # showDistribution(xp, Q, NG, L, 64)
    picnum=0

    for it in range(NT):
        print(it)

        # Update xp
        xp = xp + vp * DT

        # Apply bc on the particle position, periodic
        out = (xp < 0)
        xp[out] = xp[out] + L
        out = (xp >= L)
        xp[out] = xp[out] - L

        # projection p->g
        g1 = np.floor(xp / dx - 0.5).astype(int)  # which grid point to project onto
        g = np.array([g1, g1 + 1])  # used to determine bc
        fraz1 = 1 - abs(xp / dx - g1 - 0.5)
        fraz = np.array([fraz1, 1 - fraz1])

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
        vp = vp + np.transpose(mat * Eg)[0] * QM * DT

        # energies
        kinetic = sum(Q * vp * vp * 0.5 / QM)
        potential = sum(mat * Phi * Q)

        Ek.append(kinetic)
        Ep.append(potential)
        E.append(kinetic + potential)
        momentum.append(sum(Q * vp / QM))
        PhiMax.append(np.max(np.abs(Phi)))

    # plt.plot(np.linspace(1,L,NG),Phi)    # Discrete Potential Field at a given time
    a = np.linspace(0, (NT - 1) * DT, NT)
    plt.subplot(2,3,m+1)
    plt.plot(a, PhiMax, label='$\phi_{max}$')
    gamma = float(np.imag(findroot(
        lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                    math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
    print(gamma)
    b = PhiMax[0] * np.exp(a[0:60] * gamma)
    plt.plot(a[0:60], b, label='predicted decay rate')
    plt.yscale('log')
    plt.title('N=%s' % N)
plt.show()

