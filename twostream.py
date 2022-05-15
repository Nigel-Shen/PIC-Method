import numpy as np
import scipy.sparse.linalg
from scipy import sparse
import matplotlib.pyplot as plt

# I am not sure yet how to unify the units of kinetic energy and potential energy, but the shape of the plots should
# be correct


def phaseSpace(xp, vp, pixel = 200):
    X = np.max(np.abs(xp)) * (1 + 1 / pixel)
    V = np.max(np.abs(vp)) * (1 + 1 / pixel)
    M = np.zeros([pixel, pixel])
    x = X / pixel
    v = 2 * V / pixel
    for i in range(len(xp)):
        M[int((vp[i] + V)//v), int(xp[i]//x)] = M[int((vp[i] + V)//v), int(xp[i]//x)] + 1
    plt.imshow(M, vmin=0, vmax=np.max(M), cmap='plasma', interpolation="nearest", extent=[0, X, -V, V])
    plt.colorbar()
    plt.axis('off')

L = 2 * np.pi  # Length of the container
DT = .5  # Length of a time step
NT = 400  # number of time steps
NG = 64  # Number of Grid points
N = 1000000  # Number of simulation particles
WP = 1  # ???
QM = -1  # Charge/Mass, not sure
V0 = 1  # drift velocities ue to 2 stream instability
VT = 0  # magnitude of random fluctuation of velocities
XP1 = 1  # Magnitude of perturbation in x
V1 = 0  # Magnitude of perturbation in v
mode = 1  # Mode of the sin wave in perturbation
Q = WP ** 2 / (QM * N / L)  # Charge of a particle
rho_back = -Q * N / L  # background rho
dx = L / NG  # cell length

# initial loading for the 2 Stream instability
xp = np.linspace(0, L - L / N, N)  # initial location of particles
vp = VT * np.random.randn(N)
pm = 1 - 2 * np.mod(np.linspace(1, N, N), 2)  # 2 stream instability [1 -1 1 -1 ...]
vp = vp + V0 * pm

# Perturbation
vp = vp + V1 * np.sin(2 * np.pi * xp / L * mode)
xp = xp + XP1 * (L / N) * np.sin(2 * np.pi * xp / L * mode)  # sin wave perturbation
p = np.linspace(0, N - 1, N).astype(int)
p = np.array([p, p])
un = np.ones(NG - 1)
Poisson = sparse.spdiags(np.array([un, -2 * un, un]), [-1, 0, 1], NG - 1, NG - 1)  # Matrix to represent Poisson
# Energy
Ek = [] # Kinetic Energy
Ep = [] # Potential Energy
E = [] # Total Energy
# Main Computation Cycle
for it in range(NT):
    # Update xp
    xp = xp + vp * DT
    print(it)
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
    mat = sparse.csr_matrix((fraz[0], (p[0], g[0]))) + sparse.csr_matrix((fraz[1], (p[1], g[1])))  # interpolation
    rho = np.asarray((Q / dx) * mat.sum(0) + rho_back * np.ones([1, NG]))

    # computing fields
    Phi = scipy.sparse.linalg.splu(Poisson).solve(-rho[0, 0:NG - 1] * dx ** 2)
    Phi = np.append(Phi, [0])
    Eg = np.transpose([np.append(Phi[NG - 1], Phi[0:NG - 1]) - np.append(Phi[1:NG], Phi[0])]) / (2 * dx)

    # projection p -> q and update of vp
    vp = vp + np.transpose(mat * Eg)[0] * QM * DT

    # energies
    kinetic = sum(vp * vp * 0.5)
    potential = sum(mat * Phi * Q)
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    if it % 25 == 0:
        plt.subplot(4, 4, it // 25 + 1)
        phaseSpace(xp, vp)
        plt.title('$t$=%s' % str(it * 0.25))
plt.show()

# plt.scatter(xp,vp)    # Phase space at a given time
plt.plot(np.linspace(1,NT * DT, NT), E)    # Total Energy at a given time
# plt.plot(np.linspace(1,L,NG),Phi)    # Discrete Potential Field at a given time
x = np.linspace(0, L, N)
y = np.zeros(N)
for i in range(N):
    for item in xp:
        if item - 0.5 < x[i] < item + 0.5:
            y[i] = y[i] + 1
        elif x[i] < item - L + 0.5 or x[i] > item + L - 0.5:
            y[i] = y[i] + 1
#plt.plot(x, y)    # Density at given time
plt.show()
