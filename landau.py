import scipy.sparse.linalg
from toolbox import *


L = 16  # Length of the container
DT = .2  # Length of a time step
NT = 400  # number of time steps
NG = 128  # Number of Grid points
N = 1000000  # Number of simulation particles
WP = 1  # omega p
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
XP1 = 0.03  # Magnitude of perturbation in x
mode = 1  # Mode of the sin wave in perturbation
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
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
# Perturbation
p = np.linspace(0, N - 1, N).astype(int)
un = np.ones(NG - 1)
Poisson = sparse.spdiags(np.array([un, -2 * un, un]), [-1, 0, 1], NG - 1, NG - 1,
                         format='csc')  # Matrix to represent Poisson
# Energy
Ek = []  # Kinetic Energy
Ep = []  # Potential Energy
E = []  # Total Energy
momentum = []
PhiMax = []
# showDistribution(xp, Q, NG, L, 64)
picnum=0
plt.rcParams['figure.dpi'] = 300
for it in range(NT):
    print(it)
    if it % 25 == 1 and picnum < 16:
        picnum = picnum + 1
        plt.subplot(4, 4, picnum)
        phaseSpace(g, fraz, vp, Q)
        plt.title('$t$=%s' % str(np.round(it * DT,4)))

    xp = toPeriodic(xp, L)

    # projection p->g
    g1 = np.floor(xp / dx).astype(int)  # which grid point to project onto
    g = np.array([g1 - 1, g1, g1 + 1])  # used to determine bc
    delta = xp % dx
    fraz0 = (1 - delta) ** 2 / 2
    fraz2 = delta ** 2 / 2
    fraz1 = 1 - (fraz0 + fraz2)
    fraz = np.array([fraz0, fraz1, fraz2])

    # apply bc on the projection
    g = toPeriodic(g, NG, True)
    mat = sparse.csr_matrix((fraz[0], (p, g[0]))) + sparse.csr_matrix((fraz[1], (p, g[1]))) + sparse.csr_matrix(
        (fraz[2], (p, g[2])))  # interpolation
    rho = np.asarray((Q / dx) * mat.sum(0) + rho_back * np.ones([1, NG]))[0]

    # computing fields
    Phi = fieldSolve(rho, L)
    Eg = np.transpose([np.append(Phi[NG - 1], Phi[0:NG - 1]) - np.append(Phi[1:NG], Phi[0])]) / (2 * dx)
    # projection p -> q and update of vp

    if it == 0:
        vp = vp + np.transpose(mat * Eg)[0] * QM * DT / 2
    else:
        vp = vp + np.transpose(mat * Eg)[0] * QM * DT
    xp = xp + vp * DT

    # energies
    kinetic = sum(Q * vp ** 2 * 0.5 / QM)
    potential = sum(mat * Phi * Q / 2)

    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    momentum.append(sum(Q * vp / QM))
    PhiMax.append(np.max(np.abs(Phi)))

plt.show()
plt.plot(np.linspace(1, NT * DT, NT), E, label='Total Energy')
plt.plot(np.linspace(1, NT * DT, NT), Ek, label='Kinetic Energy')
# plt.plot(np.linspace(1, NT * DT, NT), Ep, label='Potential Energy')  # Total Energy at a given time
plt.legend()
plt.show()
# plt.plot(np.linspace(1,L,NG),Phi)    # Discrete Potential Field at a given time
a = np.linspace(0, (NT - 1) * DT, NT)
plt.plot(a, PhiMax, label='$\phi_{max}$')
gamma = float(np.imag(findroot(
    lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
print(gamma)
b = PhiMax[0] * np.exp(a[0:60] * gamma)
plt.plot(a[0:60], b, label='predicted decay rate')
plt.yscale('log')
plt.legend()
plt.show()
plt.plot(np.linspace(1, NT * DT, NT), momentum, label='Momentum')
plt.legend()
plt.show()
