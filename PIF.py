import scipy.sparse.linalg
from toolbox import *


L = 32  # Length of the container
DT = .02  # Length of a time step
NT = 1000  # number of time steps
NF = 16 # Number of Fourier Modes
PL = 2
Ka = np.arange(1, NF // 2)
Kb = Ka[::-1]
K = np.append(np.append(Ka, [- NF // 2]), - Kb)
Shat = (L * np.sin(np.pi * K * PL / L) / (np.pi * K * PL)) ** 2
Shat = np.append([1], Shat)
K = np.append([0], K)
N = 10000  # Number of simulation particles
WP = 1  # omega p
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
XP1 = 0.2  # Magnitude of perturbation in x
mode = 2  # Mode of the sin wave in perturbation
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
k = lambdaD * mode * 2 * np.pi / L
period = 2 * np.pi / float(np.real(findroot(
    lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
i = 0
#xp = np.zeros(N)
#while i < N:
#    x = np.random.rand(1, 2) * np.array([L, N / L * (1 + XP1)])
#    if x[0, 1] < N / L * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L * mode)):
#        xp[i] = x[0, 0]
#        i = i + 1
#vp = VT * np.random.randn(N)
xp0 = np.zeros(int(N / mode))
vp0 = VT * np.random.randn(int(N / mode))
while i < int(N / mode):
    x = np.random.rand(1, 2) * np.array([L / mode, N / L * (1 + XP1)])
    if x[0, 1] < N / L * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L * mode)):
        xp0[i] = x[0, 0]
        i = i + 1
xp = xp0
vp = vp0
for j in range(mode - 1):
    xp = np.append(xp, xp0 + L * (j + 1) / mode)
    vp = np.append(vp, vp0)
# Energy
Ek = []  # Kinetic Energy
Ep = []  # Potential Energy
Ep2 = []
E = []  # Total Energy
momentum = []
PhiMax = []
# showDistribution(xp, Q, NG, L, 64)
picnum=0
plt.rcParams['figure.dpi'] = 300
for it in range(NT):
    print(it)
    xp = toPeriodic(xp, L)
    rhoHat = Q * Shat * np.sum(np.exp(-2j * np.pi * np.kron(xp, np.transpose([K])) / L), axis = 1)
    
    # computing fields
    Phihat, Ehat = fieldSolve(rhoHat, L, hat=True)
    
    # projection p -> q and update of vp
    a = np.sum(np.transpose([Ehat * Shat]) * np.exp(2j * np.pi * np.kron(xp, np.transpose([K])) / L), axis = 0) * QM / L
    
    if it == 0:
        vp = vp + np.real(a) * DT / 2
    else:
        vp = vp + np.real(a) * DT
    xp = xp + vp * DT

    # energies
    kinetic = sum(Q * vp ** 2 * 0.5 / QM)
    potential1 = sum(rhoHat * np.conjugate(Phihat) / (2 * L))
    Phi = np.fft.ifft(Phihat) * NF / L
    Ek.append(kinetic)
    Ep.append(potential1)
    #Ep2.append(potential1)
    E.append(kinetic + potential1)
    momentum.append(sum(Q * vp / QM))
    PhiMax.append(np.max(Phi))
plt.plot(np.linspace(0, NT * DT, NT), E / E[0], label='Total Energy')
#plt.plot(np.linspace(0, NT * DT, NT), Ek, label='Kinetic Energy')
#plt.plot(np.linspace(0, NT * DT, NT), Ep, label='Potential Energy')  # Total Energy at a given time
#plt.plot(np.linspace(0, NT * DT, NT), Ep2, label='Potential Energy') 
plt.legend()
plt.show()
plt.close()
plt.plot(np.arange(NF), np.fft.ifft(rhoHat))
plt.show()
plt.close()
a = np.linspace(0, (NT - 1) * DT, NT)
plt.plot(a, PhiMax, label='$\phi_{max}$')
gamma = np.sqrt(2) * float(np.imag(findroot(
    lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
                math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
print(gamma)
b = PhiMax[int(period // (2 * DT))] * np.exp((a[0:600]- period / 2) * gamma)
plt.plot(a[0:600], b, label='predicted decay rate')
plt.yscale('log')
plt.legend()
plt.show()
plt.close()
plt.plot(np.linspace(1, NT * DT, NT), momentum, label='Momentum')
plt.legend()
plt.show()

