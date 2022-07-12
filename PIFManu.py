import scipy.sparse.linalg
from toolbox import *
from finufft import nufft1d1, nufft1d2

def findsource(xp, vp, L, it, DT):
     E = Q * N * np.sin(np.pi * it * DT / 4) * np.cos(2 * np.pi * xp / L) / (4 * np.pi)
     S1 = - 0.5 * (np.sqrt(np.pi * VT / 2) / L) * np.exp(- vp ** 2 / (2 * VT)) * np.cos(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L) / 4
     S2 = - 0.5 * (np.sqrt(2 * np.pi * VT) * vp / L ** 2) * np.exp(- vp ** 2 / (2 * VT)) * np.sin(np.pi * it * DT / 4) * np.cos(2 * np.pi * xp / L)
     S3 = - (QM * E / L) * (1 - 0.5 * np.sin(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L)) * (np.exp(- vp ** 2 / (2 * VT)) * vp / np.sqrt(2 * np.pi))
     return S1 + S2 + S3


L = 32  # Length of the container
DT = .01  # Length of a time step
NT = 800  # number of time steps
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
QM = -1 # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
mode = 1  # Mode of the sin wave in perturbation
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
xp = np.linspace(0, 1, N, endpoint=False) * L
vp = np.random.randn(N) * VT
wp = np.ones(N)
f0 = np.exp(- vp ** 2 / (2 * VT)) * np.sqrt(VT) / (L * np.sqrt(2 * np.pi))
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
    rhoHat = Q * Shat * finufft.nufft1d1(xp * 2 * np.pi / L, wp + 0j, NF, eps=1e-12, modeord=1)
    rhoHat = np.append(rhoHat[0], rhoHat[:0:-1]) # Somehow the documentation was wrong
    # computing fields
    Phihat, Ehat = fieldSolve(rhoHat, L, hat=True)

    # projection p -> q and update of vp
    coeff = Ehat * Shat
    coeff = np.append(coeff[0], coeff[:0:-1])
    a = finufft.nufft1d2(xp * 2 * np.pi / L, coeff, eps=1e-12, modeord=1) * QM / L
    if it == 0:
        vp = vp + np.real(a) * DT / 2
    else:
        vp = vp + np.real(a) * DT
    wp = wp + DT * findsource(xp, vp, L, it + 0.5, DT) / f0
    xp = xp + vp * DT

    # energies
    kinetic = sum(Q * vp ** 2 * 0.5 / QM)
    potential1 = sum(rhoHat * np.conjugate(Phihat) * L / (2 * NF ** 2))
    Phi = np.fft.ifft(Phihat)
    Ek.append(kinetic)
    Ep.append(potential1)
    # Ep2.append(potential1)
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
