import scipy.sparse.linalg
from toolbox import *
import finufft
from true import trueField

def findsource(xp, vp, L, it, DT):
     E = Q * N * np.sin(np.pi * it * DT / 4) * np.cos(2 * np.pi * xp / L) / (4 * np.pi)
     S1 = - 0.5 * (np.sqrt(np.pi * VT / 2) / L) * np.exp(- vp ** 2 / (2 * VT)) * np.cos(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L) / 4
     S2 = - 0.5 * (np.sqrt(2 * np.pi * VT) * vp / L ** 2) * np.exp(- vp ** 2 / (2 * VT)) * np.sin(np.pi * it * DT / 4) * np.cos(2 * np.pi * xp / L)
     S3 = - (QM * E / L) * (1 - 0.5 * np.sin(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L)) * (np.exp(- vp ** 2 / (2 * VT)) * vp / np.sqrt(2 * np.pi))
     return S1 + S2 + S3


L = 4 * np.pi  # Length of the container
DT = .01  # Length of a time step
NT = 400  # number of time steps
NF = 32  # Number of Fourier Modes
PL = L / NF
Ka = np.arange(1, NF // 2)
Kb = Ka[::-1]
K = np.append(np.append(Ka, [- NF // 2]), - Kb)
Shat = (L * np.sin(np.pi * K * PL / L) / (np.pi * K * PL)) ** 2
Shat = np.append([1], Shat)
K = np.append([0], K)
N = 100000  # Number of simulation particles
WP = 1  # omega p
QM = -1 # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
mode = 1  # Mode of the sin wave in perturbation
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
xp = np.random.rand(N) * L
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
dx = PL
# showDistribution(xp, Q, NG, L, 64)
picnum=0
plt.rcParams['figure.dpi'] = 300
for it in range(NT):
    print(it)

    if it % 25 == 1 and picnum < 16:
        picnum = picnum + 1
        plt.subplot(4, 4, picnum)
        phaseSpace(g, fraz, vp, Q, VT)
        plt.title('$t$=%s' % str(np.round(it * DT, 4)))

    # Apply bc on the particle position, periodic

    xp = toPeriodic(xp, L)

    # projection p->g
    g1 = np.floor(xp / dx).astype(int)  # which grid point to project onto
    g = np.array([g1 - 1, g1, g1 + 1])  # used to determine bc
    delta = xp % dx
    fraz0 = (1 - delta) ** 2 / 2
    fraz2 = delta ** 2 / 2
    fraz1 = 1 - (fraz0 + fraz2)
    fraz = np.array([fraz0, fraz1, fraz2] * wp)

    # apply bc on the projection
    g = toPeriodic(g, NF, True)



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
    xp = xp + vp * DT / 2
    wp = wp + DT * findsource(xp, vp, L, it + 0.5, DT) / f0
    xp = xp + vp * DT / 2

    # if it % 25 == 0 and picnum < 16:
    #     picnum = picnum + 1
    #     plt.subplot(4, 4, picnum)
    #     plt.plot(np.linspace(0, NF - 1, NF), np.fft.ifft(Ehat) * NF / L - trueField(it * DT, NF, N, Q), label='%s' % N)
    #     plt.title('$t$=%s' % str(np.round(it * DT, 4)))
    #     plt.savefig('field.png')

    kinetic = sum(Q * wp * vp ** 2 * 0.5 / QM)
    potential1 = sum(rhoHat * np.conjugate(Phihat) / (2 * L))
    Phi = np.fft.ifft(Phihat)
    Ek.append(kinetic - 0.5 * Q * N / QM)
    Ep.append(potential1 - L * (Q * N) ** 2 * np.sin(np.pi * it * DT / 4) ** 2 / (64 * np.pi ** 2))
    # Ep2.append(potential1)
    E.append(kinetic + potential1)
    momentum.append(sum(Q * vp / QM))
    PhiMax.append(np.max(Phi))
plt.show()
# plt.plot(np.linspace(0, NT * DT, NT), E, label='Total Energy')
plt.plot(np.linspace(0, NT * DT, NT), Ek, label='Kinetic Energy')
plt.plot(np.linspace(0, NT * DT, NT), Ep, label='Potential Energy')  # Total Energy at a given time
#plt.plot(np.linspace(0, NT * DT, NT), Ep2, label='Potential Energy')
plt.legend()
plt.show()
plt.close()
plt.plot(np.arange(NF), np.fft.ifft(rhoHat))
plt.show()
plt.close()
plt.scatter(np.arange(N), vp)# Discrete Potential Field at a given time
plt.show()
plt.close()
plt.plot(np.linspace(1, NT * DT, NT), momentum, label='Momentum')
plt.legend()
plt.show()
