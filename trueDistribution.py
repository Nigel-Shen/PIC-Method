import numpy as np
import matplotlib.pyplot as plt


def trueDistribution(t, NG):
    image = np.zeros([NG, NG])
    for i in range(NG):
        for j in range(NG):
            image[j,i] = np.exp(-((NG/2-(j+0.5))/(NG/20))**2/2)*(1-0.5*np.sin(np.pi * t/4)*np.sin(2*np.pi*(i+0.5)/NG))/(NG*np.sqrt(np.pi*2))
    plt.imshow(image, 'plasma', interpolation='nearest')

def trueRho(t, NG, N, Q, L):
    y = np.linspace(0, 1, NG, endpoint=False) + 1 / (2 * NG)
    return N * Q * (1 - 0.5 * np.sin(np.pi * t / 4) * np.sin(2*np.pi*y)) / L

def trueField(t, NG):
    y=np.zeros(NG)
    for i in range(NG):
        y[i] = np.sin(np.pi * t/4)*np.cos(2*np.pi*(i+0.5)/NG)
    return y

def truePhi(t, NG, N, Q, L):
    y = np.linspace(0, 1, NG, endpoint=False) + 1 / (2 * NG)
    return N * Q * L * (- np.sin(np.pi * t / 4) * np.sin(2 * np.pi * y)) / (8 * np.pi ** 2)
