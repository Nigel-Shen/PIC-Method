import numpy as np
import matplotlib.pyplot as plt


def trueDistribution(t):
    image = np.zeros([128, 128])
    for i in range(128):
        for j in range(128):
            image[j,i] = np.exp(-((64-j)/6.4)**2/2)*(1-0.5*np.sin(np.pi * t/4)*np.sin(2*np.pi*i/128))/(128*np.sqrt(np.pi*2))
    plt.imshow(image, 'plasma', interpolation='nearest')



def trueField(t):
    y=np.zeros(128)
    for i in range(128):
        y[i] = np.sin(np.pi * t/4)*np.cos(2*np.pi*i/128)
    plt.plot(np.linspace(0,127,128), y)
    plt.ylim([-1.2,1.2])

for t in [0,1,2,3,4,5,6,7]:
    trueField(t)
    plt.savefig('E%s.png' %t)
    plt.clf()