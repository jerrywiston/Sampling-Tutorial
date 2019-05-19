import numpy as np
import matplotlib.pyplot as plt

def BasicSample(pdf):
    cdf = np.zeros(pdf.shape[0])
    cdf[0] = pdf[0]
    for i in range(1,pdf.shape[0]):
        cdf[i] = pdf[i] + cdf[i-1]
    r = np.random.random()
    for i in range(cdf.shape[0]):
        if r<=cdf[i]:
            return i

def Gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * \
    np.exp(-0.5*np.square((x-mu)/sigma))

def Target(x):
    return Gaussian(x,0,1)

def Proposal(x):
    return Gaussian(x,1,2)

def Rate(x):
    return 2 * Target(x) / Proposal(x)

if __name__ == '__main__':
    x = np.linspace(-10,10,100)
    plt.plot(x,Target(x),'g')
    plt.plot(x,Proposal(x),'b')
    plt.show()

    # Sample Particle
    total = 1000
    samp = np.random.normal(1,2,total)
    importance = Rate(samp)
    importance /= np.sum(importance)

    # Importance Sample
    slist = []
    for i in range(1000):
        s = BasicSample(importance)
        slist.append(samp[s])
        print(i,samp[s])

    slist = np.asarray(slist)
    print('Mean:', np.mean(slist))
    print('Var:', np.var(slist))

