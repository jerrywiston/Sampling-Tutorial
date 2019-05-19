import numpy as np
import matplotlib.pyplot as plt

def Gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * \
    np.exp(-0.5*np.square((x-mu)/sigma))

def Target(x):
    return Gaussian(x,0,1)

def Proposal(x):
    return 2.5*Gaussian(x,1,2)

def Rate(x):
    return Target(x) / Proposal(x)

if __name__ == '__main__':
    x = np.linspace(-10,10,100)
    plt.plot(x,Target(x),'g')
    plt.plot(x,Proposal(x),'b')
    plt.show()
    
    slist = []
    while(1):
        samp = np.random.normal(1,2,1)
        reject = Rate(samp)
        r = np.random.random()
        if r<reject:
            print(len(slist), samp)
            slist.append(samp)
        if len(slist) >= 1000:
            break

    slist = np.asarray(slist)
    print('Mean:', np.mean(slist))
    print('Var:', np.var(slist))

