import numpy as np

def BasicSample(pdf):
    # Construct CDF
    cdf = np.zeros(pdf.shape[0])
    cdf[0] = pdf[0]
    for i in range(1,pdf.shape[0]):
        cdf[i] = pdf[i] + cdf[i-1]
    
    # Sample
    r = np.random.random()
    for i in range(cdf.shape[0]):
        if r<=cdf[i]:
            return i

if __name__ == '__main__':
    pdf = np.array([0.1,0.2,0.3,0.4])
    count = [0,0,0,0]
    for i in range(10000):
        s = BasicSample(pdf)
        count[s] += 1
    print(count)

