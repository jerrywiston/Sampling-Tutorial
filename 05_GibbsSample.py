import numpy as np
import matplotlib.pyplot as plt

def MrfGibbsSample(img, px, py):
    size = img.shape[0]
    rx = np.random.randint(size)
    ry = np.random.randint(size)
    pot = np.ones(2, dtype=np.int)

    if rx-1 > 0:
        pot[0] *= px[0,img[rx-1,ry]]
        pot[1] *= px[1,img[rx-1,ry]]
        
    if rx+1 < size:
        pot[0] *= px[0,img[rx+1,ry]]
        pot[1] *= px[1,img[rx+1,ry]]
        
    if ry-1 > 0:
        pot[0] *= py[0,img[rx,ry-1]]
        pot[1] *= py[1,img[rx,ry-1]]
        
    if ry+1 < size:
        pot[0] *= py[0,img[rx,ry+1]]
        pot[1] *= py[1,img[rx,ry+1]]
        
    prob = pot / np.sum(pot)
    img[rx,ry] = np.random.choice(2,1,p=prob)

px = np.array([[2,1],[1,2]])
py = np.array([[2,1],[1,2]])

#px = np.array([[2,1],[1,2]])
#py = np.array([[1,2],[2,1]])

# Initial
size = 32
img = []
img.append(np.random.choice(2,[size,size]))
img.append(np.random.choice(2,[size,size], p=[0.8,0.2]))
img.append(np.ones([size,size], dtype=np.int))
img.append(np.zeros([size,size], dtype=np.int))

view = np.zeros([size*4,size*4])
for i in range(100001):
    if i%1000 == 0:
        print('Iter',i)
        
    for j in range(4):
        if i==0:
            view[0:size, size*j:size*(j+1)] = img[j]
        elif i==1000:
            view[size:size*2, size*j:size*(j+1)] = img[j]
        elif i==10000:
            view[size*2:size*3, size*j:size*(j+1)] = img[j]
        elif i==100000:
            view[size*3:size*4, size*j:size*(j+1)] = img[j]
        MrfGibbsSample(img[j], px, py)

plt.imshow(view, cmap='Greys_r')
plt.show()
