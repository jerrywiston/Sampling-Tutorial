import numpy as np
import matplotlib.pyplot as plt

total = 3000
count = 0

x_in, y_in, x_out, y_out = [], [], [], []
for i in range(total):
    rx = np.random.random()
    ry = np.random.random()

    if rx*rx + ry*ry < 1:
        count += 1
        x_in.append(rx)
        y_in.append(ry)
    else:
        x_out.append(rx)
        y_out.append(ry)

pi = 4 * count / total
print("PI:", pi)
plt.plot(x_in,y_in,'r.')
plt.plot(x_out,y_out,'b.')
plt.axis('equal')
plt.show()
