import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


data = np.loadtxt('pts-BasicSimplePendula.txt')

# plt.gca().set_aspect('equal', adjustable='box')
data = data[::10]


ts = [d[0] for d in data]
xs = [d[1] for d in data]
ys = [d[2] for d in data]


ax.plot(xs, ys, ts)
min_b = np.min([plt.xlim()[0], plt.ylim()[0]])
max_b = np.max([plt.xlim()[1], plt.ylim()[1]])
plt.xlim((min_b, max_b))
plt.ylim((min_b, max_b))
plt.show()
