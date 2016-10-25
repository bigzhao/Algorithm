import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plt.axis([0, 100, 0, 1])
fig = plt.figure()
ax = Axes3D(fig)
plt.ion()

for i in range(100):
    y = np.random.random()
    ax.scatter(i, y, 0)
    plt.pause(0.1)