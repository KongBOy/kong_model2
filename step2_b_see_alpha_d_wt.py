import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

split = 30

alpha_max = 0.1
alpha = np.linspace(0, alpha_max, split)
alpha_map = np.tile(alpha, (split, 1))

d = np.arange(split) / split
d_map = np.tile(d, (split, 1)).T

w_fold_map = alpha_map / (d_map + alpha_map + 0.000001)
w_curl_map = 1 - (d_map + 0.01) ** alpha_map


fig = plt.figure(0)
ax = Axes3D(fig)
ax.set_xlabel("d")
ax.set_ylabel("alpha")
ax.set_zlabel("w_fold")
# ax.scatter(d_map, alpha_map, w_fold_map)
ax.bar3d(d_map.ravel(), alpha_map.ravel(), np.zeros(split**2), 1 / split, alpha_max / split, w_fold_map.ravel())


fig = plt.figure(1)
ax = Axes3D(fig)
ax.set_xlabel("d")
ax.set_ylabel("alpha")
ax.set_zlabel("w_curl")
# ax.scatter(d_map, alpha_map, w_fold_map)
ax.bar3d(d_map.ravel(), alpha_map.ravel(), np.zeros(split**2), 1 / split, alpha_max / split, w_curl_map.ravel())
plt.show()

plt.show()
