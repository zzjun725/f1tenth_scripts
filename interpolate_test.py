from operator import ne
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt


x = np.array([-13.53, -13.53, 0,  9.5920, 9.5920, 0])
y = np.array([-0.1233, 0.5883, 0.5883, 0.5883, -0.1233, -0.1233])

x2 = np.linspace(-15, 0, 20)

spline, u = splprep([x, y], s=0)

new_points = splev(u, spline)
print(len(new_points[0]))

fig, ax = plt.subplots()
ax.plot(x, y, 'ro')
ax.plot(new_points[0], new_points[1], 'ro')
plt.show()


