import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep
from torch import ne


left_up = np.array([9.65, 8.52])  # (x2, y2)
right_down = np.array([-13.7, -0.1])  # (x1, y1)
y1, y2 = right_down[1], left_up[1]  # y1 < y2
x1, x2 = right_down[0], left_up[0]  # x1 < x2
vertical = np.linspace(x1, x2, 500)
horizon = np.linspace(y1, y2, 250)
waypoints = []
for v in vertical:
    waypoints.append(np.array([v, y1]))
for h in horizon:
    waypoints.append(np.array([x2, h]))
for v in vertical[::-1]:
    waypoints.append(np.array([v, y2]))
for h in horizon[::-1]:
    waypoints.append(np.array([x1, h]))
wp = np.array(waypoints)
# print(wp.shape)
# plt.plot(-wp[:, 1], wp[:, 0])
# plt.show()

left_up = np.array([9.65, 8.52])  # (x2, y2)
right_down = np.array([-13.7, -0.1])  # (x1, y1)
y1, y2 = right_down[1], left_up[1]  # y1 < y2
x1, x2 = right_down[0], left_up[0]  # x1 < x2


# x = np.array([x1, 2*(x1+x2)/3, (x1+x2)/3, x2, x2, x2, x2, (x1+x2)/3, 2*(x1+x2)/3, x1, x1, x1, x1])
# y = np.array([y1, y1, y1,y1, (y1+y2)/3, 2*(y1+y2)/3, y2, y2, y2, y2, 2*(y1+y2)/3, (y1+y2)/3,y1])
x = np.array([x1, x2, x2])
y = np.array([y1, y1, y2])

phi = np.linspace(0, 2.*np.pi, 20)
r = 0.5 + np.cos(phi)
# x, y = r * np.cos(phi), r*np.sin(phi)

tck, u = splprep([x, y], s=0, k=2)
test_x = np.linspace(0,1, 1000)
new_points = splev(test_x, tck)
fig, ax = plt.subplots()
print(new_points[0])
ax.plot(new_points[0], new_points[1], 'r-')
plt.show()
