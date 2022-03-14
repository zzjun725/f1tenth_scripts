import numpy as np
import matplotlib.pyplot as plt

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
print(wp.shape)
plt.plot(-wp[:, 1], wp[:, 0])
plt.show()
