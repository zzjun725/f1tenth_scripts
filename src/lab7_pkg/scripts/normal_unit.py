import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from numba import njit
import time


normal_span = 2.5
normal_interpScale = 15

# @njit(fastmath=False, cache=True)
def get_span_from_two_point(span_L=0.3, interp_num=10, pA=None, pB=None):
    # vector from A to B
    x_axisVector = np.array([1.0, 0.0])
    AB_Vector = pB - pA
    AB_th = np.arccos(np.dot(AB_Vector, x_axisVector) / (LA.norm(AB_Vector, ord=2)*1.0))
    # print(np.dot(AB_Vector, x_axisVector))
    # print(LA.norm(AB_Vector, ord=2))
    # print(AB_th*180/np.pi)
    AB_interpX = np.linspace(pA[0], pB[0], num=interp_num)
    AB_interpY = np.linspace(pA[1], pB[1], num=interp_num)
    AB_interp = np.vstack([AB_interpX, AB_interpY])
    AB_span1 = np.vstack([AB_interpX + span_L*np.cos(AB_th-np.pi/2), AB_interpY + span_L*np.sin(AB_th-np.pi/2) ])
    AB_span2 = np.vstack([AB_interpX + span_L*np.cos(AB_th+np.pi/2), AB_interpY + span_L*np.sin(AB_th+np.pi/2) ])
    return np.hstack([AB_interp, AB_span1, AB_span2])

def get_normal_from_two_point(normal_L=normal_span, interp_num = normal_span*normal_interpScale, pA=None, pB=None):
    x_axisVector = np.array([1.0, 0.0])
    AB_Vector = pB - pA
    AB_midpoint = (pB+pA) / 2
    AB_th = np.arccos(np.dot(AB_Vector, x_axisVector) / (LA.norm(AB_Vector, ord=2)*1.0))
    AB_norm1 = np.array([AB_midpoint[0] + normal_L*np.cos(AB_th-np.pi/2), AB_midpoint[1] + normal_L*np.sin(AB_th-np.pi/2)])
    AB_norm2 = np.array([AB_midpoint[0] + normal_L*np.cos(AB_th+np.pi/2), AB_midpoint[1] + normal_L*np.sin(AB_th+np.pi/2)])
    norm_interp_X = np.linspace(AB_norm1[0], AB_norm2[0], num=int(interp_num))
    norm_interp_Y = np.linspace(AB_norm1[1], AB_norm2[1], num=int(interp_num))
    norm_interp = np.vstack([norm_interp_X, norm_interp_Y])
    # print(interp_num)
    # print(norm_interp.shape)
    return norm_interp

def find_max_gap(block_mask):
    cur_gap, i, max_gap = 0, 0, 0
    n = len(block_mask)
    while i < n:
        if block_mask[i]:
            i += 1
        else:
            cur_gap = 0
            while i < n and not block_mask[i]:
                cur_gap += 1
                i += 1
            max_gap = max(max_gap, cur_gap)
    return max_gap


# pA = np.array([0.0, 0.0])
# pB = np.array([3.0, 1.0])

# spanPoints = get_span_from_two_point(pA=pA, pB=pB)
# normPoints = get_normal_from_two_point(pA=pA, pB=pB)
# plt.plot(spanPoints[0], spanPoints[1], '-o')
# plt.plot(normPoints[0], normPoints[1], 'ro')
# plt.axis('equal')
# plt.show()

# mask = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
# max_gap = 

def generateDir(leftRanges, rightRanges):
    left = []; right = [];
    for l in leftRanges:
        left.extend(list(range(*l)))
    for r in rightRanges:
        right.extend(list(range(*r)))
    return left, right

# generateDir([(0, 8), (13, 25), (31, 45)], [(8, 13), (25, 31), (45, 55)])
# set(list([[1, 3], [2]]))
