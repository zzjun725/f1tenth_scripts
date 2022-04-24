
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
 
 
def f(x):
    return 1 / (1 + x ** 2)
 
 
def cal(begin, end, i):
    by = f(begin)
    ey = f(end)
    i = ms[i] * ((end - n) ** 3) / 6 + ms[i + 1] * ((n - begin) ** 3) / 6 + (by - ms[i] / 6) * (end - n) + (ey - ms[i + 1] / 6) * (n - begin)
    return i
 
 
def ff(x): # f[x0, x1, ..., xk]
    ans = 0
    for i in range(len(x)):
    temp = 1
    for j in range(len(x)):
    if i != j:
    temp *= (x[i] - x[j])
    ans += f(x[i]) / temp
    return ans
 
 
def calm():
 lam = [1] + [1 / 2] * 9
 miu = [1 / 2] * 9 + [1]
 # y = 1 / (1 + n ** 2)
 # df = diff(y, n)
 x = np.array(range(11)) - 5
 # ds = [6 * (ff(x[0:2]) - df.subs(n, x[0]))]
 ds = [6 * (ff(x[0:2]) - 1)]
 for i in range(9):
 ds.append(6 * ff(x[i: i + 3]))
 # ds.append(6 * (df.subs(n, x[10]) - ff(x[-2:])))
 ds.append(6 * (1 - ff(x[-2:])))
 mat = np.eye(11, 11) * 2
 for i in range(11):
 if i == 0:
  mat[i][1] = lam[i]
 elif i == 10:
  mat[i][9] = miu[i - 1]
 else:
  mat[i][i - 1] = miu[i - 1]
  mat[i][i + 1] = lam[i]
 ds = np.mat(ds)
 mat = np.mat(mat)
 ms = ds * mat.i
 return ms.tolist()[0]
 
 
def calnf(x):
 nf = []
 for i in range(len(x) - 1):
 nf.append(cal(x[i], x[i + 1], i))
 return nf
 
 
def calf(f, x):
 y = []
 for i in x:
 y.append(f.subs(n, i))
 return y
 
 
def nfsub(x, nf):
 tempx = np.array(range(11)) - 5
 dx = []
 for i in range(10):
 labelx = []
 for j in range(len(x)):
  if x[j] >= tempx[i] and x[j] < tempx[i + 1]:
  labelx.append(x[j])
  elif i == 9 and x[j] >= tempx[i] and x[j] <= tempx[i + 1]:
  labelx.append(x[j])
 dx = dx + calf(nf[i], labelx)
 return np.array(dx)
 
 
def draw(nf):
 plt.rcparams['font.sans-serif'] = ['simhei']
 plt.rcparams['axes.unicode_minus'] = false
 x = np.linspace(-5, 5, 101)
 y = f(x)
 ly = nfsub(x, nf)
 plt.plot(x, y, label='原函数')
 plt.plot(x, ly, label='三次样条插值函数')
 plt.xlabel('x')
 plt.ylabel('y')
 plt.legend()
 
 plt.savefig('1.png')
 plt.show()
 
 
def losscal(nf):
 x = np.linspace(-5, 5, 101)
 y = f(x)
 ly = nfsub(x, nf)
 ly = np.array(ly)
 temp = ly - y
 temp = abs(temp)
 print(temp.mean())
 
 
if __name__ == '__main__':
 x = np.array(range(11)) - 5
 y = f(x)
 
 n, m = symbols('n m')
 init_printing(use_unicode=true)
 ms = calm()
 nf = calnf(x)
 draw(nf)
 losscal(nf)
