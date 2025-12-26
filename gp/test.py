import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import math

T = 6
n, m = 2,2
Q = np.diag([2.0, 3.0])
Q_diag = block_diag(*([Q] * T))


def cost(Q_diag, u):
    c = u.T @ Q_diag @ u
    return(c)


y = []
for i in range(0, 1000):
    u = np.random.randn(m*T, 1)
    c = cost(Q_diag, u)
    c = c.reshape(-1)
    y.append(c)

values = [float(a) for a in y]

print(values)

N = np.arange(0, len(values))
plt.figure(figsize = (10, 6))
plt.plot(N, values, 'o')
plt.show()
