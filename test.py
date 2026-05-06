import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 10)
y = np.sin(x)
ylow = y - 0.2
yup = y + 0.2

# plt.figure(figsize = (10, 6))
# plt.fill_between(x, ylow, yup, alpha =0.2)
# plt.plot(x, y)
# plt.legend(labels = ["fill between"])
# plt.show()

# U = np.linspace(-5, 5, 3)[:, None]
# S = np.linspace(-5, 5, 3)[:, None]
# U_grid, S_grid = np.meshgrid(U, S)

# points = np.column_stack([U_grid.ravel(), S_grid.ravel()])
# print(points)

# x = np.linspace(-100, 100, 1000)
# y = -x**2 + 4*x
# plt.plot(x, y)
# plt.savefig('test.jpeg')

cols = []
m , n = 2,2
M = np.zeros((m, n))

col = np.array([1, 2]).reshape(-1, 1)
M = np.hstack((M, col))
print(M)