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

U = np.linspace(-5, 5, 3)[:, None]
S = np.linspace(-5, 5, 3)[:, None]
U_grid, S_grid = np.meshgrid(U, S)

points = np.column_stack([U_grid.ravel(), S_grid.ravel()])
print(points)

x = np.linspace(-10, 100, 1000)
y = 3* x**3 + 2
plt.plot(x, y)
plt.show()