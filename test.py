import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 10)
y = np.sin(x)
ylow = y - 0.2
yup = y + 0.2

plt.figure(figsize = (10, 6))
plt.fill_between(x, ylow, yup, alpha =0.2)
plt.plot(x, y)
plt.legend(labels = ["fill between"])
plt.show()