from scipy.stats import ncx2
import numpy as np
import matplotlib.pyplot as plt

epsilon = 2.4
n_list = np.arange(2, 41)

lambda_val = 5.0      # example value

delta_min = []

for n in n_list:
    delta = 1 - ncx2.cdf(epsilon**2,
                         df=n,
                         nc=lambda_val)
    delta_min.append(delta)

plt.figure(figsize=(7,5))
plt.plot(n_list, delta_min, 'o-', linewidth=2)
plt.xlabel(r"$n$")
plt.ylabel(r"Minimum feasible $\delta$")
plt.grid(True)
plt.savefig('delta_n_chi.jpeg')