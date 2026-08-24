from scipy.stats import ncx2
import numpy as np
import matplotlib.pyplot as plt

epsilon_1 = 0.7
epsilon_2 = 1.4
epsilon_3 = 2.1

n_list = np.arange(2, 41)
lambda_list = np.arange(2, 41)

lambda_val = 5     # example value

delta_min1 = []
delta_min2 = []
delta_min3 = []

for n in n_list:
    delta1 = 1 - ncx2.cdf(epsilon_1**2,
                         df=n,
                         nc=lambda_val)
    delta2 = 1 - ncx2.cdf(epsilon_2**2, df=n, nc=lambda_val)
    delta3 = 1 - ncx2.cdf(epsilon_3**2, df=n, nc=lambda_val)
    delta_min1.append(delta1)
    delta_min2.append(delta2)
    delta_min3.append(delta3)

plt.figure(figsize=(7,5))
plt.plot(n_list, delta_min1, 'o-', linewidth=2, label =r'$\epsilon$ = {}'.format(epsilon_1))
plt.plot(n_list, delta_min2, 'o-', linewidth =2, label =r'$\epsilon$ = {}'.format(epsilon_2))
plt.plot(n_list, delta_min3, 'o-', linewidth =2, label =r'$\epsilon$ = {}'.format(epsilon_3))
plt.xlabel(r"$n$")
plt.ylabel(r"Minimum feasible $\delta$")
plt.grid(True)
plt.legend()
plt.savefig('delta_n_chi.jpeg')

delta_min1 = []
delta_min2 = []
delta_min3 = []
n = 4
for l in lambda_list:
    delta1 = 1- ncx2.cdf(epsilon_1**2,
                         df=n,
                         nc=l)
    delta2 = 1 - ncx2.cdf(epsilon_2**2, df=n, nc=l)
    delta3 = 1 - ncx2.cdf(epsilon_3**2, df=n, nc=l)
    delta_min1.append(delta1)
    delta_min2.append(delta2)
    delta_min3.append(delta3)

plt.figure(figsize=(7,5))
plt.plot(lambda_list, delta_min1, 'o-', linewidth=2, label =r'$\epsilon$ = {}'.format(epsilon_1))
plt.plot(lambda_list, delta_min2, 'o-', linewidth =2, label =r'$\epsilon$ = {}'.format(epsilon_2))
plt.plot(lambda_list, delta_min3, 'o-', linewidth =2, label =r'$\epsilon$ = {}'.format(epsilon_3))
plt.xlabel(r"$\lambda$")
plt.ylabel(r"Minimum feasible $\delta$")
plt.grid(True)
plt.legend()
plt.savefig('delta_lambda_chi.jpeg')    
    
    