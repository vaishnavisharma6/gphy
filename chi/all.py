import numpy as np
import matplotlib.pyplot as plt
import math
import os
from datagen import get_system
#code to visualize ellipsoid for all constraints obtained.

#para
name = 'stable'
n = 2
m = 2
T = 1
A, B, C_T = get_system(name, n, m, T, seed=None) 
x_0 = np.array([0.0, 0.0])
U = np.random.randn(m * T, 1)
x_f = x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + C_T @ U
x_f =  x_f.reshape(-1)
b = np.linalg.matrix_power(A, T) @ x_0 - x_f
N_T = np.eye(n)
for i in range(1, T):
    N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
sigma_w = np.eye(n) * 0.1    
sigma_z = N_T @ sigma_w @ N_T.T
delta = 0.9

eps = 1.0

# Quadratic-ellipsoid parameters
A_mat = C_T.T @ np.linalg.inv(sigma_z) @ C_T
f = C_T.T @ np.linalg.inv(sigma_z) @ b.reshape(-1,1)
c = eps**2 - b.T @ np.linalg.inv(sigma_z) @ b

A_mat = np.array(A_mat, dtype=float)
f = np.array(f, dtype=float)
c = float(c)

# Center of ellipsoid
A_inv = np.linalg.inv(A_mat)
u0 = -A_inv @ f   # center

gamma = c + (f.T @ A_inv @ f).item()
Q = gamma * A_inv   # shape matrix

# Ellipsoid boundary
theta = np.linspace(0, 2*np.pi, 300)
circle = np.vstack((np.cos(theta), np.sin(theta)))

eigvals, eigvecs = np.linalg.eigh(Q)
axes_lengths = np.sqrt(eigvals)

ellip = (eigvecs @ np.diag(axes_lengths) @ circle) + u0

# ------------------------------------------------------------
# 1) Chebyshev ball: ||u|| <= eps
# ------------------------------------------------------------
cheby = eps * circle

# ------------------------------------------------------------
# 2) Second-order set: ||C_T u + b|| <= eps
# ------------------------------------------------------------
# boundary curve parameterization:
v = np.linspace(0, 2*np.pi, 300)
unit_v = np.vstack((np.cos(v), np.sin(v)))

# Solve C_T u + b = eps * v  -->  u = C_T^{-1}(eps v - b)
# Only works if C_T is invertible (2x2 in your system)
C_inv = np.linalg.inv(C_T)
soc = C_inv @ (eps * unit_v - b.reshape(-1,1))

# ------------------------------------------------------------
# Plot all three
# ------------------------------------------------------------
plt.figure(figsize=(7,7))

plt.plot(ellip[0], ellip[1], 'b', label='Quadratic Ellipsoid')
plt.scatter(u0[0], u0[1], c='b')

plt.plot(cheby[0], cheby[1], 'g', label='Chebyshev Ball')

plt.plot(soc[0], soc[1], 'r', label='Second-Order Set')

plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title("Comparison of Sets in Control Space")
plt.show()

