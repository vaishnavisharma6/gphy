import numpy as np
import matplotlib.pyplot as plt
import math
import os
from datagen import get_system

n,m = 2,2
x_0 = np.array([0.2, 0.0])

T = 4
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
norm_folder = 'gp_norms_{}'.format(name)
state_folder = 'gp_states_{}'.format(name)
os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)

A, B, CT = get_system(name, n, m, T, seed=2)
u = np.random.randn(m*T, 1)

x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + CT @ u
x_f =  x_f.reshape(-1)

N_T = np.eye(n)

for i in range(1, T):
     N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
b = np.linalg.matrix_power(A, T)@x_0 - x_f     

AT = np.linalg.matrix_power(A, T)
u_det = np.linalg.pinv(CT) @(x_f - AT@x_0)

u_det = u_det.reshape(T,m)

xp = x_0
xpd = xp
xps = xp
d = []
s = []
rng = np.random.default_rng()
#compute state trajectory
plt.figure(figsize = (10,6))
plt.scatter(x_f[0], x_f[1], label = 'desired_state')
for j in range(0,1000):
    d = []
    s = []
    d = [x_0]
    s = [x_0]
    for i in range(0, T):
        w = rng.normal()
        print(w)
        xnd = A @ xpd + B @ u_det[i]
        d.append(xnd.reshape(-1))
        xs = A @ xps + B @ u_det[i] + w
        s.append(xs.reshape(-1))
        xpd = xnd
        xps = xs

    d = np.array(d)
    s = np.array(s)
    
    plt.scatter(d[-1,0], d[-1, 1], marker = 'x')
    plt.scatter(s[-1,0], s[-1, 1], marker = 'o')

    plt.legend()
plt.savefig('traj.jpeg')     