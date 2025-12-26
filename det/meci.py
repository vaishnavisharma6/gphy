import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.linalg import block_diag
from datagen import get_system

n,m = 2,2
x_0 = np.array([0.0, 0.0])

t = 4

name = 'stable'
A, B= get_system(name, n, m, t, seed=2)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])

xp = x_0
xpd = xp
xps = xp
d = []
s = []
rng = np.random.default_rng()
#compute state trajectory
plt.figure(figsize = (10,6))
# plt.scatter(x_f[0], x_f[1], label = 'desired_state', marker = 'x', color = 'red')
plt.scatter(x_0[0], x_0[1], label = 'initial')
devd = []
devs = []
dn = []
sn = []
for T in range(1,100):
    u = np.random.randn(m*T, 1)
    AT = np.linalg.matrix_power(A, T)
    CT = B
    for i in range(1, T):
        CT = np.hstack((CT, np.linalg.matrix_power(A, i) @ B))
        
    Gramian = np.linalg.matrix_power(A, 0)@ B @ B.T @ np.linalg.matrix_power(A.T, 0)
    for i in range(0, T):
        Gramian = Gramian + np.linalg.matrix_power(A, i)@ B @ B.T @ np.linalg.matrix_power(A.T, i)
    a = np.random.randn(Gramian.shape[1])
    v = Gramian @ a

    x_f = AT @ x_0 + v

    N_T = np.eye(n)
    w = np.random.randn(n*T)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
    b = np.linalg.matrix_power(A, T)@x_0 - x_f     
    x = x_f - N_T @ w
    Q_tilde = block_diag(*([Q] * T))
    u_det = np.linalg.pinv(CT) @(x_f - AT@x_0)
    dn.append(u_det.T @ Q_tilde @ u_det)
    u_ss = np.linalg.pinv(CT) @ (x - AT@ x_0)
    sn.append(u_ss.T @ Q_tilde @ u_ss)
    u_det = u_det.reshape(T, m)
    u_ss = u_ss.reshape(T, m)
    
    w = w.reshape(T, n)
    d = []
    s = []
    d = [x_0]
    sd = [x_0]
    s = [x_0]

    for i in range(0, T):
        # xnd = A @ xpd + B @ u_det[i]
        
        # d.append(xnd)
        wk = np.random.randn(n)
        xs = A @ xp + B @ u_det[i] + wk
        xsd = A @xps + B @ u_ss[i] + w[i]
        s.append(xs)
        sd.append(xsd)
        # xpd = xnd
        xp = xs
        xps = xsd

    devd.append(np.sqrt(np.linalg.norm(s[-1]-x_f)))
    devs.append(np.sqrt(np.linalg.norm(sd[-1]-x_f)))
    d = np.array(d)
    print(np.shape(d))
    s = np.array(s)
    sd = np.array(sd)

    # plt.plot(d[-1, 0], d[-1, 1], marker = 'x')
    # plt.plot(s[:,0], s[:, 1], marker = 'o', label = 'deti-sto')
    plt.plot(s[-1, 0], s[-1, 1], marker = 'o', color = 'blue')
    plt.plot(sd[-1,0], sd[-1,1], marker = 'x', color = 'green')

  
    plt.legend()
plt.savefig('traj.jpeg')     

plt.figure(figsize = (10,6))
N = np.arange(0, len(devd))
plt.semilogy(N, devd, label = 'detu')
plt.semilogy(N, devs, label = 'stochu')
plt.xlabel('Time horizon')
plt.ylabel('state dev')
plt.legend()
plt.savefig('dev-T.jpeg')

plt.figure(figsize = (10, 6))
N = np.arange(0, len(sn))
plt.semilogy(N, sn, label = 'stoch norm')
plt.semilogy(N, dn, label = 'det norm')
plt.xlabel('Time horizon')
plt.ylabel('input norm')
plt.legend()
plt.savefig('norm.jpeg')