import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib.patches as patches
import math
import os
from datagen import get_system
from functools import reduce
from plot import opt, diff
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib.patches import Circle
from scipy.stats import ncx2
from scipy.optimize import brentq


n, m = 2, 2
x_0 = np.array([0.0, 0.0])
sigma_0 = np.eye(n) * 0.1
errors = []

delta = 0.9
Q = np.diag([2.0, 3.0])
# T = 10
name = 'stable'
gamma = 0.1
# N = 21
M = 1000

T_list = range(1, 40, 5)
data_folder = 'data_{}'.format(name)
os.makedirs(data_folder, exist_ok = True)
dev = []
dev1 = []
cost = []
cost1 = []

for T in T_list:

    A, B, CT = get_system(name, n, m, T, seed = 0)
    NT = np.eye(n)
    for i in range(1, T):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    N = (2* T) + 1
    u = np.random.randn(m * T, 1)
    Q_tilde = block_diag(*([Q] * T))
    # Compute reachable final state
    x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + NT @ u
    x_f =  x_f.reshape(-1, 1)

    U  = np.random.rand(m*T, N)

    mu = np.zeros((n, n))
    var_sum = np.zeros((n, n))
    X = np.zeros((n, n))
    for i in range(0, N):
        sum = np.zeros(n)
        for j in range(0, M):
            w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
            xT = CT @ U[:, i] + NT @ w
            if i == N - 1:
                X = np.hstack((X, xT.reshape(-1,1)))
            sum = sum + xT
        mu_i = (1/M) * sum   
        mu = np.hstack((mu, mu_i.reshape(-1, 1))) 


    for j in range(0, M):
        var_sum = var_sum + (X[:, n+j]- mu_i) @ (X[:, n+j] - mu_i).T
    Sigma = (1/M) * var_sum
    
    b =  -1 *  x_f
    b = b.reshape(-1, 1)
    sigma_w_base = np.eye(n*T) * 0.1
    # --------- CVXPY FIXED PART ---------

    alpha = cp.Variable((N, 1))
    u = cp.Variable((m*T, 1))
    # ---- Objective (make it DCP safe) ----
    P = U.T @ Q_tilde @ U
    P = 0.5 * (P + P.T)                    # enforce symmetry
    P = P + 1e-8 * np.eye(N)               # ensure PSD

    objective = cp.Minimize(cp.quad_form(alpha, P))
    objective1 = cp.Minimize(cp.quad_form(u, Q_tilde))

    # ---- Constraint (Mahalanobis term) ----
    mu_eff = mu[:, n:]                     # your intended slice
    x_f_col = x_f.reshape(-1, 1)

    Sigma_reg = Sigma + 1e-6 * np.eye(n)   # stabilize
    Sigma_inv = np.linalg.inv(Sigma_reg)
    Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)  # enforce symmetry
    
    sigma_x = NT @ sigma_w_base @ NT.T
    f = cp.quad_form(mu_eff @ alpha - x_f_col, Sigma_inv)
    f1 = cp.quad_form(CT@ u + b, np.linalg.inv(sigma_x))

    # ---- Problem ----
    prob = cp.Problem(objective, [f <= gamma])
    prob1 = cp.Problem(objective1, [f1 <= gamma])
    prob.solve(solver=cp.SCS, verbose=False)
    prob1.solve(solver = cp.SCS, verbose=False)

    alpha_hat = alpha.value
    u_star = u.value.reshape(-1,1)
    errors.append(np.linalg.norm(U @ alpha_hat - u_star))

    if alpha_hat is None:
           print(f"Convex solver failed at T={T}")

    # -------------------------------------
   
    
    x1 = []
    x2 = []
    for j in range(0, 100):
        w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        x1T = CT @ U @ alpha_hat + NT @ w
        x2T = CT @ u_star + NT @ w
        x1.append(np.linalg.norm(x1T - x_f))
        x2.append(np.linalg.norm(x2T - x_f))

    hat_norm = float(alpha_hat.T @ U.T @ Q_tilde @ U @ alpha_hat)
    u_norm = float(u_star.T @ Q_tilde @ u_star)
    cost.append(hat_norm)
    cost1.append(u_norm)

    dev.append(np.average(x1))
    dev1.append(np.average(x2))

    center = (x_f[0], x_f[1])

    theta = np.linspace(0, 2*np.pi, 400)
    x = center[0] + gamma * np.cos(theta)
    y = center[1] + gamma * np.sin(theta)




    # filename1 = os.path.join(data_folder, 'data-{}-{}.jpeg'.format(delta, T))
    # plt.figure(figsize = (10,6))
    # plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
    # plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    # plt.plot(x, y)
    # plt.scatter(*center)
    # plt.gca().set_aspect('equal', adjustable='box')
    # circle = patches.Circle((x_f[0], x_f[1]), gamma, 
    #                     fill=False, linestyle='--', color='k', label=f'ε-ball (ε={gamma})')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # # plt.title('second order moment, horizon = {}, ratio = {}'.format(t, count1/2000))

    # plt.savefig(filename1)



plt.figure(figsize = (10, 6))
plt.plot(T_list, cost, label = 'data')
plt.plot(T_list, cost1, label = 'true')
plt.xlabel('time horizon')
plt.ylabel('cost')
plt.legend()
plt.savefig('cost_data.jpeg')

plt.figure(figsize = (10, 6))
plt.plot(T_list, dev, label = 'deviation')
plt.plot(T_list, dev1, label = 'true')
plt.xlabel('time horizon')
plt.ylabel('dev')
plt.legend()
plt.savefig('data_dev.jpeg')    

plt.figure(figsize = (10, 6))
plt.plot(T_list, errors, label = 'errors')
plt.xlabel('time horizon')
plt.ylabel('error')
plt.legend()
plt.savefig('error.jpeg')