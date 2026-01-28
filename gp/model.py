import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from datagen import get_system
from scipy.linalg import solve
import math
import cvxpy as cp
from scipy.optimize import least_squares
import os


def quad_ls_symmetric(U, y):
    """
    Least squares assuming Q is symmetric
    """
    N, d = U.shape
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    p = len(idx)

    X = np.zeros((N, p))

    for n in range(N):
        row = []
        for (i, j) in idx:
            if i == j:
                row.append(U[n, i] ** 2)
            else:
                row.append(2 * U[n, i] * U[n, j])
        X[n] = row

    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    Q_hat = np.zeros((d, d))
    for k, (i, j) in enumerate(idx):
        Q_hat[i, j] = theta[k]
        Q_hat[j, i] = theta[k]

    return Q_hat


def quad_ridge(U, y, lam=1e-3):
    """
    Ridge-regularized symmetric LS
    """
    N, d = U.shape
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    p = len(idx)

    X = np.zeros((N, p))

    for n in range(N):
        row = []
        for (i, j) in idx:
            if i == j:
                row.append(U[n, i] ** 2)
            else:
                row.append(2 * U[n, i] * U[n, j])
        X[n] = row

    A = X.T @ X + lam * np.eye(p)
    b = X.T @ y
    theta = np.linalg.solve(A, b)

    Q_hat = np.zeros((d, d))
    for k, (i, j) in enumerate(idx):
        Q_hat[i, j] = theta[k]
        Q_hat[j, i] = theta[k]

    return Q_hat


def quad_psd(U, y):
    """
    PSD-constrained least squares
    """
    N, d = U.shape
    Q = cp.Variable((d, d), PSD=True)

    loss = 0
    for i in range(N):
        loss += (cp.quad_form(U[i], Q) - y[i]) ** 2

    prob = cp.Problem(cp.Minimize(loss))
    prob.solve()

    return Q.value


def quad_nuclear(U, y, lam=1e-2):
    """
    Low-rank estimation via nuclear norm
    """
    N, d = U.shape
    Q = cp.Variable((d, d), symmetric=True)

    loss = 0
    for i in range(N):
        loss += (cp.quad_form(U[i], Q) - y[i]) ** 2

    prob = cp.Problem(cp.Minimize(loss + lam * cp.normNuc(Q)))
    prob.solve()

    return Q.value

def quad_factorized(U, y, r, lr=1e-2, steps=1000):
    """
    Nonconvex low-rank PSD estimation
    """
    N, d = U.shape
    L = np.random.randn(r, d)

    for _ in range(steps):
        grad = np.zeros_like(L)
        for i in range(N):
            Lu = L @ U[i]
            err = Lu @ Lu - y[i]
            grad += 4 * err * np.outer(Lu, U[i])
        L -= lr * grad / N

    Q_hat = L.T @ L
    return Q_hat

def quad_huber(U, y, delta=1.0):
    """
    Robust symmetric estimation with Huber loss
    """
    N, d = U.shape
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    p = len(idx)

    def residual(theta):
        Q = np.zeros((d, d))
        for k, (i, j) in enumerate(idx):
            Q[i, j] = theta[k]
            Q[j, i] = theta[k]
        return np.array([U[i] @ Q @ U[i] - y[i] for i in range(N)])

    theta0 = np.zeros(p)
    res = least_squares(residual, theta0, loss='huber', f_scale=delta)

    Q_hat = np.zeros((d, d))
    for k, (i, j) in enumerate(idx):
        Q_hat[i, j] = res.x[k]
        Q_hat[j, i] = res.x[k]

    return Q_hat

def sample_u(C, b, gamma, n, nullscale=1.0):
   
    m, d = C.shape
    b = b.reshape(-1)

    # SVD for nullspace
    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    rank = np.linalg.matrix_rank(C)
    N = Vt[rank:].T   # nullspace basis

    C_pinv = np.linalg.pinv(C)

    u_samples = []

    while len(u_samples) < n:
        # 1. sample y uniformly from ball ||y||^2 <= gamma
        z = np.random.randn(m)
        z = z / np.linalg.norm(z)
        # r = np.random.rand() ** (1.0 / m)     # correct radius scaling
        y = np.sqrt(gamma)  * z

        # 2. particular solution
        u_p = C_pinv @ (y - b)

        # 3. add bounded nullspace component
        if N.size > 0:
            w = nullscale * np.random.randn(N.shape[1])
            u = u_p + N @ w
        else:
            u = u_p

        # 4. safety check (numerical)
        if (C @ u + b).T @ (C @ u + b) <= gamma + 1e-8:
            u_samples.append(u)

    return np.array(u_samples)

def cost(Q, T, U):
    Q_tilde = block_diag(*([Q] * T))
    sigma = 0.1
    y = np.einsum('ni,ij,nj->n', U, Q_tilde, U)
    noise   = np.random.normal()
    return (y+ noise)



n,m = 2,2
x_0 = np.array([0.0, 0.0])

T_list  = range(1, 30, 2)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 0.85
delta = 0.9
j = 10000
norm = []
chebu = []

for T in T_list:
    A, B, CT = get_system(name, n, m, T,  seed=0)

    u = np.random.randn(m*T, 1)

    x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + CT @ u
    x_f =  x_f.reshape(-1)

    N_T = np.eye(n)

    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

        
    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T
    
    traceSigma = np.trace(sigma_z)
  
    b = (np.linalg.matrix_power(A, T)@x_0) - x_f
    alpha = epsilon**2  + (traceSigma/delta) - 2*epsilon*math.sqrt(traceSigma/delta)

    U = sample_u(CT, b, alpha, j)
    y = cost(Q, T, U)

    proj = lambda u: u / np.linalg.norm(u)
    u_star, min_cost = unconstrained_min_cost(U, y)
    print("Approx. min cost:", min_cost)
    norm.append(min_cost)
    

    Q_tilde = block_diag(*([Q] * T))
    u = cp.Variable(m*T)
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    f = cp.quad_form(CT @ u + b, np.eye(CT.shape[0]))
    prob = cp.Problem(objective, [f <= alpha])
    prob.solve(solver=cp.SCS, verbose=False)

    uc = u.value 
    uc     = u.value.reshape(-1, 1)
    cheb_norm = float(uc.T @ Q_tilde @ uc)
    chebu.append(cheb_norm)

plt.figure(figsize=(10, 6))
plt.plot(T_list, norm, 'o-', label = 'model')
plt.plot(T_list, chebu, 'o-', label = 'analytical')
plt.xlabel('Horizon')
plt.ylabel('u* norm')
plt.xticks(T_list)
plt.grid(True)
plt.legend()
plt.savefig('model.jpeg')    

