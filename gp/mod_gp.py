import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import block_diag
import GPy

from datagen import get_system

def sample_u_interior(C, b, gamma, n, nullscale=0.1):
    m, d = C.shape
    b = b.reshape(-1)

    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    rank = np.linalg.matrix_rank(C)
    N = Vt[rank:].T
    C_pinv = np.linalg.pinv(C)

    u_samples = []

    while len(u_samples) < n:
        z = np.random.randn(m)
        z /= np.linalg.norm(z)
        r = np.random.rand() ** (1.0 / m)
        y = np.sqrt(gamma) * r * z

        u_p = C_pinv @ (y - b)

        if N.size > 0:
            w = nullscale * np.random.randn(N.shape[1])
            u = u_p + N @ w
        else:
            u = u_p

        if (C @ u + b).T @ (C @ u + b) <= gamma + 1e-8:
            u_samples.append(u)

    return np.array(u_samples)


class gp_ucb:
    def __init__(self, domain, C, b, alpha, lambda_feas=0.05):
        self.X = domain
        self.C = C
        self.b = b.reshape(-1)
        self.alpha = alpha
        self.lambda_feas = lambda_feas
    
    def initialize(self, k):
        mean = np.zeros((len(self.X), 1))
        cov = k.K(self.X, self.X)
        return mean, cov
        
    def sample(self, mean, cov):
        beta = 0.5
        var = np.diag(cov).reshape(-1, 1)

        scores = []

        for i, x in enumerate(self.X):
            x = x.reshape(-1, 1)
            ucb = -mean[i] + np.sqrt(beta) * var[i]

            margin = self.alpha - (self.C @ x + self.b.reshape(-1,1)).T @ (self.C @ x + self.b.reshape(-1,1))

            if margin <= 0:
                score = -np.inf
            else:
                barrier = self.lambda_feas / (margin + 1e-6)
                score = ucb - barrier

            scores.append(score)

        scores = np.array(scores).flatten()
        idx = np.random.choice(np.where(scores == scores.max())[0])
        return self.X[idx], idx

    def cost(self, Q, x, T):
        Q_tilde = block_diag(*([Q] * T))
        x = x.reshape(-1,1)
        c = x.T @ Q_tilde @ x
        return (c + np.random.normal()).flatten()
    
    def updation(self, Y_obs, X_obs, k):
        Kxx = k.K(X_obs, X_obs) + 1e-6 * np.eye(len(X_obs))
        KxX = k.K(self.X, X_obs)

        mean = KxX @ np.linalg.inv(Kxx) @ Y_obs
        cov = k.K(self.X, self.X) - KxX @ np.linalg.inv(Kxx) @ KxX.T
        return mean, cov


# -------------------- system parameters --------------------
n, m = 2, 2
x_0 = np.zeros(n)
Q = np.diag([2.0, 3.0])
epsilon = 0.85
delta = 0.7
T_list = range(1, 20)
j = 500

dev, ratios, norms = [], [], []

for T in T_list:

    A, B, CT = get_system('stable', n, m, T, seed=0)

    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)

    x_f = np.zeros(n)
    b = np.linalg.matrix_power(A, T) @ x_0 - x_f

    alpha = (
        epsilon**2
        + traceSigma / delta
        - 2 * epsilon * math.sqrt(traceSigma / delta)
    )

    X = sample_u_interior(CT, b, alpha, j, nullscale=0.1)

    k = GPy.kern.Poly(m*T)
    algo = gp_ucb(X, CT, b, alpha, lambda_feas=0.05)

    X_obs, Y_obs, indices = [], [], []
    mean, cov = algo.initialize(k)

    for _ in range(700):
        x, idx = algo.sample(mean, cov)
        X_obs.append(x)
        Y_obs.append(algo.cost(Q, x, T))
        indices.append(idx)
        mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs), k)

    u_star = X[indices[-1]]
    norms.append(np.linalg.norm(u_star))

    # Monte-Carlo validation
    sim = 3000
    count = 0
    deviation = 0

    for _ in range(sim):
        w = np.random.normal(size=n*T)
        xT = np.linalg.matrix_power(A, T) @ x_0 + CT @ u_star + N_T @ w
        d = np.linalg.norm(xT - x_f)
        deviation += d
        if d <= epsilon:
            count += 1

    dev.append(deviation / sim)
    ratios.append(count / sim)


N = np.arange(len(dev))

plt.figure(figsize=(10,6))
plt.plot(N, dev, marker='o')
plt.xlabel('time horizon T')
plt.ylabel('average ||x_T - x_f||')
plt.title('Average final-state deviation vs horizon')
plt.grid(True)
plt.tight_layout()
plt.savefig('dev_vs_T_feasibility_aware_BO.jpeg')
plt.close()


plt.figure(figsize=(10,6))
plt.plot(N, ratios, marker='o')
plt.axhline(1 - delta, linestyle='--', color='k', label='1 - δ')
plt.xlabel('time horizon T')
plt.ylabel('P(||x_T - x_f|| ≤ ε)')
plt.title('Chance-constraint satisfaction vs horizon')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('chance_constraint_ratio_vs_T.jpeg')
plt.close()


plt.figure(figsize=(10,6))
plt.plot(N, norms, marker='o')
plt.xlabel('time horizon T')
plt.ylabel('||u*||')
plt.title('Optimal input norm vs horizon')
plt.grid(True)
plt.tight_layout()
plt.savefig('u_norm_vs_T.jpeg')
plt.close()


if T == 10:
    s = []
    for _ in range(3000):
        w = np.random.normal(size=n*T)
        xT = np.linalg.matrix_power(A, T) @ x_0 + CT @ u_star + N_T @ w
        s.append(xT)
    s = np.array(s)

    plt.figure(figsize=(6,6))
    plt.scatter(s[:,0], s[:,1], s=5, alpha=0.3)
    plt.scatter(x_f[0], x_f[1], c='red', marker='x', label='x_f')

    circle = patches.Circle(
        (x_f[0], x_f[1]), epsilon,
        fill=False, linestyle='--', color='k'
    )
    plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Final-state distribution (T=10)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('final_state_scatter_T10.jpeg')
    plt.close()
