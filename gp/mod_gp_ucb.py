import numpy as np
import matplotlib.pyplot as plt
import math
import os
import GPy
import matplotlib.patches as patches
from scipy.linalg import block_diag
from datagen import get_system

# ===================== SAMPLING =====================

def sample_u(C, b, gamma, n, nullscale=1.0):
    m, d = C.shape
    b = b.reshape(-1)

    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    rank = np.linalg.matrix_rank(C)
    N = Vt[rank:].T

    C_pinv = np.linalg.pinv(C)
    u_samples = []

    while len(u_samples) < n:
        z = np.random.randn(m)
        z = z / np.linalg.norm(z)
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

# ===================== GP-UCB =====================

class gp_ucb:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k.K(self.X, self.X)
        return pr_mean, pr_cov
    
    def sample(self, mean, cov):
        beta = 0.5
        var = np.diag(cov).reshape(-1, 1)
        score = -mean + np.sqrt(beta) * var
        i = np.random.choice(np.where(score == score.max())[0])
        return self.X[i], i

    def cost(self, Q, x, T):
        Q_tilde = block_diag(*([Q] * T))
        x = x.reshape(-1, 1)
        c = x.T @ Q_tilde @ x
        return (c + np.random.normal()).flatten()

    def updation(self, Y_obs, X_obs):
        mean = k.K(self.X, X_obs) @ np.linalg.inv(
            k.K(X_obs, X_obs) + np.eye(len(X_obs))
        ) @ Y_obs
        cov = k.K(self.X, self.X) - k.K(self.X, X_obs) @ np.linalg.inv(
            k.K(X_obs, X_obs) + np.eye(len(X_obs))
        ) @ k.K(X_obs, self.X)
        return mean, cov

# ===================== NEW: TRAJECTORY SIMULATION =====================

def simulate_full_trajectory(A, B, x0, u_seq, sigma_w):
    T = u_seq.shape[0]
    n = x0.shape[0]

    x = np.zeros((T + 1, n))
    x[0] = x0

    for t in range(T):
        w = np.random.multivariate_normal(np.zeros(n), sigma_w)
        x[t+1] = A @ x[t] + B @ u_seq[t] + w

    return x

# ===================== MAIN =====================

n, m = 2, 2
x_0 = np.array([0.0, 0.0])
T_list = range(1, 20)
Q = np.diag([2.0, 3.0])
epsilon = 0.85
delta = 0.9
j = 1000

dev, norm, ratios = [], [], []

for T in T_list:

    A, B, CT = get_system('stable', n, m, T, seed=0)

    u = np.random.randn(m*T, 1)
    x_f = (np.linalg.matrix_power(A, T) @ x_0.reshape(-1,1) + CT @ u).flatten()

    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)

    b = np.linalg.matrix_power(A, T) @ x_0 - x_f
    alpha = epsilon**2 + traceSigma/delta - 2*epsilon*np.sqrt(traceSigma/delta)

    k = GPy.kern.Poly(m*T)
    X = sample_u(CT, b, alpha, j)

    algo = gp_ucb(X)
    mean, cov = algo.initialize(k)

    X_obs, Y_obs, indices = [], [], []

    for _ in range(700):
        x, i = algo.sample(mean, cov)
        X_obs.append(x)
        Y_obs.append(algo.cost(Q, x, T))
        indices.append(i)
        mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))

    index = indices[-1]
    u_star = X.reshape(j, T, m)[index]

    # ===================== MONTE CARLO TRAJECTORIES =====================

    sim = 3000
    trajectories = []

    for _ in range(sim):
        traj = simulate_full_trajectory(
            A, B, x_0, u_star, sigma_w=np.eye(n)*0.1
        )
        trajectories.append(traj)

    trajectories = np.array(trajectories)  # (sim, T+1, n)

    # ===================== PLOT ALL INTERMEDIATE STATES =====================

    plt.figure(figsize=(7, 7))

    for t in range(T+1):
        plt.scatter(
            trajectories[:, t, 0],
            trajectories[:, t, 1],
            s=5,
            alpha=0.3
        )

    plt.scatter(x_f[0], x_f[1], color='red', label='target')

    circle = patches.Circle(
        (x_f[0], x_f[1]),
        epsilon,
        fill=False,
        linestyle='--',
        color='k',
        label='terminal ε-ball'
    )
    plt.gca().add_patch(circle)

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'All intermediate states (T={T})')
    plt.legend()
    plt.savefig(f'all_states_T{T}.jpeg')
    plt.close()
