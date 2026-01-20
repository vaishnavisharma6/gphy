import numpy as np
import matplotlib.pyplot as plt
from datagen import get_system
from scipy.linalg import block_diag
from scipy.stats import norm
import GPy
import math

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
        r = np.random.rand() ** (1.0 / m)     # correct radius scaling
        y = np.sqrt(gamma) * r * z

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

class gp_ucb:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k.K(self.X, self.X)
        return(pr_mean, pr_cov)
    
    def sample_ucb(self, mean, cov, beta=0.5):
        var = np.diag(cov).reshape(-1,1)
        score = mean + np.sqrt(beta) * var
        max_indices = np.where(score == score.max())[0]
        i = np.random.choice(max_indices)
        return self.X[i], i

    def sample_pi_min(self, mean, cov, Y_obs, xi=0.01):
        f_best = np.min(Y_obs)
        var = np.diag(cov).reshape(-1,1)
        std = np.sqrt(np.maximum(var, 1e-9))
        z = (f_best - mean - xi) / std
        score = norm.cdf(z)
        max_indices = np.where(score == score.max())[0]
        i = np.random.choice(max_indices)
        return self.X[i], i

    def sample_ei_min(self, mean, cov, Y_obs, xi=0.01):
        f_best = np.min(Y_obs)
        var = np.diag(cov).reshape(-1,1)
        std = np.sqrt(np.maximum(var, 1e-9))
        z = (f_best - mean - xi) / std
        score = (f_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        max_indices = np.where(score == score.max())[0]
        i = np.random.choice(max_indices)
        return self.X[i], i

    def sample_thompson(self, mean, cov):
        sample = np.random.multivariate_normal(mean.flatten(), cov)
        max_indices = np.where(sample == sample.max())[0]
        i = np.random.choice(max_indices)
        return self.X[i], i

    def sample_uncertainty(self, mean, cov):
        var = np.diag(cov).reshape(-1,1)
        max_indices = np.where(var == var.max())[0]
        i = np.random.choice(max_indices)
        return self.X[i], i

    
    def cost(self, Q, x, T):
        Q_tilde = block_diag(*([Q] * T))
        x = x.reshape(-1,1)
        c = np.transpose(x) @ Q_tilde @ x
        w = np.random.normal()
        y = c + w
        return(y.flatten())
    

    def updation(self, Y_obs, X_obs):
            mean =  k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k.K(self.X, self.X) - k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k.K(X_obs, self.X)
            return(mean, cov)



n,m = 2,2
x_0 = np.array([0.0, 0.0])

T_list  = range(1, 40)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 0.85
delta = 0.9
dev = []
state_norm = []
ratios = []
j = 500


methods = ['ucb', 'ei', 'pi', 'thompson', 'uncertainty']

# dictionaries to store results for each T
final_u_norms_T = {method: [] for method in methods}
avg_state_devs_T = {method: [] for method in methods}

sim = 3000  # number of simulations for final state

for T in T_list:

    # --- system setup as in your existing code ---
    A, B, CT = get_system(name, n, m, T, seed=0)
    u = np.random.randn(m*T, 1)

    x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + CT @ u
    x_f = x_f.reshape(-1)

    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)

    b = (np.linalg.matrix_power(A, T) @ x_0) - x_f
    alpha = epsilon**2 + (traceSigma/delta) - 2*epsilon*math.sqrt(traceSigma/delta)

    k = GPy.kern.Poly(m*T)
    X = sample_u(CT, b, alpha, j)

    # --- loop over acquisition strategies ---
    for method in methods:

        algo = gp_ucb(X)
        X_obs = []
        Y_obs = []
        indices = []

        mean, cov = algo.initialize(k)

        for t_iter in range(50):
            if len(Y_obs) == 0:
                x, i = algo.sample_ucb(mean, cov)
            else:
                if method == 'ucb':
                    x, i = algo.sample_ucb(mean, cov)
                elif method == 'ei':
                    x, i = algo.sample_ei_min(mean, cov, Y_obs)
                elif method == 'pi':
                    x, i = algo.sample_pi_min(mean, cov, Y_obs)
                elif method == 'thompson':
                    x, i = algo.sample_thompson(mean, cov)
                elif method == 'uncertainty':
                    x, i = algo.sample_uncertainty(mean, cov)

            X_obs.append(x)
            y = algo.cost(Q, x, T)
            Y_obs.append(y)
            indices.append(i)

            mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
    
        # --- final metrics for this method and T ---
        index = indices[-1]
        u_star = X[index]
        final_u_norms_T[method].append(np.linalg.norm(u_star))

        # simulate final state deviation
        deviation = 0
        for t_sim in range(sim):
            w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
            xT = np.linalg.matrix_power(A, T) @ x_0 + CT @ u_star + N_T @ w
            deviation += np.linalg.norm(xT - x_f)
        deviation /= sim
        avg_state_devs_T[method].append(deviation)


plt.figure(figsize=(10,6))
for method in methods:
    plt.plot(T_list, final_u_norms_T[method], marker='o', label=method)
plt.xlabel('Time horizon T')
plt.ylabel('||u*||')
plt.title('Final control input norm vs T')
plt.legend()
plt.grid(True)
plt.savefig('norm.jpeg')


plt.figure(figsize=(10,6))
for method in methods:
    plt.plot(T_list, avg_state_devs_T[method], marker='o', label=method)
plt.xlabel('Time horizon T')
plt.ylabel('Average state deviation')
plt.title('Average state deviation vs T')
plt.legend()
plt.grid(True)
plt.savefig('dev.jpeg')
