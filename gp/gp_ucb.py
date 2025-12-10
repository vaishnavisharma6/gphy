import numpy as np
import matplotlib.pyplot as plt
import GPy
import os
import sys
import math

from datagen import get_system
#GP-UCB Class


class gp_ucb:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k.K(self.X, self.X)
        return(pr_mean, pr_cov)
        
    
    def sample(self, mean, cov):
        beta = 0.5
        var = np.diag(cov).reshape(-1, 1)
        score = mean + (beta**0.5 * var)
        # print(np.shape(score))
        max_indices = np.where(score == score.max())[0]
        i = np.random.choice(max_indices)
        print(i)
        x = self.X[i]
        return(x, i)

    def cost(self, Q, x):
        c = np.transpose(x) @ Q @ x
        w = np.random.normal(loc = 0, scale = 1, size =1)
        y = c + w
        # print(w)
        return(y)
    

    def updation(self, Y_obs, X_obs):
            mean =  k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k.K(self.X, self.X) - k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k.K(X_obs, self.X)
            
            return(mean, cov)

#system parameters
n,m = 2,2
x_0 = np.array([0.2, 0.0])

T = 4
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
norm_folder = 'gp_norms_{}.'.format(name)
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
     
sigma_w = np.eye(n*T) * 0.1
sigma_z = np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T + N_T @ sigma_w @ N_T.T
traceSigma = np.trace(sigma_z)
tau = 2 * np.trace(sigma_z @ sigma_z)
b = (np.linalg.matrix_power(A, T) @ x_0) - x_f
epsilon = 0.001
delta = 0.01


def sample_u(CT, b, alpha, n):
    m, d = CT.shape  # C is m x d
    u_samples = []

    while len(u_samples) < n:
        # sample x in the ball ||x||^2 < alpha
        x = np.random.normal(size=m)
        x = x / np.linalg.norm(x)  # unit vector
        r = np.sqrt(alpha) * np.random.rand()**(1/m)
        x = r * x

        # solve for u: C^T u = x - b
        # u = np.linalg.pinv(C.T) @ (x - b)
        u = np.linalg.lstsq(CT, x - b, rcond=None)[0]

        u_samples.append(u)

    return np.array(u_samples)



b = np.linalg.matrix_power(A, T)@x_0 - x_f
alpha = epsilon**2 - b.T@b + (traceSigma/delta) - 2*epsilon*math.sqrt(traceSigma/delta)
print(alpha)
X =  sample_u(CT, b, alpha, 10) # set of inputs from constraint (only change)
k = GPy.kern.RBF(m*T)

print(X)
algo = gp_ucb(X)
X_obs = []
Y_obs = []
# reg = []
indices = []
mean, cov = algo.initialize(k)
print(mean)
# Y = algo.cost(X)
# yb = np.max(Y)
plt.figure(figsize = (10, 6))
# sr = 0
N = np.arange(0, np.shape(X)[0])
for t in range(0,100):
    x = algo.sample(mean, cov)
    X_obs.append(x)
    y, i = algo.cost(Q, x)
    indices.append(i)
    Y_obs.append(y)
    # regret = yb - y
    # sr = sr + regret
    # reg.append(sr)
    mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
    plt.fill_between(N, mean[:, 0] - 1*np.sqrt(np.diag(cov)), mean[:, 0] + 1*np.sqrt(np.diag(cov)), alpha=0.2, color='blue')
    plt.plot(N, mean, color = 'red')
    plt.scatter(indices, Y_obs, color ='black')
    plt.xlabel('X')
    plt.ylabel('Mean function')
    plt.legend()
    plt.pause(0.4)
    plt.savefig('gp_ucb.jpeg')
    
plt.figure(figsize=(10, 6))
N = np.arange(0, len(reg))
plt.semilogy(N, reg, label = 'regret')
plt.legend()
plt.savefig('regret.jpeg')
