import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.linalg import block_diag
import os
import sys
import math

from datagen import get_system
#GP-UCB Class

def sample_u(CT, b, alpha, n):
    a, d = CT.shape  # C is m x d
    u_samples = []

    while len(u_samples) < n:
        # sample x in the ball ||x||^2 < alpha
        x = np.random.normal(size=a)
        x = x / np.linalg.norm(x)  # unit vector
        r = np.sqrt(alpha) 
        x = r * x

        # solve for u: C^T u = x - b
        # u = np.linalg.pinv(C.T) @ (x - b)
        u = np.linalg.lstsq(CT, x - b, rcond=None)[0]

        u_samples.append(u)

    return np.array(u_samples)

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
        score = -1*mean + (beta**0.5 * var)
        max_indices = np.where(score == score.max())[0]
        i = np.random.choice(max_indices)
        x = self.X[i]
        return(x, i)

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




#system parameters
n,m = 2,2
x_0 = np.array([0.2, 0.0])

T_list  = range(1, 100)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'

dev = []
norm = []
for T in T_list:

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
    epsilon = 0.5
    delta = 0.99
    b = np.linalg.matrix_power(A, T)@x_0 - x_f
    alpha = epsilon**2  + (traceSigma/delta) - 2*epsilon*math.sqrt(traceSigma/delta)
    k = GPy.kern.RBF(m*T)
   
    plt.figure(figsize = (10, 10))
    j = 2000
    X =  sample_u(CT, b, alpha, j) # set of inputs from constraint (only change)
    
    algo = gp_ucb(X)
    X_obs = []
    Y_obs = []
    indices = []
    mean, cov = algo.initialize(k)
 

    N = np.arange(0, np.shape(X)[0])
    for t in range(0,50):
        x, i = algo.sample(mean, cov)
        X_obs.append(x)
        y = algo.cost(Q, x, T)
        indices.append(i)
        Y_obs.append(y)
        mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
    index = indices[-1]    
    deviation = 0
    sim = 100
    for t in range(0, sim):
        w = np.random.normal(size = m*T)
        xT = np.linalg.matrix_power(A, T)@ x_0 + CT @ X[index, :] + N_T @ w
        deviation = deviation + np.linalg.norm(x_f - xT)
    deviation = deviation /sim       
    dev.append(deviation)
    norm.append(np.linalg.norm(X[index]))
    
N = np.arange(0, len(dev))
plt.figure(figsize = (10,6))
plt.plot(N, dev)
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.savefig('dev-T-{}-{}.jpeg'.format(epsilon, delta))


plt.figure(figsize = (10,6))
plt.plot(N, norm)
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.savefig('norm-T-{}-{}.jpeg'.format(epsilon, delta))
    



  

