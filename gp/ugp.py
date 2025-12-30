import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.linalg import block_diag
import os
import sys
import math
import matplotlib.patches as patches
from matplotlib.patches import Circle

from datagen import get_system


class gphy:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k3):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k3.K(self.X, self.X)
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
        # Q_tilde = block_diag(*([Q] * T))
        x = x.reshape(-1,1)
        c = np.transpose(x) @ Q @ x
        w = np.random.normal()
        y = c + w
        return(y.flatten())
    
    def str(self, xk, xn, x_f, y):
        gamma = 0.8
        l = 0.5
        const = np.linalg.norm(x_f - x_0)
        ratio = (gamma * np.linalg.norm(xn- xk)+ np.linalg.norm(x_f - xn))/const
        final = y + l* ratio
        return(final)

    def updation(self, Y_obs, X_obs):
            mean =  k3.K(self.X, X_obs) @ np.linalg.inv(k3.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k3.K(self.X, self.X) - k3.K(self.X, X_obs) @ np.linalg.inv(k3.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k3.K(X_obs, self.X)
            return(mean, cov)




#system parameters
n,m = 2,2
x_0 = np.array([0.0, 0.0])

T_list  = range(1, 30)
# sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 0.85
delta = 0.9
# delta_folder = 'traj_{}'.format(delta)
folder = 'traj_un'
# idelta_folder = 'traj_{}'.format(delta)
os.makedirs(folder, exist_ok = True)

dev = []
norm = []  
# j = 1000
for T in T_list:
    A, B, CT = get_system(name, n, m, T,  seed=0)
   
    u = np.random.randn(m*T, 1)
    # CT = B
    # for i in range(1, T):
    #     CT = np.hstack((CT, np.linalg.matrix_power(A, i) @ B)) 

    x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + CT @ u
    x_f =  x_f.reshape(-1)

    N_T = np.eye(n)

    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

        
    # sigma_w = np.eye(n*T) * 0.1
    # sigma_z = np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T + N_T @ sigma_w @ N_T.T
    # traceSigma = np.trace(sigma_z)
  
    # b = np.linalg.matrix_power(A, T)@x_0 - x_f
    # alpha = epsilon**2  + (traceSigma/delta) - 2*epsilon*math.sqrt(traceSigma/delta)
    k3 = GPy.kern.RBF(m)
    U = np.random.randn(500, m)
    
    U = U.reshape(U.shape[0], -1)
    # X =  sample_u(CT, b, alpha, j) # set of inputs from constraint (only change)
    
    algo = gphy(U)
  
    indices = []
    final_u = []
    mean, cov = algo.initialize(k3)
    s = []
    xk = x_0
    for k in range(0, T):
        X_obs = []
        Y_obs = []
        for t in range(0,100):
            x, i = algo.sample(mean, cov)
            X_obs.append(x)
            wk = np.random.normal(n)
            xn = A @xk + B @ x + wk
            y = algo.cost(Q, x, T)
            indices.append(i)
            fy = algo.str(xk, xn, x_f, y)
            Y_obs.append(fy)
            mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
        index = indices[-1] 
        final_u.append(X_obs[-1])  
        wk = np.random.normal(n)
        xk = A @ x_0 + B @ X_obs[-1] + wk 
        x_0 = xk
    final_u = [x.tolist() for x in final_u]    
   
    
    #record metrics----------------------------------------
    deviation = 0
    sim = 2000
    for t in range(0, sim):
        w = np.random.normal(size = n*T)
        xT = np.linalg.matrix_power(A, T)@ x_0 + CT @ np.concatenate(final_u[::-1]) + N_T @ w
        s.append(xT)
        deviation = deviation + np.linalg.norm(x_f - xT)
    deviation = deviation /sim       
    dev.append(deviation)
    norm.append(np.linalg.norm(final_u))
    s = np.array(s)
    filename4 = os.path.join(folder, 'ufstate.jpeg')
    plt.figure(figsize = (10,6))
    plt.scatter(s[:, 0], s[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
   
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('horizon = {}'.format(T))
    plt.legend()
    plt.savefig(filename4)
    plt.close()
N = np.arange(0, len(dev))
plt.figure(figsize = (10,6))
plt.plot(N, dev, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.savefig('udev-T.jpeg')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, norm, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.savefig('unorm-T.jpeg')
plt.close()    



#transfer learning
