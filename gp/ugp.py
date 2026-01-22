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
        # ratio = (gamma * np.linalg.norm(xn- xk)+ np.linalg.norm(x_f - xn))/const
        ratio = np.linalg.norm(xn - xk)/np.linalg.norm(x_f - xn)
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

    x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + CT @ u
    x_f =  x_f.reshape(-1)

    N_T = np.eye(n)

    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

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
        w =  np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        xT = np.linalg.matrix_power(A, T)@ x_0 + CT @ np.concatenate(final_u[::-1]) + N_T @ w
        s.append(xT)
        deviation = deviation + np.linalg.norm(x_f - xT)
    deviation = deviation /sim       
    dev.append(deviation)
    norm.append(np.linalg.norm(final_u))
    s = np.array(s)
    filename4 = os.path.join(folder, 'on_state.jpeg')
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
 
N = np.arange(0, len(dev))
plt.figure(figsize = (10,6))
plt.plot(N, dev, label = 'online')
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.savefig('udev-T.jpeg')
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, norm, label = 'online')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.savefig('unorm-T.jpeg')
plt.close()    



#GP training then prediction

#set of inputs u
#cost values with str and u cost
#states achieved
# final posterior

class train_gphy:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k1):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k1.K(self.X, self.X)
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
        # ratio = (gamma * np.linalg.norm(xn- xk)+ np.linalg.norm(x_f - xn))/const
        ratio = np.linalg.norm(xn - xk)/np.linalg.norm(x_f - xn)
        final = y +  ratio
        return(final)

    def updation(self, Y_obs, X_obs):
            mean =  k1.K(self.X, X_obs) @ np.linalg.inv(k1.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k1.K(self.X, self.X) - k1.K(self.X, X_obs) @ np.linalg.inv(k1.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k1.K(X_obs, self.X)
            return(mean, cov)

U = []
X = []
y = []
nsim = 100
t = 4
A, B, CT = get_system(name, n, m, t, seed = 0)
u = np.random.randn(m*t)
x_f = np.linalg.matrix_power(A, t) @ x_0.reshape(-1, 1) + CT @ u
x_f.reshape(-1)
x_0 = np.array([0.0, 0.0])
#generate training data
for i in range(0, nsim):
        u = np.random.randn(m)
        U.append(u)
        w = np.random.normal(n)
        xn = A @ x_0 + B @ u + w
        X.append([x_0, xn])
        c = algo.cost(Q, u, T)
        fc = algo.str(x_0, xn, x_f, c)
        y.append(fc)
        x_0 = xn
   
k1 = GPy.kern.RBF(m)
U = np.vstack(U)
model = train_gphy(U)
pr_mean, pr_cov = model.initialize(k1)
mean, cov = model.updation(y, U)


T_list = range(1, 30)
norm1 = []
dev1 = []

deviation = 0
for T in T_list:
    Y_obs = []
    X_obs = []
    s = []
    A, B, CT = get_system(name, n, m, T,  seed=0)
    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    for i in range(0, T):
        u, i = model.sample(mean, cov)
        X_obs.append(u)
        c = model.cost(Q, u, T)
        wk = np.random.normal(n)
        xn = A @ x_0 + B @ u + wk
        fy = model.str(x_0, xn, x_f, c)
        Y_obs.append(fy)
        mean, cov = model.updation(np.array(Y_obs), np.array(X_obs))
        x_0 = xn
    norm1.append(np.linalg.norm(X_obs))
    X_obs = np.asarray(X_obs).reshape(-1)
    nsim = 2000
    for i in range(0, nsim):
        w = np.random.normal(size = n*T)
        print(np.shape(CT))
        print(np.shape(X_obs))
        xT = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + CT @ X_obs[::-1] + N_T @ w
        s.append(xT)
        deviation = deviation + np.linalg.norm(x_f - xT)
    deviation = deviation/nsim
    dev1.append(deviation)
    s = np.array(s)
    # filename3 = os.path.join(folder, 'train_state.jpeg')
    # plt.figure(figsize = (10,6))
    # plt.scatter(s[:, 0], s[:, 1], marker = 'x')
    # plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
   
    # plt.gca().set_aspect('equal', adjustable='box')
    # circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
    #                     fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    # plt.gca().add_patch(circle)
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.title('horizon = {}'.format(T))
    # plt.legend()
    # plt.savefig(filename3)
    

N = np.arange(0, len(norm))
plt.figure(figsize = (10,6))
plt.plot(N, norm1, label = 'batch')
plt.plot(N, norm, label = 'online')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.savefig('train_norm.jpeg')
plt.close()

plt.figure(figsize = (10, 6))
plt.plot(N, dev1, label = 'batch')
plt.plot(N, dev, label = 'online')
plt.xlabel('time horizon')
plt.ylabel('final state deviation')
plt.legend()
plt.savefig('train_dev.jpeg')
plt.close()