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
#GP-UCB Class

#unknown cost, known model

# def sample_u(C, b, gamma, n):
#     m, d = C.shape
#     u_samples = []

#     # step 1: sample y on sphere
#     while len(u_samples) < n:
#         z = np.random.randn(m)
#         y = np.sqrt(gamma) * (z / np.linalg.norm(z))

#         # step 2: particular solution
#         Cp = np.linalg.pinv(C)
#         u_p = Cp @ (y - b)

#         # step 3: nullspace
#         U, S, Vt = np.linalg.svd(C)
#         N = Vt[m:].T  # nullspace basis

#         w = np.random.randn(N.shape[1])
#         u = u_p + N @ w
#         v = (C @ u + b).T @ (C @ u + b)

#         u_samples.append(u)

#     return np.array(u_samples)

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


#algorithm 1: Find set of feasible inputs and using BO (GP-UCB), obtain MECI

#system parameters
n,m = 2,2
x_0 = np.array([0.0, 0.0])

T_list  = range(1, 20)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 0.85
delta = 0.9
delta_folder = 'traj_{}'.format(delta)
idelta_folder = 'traj_{}'.format(delta)
os.makedirs(delta_folder, exist_ok = True)
dev = []
norm = []
ratios = []
j = 1000
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

        
    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)
  
    b = (np.linalg.matrix_power(A, T)@x_0) - x_f
    alpha = epsilon**2  + (traceSigma/delta) - 2*epsilon*math.sqrt(traceSigma/delta)
    # alpha = epsilon**2 - traceSigma
    k = GPy.kern.Poly(m*T)
   
   
    X =  sample_u(CT, b, alpha, j) # set of inputs from constraint (only change)

    # ======== PLOTTING FEASIBLE CONTROLS (NO CODE STRUCTURE CHANGED) ========

# X has shape (j, m*T)
# reshape safely
 # ======== SINGLE-PLOT FEASIBLE CONTROLS FOR ALL TIME STEPS ========

    X_seq = X.reshape(j, T, m)

    plt.figure(figsize=(7, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, T))

    for t_step in range(T):
        if t_step == 0 or t_step == T-1:
            plt.scatter(
                X_seq[:, t_step, 0],
                X_seq[:, t_step, 1],
                s=8,
                alpha=0.35,
                color=colors[t_step],
                label=f't={t_step}' if T <= 6 else None
            )

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel(r'$u^{(1)}$')
    plt.ylabel(r'$u^{(2)}$')
    plt.title(f'Feasible control inputs across all time steps (T={T})')

    if T <= 6:
        plt.legend()

    plt.savefig(f'feasible_inputs_all_t_T{T}.jpeg')
    plt.close()

# ==================================================================



    
    

    algo = gp_ucb(X)
    X_obs = []
    Y_obs = []
    indices = []
    mean, cov = algo.initialize(k)
    s = []
    x1 = []
    # Y = algo.cost(Q, X, Y)
    # model = GPy.models.GPRegression(X, Y)
    # model.optimize
    for t in range(0,700):
        x, i = algo.sample(mean, cov)
        X_obs.append(x)
        y = algo.cost(Q, x, T)
        indices.append(i)
        Y_obs.append(y)
        mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
    index = indices[-1]    
   
    if T == 1:
        
        plt.figure(figsize = (10,6))
        plt.plot(X[:, 0], X[:, 1], 'x')
        plt.plot(X[index, 0], X[index, 1], 'o')
        plt.savefig('inputs.jpeg')
        plt.close()
    #final state distribution
    deviation = 0
    sim = 3000
    count1 = 0
    for t in range(0, sim):
        w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        xT = np.linalg.matrix_power(A, T)@ x_0 + CT @ X[index, :] + N_T @ w
        s.append(xT)
        x1.append(np.linalg.norm(xT-x_f))
        deviation = deviation + np.linalg.norm(x_f - xT)
    deviation = deviation /sim       
    dev.append(deviation)
    count1 = sum(d <= epsilon for d in x1)
    ratios.append(count1/sim)
    norm.append(np.linalg.norm(X[index]))
    s = np.array(s)
    filename2 = os.path.join(delta_folder, 'fstate-{}-{}-{}.jpeg'.format(delta, epsilon, T))
    plt.figure(figsize = (10,6))
    plt.scatter(s[:, 0], s[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
   
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('horizon = {}, ratio = {}'.format(T, count1/sim))
    plt.legend()
    plt.savefig(filename2)
    plt.close()

N = np.arange(0, len(dev))
plt.figure(figsize = (10,6))
plt.plot(N, dev, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.savefig('dev-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, norm, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.savefig('norm-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()
    

N = np.arange(0, len(ratios))
plt.figure(figsize = (10, 6))
plt.plot(N, ratios)
plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('final state ratio inside epsilon ball')
plt.legend()
plt.savefig('off_ratio.jpeg')
plt.close()

#------------------------------------------------------------------------------------------
