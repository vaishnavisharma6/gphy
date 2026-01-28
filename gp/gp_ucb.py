import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.linalg import block_diag
import os
import sys
import math
from functools import reduce
import matplotlib.patches as patches
from matplotlib.patches import Circle
import cvxpy as cp

from datagen import get_system
#GP-UCB Class

def compute_W_from_CT(C_T, tol=1e-8):

    U, S, Vt = np.linalg.svd(C_T, full_matrices=False)

    # determine numerical rank
    r = np.sum(S > tol)

    # right singular vectors
    W = Vt[:r].T   # shape (n, r)

    return W, S[:r]

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



class gp_ucb:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k.K(self.X, self.X)
        return(pr_mean, pr_cov)
        
    
    def sample(self, mean, cov, t, j, d):
        c = 3
        # beta = 2 * np.log((j * t**2 * np.pi**2)/d*6)
        beta = 0
        # beta = c * np.log(t+1)
        var = np.sqrt(np.diag(cov).reshape(-1, 1))
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
        sigma_noise = 0.5
        y = x.T @ Q_tilde @ x + np.random.normal(0, sigma_noise)
        return(y.flatten())
    

    def updation(self, Y_obs, X_obs):
            mean =  k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k.K(self.X, self.X) - k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k.K(X_obs, self.X)
            return(mean, cov)


class gp_st:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k1):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k1.K(self.X, self.X)
        return(pr_mean, pr_cov)
        
    
    def sample(self, mean, cov, t, j, d):
        c = 3
        beta = 2 * np.log((j * t**2 * np.pi**2)/d*6)
        # beta = 0
        # beta = c * np.log(t+1)
        var = np.sqrt(np.diag(cov).reshape(-1, 1))
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
        sigma_noise = 0.5
        y = x.T @ Q_tilde @ x + np.random.normal(0, sigma_noise)
        return(y.flatten())
    

    def updation(self, Y_obs, X_obs):
            mean =  k1.K(self.X, X_obs) @ np.linalg.inv(k1.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k1.K(self.X, self.X) - k1.K(self.X, X_obs) @ np.linalg.inv(k1.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k1.K(X_obs, self.X)
            return(mean, cov)


#algorithm 1: Find set of feasible inputs and using BO (GP-UCB), obtain MECI

#system parameters
n,m = 2,2
x_0 = np.array([0.0, 0.0])

T_list  = range(1, 50, 5)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 0.85
delta = 0.9
delta_folder = 'traj_{}'.format(delta)
chebu = []

os.makedirs(delta_folder, exist_ok = True)
dev = []
dev1 = []
dev2 = []
norm = []
norm1 = []
norm2 = []
ratios = []
ratios1 = []
ratios2 = []
# reg = []
j = 200
plt.figure(figsize = (10,6))
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
    
    W, S = compute_W_from_CT(CT)
    C_tilde = CT @ W
    r = W.shape[1]

    M = np.linalg.pinv(C_tilde.T @ C_tilde)   # r x r
    L = np.linalg.cholesky(M)

    X_st_raw = sample_u(C_tilde, b, alpha, j)   # true reduced controls
    X_st_tilde = X_st_raw @ L.T 

    k1 = GPy.kern.Poly(input_dim=r, variance=1.0, scale=1.0, bias=1.0, order=2)  
    k2 = GPy.kern.Poly(input_dim=r, variance=1.0, scale=1.0, bias=1.0, order=2)  

    k = GPy.kern.Poly(input_dim=m*T, variance=1.0, scale=1.0, bias=1.0, order=2)  
 
    X =  sample_u(CT, b, alpha, j) # set of inputs from constraint (only change)
    X_st = sample_u(C_tilde, b, alpha, j)

    algo = gp_ucb(X)
    algo1 = gp_st(X_st)
    algo2 = gp_st(X_st_tilde)
    X_obs = []
    Y_obs = []
    X_obs1 = []
    Y_obs1 = []
    indices = []
    indices1 = []
  
    X_obs2, Y_obs2 = [], []
    indices2 = []
    reg2 = []
    cr2 = 0

    mean, cov = algo.initialize(k)
    mean1, cov1 = algo1.initialize(k1)
    mean2, cov2 = algo2.initialize(k2)
    s = []
    s1 = []
    x1 = []
    x1st = []
    x2 = []
    reg = []
    reg1 = []
    d = 0.1
    Q_tilde = block_diag(*([Q] * T))
    X_full = X_st @ W.T
    values = np.einsum('ij,jk,ik->i', X, Q_tilde, X)
    values1 = np.einsum('ij,jk,ik->i', X_full, Q_tilde, X_full)
    cr = 0
    cr1 = 0
    min_value = np.min(values)
    min_value1 = np.min(values1)
    for t in range(0,100):
        x, i = algo.sample(mean, cov, t, j, d)
        xst, i1 = algo1.sample(mean1, cov1, t, j, d)
        xst_tilde, i2 = algo2.sample(mean2, cov2, t, j, d)
        x_full = W @ xst
        xst_metric = np.linalg.solve(L.T, xst_tilde)
        x_full2 = W @ xst_metric
        X_obs.append(x)
        X_obs1.append(xst)
        X_obs2.append(xst_tilde)
        y = algo.cost(Q, x, T)
        y1 = algo1.cost(Q, x_full, T)
        y2 = algo2.cost(Q, x_full2, T)
        indices.append(i)
        indices1.append(i1)
        indices2.append(i2)
        Y_obs.append(y)
        Y_obs1.append(y1)
        Y_obs2.append(y2)
        mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
        mean1, cov1 = algo1.updation(np.array(Y_obs1), np.array(X_obs1))
        mean2, cov2 = algo2.updation(np.array(Y_obs2), np.array(X_obs2))
        regret = (x.T @ Q_tilde @ x) - min_value
        regret1 = (x_full.T @ Q_tilde @ x_full) - min_value1
        regret2 = (x_full2.T @ Q_tilde @ x_full2) - min_value1
        cr = cr + regret
        cr1 = cr1 + regret1
        cr2 = cr2 + regret2
        # print(regret)
        reg.append(cr)
        reg1.append(cr1)
        reg2.append(cr2)
    index = indices[-1]    
    index1 = indices1[-1]
    index2 = indices2[-1]
    N = np.arange(0, len(reg))
    if T%5 == 0:
    
        # plt.semilogy(N, reg, label = 'offline')
        # plt.semilogy(N, reg1, label = 'constraint')
        plt.semilogy(N, reg2, label = 'kernel')
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend()
        plt.savefig('reg.jpeg')
    
    u_r = np.linalg.solve(L.T, X_st_tilde[index2, :])
    u_metric = W @ u_r
    # print(index)
    if T == 1:
        xstar = W @ X_st[index, :]
        plt.figure(figsize = (10,6))
        plt.plot(X[:, 0], X[:, 1], 'x')
        plt.plot(X_st[:, 0], X_st[:, 1], 'x', label = 'structured')
        plt.plot(X_st_tilde[:, 0], X_st_tilde[:, 1], 'x', label = 'geometry')
        plt.plot(X[indices, 0], X[indices, 1], 'o', label = 'sampled')
        plt.plot(X_st_tilde[indices2, 0], X_st_tilde[indices2, 1], 'o', label = ' geometric sampled')
        plt.plot(X_st[indices1, 0], X_st[indices1, 1], 'o', label = 'structured sampled')
        plt.plot(X[index, 0], X[index, 1], 'o', color = 'red', label = 'u*')
        plt.plot(u_metric[0], u_metric[1], 'o', color = 'black', label = 'u* geometric')
        plt.plot(xstar[0], xstar[1], 'o', color = 'yellow', label = 'u* structured')
        plt.xlabel('u_0')
        plt.ylabel('u_1')
        plt.title('Feasible set of u for T = 1')
        plt.legend()
        plt.savefig('inputs.jpeg')
        plt.close()
    u = cp.Variable(m*T)
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    f = cp.quad_form(CT @ u + b, np.eye(CT.shape[0]))
    prob = cp.Problem(objective, [f <= alpha])
    prob.solve(solver=cp.SCS, verbose=False)

    uc = u.value 
    uc     = u.value.reshape(-1, 1)
    cheb_norm = float(uc.T @ Q_tilde @ uc)
    chebu.append(cheb_norm)     
    #final state distribution
    deviation = 0
    deviation1 = 0
    deviation2 = 0
    sim = 3000
    count1 = 0
    count2 = 0
    count3 = 0
   
    for t in range(0, sim):
        w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        xT = np.linalg.matrix_power(A, T)@ x_0 + CT @ X[index, :] + N_T @ w
        xT1 = np.linalg.matrix_power(A, T)@x_0 + CT @ W @ X_st[index1, :] + N_T @ w
        xT2 = np.linalg.matrix_power(A, T)@x_0 + CT @ u_metric + N_T @ w
        s.append(xT)
        s1.append(xT1)
        x1.append(np.linalg.norm(xT-x_f))
        x1st.append(np.linalg.norm(xT1 - x_f))
        x2.append(np.linalg.norm(xT2 - x_f))
        deviation = deviation + np.linalg.norm(x_f - xT)
        deviation1 = deviation1 + np.linalg.norm(x_f - xT1)
        deviation2 = deviation2 + np.linalg.norm(x_f - xT2)
    deviation = deviation /sim       
    deviation1 = deviation1/sim
    deviation2 = deviation2/sim
    dev.append(deviation)
    dev1.append(deviation1)
    dev2.append(deviation2)

    count1 = sum(d <= epsilon for d in x1)
    count2 = sum(d <= epsilon for d in x1st)
    count3 = sum(d <= epsilon for d in x2)
    ratios.append(count1/sim)
    ratios1.append(count2/sim)
    ratios2.append(count3/sim)
 
   
    norm.append(X[index, :].reshape(T, m)[::-1].reshape(-1).T @ Q_tilde @ X[index, :].reshape(T, m)[::-1].reshape(-1))
    norm1.append((W @ X_st[index1, :].T) @ Q_tilde @  (W @ X_st[index1, :]))
    norm2.append(u_metric.T @ Q_tilde @ u_metric)


N = np.arange(0, len(ratios))
plt.figure(figsize = (10, 6))
plt.plot(N, ratios2, label = 'geometry aware')
plt.plot(N, ratios1, label = 'structured')
plt.plot(N, ratios, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('final state ratio inside epsilon ball')
plt.legend()
plt.grid(True)
plt.savefig('ratio.jpeg')
plt.close()

# #--------------------------------------------------------------------


# --- Plotting the Results ---
plt.figure(figsize=(10, 6))
plt.semilogy(T_list, norm1, 'x-', label='Structure-Aware gp', color='blue')
plt.semilogy(T_list, norm, 'x-', label = 'offline gp')
plt.semilogy(T_list, norm2, 'x-', label = 'kernel')
plt.semilogy(T_list, chebu, 'x-', label='analytical', color='red')
plt.xlabel('Time Horizon (T)')
plt.ylabel('log(Minimum Energy Cost)')
plt.title('Minimum Energy Control Cost vs Time Horizon')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('energy_vs_T.png')
plt.show()


N = np.arange(0, len(dev))
plt.figure(figsize = (10,6))
plt.plot(T_list, dev2, label = 'geometry aware')
plt.plot(T_list, dev1, label = 'structure aware')
plt.plot(T_list, dev, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.grid(True)
plt.savefig('dev-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()


