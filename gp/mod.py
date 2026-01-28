import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.linalg import block_diag
import os
import math
import matplotlib.patches as patches
import cvxpy as cp


from datagen import get_system

class PolyMetricKernel(GPy.kern.Kern):
    def __init__(self, input_dim, M, variance=1.0, bias=1.0, order=2):
        super().__init__(input_dim, active_dims=None, name='poly_metric')
        self.M = M
        self.variance = variance
        self.bias = bias
        self.order = order

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        G = X @ self.M @ X2.T
        return self.variance * (G + self.bias) ** self.order

    def Kdiag(self, X):
        G = np.sum((X @ self.M) * X, axis=1)
        return self.variance * (G + self.bias) ** self.order



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
        beta = 2 * np.log((j * t**2 * np.pi**2)/d*6)
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
        sigma_noise = 1.0
        y = x.T @ Q_tilde @ x + np.random.normal(0, sigma_noise)
        return(y.flatten())
    

    def updation(self, Y_obs, X_obs):
            mean =  k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k.K(self.X, self.X) - k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k.K(X_obs, self.X)
            return(mean, cov)
    

n,m = 2,2
x_0 = np.array([0.0, 0.0])
chebu = []
T_list  = range(1, 50, 1)
sigma_0 = np.eye(n) *0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 0.85
delta = 0.9
delta_folder = 'traj_{}'.format(delta)

os.makedirs(delta_folder, exist_ok = True)
dev = []
norm = []
ratios = []
# reg = []
j = 600
plt.figure(figsize = (10,6))



for T in T_list:
    A, B, CT = get_system(name, n, m, T,  seed=0)
    # ---- kernel metric definition ----
    M = CT.T @ CT
    M += 1e-8 * np.eye(M.shape[0])   # numerical stability
    L = np.linalg.cholesky(M)        # M = L.T @ L


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
  


    # k = GPy.kern.Poly(input_dim=m*T, variance=1.0, scale=1.0, bias=1.0, order=2)  
    k = PolyMetricKernel(
    input_dim=m*T,
    M=M,
    variance=1.0,
    bias=1.0,
    order=2
     )
   
    
  
    X =  sample_u(CT, b, alpha, j) # set of inputs from constraint (only change)
    # X = (L @ X_raw.T).T 
    print(np.shape(X))
    algo = gp_ucb(X)
    X_obs = []
    Y_obs = []
    indices = []
    mean, cov = algo.initialize(k)
    s = []
    x1 = []
    reg = []
    d = 0.1
    Q_tilde = block_diag(*([Q] * T))
    # X_full = X @ W.T
    values = np.einsum('ij,jk,ik->i', X, Q_tilde, X)
    cr = 0
    min_value = np.min(values)
    for t in range(0,100):
        x, i = algo.sample(mean, cov, t, j, d)
        X_obs.append(x)
   
        y = algo.cost(Q, x, T)
        indices.append(i)
        Y_obs.append(y)
        mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
        regret = (x.T @ Q_tilde @ x) - min_value
        cr = cr + regret
        # print(regret)
        reg.append(cr)
    index = indices[-1]    
    N = np.arange(0, len(reg))
    if T%5 == 0:
    
        plt.plot(N, reg, label = '{}'.format(T))
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend()
        plt.savefig('reg.jpeg')
    
   
    # print(index)
    if T == 2:
        
        plt.figure(figsize = (10,6))
        plt.plot(X[:, 0], X[:, 1], 'x')
        plt.plot(X[indices, 0], X[indices, 1], 'o', label = 'sampled')
        plt.plot(X[index, 0], X[index, 1], 'o', color = 'red', label = 'u*')
        plt.xlabel('u_0')
        plt.ylabel('u_1')
        plt.title('Feasible set of u for T = 2')
        plt.legend()
        plt.savefig('inputs.jpeg')
        plt.close()

    N = np.arange(0, len(indices))
    plt.figure(figsize = (10, 6))
    plt.scatter(N, indices, marker = 'o')    
    plt.xlabel('time step')
    plt.ylabel('index')
    plt.legend()
    plt.savefig('indices_{}.jpeg'.format(T))
    plt.close()    

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
    print( X[index, :])

    norm.append((X[index, :].T) @ Q_tilde @  (X[index, :]))
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

    

N = np.arange(0, len(ratios))
plt.figure(figsize = (10, 6))
# plt.plot(N, ratios1, label = 'online')
plt.plot(N, ratios, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('final state ratio inside epsilon ball')
plt.legend()
plt.grid(True)
plt.savefig('ratio.jpeg')
plt.close()



plt.figure(figsize=(10, 6))
# plt.plot(T_list, final_energies, 'o-', label='Structure-Aware BO (u*TQ\'u*)', color='blue')
plt.plot(T_list, norm, label = 'offline')
plt.plot(T_list, chebu, '--', label='Theoretical Analytical Min', color='red')
plt.xlabel('Time Horizon (T)')
plt.ylabel('Minimum Energy Cost')
plt.title('Minimum Energy Control Cost vs Time Horizon')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('energy_vs_T.png')
plt.show()


N = np.arange(0, len(dev))
plt.figure(figsize = (10,6))
# plt.plot(T_list, dev1, label = 'structure aware')
plt.plot(T_list, dev, label = 'offline')
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.grid(True)
plt.savefig('dev-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()


