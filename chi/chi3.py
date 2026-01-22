import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from datagen import get_system
from functools import reduce
from plot import opt, diff
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib.patches import Circle
import GPy

#----------------------------------------------------------------------------------------
# fixed system parameters
n, m = 2, 2
x_0 = np.array([0.0, 0.0])

T_list = range(1, 15)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
delta = 0.9
Q = np.diag([2.0, 3.0])
q = norm.ppf(1 - delta)
u_norm_diff_list = []
state_dev_diff_list = []
name = 'stable'
norm_folder = 'norms_{}'.format(name)
state_folder = 'states_{}'.format(name)
diff_folder = 'diffs_{}'.format(name)

delta_folder = 'traj_{}'.format(delta)
epsilon = 0.85
os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)
os.makedirs(delta_folder, exist_ok = True)
chebd = []
secd = []
epd = []
firstd = []
modeld = []

chebu = []
secu = []
epu = []
firstu = []
modelu = []
staru = []
alleps = []

x1_dev = []
x2_dev = []
x3_dev = []
x4_dev = []
x5_dev = []

all_lam = []
g = []

ratios1 = []
ratios2 = []
ratios3 = []
ratios4 = []
ratios5 = []

def solve_lambda(A, b, gamma, tol=1e-8, max_iter=100):
    # 1. Eigenvalue Decomposition
    # A is CT @ inv(Q) @ CT.T
    eigenvalues, V = np.linalg.eigh(A)
    
    # 2. Transform b into the eigenvector space (b_tilde)
    b_tilde = V.T @ b
    b_sq = b_tilde**2
    
    # Check if a solution exists: gamma must be between the 'floor' and ||b||^2
    floor = np.sum(b_sq[eigenvalues < 1e-12])
    b_norm_sq = np.sum(b_sq)
    
    if gamma <= floor or gamma > b_norm_sq:
        return None, f"No solution: gamma={gamma} is outside range ({floor}, {b_norm_sq})"

    # 3. Newton-Raphson Solver
    # We want to find x where f(x) = sum(b_sq / (1 + x*delta)^2) - gamma = 0
    # Start with a guess (e.g., 0.1)
    lam = 0.1 
    
    for i in range(max_iter):
        denom = (1 + lam * eigenvalues)
        f = np.sum(b_sq / (denom**2)) - gamma
        
        # Derivative f'(x) = sum(-2 * delta * b_sq / (1 + x*delta)^3)
        f_prime = np.sum(-2 * eigenvalues * b_sq / (denom**3))
        
        new_lam = lam - f/f_prime
        
        # Convergence check
        if abs(new_lam - lam) < tol:
            return new_lam, "Success"
        
        lam = new_lam
        
    return lam, "Did not converge"


for t in T_list:
    A,B, NT = get_system(name, n, m, t, seed=0)
    U = np.random.randn(m * t, 1)

        # Compute reachable final state
    x_f = np.linalg.matrix_power(A, t) @ x_0.reshape(-1, 1) + NT @ U
    x_f =  x_f.reshape(-1)
   
    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = NT
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
        # C_T = np.hstack((C_T, np.linalg.matrix_power(A, i) @ B))

    # covariance and bias
    # D = np.random.randn(n*t, n*t)
    # sigma_w = D @ D.T
    sigma_w = np.eye(n*t) * 0.1
    sigma_z = np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T + N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)

    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = (np.linalg.matrix_power(A, t) @ x_0) - x_f
    b_cheb =  x_f - (np.linalg.matrix_power(A, t) @ x_0) 
    # convex constraint gamma
    gamma = epsilon**2 - traceSigma  #second order constraint
    r_cheb = epsilon**2 - 2* epsilon * math.sqrt(traceSigma/delta) + (traceSigma/delta)  #chebyshev
    dev = epsilon**2  #epsilon sd
    

    g.append(r_cheb)
    #choose proper epsilon
   
    
    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    u3 = cp.Variable(m*t)
    u4 = cp.Variable(m*t)
    Q_tilde = block_diag(*([Q] * t))
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))
    f = cp.quad_form(C_T @ u + b, np.eye(C_T.shape[0]))
    f2 = cp.quad_form(C_T @ u2 + b, np.eye(C_T.shape[0]))
    f3 = cp.quad_form(C_T @ u3 + b, np.linalg.inv(sigma_z))
    f4 = C_T @ u4 + b
    prob = cp.Problem(objective, [f <= gamma])
    prob2 = cp.Problem(objective2, [f2 <= r_cheb])
    prob3 = cp.Problem(objective3, [f3 <= dev])
    prob4 = cp.Problem(objective4, [f4 == 0])
    prob.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver = cp.SCS, verbose = False)
    prob3.solve(solver = cp.SCS, verbose = False)
    prob4.solve(solver = cp.SCS, verbose = False)
    u_convex = u.value  #second order
    u_c2 = u2.value #chebyshev
    u_c3 = u3.value #epsilon sd
    u_c4 = u4.value # first order
    # print('RHS, 2nd order:', gamma)
    # print('residue 2nd order:', (C_T @ u_convex + b).T @ (C_T @ u_convex + b))
    # print('RHS, chebyshev:', r_cheb)
    # print('residue, chebyshev:', (C_T @ u_c2 + b).T @ (C_T @ u_c2 + b))
    # print('RHS, epsilon sd:', dev)
    # print('residue, epsilon sd:', (C_T @ u_c3 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c3 + b))
    w = np.random.randn(n*t)
    x = x_f - N_T @ w
    AT = np.linalg.matrix_power(A, t)
    u_model =  np.linalg.pinv(C_T) @ (x - AT@ x_0)
    if u_convex is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c2 is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c3 is None:
        print(f'Convex solver failed at T={t}')
    if u_c4 is None:
        print(f'Convex solver failed at T={t}')
    #expression u
    lam, status = solve_lambda(C_T @ np.linalg.inv(Q_tilde) @ np.transpose(C_T), b, r_cheb, 1e-8, 100)
    
    u_star = -lam* np.linalg.inv(Q_tilde + lam * np.transpose(C_T) @ C_T) @ np.transpose(C_T) @ b
    all_lam.append(lam)
    # --- Simulate both systems ---
    np.random.seed(0)
   
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []

    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    for j in range(0, 2000):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_convex + N_T @ w  # second order
        x2T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c2 + N_T @ w  # chebyshev
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c3 + N_T @ w  # epsilon 
        x4T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c4 + N_T @ w
        x5T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_star + N_T @ w
        x1.append(np.linalg.norm(x1T-x_f))
        
        
        x2.append(np.linalg.norm(x2T-x_f))
      
       
        x3.append(np.linalg.norm(x3T-x_f))
   
       
        x4.append(np.linalg.norm(x4T - x_f))
      
     
        x5.append(np.linalg.norm(x5T - x_f))
        # count5 = sum(d <= epsilon for d in x5)
        
        s1.append(x1T)
        s2.append(x2T)
        s3.append(x3T)
        s4.append(x4T)
        s5.append(x5T)

    count1 = sum(d <= epsilon for d in x1)    
    ratios1.append(count1/2000)
    count2 = sum(d <= epsilon for d in x2)
    ratios2.append(count2/2000)
    count3 = sum(d <= epsilon for d in x3)
    ratios3.append(count3/2000)
    count4 = sum(d <= epsilon for d in x4)
    ratios4.append(count4/2000)
    # ratios5.append(count5/5000)



    x1_dev.append(np.average(x1))
    x2_dev.append(np.average(x2))
    x3_dev.append(np.average(x3))
    x4_dev.append(np.average(x4))
    x5_dev.append(np.average(x5))
   
    s1 = np.array(s1)
    s2 = np.array(s2)
    s3 = np.array(s3)
    s4 = np.array(s4)
    s5 = np.array(s5)

    # --- Record metrics ---
    u_convex = u.value.reshape(-1, 1)
    u_c2     = u2.value.reshape(-1, 1)
    u_c3     = u3.value.reshape(-1, 1)
    u_c4     = u4.value.reshape(-1, 1)

 
    cheb_norm = float(u_c2.T @ Q_tilde @ u_c2)
    ep_norm = float(u_c3.T @ Q_tilde @ u_c3)
    sec_norm = float(u_convex.T @ Q_tilde @ u_convex)
    det_norm = float(u_c4.T @ Q_tilde @ u_c4)
    model_norm = float(u_model.T @ Q_tilde @ u_model)
    star_norm = float(u_star.T @ Q_tilde @ u_star)
    # u_norm_diff = np.linalg.norm(u_c3) - np.linalg.norm(u_convex)
    


    chebu.append(cheb_norm)
    epu.append(ep_norm)
    secu.append(sec_norm)
    firstu.append(det_norm)
    staru.append(star_norm)

    center = (x_f[0], x_f[1])

    theta = np.linspace(0, 2*np.pi, 400)
    x = center[0] + epsilon * np.cos(theta)
    y = center[1] + epsilon * np.sin(theta)



    # modelu.append(model_norm)
    filename1 = os.path.join(delta_folder, '2nd-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('2nd order, horizon = {}, ratio = {}'.format(t, count1/5000))
    plt.legend()
    plt.savefig(filename1)


    filename2 = os.path.join(delta_folder, 'cheb-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s2[:, 0], s2[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('chebyshev, horizon = {}, ratio = {}'.format(t, count2/5000))
    plt.legend()
    plt.savefig(filename2)


    filename3 = os.path.join(delta_folder, 'eps-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s3[:, 0], s3[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('epsilon, horizon = {}, ratio = {}'.format(t, count3/5000))
    plt.legend()
    plt.savefig(filename3)


    filename4 = os.path.join(delta_folder, 'det-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s4[:, 0], s4[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('det, horizon = {}, ratio = {}'.format(t, count4/5000))
    plt.legend()
    plt.savefig(filename4)

plt.close()

plt.figure(figsize = (10, 6))
N = np.arange(1, len(alleps)+1)
plt.plot(N, alleps, label = 'feasible epsilon vs Time horizon')
plt.xlabel('time horizon')
plt.ylabel('epsilon')
plt.title('delta = {}'.format(delta))
plt.legend()
plt.savefig('eps-T.jpeg')
plt.close()

plt.figure(figsize = (10, 6))
N = np.arange(0, len(all_lam))
plt.plot(N, all_lam, label = 'lambda*')
plt.title('lambda* vs time horizon')
plt.xlabel('time horizon')
plt.ylabel('lambda*')
plt.savefig('lambda.jpeg')
plt.close()
   

plt.figure(figsize = (10,6))
N = np.arange(0, len(g))
plt.plot(N, g, '--',label = 'gamma')
plt.savefig('gamma.jpeg')
plt.close()

N = np.arange(0, len(ratios1))
plt.figure(figsize = (10, 6))
plt.plot(N, ratios1, label = 'second order')
plt.plot(N, ratios2, label = 'chebyshev')
plt.plot(N, ratios3, label = 'epsilon')
plt.plot(N, ratios4, label = 'det')

plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('final state ratio inside epsilon ball')
plt.legend()
plt.savefig('ratio.jpeg')
plt.close()
   


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
        y = np.sqrt(gamma) * z

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
        beta = 0.5
        # beta = 2 * np.log((j * t**2 * np.pi**2)/d*6)
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


#algorithm 1: Find set of feasible inputs and using BO, obtain MECI

#system parameters
n,m = 2,2
x_0 = np.array([0.0, 0.0])

T_list  = range(1, 40)
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
j = 500
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

    k = GPy.kern.RBF(
    input_dim=m*T,
    variance=1.0,        
    lengthscale=np.ones(m*T),       
    ARD=True
    )        



    X =  sample_u(CT, b, alpha, j) # set of inputs from constraint (only change)
  
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
    values = np.einsum('ij,jk,ik->i', X, Q_tilde, X)
    cr = 0
    min_value = np.min(values)
    for t in range(0,500):
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
        
    N = np.arange(0, len(indices))
    plt.figure(figsize = (10, 6))
    plt.scatter(N, indices, marker = 'o')    
    plt.xlabel('time step')
    plt.ylabel('index')
    plt.legend()
    plt.savefig('indices_{}.jpeg'.format(T))
    plt.close()
   
    if T == 1:
        
        plt.figure(figsize = (10,6))
        plt.plot(X[:, 0], X[:, 1], 'x')
        plt.plot(X[indices, 0], X[indices, 1], 'o', label = 'sampled')
        plt.plot(X[index, 0], X[index, 1], 'o', color = 'red', label = 'u*')
        plt.xlabel('u_0')
        plt.ylabel('u_1')
        plt.title('Feasible set of u for T = 1')
        plt.legend()
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
    print(X[index, :])
    print(X[index, :].reshape(T, m)[::-1].reshape(-1))
    norm.append(X[index, :].reshape(T, m)[::-1].reshape(-1).T @ Q_tilde @ X[index, :].reshape(T, m)[::-1].reshape(-1))
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
plt.plot(N, dev,'x--', label = 'offline_gp')
# N = np.arange(0, len(x2_dev))
plt.plot(N, x2_dev, label = 'chebyshev')
plt.plot(N, x3_dev, label = 'epsilon sd')
plt.plot(N, x1_dev, label = '2nd order state dev')
plt.plot(N, x4_dev, label = 'first order')
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.savefig('dev-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()

plt.figure(figsize = (10,6))
plt.plot(N, norm, 'x--', label = 'offline gp')
plt.plot(N, chebu, label = 'cheby norm')
plt.plot(N, epu, label = 'epsilon norm')
plt.plot(N, secu, label = 'second order norm')
plt.plot(N, firstu, label = 'first order norm')
# plt.plot(N, staru, label = 'lambda u')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.grid(True)
plt.savefig('norm-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()


N = np.arange(0, len(ratios))
plt.figure(figsize = (10, 6))
plt.plot(N, ratios)
plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('final state ratio inside epsilon ball')
plt.legend()
plt.grid(True)
plt.savefig('off_ratio.jpeg')
plt.close()

plt.figure(figsize = (10,6))
diff = np.abs(np.array(norm) - np.array(chebu))
plt.plot(N, diff, label = 'norm difference')
plt.legend()
plt.savefig('diff.jpeg')
plt.close()
