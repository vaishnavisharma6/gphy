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
from scipy.stats import ncx2
from scipy.optimize import brentq
import GPy

#----------------------------------------------------------------------------------------
# fixed system parameters
n, m = 2, 2
x_0 = np.array([0.0, 0.0])

T_list = range(1, 50, 5)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
delta = 0.25
Q = np.diag([2.0, 3.0])
q = norm.ppf(1 - delta)
u_norm_diff_list = []
state_dev_diff_list = []
name = 'stable'
norm_folder = 'norms_{}'.format(name)
state_folder = 'states_{}'.format(name)
diff_folder = 'diffs_{}'.format(name)
delta_folder = 'traj_{}'.format(delta)
exact_folder = 'traj_{}'.format(delta)
epsilon = 0.8
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
exact = []
marcu = []
alleps = []

x1_dev = []
x2_dev = []
x3_dev = []
x4_dev = []
x5_dev = []
x6_dev = []


all_lam = []
g = []
g1 = []
g2 = []

ratios1 = []
ratios2 = []
ratios3 = []
ratios4 = []
ratios5 = []
ratios6 = []

def find_lambda_star(n, epsilon, delta):
    """
    Solves F_{chi^2_n(lambda)}(epsilon^2) = delta
    for lambda >= 0.
    
    Parameters:
    n       : degrees of freedom
    epsilon : radius (so we evaluate CDF at epsilon^2)
    delta   : probability level
    
    Returns:
    lambda_star
    """
    
    x = epsilon**2

    # Define equation to solve
    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - delta

    # We search in a reasonable interval for lambda
    # Lower bound = 0
    # Upper bound chosen large enough
    lam_lower = 0.0
    lam_upper = 1e4

    # Ensure bracket contains root
    if equation(lam_lower) * equation(lam_upper) > 0:
        raise ValueError("Root not bracketed. Increase lam_upper.")

    lambda_star = brentq(equation, lam_lower, lam_upper)
    
    return lambda_star

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
    vals = np.linalg.eigvals(sigma_z)
    eig = np.max(vals)
    # convex constraint gamma
    gamma = epsilon**2 - traceSigma  #second order constraint
    r_cheb = epsilon**2 - 2* epsilon * math.sqrt(traceSigma/delta) + (traceSigma/delta)  #chebyshev
    dev = epsilon**2  #epsilon sd
    lambda_star = find_lambda_star(n, epsilon, delta)
    marc = (epsilon - np.sqrt(-2 * math.log(1-delta)))**2
    g.append(r_cheb)
    g1.append(gamma)

    #choose proper epsilon
   
    
    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    u3 = cp.Variable(m*t)
    u4 = cp.Variable(m*t)
    u5 = cp.Variable(m*t)
    u6 = cp.Variable(m*t)
    Q_tilde = block_diag(*([Q] * t))
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))
    objective5 = cp.Minimize(cp.quad_form(u5, Q_tilde))
    objective6 = cp.Minimize(cp.quad_form(u6, Q_tilde))
    f = cp.quad_form(C_T @ u + b, np.eye(C_T.shape[0]))
    f2 = cp.quad_form(C_T @ u2 + b, np.eye(C_T.shape[0]))
    f3 = cp.quad_form(C_T @ u3 + b, np.linalg.inv(sigma_z))
    f4 = C_T @ u4 + b
    f5 = cp.quad_form(C_T@ u5 + b, np.linalg.inv(sigma_z))
    f6 = cp.quad_form(C_T@ u6 + b, np.linalg.inv(sigma_z))
    # f4 = cp.quad_form(C_T @ u4 + b, np.eye(C_T.shape[0]))
    prob = cp.Problem(objective, [f <= gamma])
    prob2 = cp.Problem(objective2, [f2 <= r_cheb])
    prob3 = cp.Problem(objective3, [f3 <= dev])
    prob4 = cp.Problem(objective4, [f4 == 0])
    prob5 = cp.Problem(objective5, [f5 <= lambda_star])
    prob6 = cp.Problem(objective6, [f6 <= marc])
    prob.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver = cp.SCS, verbose = False)
    prob3.solve(solver = cp.SCS, verbose = False)
    prob4.solve(solver = cp.SCS, verbose = False)
    prob5.solve(solver = cp.SCS, verbose = False)
    prob6.solve(solver = cp.SCS, verbose = False)
    u_convex = u.value  #second order
    u_c2 = u2.value #chebyshev
    u_c3 = u3.value #epsilon sd
    u_c4 = u4.value # first order
    u_c5 = u5.value
    u_c6 = u6.value
  
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
    if u_c5 is None:
        print(f'Convex solver failed at T={t}')    
    if u_c6 is None:
        print(f'Convex solver failed at T={t}')       
    #expression u
    lam, status = solve_lambda(C_T @ np.linalg.inv(Q_tilde) @ np.transpose(C_T), b, r_cheb, 1e-8, 100)
    
    # u_star = -lam* np.linalg.inv(Q_tilde + lam * np.transpose(C_T) @ C_T) @ np.transpose(C_T) @ b
    all_lam.append(lam)
    # --- Simulate both systems ---
    np.random.seed(0)
   
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []

    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 = []
    for j in range(0, 2000):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_convex + N_T @ w  # second order
        x2T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c2 + N_T @ w  # chebyshev
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c3 + N_T @ w  # epsilon 
        x4T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c4 + N_T @ w
        x5T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c5 + N_T @ w
        x6T = np.linalg.matrix_power(A, t)@x_0 + C_T @ u_c6 + N_T @ w
        x1.append(np.linalg.norm(x1T-x_f))
        
        
        x2.append(np.linalg.norm(x2T-x_f))
      
       
        x3.append(np.linalg.norm(x3T-x_f))
   
       
        x4.append(np.linalg.norm(x4T - x_f))
      
     
        x5.append(np.linalg.norm(x5T - x_f))
        count5 = sum(d <= epsilon for d in x5)

        x6.append(np.linalg.norm(x6T - x_f))
        count6 = sum(d <= epsilon for d in x6)
        
        s1.append(x1T)
        s2.append(x2T)
        s3.append(x3T)
        s4.append(x4T)
        s5.append(x5T)
        s6.append(x6T)

    count1 = sum(d <= epsilon for d in x1)    
    ratios1.append(count1/2000)
    count2 = sum(d <= epsilon for d in x2)
    ratios2.append(count2/2000)
    count3 = sum(d <= epsilon for d in x3)
    ratios3.append(count3/2000)
    count4 = sum(d <= epsilon for d in x4)
    ratios4.append(count4/2000)
    ratios5.append(count5/2000)
    ratios6.append(count6/2000)



    x1_dev.append(np.average(x1))
    x2_dev.append(np.average(x2))
    x3_dev.append(np.average(x3))
    x4_dev.append(np.average(x4))
    x5_dev.append(np.average(x5))
    x6_dev.append(np.average(x6))
   
    s1 = np.array(s1)
    s2 = np.array(s2)
    s3 = np.array(s3)
    s4 = np.array(s4)
    s5 = np.array(s5)
    s6 = np.array(s6)

    # --- Record metrics ---
    u_convex = u.value.reshape(-1, 1)
    u_c2     = u2.value.reshape(-1, 1)
    u_c3     = u3.value.reshape(-1, 1)
    u_c4     = u4.value.reshape(-1, 1)
    u_c5 = u5.value.reshape(-1, 1)
    u_c6 = u6.value.reshape(-1, 1)

 
    cheb_norm = float(u_c2.T @ Q_tilde @ u_c2)
    ep_norm = float(u_c3.T @ Q_tilde @ u_c3)
    sec_norm = float(u_convex.T @ Q_tilde @ u_convex)
    det_norm = float(u_c4.T @ Q_tilde @ u_c4)
    model_norm = float(u_model.T @ Q_tilde @ u_model)
    exact_norm = float(u_c5.T @ Q_tilde @ u_c5)
    marc_norm = float(u_c6.T @ Q_tilde @ u_c6)

    # u_norm_diff = np.linalg.norm(u_c3) - np.linalg.norm(u_convex)
    


    chebu.append(cheb_norm)
    epu.append(ep_norm)
    secu.append(sec_norm)
    firstu.append(det_norm)
    exact.append(exact_norm)
    marcu.append(marc_norm)

    center = (x_f[0], x_f[1])

    theta = np.linspace(0, 2*np.pi, 400)
    x = center[0] + epsilon * np.cos(theta)
    y = center[1] + epsilon * np.sin(theta)




    filename1 = os.path.join(delta_folder, '2nd-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('2nd order, horizon = {}, ratio = {}'.format(t, count1/2000))

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
    plt.title('chebyshev, horizon = {}, ratio = {}'.format(t, count2/2000))
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
    plt.title('epsilon, horizon = {}, ratio = {}'.format(t, count3/2000))
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
    plt.title('det, horizon = {}, ratio = {}'.format(t, count4/2000))
    plt.legend()
    plt.savefig(filename4)

    filename5 = os.path.join(delta_folder, 'exact-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s5[:, 0], s5[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('exact, horizon = {}, ratio = {}'.format(t, count5/2000))
    plt.legend()
    plt.savefig(filename5)

    filename6 = os.path.join(delta_folder, 'marc-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s6[:, 0], s6[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('marc, horizon = {}, ratio = {}'.format(t, count6/2000))
    plt.legend()
    plt.savefig(filename6)

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
plt.plot(N, g, '--',label = 'cheb')
plt.plot(N, g1, '-', label = 'second')
# plt.plot(N, g2, '--', label = 'bern')
plt.legend()
plt.savefig('gamma.jpeg')
plt.close()

N = np.arange(0, len(ratios1))
plt.figure(figsize = (10, 6))
plt.plot(T_list, ratios1, label = 'second order')
plt.plot(T_list, ratios2, label = 'chebyshev')
plt.plot(T_list, ratios3, label = 'epsilon')
plt.plot(T_list, ratios4, label = 'det')
plt.plot(T_list, ratios5, label = 'exact')
plt.plot(T_list, ratios6, label = 'marc')

plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('final state ratio inside epsilon ball')
plt.legend()
plt.savefig('ratio.jpeg')
plt.close()
   

plt.figure(figsize = (10,6))

plt.plot(T_list, x2_dev, 'o-', label = 'chebyshev')
plt.plot(T_list, x3_dev, 'o-', label = 'epsilon sd')
plt.plot(T_list, x1_dev,'o-',  label = '2nd order state dev')
plt.plot(T_list, x4_dev, 'o-', label = 'first order')
plt.plot(T_list, x5_dev, 'o-', label = 'exact')
plt.plot(T_list, x6_dev, 'o-', label = 'marc' )
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.title('avg state deviation')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('dev-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()

plt.figure(figsize = (10,6))

plt.plot(T_list, chebu, 'o-', label = 'cheby norm')
plt.plot(T_list, epu, 'o-', label = 'epsilon norm')
plt.plot(T_list, secu, 'o-', label = 'second order norm')
plt.plot(T_list, firstu, 'o-', label = 'first order norm')
plt.plot(T_list, exact, 'o-', label = 'exact')
plt.plot(T_list, marcu, 'o-', label = 'marc')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('norm-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()





