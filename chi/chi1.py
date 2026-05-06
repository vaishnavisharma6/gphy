# import numpy as np
# import matplotlib.pyplot as plt
# import cvxpy as cp
# from scipy.optimize import minimize
# from scipy.stats import norm
# from scipy.linalg import block_diag
# import matplotlib.patches as patches
# import math
# import os
# from datagen import get_system
# from functools import reduce
# from plot import opt, diff
# from scipy.integrate import quad
# from scipy.stats import chi2
# from scipy.stats import norm
# from matplotlib.patches import Circle
# from scipy.stats import ncx2
# from scipy.optimize import brentq


# m = 2
# x_0 = np.array([0.0, 0.0])

# delta = 0.9
# epsilon = 8.0
# Q = np.diag([2.0, 3.0])
# T = 10
# name = 'stable'

# N = 21
# M = 1000

# # N_list = range(2*T +1, 100, 3)
# N_list = range(2*T+1, 100, 3)
# data_folder = 'data_{}'.format(name)
# os.makedirs(data_folder, exist_ok = True)


# n_list = range(2, 7, 1)




# plt.figure(figsize = (10, 6))
# for n in n_list:
#     errors = []
#     dev = []
#     dev1 = []
#     cost = []
#     cost1 = []
#     sigma_0 = np.eye(n) * 0.1
#     gamma = find_lambda_star(n, epsilon, delta)
#     for N in N_list:

#         A, B, CT = get_system(name, n, m, T, seed = 0)
#         NT = np.eye(n)
#         for i in range(1, T):
#             NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
#         # N = (2* T) + 1
#         u = np.random.randn(m * T, 1)
#         Q_tilde = block_diag(*([Q] * T))
#         # Compute reachable final state
#         w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
#         x_f = NT @ w
#         x_f =  x_f.reshape(-1, 1)

#         U  = np.random.rand(m*T, N)

#         mu = np.zeros((n, n))
#         var_sum = np.zeros((n, n))
#         X = np.zeros((n, n))
#         for i in range(0, N):
#             sum = np.zeros(n)
#             for j in range(0, M):
#                 w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
#                 xT = CT @ U[:, i] + NT @ w
#                 if i == N - 1:
#                     X = np.hstack((X, xT.reshape(-1,1)))
#                 sum = sum + xT
#             mu_i = (1/M) * sum   
#             mu = np.hstack((mu, mu_i.reshape(-1, 1))) 


#         for j in range(0, M):
#             var_sum = var_sum + (X[:, n+j]- mu_i) @ (X[:, n+j] - mu_i).T
#         Sigma = (1/M) * var_sum
        
#         b =  -1 *  x_f
#         b = b.reshape(-1, 1)
#         sigma_w_base = np.eye(n*T) * 0.1
#         # --------- CVXPY FIXED PART ---------

#         alpha = cp.Variable((N, 1))
#         u = cp.Variable((m*T, 1))
#         # ---- Objective (make it DCP safe) ----
#         P = U.T @ Q_tilde @ U
#         P = 0.5 * (P + P.T)                    # enforce symmetry
#         P = P + 1e-8 * np.eye(N)               # ensure PSD

#         objective = cp.Minimize(cp.quad_form(alpha, P))
#         objective1 = cp.Minimize(cp.quad_form(u, Q_tilde))

#         # ---- Constraint (Mahalanobis term) ----
#         mu_eff = mu[:, n:]                     # your intended slice
#         x_f_col = x_f.reshape(-1, 1)

#         Sigma_reg = Sigma + 1e-6 * np.eye(n)   # stabilize
#         Sigma_inv = np.linalg.inv(Sigma_reg)
#         Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)  # enforce symmetry
        
#         sigma_x = NT @ sigma_w_base @ NT.T
#         f = cp.quad_form(mu_eff @ alpha - x_f_col, Sigma_inv)
#         f1 = cp.quad_form(CT@ u + b, np.linalg.inv(sigma_x))

#         # ---- Problem ----
#         prob = cp.Problem(objective, [f <= gamma])
#         prob1 = cp.Problem(objective1, [f1 <= gamma])
#         prob.solve(solver=cp.SCS, verbose=False)
#         prob1.solve(solver = cp.SCS, verbose=False)

#         alpha_hat = alpha.value
#         u_star = u.value.reshape(-1,1)
#         errors.append(np.linalg.norm(U @ alpha_hat - u_star))

#         if alpha_hat is None:
#             print(f"Convex solver failed at T={T}")

#         # -------------------------------------
    
        
#         x1 = []
#         x2 = []
#         for j in range(0, 100):
#             w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
#             x1T = CT @ U @ alpha_hat + NT @ w
#             x2T = CT @ u_star + NT @ w
#             x1.append(np.linalg.norm(x1T - x_f))
#             x2.append(np.linalg.norm(x2T - x_f))

#         hat_norm = float(alpha_hat.T @ U.T @ Q_tilde @ U @ alpha_hat)
#         u_norm = float(u_star.T @ Q_tilde @ u_star)
#         cost.append(hat_norm)
#         cost1.append(u_norm)

#         dev.append(np.average(x1))
#         dev1.append(np.average(x2))

#         center = (x_f[0], x_f[1])

#         theta = np.linspace(0, 2*np.pi, 400)
#         x = center[0] + gamma * np.cos(theta)
#         y = center[1] + gamma * np.sin(theta)

#     plt.semilogy(N_list, errors, label = 'error for n = {}'.format(n))
#     plt.xlabel('N')
#     plt.ylabel('error')
#     plt.legend()
#     plt.savefig('error.jpeg')


#     # filename1 = os.path.join(data_folder, 'data-{}-{}.jpeg'.format(delta, T))
#     # plt.figure(figsize = (10,6))
#     # plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
#     # plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
#     # plt.plot(x, y)
#     # plt.scatter(*center)
#     # plt.gca().set_aspect('equal', adjustable='box')
#     # circle = patches.Circle((x_f[0], x_f[1]), gamma, 
#     #                     fill=False, linestyle='--', color='k', label=f'ε-ball (ε={gamma})')
#     # plt.xlabel('x1')
#     # plt.ylabel('x2')
#     # # plt.title('second order moment, horizon = {}, ratio = {}'.format(t, count1/2000))

#     # plt.savefig(filename1)





  

   
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
import math
import os
from datagen import get_system
from plot import opt, diff
from scipy.stats import chi2, ncx2
from scipy.optimize import brentq

m = 2
x_0 = np.array([0.0, 0.0])

delta = 0.9
epsilon = 2.4
Q = np.diag([2.0, 3.0])
T = 10
name = 'stable'

# N = 21
M = 100

N_list = range(2*T+1, 100, 3)
data_folder = 'data_{}'.format(name)
os.makedirs(data_folder, exist_ok=True)

n = 2
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


plt.figure(figsize=(10, 6))


errors = []
errors1 = []
dev = []
dev1 = []
dev2 = []
cost = []
cost1 = []
cost2 = []
   

sigma_0 = np.eye(n) * 0.1
gamma = find_lambda_star(n, epsilon, delta)

for N in N_list:

    A, B, CT = get_system(name, n, m, T, seed=0)

    NT = np.eye(n)
    for i in range(1, T):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

    Q_tilde = block_diag(*([Q] * T))

    w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
    x_f = NT @ w
    x_f = x_f.reshape(-1, 1)

    U = np.random.rand(m*T, N)

    # ================== NEW METHOD: C_T ESTIMATION ==================

# Step 1: collect one sample per input (consistent with your derivation)
    X_single = np.zeros((n, N))

    for i in range(N):
        w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        xT = CT @ U[:, i] + NT @ w
        X_single[:, i] = xT

    # Step 2: estimate C_T
    UUT_inv = np.linalg.inv(U @ U.T + 1e-8 * np.eye(m*T))   # regularization
    C_hat = X_single @ U.T @ UUT_inv

    # Step 3: estimate Sigma_xT
    Sigma_CT = np.zeros((n, n))

    for i in range(N):
        err = (X_single[:, i] - C_hat @ U[:, i]).reshape(-1, 1)
        Sigma_CT += err @ err.T

    Sigma_CT = Sigma_CT / N
    # Sigma_CT_reg = Sigma_CT + 1e-6 * np.eye(n)
    Sigma_CT_inv = np.linalg.inv(Sigma_CT)
    # Sigma_CT_inv = 0.5 * (Sigma_CT_inv + Sigma_CT_inv.T)

# ===============================================================

    # 🔧 FIX: correct initialization
    mu = np.zeros((n, 0))
    var_sum = np.zeros((n, n))
    
    for i in range(0, N):
        sum_vec = np.zeros(n)  # 🔧 FIX: don't overwrite built-in sum

        # store samples for covariance
        Xi = []

        for j in range(0, M):
            w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
            xT = CT @ U[:, i] + NT @ w

            sum_vec += xT
            Xi.append(xT)
        
        mu_i = (1/M) * sum_vec
        mu = np.hstack((mu, mu_i.reshape(-1, 1)))

        # 🔧 FIX: correct covariance accumulation
        for xT in Xi:
            diff_vec = (xT - mu_i).reshape(-1, 1)
            var_sum += diff_vec @ diff_vec.T

    # 🔧 FIX: normalize properly
    Sigma = var_sum / (M * N)

    b = -1 * x_f

    sigma_w_base = np.eye(n*T) * 0.1

    alpha = cp.Variable((N, 1))
    alpha_true = cp.Variable((N, 1))
    alpha_c = cp.Variable((N, 1))

    P = U.T @ Q_tilde @ U
    P = 0.5 * (P + P.T)
    P = P + 1e-8 * np.eye(N)

    objective = cp.Minimize(cp.quad_form(alpha, P))
    objective1 = cp.Minimize(cp.quad_form(alpha_true, P))
    objective2 = cp.Minimize(cp.quad_form(alpha_c, P))
    # 🔧 FIX: use full mu (no slicing)
    mu_eff = mu

    # Sigma_reg = Sigma + 1e-6 * np.eye(n)
    Sigma_inv = np.linalg.inv(Sigma)
    # Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)

    sigma_x = NT @ sigma_w_base @ NT.T

    f = cp.quad_form(mu_eff @ alpha - x_f, Sigma_inv)
    f1 = cp.quad_form(CT @ U @ alpha_true -x_f, np.linalg.inv(sigma_x))
    f2 = cp.quad_form(C_hat @ U @ alpha_c - x_f, Sigma_CT_inv)

    prob = cp.Problem(objective, [f <= gamma])
    prob1 = cp.Problem(objective1, [f1 <= gamma])
    prob2 = cp.Problem(objective2, [f2 <= gamma])

    prob.solve(solver=cp.SCS, verbose=False)
    prob1.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver = cp.SCS, verbose = False)

    alpha_hat = alpha.value
    alpha_star = alpha_true.value
    calpha = alpha_c.value

    if alpha_hat is None:
        print(f"Convex solver failed at n={n}, N={N}")
        continue

    if alpha_hat is None or alpha_star is None or calpha is None:
        print(f"Solver failed at n={n}, N={N}")
        continue
    
    errors.append(np.linalg.norm(U @ alpha_hat - U @ alpha_star))
    errors1.append(np.linalg.norm(U @ calpha - U @alpha_star))

    x1 = []
    x2 = []
    x3 = []

    for j in range(0, 100):
        w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        x1T = CT @ U @ alpha_hat + NT @ w
        x2T = CT @ U @ alpha_star + NT @ w
        x3T = CT @ U @ calpha + NT @ w
        x1.append(np.linalg.norm(x1T - x_f))
        x2.append(np.linalg.norm(x2T - x_f))
        x3.append(np.linalg.norm(x3T - x_f))
    
   
    hat_norm = float(alpha_hat.T @ U.T @ Q_tilde @ U @ alpha_hat)
    u_norm = float(alpha_star.T @ U.T @ Q_tilde @ U @ alpha_star)
    c_norm = float(calpha.T @ U.T @ Q_tilde @ U @ calpha)
    
    print('U alpha_hat:', hat_norm)
    print('U alpha_star:', u_norm)
    print('U C_T:', c_norm)
    cost.append(hat_norm)
    cost1.append(u_norm)
    cost2.append(c_norm)

    dev.append(np.average(x1))
    dev1.append(np.average(x2))
    dev2.append(np.average(x3))

plt.semilogy(N_list, errors, label=f'n = {n}')
plt.semilogy(N_list, errors1, label = f'CT based n = {n}')
plt.xlabel('N')
plt.ylabel('error')
plt.legend()
plt.savefig('error.jpeg')   


plt.figure(figsize = (10, 6))
plt.plot(N_list, cost, label = 'data')
plt.plot(N_list, cost1, label = 'true')
plt.plot(N_list, cost2, label = 'CT based')
plt.xlabel('N')
plt.ylabel('cost')
plt.legend()
plt.savefig('cost_data.jpeg')

plt.figure(figsize = (10, 6))
plt.plot(N_list, dev, label = 'deviation')
plt.plot(N_list, dev1, label = 'true')
plt.plot(N_list, dev2, label = 'CT based')
plt.xlabel('N')
plt.ylabel('dev')
plt.legend()
plt.savefig('data_dev.jpeg')  