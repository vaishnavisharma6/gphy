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
from plot import opt, diff

#----------------------------------------------------------------------------------------
# fixed system parameters
n, m = 2, 2
x_0 = np.array([0.0, 0.0])

T_list = range(1, 2)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
epsilon = 0.9
delta = 0.9
Q = np.diag([2.0, 1.0])
q = norm.ppf(1 - delta)
u_norm_diff_list = []
state_dev_diff_list = []
name = 'stable'
norm_folder = 'norms_{}'.format(name)
state_folder = 'states_{}'.format(name)
diff_folder = 'diffs_{}'.format(name)

os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)
for t in T_list:
    A,B, NT = get_system(name, n=2, m=2, T = t, seed=2)
    U = np.random.randn(m * t, 1)

        # Compute reachable final state
    x_f = np.linalg.matrix_power(A, t) @ x_0.reshape(-1, 1) + NT @ U
    x_f =  x_f.reshape(-1)

    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = B
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
        C_T = np.hstack((C_T, np.linalg.matrix_power(A, i) @ B))

    # covariance and bias
    sigma_w = np.eye(n*t) * 0.1
    sigma_z = np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T + N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)
    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = (np.linalg.matrix_power(A, t) @ x_0) - x_f
    b_cheb =  x_f - (np.linalg.matrix_power(A, t) @ x_0) 
    # convex constraint gamma
    gamma = epsilon**2 - traceSigma - q * math.sqrt(tau)
    r_cheb = epsilon - math.sqrt(traceSigma/delta)

    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    Q_tilde = block_diag(*([Q] * t))
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    f = cp.quad_form(C_T @ u + b, np.eye(C_T.shape[0]))
    f2 = cp.norm2((C_T @ u2) - b_cheb)
    prob = cp.Problem(objective, [f <= gamma])
    prob2 = cp.Problem(objective2, [f2 <= r_cheb])
    prob.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver = cp.CLARABEL, verbose = False)
    u_convex = u.value
    u_c2 = u2.value
    if u_convex is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c2 is None:
        print(f"Convex solver failed at T={t}")
        continue
    # --- Exact constraint ---
    def exact_lhs_and_grad(u_flat):
        y = C_T @ u_flat + b
        qy = float(y.T @ (sigma_z @ y))
        s = max(1e-12, tau + 4.0 * qy)
        lhs = float(y.T @ y) + traceSigma + q * math.sqrt(s)
        grad = 2 * (C_T.T @ y) + q * (4 / math.sqrt(s)) * (C_T.T @ (sigma_z @ y))
        return lhs, grad

    def objective(u_flat):
        val = float(u_flat @ (Q_tilde @ u_flat))
        grad = 2 * (Q_tilde @ u_flat)
        return val, grad

    def constraint_fun(u_flat):
        lhs, _ = exact_lhs_and_grad(u_flat)
        return epsilon**2 - lhs

    def constraint_jac(u_flat):
        _, grad_lhs = exact_lhs_and_grad(u_flat)
        return -grad_lhs

    u0 = np.zeros(m*t)
    nonlin_cons = {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}
    res = minimize(lambda u: objective(u)[0], u0, jac=lambda u: objective(u)[1],
                   constraints=[nonlin_cons],
                   options={'maxiter': 500, 'ftol': 1e-8})

    if not res.success:
        print(f"Exact solver failed at T={t}: {res.message}")
        continue

    u_exact = res.x

    # --- Simulate both systems ---
    w = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.1)
    x1 = np.array(x_0).reshape(n,1)
    x2 = np.array(x_0).reshape(n,1)
    x3 = np.array(x_0).reshape(n,1)
    for i in range(t):
        x1 = np.hstack((x1, A @ x1[:, -1:] + B @ u_convex[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1)))
        x2 = np.hstack((x2, A @ x2[:, -1:] + B @ u_exact[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1)))
        x3 = np.hstack((x3, A @ x3[:, -1:] + B @ u_c2[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1)))
    # --- Record metrics ---
    u_norm_diff = np.linalg.norm(u_exact) - np.linalg.norm(u_convex)
    state_dev_diff = np.linalg.norm(x2[:, -1] - x1[:, -1])
    u_norm_diff_list.append(u_norm_diff)
    state_dev_diff_list.append(state_dev_diff)
    u_opt = u.value.reshape(t, m)
    u_n = np.linalg.norm(u_opt, axis = 1)
    U = u_exact.reshape(t, m)
    u_norms = np.linalg.norm(U, axis = 1)
    u_cheb = u2.value.reshape(t, m)
    uc_norms = np.linalg.norm(u_cheb, axis = 1)
    
   
    opt(t, x1, x2, x3, x_f, epsilon, u_n, u_norms, uc_norms, state_folder, norm_folder) #plot states, norms
    

diff(T_list, u_norm_diff_list, state_dev_diff_list, diff_folder)   #plot diffs



