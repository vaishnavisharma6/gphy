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
n, m = 2, 6
x_0 = np.array([0.2, 0.0])

T_list = range(1, 30)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
epsilon = 0.9
delta = 0.9
Q = np.diag([2.0, 3.0, 4.0, 5.0, 0.1, 0.2])
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
    A,B, NT = get_system(name, n, m, t, seed=2)
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
    u_c4 = u4.value 
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

    # --- Simulate both systems ---
    w = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.1)
    x1 = np.array(x_0).reshape(n,1)
    x2 = np.array(x_0).reshape(n,1)
    x3 = np.array(x_0).reshape(n,1)
    x4 = np.array(x_0).reshape(n,1)
    for i in range(t):
        x1 = np.hstack((x1, A @ x1[:, -1:] + B @ u_convex[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1)))  #second order
        x2 = np.hstack((x2, A @ x2[:, -1:] + B @ u_c2[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1)))  # chebyshev
        x3 = np.hstack((x3, A @ x3[:, -1:] + B @ u_c3[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1))) #epsilon sd
        x4 = np.hstack((x4, A @ x4[:, -1:] + B @ u_c4[m*i:m*(i+1)].reshape(m,1) + w.reshape(n,1)))
    # --- Record metrics ---
    u_norm_diff = np.linalg.norm(u_c3) - np.linalg.norm(u_convex)
    state_dev_diff = np.linalg.norm(x2[:, -1] - x1[:, -1])
    u_norm_diff_list.append(u_norm_diff)
    state_dev_diff_list.append(state_dev_diff)
    u_opt = u.value.reshape(t, m)
    u_n = np.linalg.norm(u_opt, axis = 1) #second order
    U = u_c3.reshape(t, m)
    u_norms = np.linalg.norm(U, axis = 1) #epsilon sd
    u_cheb = u2.value.reshape(t, m) 
    uc_norms = np.linalg.norm(u_cheb, axis = 1) #chebyshev
    u_det = u_c4.reshape(t, m)
    udet_norms = np.linalg.norm(u_det, axis = 1)
    
   
    opt(t, x1, x2, x3, x4, x_f, epsilon, u_n, u_norms, uc_norms, udet_norms, state_folder, norm_folder) #plot states, norms
    

diff(T_list, u_norm_diff_list, state_dev_diff_list, diff_folder)   #plot diffs





 