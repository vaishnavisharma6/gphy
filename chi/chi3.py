import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os


#folders 
norm_folder = 'norms'
state_folder = 'states'
diff_folder = 'diffs'

os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)

# fixed system parameters
n, m = 2, 2
A = np.random.rand(n, n)
B = np.random.rand(n, m)
x_0 = np.array([1.0, 3.0])
x_f = np.array([3.0, 2.0])
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
epsilon = 0.8
delta = 0.8
Q = np.diag([2.0, 1.0])
q = norm.ppf(1 - delta)

T_list = range(1, 20)  # vary horizon length here
u_norm_diff_list = []
state_dev_diff_list = []

for T in T_list:
    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = B
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
        C_T = np.hstack((C_T, np.linalg.matrix_power(A, i) @ B))

    # covariance and bias
    sigma_w = np.eye(n*T) * 0.1
    sigma_z = np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T + N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)
    tau = 2 * np.trace(sigma_z @ sigma_z)
    # tau2 = 2 * np.trace(sigma_z)
    b = (np.linalg.matrix_power(A, T) @ x_0) - x_f

    # convex constraint gamma
    gamma = epsilon**2 - traceSigma - q * math.sqrt(tau)
    # gamma2 = epsilon**2 -traceSigma -q *math.sqrt(tau2)

    # --- Convex QCQP ---
    u = cp.Variable(m*T)
    # u2 = cp.Variable(m*T)
    Q_tilde = block_diag(*([Q] * T))
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    # objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    f = cp.quad_form(C_T @ u + b, np.eye(C_T.shape[0]))
    # f2 = cp.quad_form(C_T @ u2 + b, np.eye(C_T.shape[0]))
    prob = cp.Problem(objective, [f <= gamma])
    # prob2 = cp.Problem(objective, [f2 <= gamma2])
    prob.solve(solver=cp.SCS, verbose=False)
    # prob2.solve(solver = cp.SCS, verbose = False)
    u_convex = u.value
    # u_c2 = u2.value
    if u_convex is None:
        print(f"Convex solver failed at T={T}")
        continue

    # if u_c2 is None:
    #     print(f"Convex solver failed at T={T}")
    #     continue

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

    u0 = np.zeros(m*T)
    nonlin_cons = {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}
    res = minimize(lambda u: objective(u)[0], u0, jac=lambda u: objective(u)[1],
                   constraints=[nonlin_cons],
                   options={'maxiter': 500, 'ftol': 1e-8})

    if not res.success:
        print(f"Exact solver failed at T={T}: {res.message}")
        continue

    u_exact = res.x

    # --- Simulate both systems ---
    w = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.1)
    x1 = np.array(x_0).reshape(n,1)
    x2 = np.array(x_0).reshape(n,1)
    # x3 = np.array(x_0).reshape(n,1)
    for t in range(T):
        x1 = np.hstack((x1, A @ x1[:, -1:] + B @ u_convex[m*t:m*(t+1)].reshape(m,1) + w.reshape(n,1)))
        x2 = np.hstack((x2, A @ x2[:, -1:] + B @ u_exact[m*t:m*(t+1)].reshape(m,1) + w.reshape(n,1)))
        # x3 = np.hstack((x3, A @ x3[:, -1:] + B @ u_c2[m*t:m*(t+1)].reshape(m,1) + w.reshape(n,1)))
    # --- Record metrics ---
    u_norm_diff = np.linalg.norm(u_exact) - np.linalg.norm(u_convex)
    state_dev_diff = np.linalg.norm(x2[:, -1] - x1[:, -1])
    u_norm_diff_list.append(u_norm_diff)
    state_dev_diff_list.append(state_dev_diff)
    
    t_axis = range(T+1)
    plt.figure(figsize = (10,6))

    plt.plot(x1[0, :], x1[1, :], 'b-o', label = 'state')
    plt.plot(x2[0, :], x2[1,:], 'g-o', label = 'state in excat cons')
    # plt.plot(x3[0, :], x3[1, :], 'r-o', label = 'no power')

    plt.scatter(x_f[0], x_f[1], color='red', marker='*', s=150, label='x_f (desired)')
    plt.scatter(x1[0, 0], x1[1, 0], color='green', marker='x', s=100, label='Start')
    plt.scatter(x1[0,-1], x1[1, -1], color = 'red', marker = 'x', s= 150, label = 'final')
    plt.scatter(x2[0, -1], x2[1, -1], color = 'red', marker='x', s = 150, label = 'final2')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    # plt.scatter(x3[0, -1], x3[1,-1], color = 'yellow', marker = 'x', s = 150, label = 'final3')

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid(True)
    plt.title('state deviation = {}, epsilon = {}'.format(np.linalg.norm(x_f - x1[-1, -1]), epsilon))
    plt.legend()
    filename = os.path.join(state_folder, 'state_{}.png'.format(T))
    plt.savefig(filename)
    
    u_opt = u.value.reshape(T, m)
    # u_2 = u2.value.reshape(T, m)

    U = u_exact.reshape(T, m)
    u_norms = np.linalg.norm(U, axis = 1)
    # un2 = np.linalg.norm(u_2, axis = 1)
    # Plot input norm on the same figure (secondary y-axis)
    plt.figure(figsize = (10, 6))

    plt.plot(range(1, T+1), np.linalg.norm(u_opt, axis=1), 'k--s', label='‖uₜ‖₂')
    plt.plot(range(1, T+1), u_norms, marker='o', linewidth=2, label ='exact constraint')
    # plt.plot(range(1, T+1), un2, marker = 'o', linewidth = 2, label = 'no power')
    plt.xlabel('time step')
    plt.ylabel('u-norm')
    plt.title('u norm = {}, u2_norm = {}:'.format(prob.value, np.linalg.norm(u_exact)**2))

    plt.grid(True)
    plt.legend()
    filename1 = os.path.join(norm_folder, 'norm_{}.png'.format(T))
    plt.savefig(filename1)


# -------------------------------------------------------------
# Plot results
plt.figure(figsize=(8,5))
plt.plot(T_list, u_norm_diff_list, 'o-', label='Δ‖u‖ (exact - convex)')
plt.xlabel('Time horizon T')
plt.ylabel('Difference in total ‖u‖')
plt.title('Control norm difference vs Time horizon')
plt.grid(True)
plt.legend()
plt.tight_layout()
filename2 = os.path.join(diff_folder, 'u_norm_diff_vs_T.png')
plt.savefig(filename2)

plt.figure(figsize=(8,5))
plt.plot(T_list, state_dev_diff_list, 's-', color='purple', label='‖x_T_exact - x_T_approx‖₂')
plt.xlabel('Time horizon T')
plt.ylabel('Final state deviation difference')
plt.title('State deviation difference vs Time horizon')
plt.grid(True)
plt.legend()
plt.tight_layout()
filename3 = os.path.join(diff_folder, 'state_dev_diff_vs_T.png' )
plt.savefig(filename3)
plt.show()
