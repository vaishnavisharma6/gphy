import numpy as np
from scipy.optimize import minimize
import cvxpy as cp
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from datagen import get_system
from functools import reduce
# from plot import opt, diff
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib.patches import Circle
from scipy.stats import ncx2
from scipy.optimize import brentq
from scipy.linalg import fractional_matrix_power

#system parameters
n, m = 4, 4
x_0 = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.2)
x_f = np.array([4.0, 5.0, 6.0, 7.0])
T_list = range(1, 100, 10)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
delta = 0.5
Q = np.diag([2.0, 3.0, 4.0, 5.0])
q = norm.ppf(1 - delta)
name = 'unstable'
norm_folder = 'unorms_{}'.format(name)
state_folder = 'ustates_{}'.format(name)
diff_folder = 'udiffs_{}'.format(name)
delta_folder = 'utraj_{}'.format(delta)
exact_folder = 'utraj_{}'.format(delta)
epsilon = 2.4
os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)
os.makedirs(delta_folder, exist_ok = True)

#deviation
secd = []
epd = []
firstd = []
modeld = []

#norms
secu = []
epu = []
exact = []
uw = []
alleps = []

x1_dev = []
x3_dev = []
x5_dev = []

ux1_dev = []
ux3_dev = []
ux5_dev = []
ux6_dev = []

tdev1 = []
tdev3 = []
tdev5 = []


ratios1 = []
ratios3 = []
ratios5 = []

bT = []
abs_bT = []

#numerically find lambda*
def find_lambda_star(n, epsilon, delta):
    """
    Solves F_{chi^2_n(lambda)}(epsilon^2) = delta
    for lambda >= 0.
    Parameters:
    n       : degrees of freedom
    epsilon : radius (so we evaluate CDF at epsilon^2)
    delta   : probability level
    """
    x = epsilon**2

    # Define equation to solve
    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - delta

    # We search in a reasonable interval for lambda
    # Lower bound = 0
    # Upper bound chosen large enough
    lam_lower = 0.0
    lam_upper = 1e11

    # Ensure bracket contains root
    if equation(lam_lower) * equation(lam_upper) > 0:
        raise ValueError("Root not bracketed. Increase lam_upper.")

    lambda_star = brentq(equation, lam_lower, lam_upper)
    
    return lambda_star


#optimal lagrange multiplier


def solve_nu(C_T, Q, Sigma_e, b, gamma, tol=1e-10, max_iter=1000):

    Sigma_inv_half = fractional_matrix_power(Sigma_e, -0.5)

    Q_inv = np.linalg.inv(Q)

    M = Sigma_inv_half @ C_T @ Q_inv @ C_T.T @ Sigma_inv_half

    y = Sigma_inv_half @ b

    I = np.eye(M.shape[0])

    def f(nu):
        X = np.linalg.inv(I + nu * M)
        return y.T @ X @ X @ y - gamma

    # ν*=0 case
    if gamma >= b.T @ np.linalg.solve(Sigma_e, b):
        return 0.0

    lo, hi = 0.0, 1.0
    while f(hi) > 0:
        hi *= 2

    for _ in range(max_iter):
        nu = 0.5 * (lo + hi)
        val = f(nu)

        if abs(val) < tol:
            return nu

        if val > 0:
            lo = nu
        else:
            hi = nu

    return nu

def solve_unu(C_T, Q, b, gamma, tol=1e-10, max_iter=1000):


    Q_inv = np.linalg.inv(Q)

    M = C_T @ Q_inv @ C_T.T 

    y = b

    I = np.eye(M.shape[0])

    def f(nu):
        X = np.linalg.inv(I + nu * M)
        return y.T @ X @ X @ y - gamma

    # ν*=0 case
    if gamma >= b.T @ b:
        return 0.0

    lo, hi = 0.0, 1.0
    while f(hi) > 0:
        hi *= 2

    for _ in range(max_iter):
        nu = 0.5 * (lo + hi)
        val = f(nu)

        if abs(val) < tol:
            return nu

        if val > 0:
            lo = nu
        else:
            hi = nu

    return nu

for t in T_list:
    A, B, NT = get_system(name, n, m, t, seed=0)
    N_T = np.eye(n)
    C_T = NT
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    sigma_w = np.eye(n*t) * 0.1
    sigma_z =  N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, t)@ sigma_0 @ np.linalg.matrix_power(A, t).T
    traceSigma = np.trace(sigma_z)
    lambda_max = np.linalg.eigvalsh(sigma_z).max()
    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = - 1* x_f
    vals = np.linalg.eigvals(sigma_z)
    eig = np.max(vals)
    gamma = epsilon**2 - n #second order
    dev = epsilon**2  #epsilon sd
    lambda_star = find_lambda_star(n, epsilon, 1-delta) #non-central
    marc = (epsilon - np.sqrt(-2 * math.log(1-delta)))**2
    Q_tilde = block_diag(*([Q] * t))
    nu1 = solve_nu(C_T, Q_tilde, sigma_z, b, lambda_star)
    nu2 = solve_nu(C_T, Q_tilde, sigma_z, b, gamma)
    nu3 = solve_nu(C_T, Q_tilde, sigma_z, b, dev)
    nu4 = solve_unu(C_T, Q_tilde, b, lambda_star)

    
    u1 = -nu1 * np.linalg.inv(Q_tilde + nu1*C_T.T @ np.linalg.inv(sigma_z) @ C_T)@C_T.T @ np.linalg.inv(sigma_z) @ b    
 
    u2 = -nu2 * np.linalg.inv(Q_tilde + nu2*C_T.T @ np.linalg.inv(sigma_z) @ C_T)@C_T.T @ np.linalg.inv(sigma_z) @ b      

    u3 =  -nu3 * np.linalg.inv(Q_tilde + nu3*C_T.T @ np.linalg.inv(sigma_z) @ C_T)@C_T.T @ np.linalg.inv(sigma_z) @ b   
    
    u4 = -nu4 * np.linalg.inv(Q_tilde + nu4* C_T.T @ C_T)@C_T.T @b
          
    
    np.random.seed(0)
    x1 = []
    x3 = []
    x5 = []
    x6 = []
    xm = []

    ux1 = []
    ux3 = []
    ux5 = []
    ux6 = []

    bT.append(b.T @ np.linalg.inv(sigma_z) @ b)  
    abs_bT.append(b.T @ b)
    
    for j in range(0, 5000):
        x_0 = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.2)
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u2 + N_T @ w  # second order
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u3 + N_T @ w  # epsilon 
        x5T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u1 + N_T @ w #non-central
        x6T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u4 + N_T @ w #unweighted
        
        x1.append((x1T-x_f).T @ np.linalg.inv(sigma_z) @ (x1T-x_f))
        x3.append((x3T-x_f))
        xm.append((x3T-x_f) @ np.linalg.inv(sigma_z) @ (x3T-x_f))
        ux1.append((x1T-x_f).T @ np.linalg.inv(sigma_z) @ (x1T - x_f))
        ux3.append((x3T-x_f).T @ np.linalg.inv(sigma_z) @(x3T - x_f))
        
        x5.append((x5T - x_f).T @ np.linalg.inv(sigma_z) @ (x5T - x_f))
        ux5.append((x5T - x_f).T @ np.linalg.inv(sigma_z) @ (x5T-x_f))
        count5 = sum(d <= epsilon**2 for d in x5)
        ux6.append((x6T-x_f).T @  (x6T - x_f))

    tdev1.append((C_T @ u1 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u1 + b)+n)
    tdev3.append((C_T @ u2 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u2 + b)+n )
    tdev5.append((C_T @ u3 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u3 + b)+n)

    count1 = np.mean(x1, axis = 0)  
    ratios1.append(count1)
    count3 = np.mean(x3, axis = 0) @ np.linalg.inv(sigma_z) @ np.mean(x3, axis = 0)
    ratios3.append(count3)
    ratios5.append(count5/5000)

    x1_dev.append((np.mean(x1, axis = 0)))
    x3_dev.append(np.mean(xm, axis = 0))
    x5_dev.append((np.mean(x5, axis = 0)))

    ux1_dev.append((np.mean(ux1, axis = 0)))
    ux3_dev.append((np.mean(ux3, axis = 0)))
    ux5_dev.append((np.mean(ux5, axis = 0)))
    ux6_dev.append((np.mean(ux6, axis = 0)))


    # --- Record metrics ---
    ep_norm = float(u3.T @ Q_tilde @ u3)
    sec_norm = float(u2.T @ Q_tilde @ u2)
    exact_norm = float(u1.T @ Q_tilde @ u1)
    unweighted_norm = float(u4.T @ Q_tilde @ u4)

    epu.append(ep_norm)
    secu.append(sec_norm)
    exact.append(exact_norm)
    uw.append(unweighted_norm)


# print(diff)
N = np.arange(0, len(ratios1))
plt.figure(figsize = (8, 6))
plt.plot(T_list, ratios3, '^-',label = r'$\mu_e^\top \Sigma_e^{-1}\mu_e$')
plt.plot(T_list, ratios1, '^-',label = r'$E[e^\top \Sigma_e^{-1}e]$')
plt.plot(T_list, ratios5,'^-', label = r'$P\{e^\top \Sigma_e^{-1}e <= \epsilon^2\}$')
plt.axhline(y=epsilon**2,
            color='black',
            linestyle=':',
            linewidth=2,
            label=r'$\epsilon^2$')

plt.xlabel('time horizon')
plt.ylabel('')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig(f'ratio_{name}.jpeg')
plt.close()


plt.figure(figsize = (8,6))

plt.plot(T_list, x3_dev, 'o-', label = 'mahalanobis')
plt.plot(T_list, tdev3, '^--', label = 'mahalanobis theoretical')
plt.plot(T_list, x1_dev,'o-',  label = 'second order moment')
plt.plot(T_list, tdev1, '^--', label = 'second order theoretical')
plt.plot(T_list, x5_dev, 'o-', label = 'non-central chi')
plt.plot(T_list, tdev5, '^--', label = 'non-central chi theoretical')
# plt.plot(T_list, x6_dev, 'o-', label = 'marc' )
plt.xlabel('time horizon')
plt.ylabel('state dev')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('dev-T--{}-{}-{}_ss{}.jpeg'.format(name, epsilon, delta, j))
plt.close()

plt.figure(figsize = (8,6))


plt.semilogy(T_list, epu, 'o-', label = 'mahalanobis')
plt.semilogy(T_list, secu, 'o-', label = 'second order moment')
plt.semilogy(T_list, exact, 'o-', label = 'non-central chi')
plt.semilogy(T_list, uw, 'o-', label = 'unweighted norm')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('norm-T-{}-{}-{}_ss{}.jpeg'.format(name, epsilon, delta, j))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(T_list, bT)  # your curve
plt.axhline(y=lambda_star, color='r', linestyle='--', label=r'$\gamma$')

plt.xlabel('T')
plt.ylabel(r'$x_f^\top \Sigma_e^{-1}x_f$')
plt.legend()
plt.savefig('bT.jpeg')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(T_list, abs_bT)  # your curve
plt.axhline(y=lambda_star, color='r', linestyle='--', label=r'$\gamma$')

plt.xlabel('T')
plt.ylabel(r'$x_f^\top \Sigma_e^{-1}x_f$')
plt.legend()
plt.savefig('abs_bT.jpeg')
plt.close()

plt.figure(figsize = (8, 6))
plt.semilogy(T_list, ux1_dev, 'o-', label = 'mahalanobis')
plt.semilogy(T_list, ux3_dev, 'o-', label = 'second order moment')
plt.semilogy(T_list, ux5_dev, 'o-', label = 'non-central chi')
plt.semilogy(T_list, ux6_dev, 'o-', label = 'unweighted norm')
plt.xlabel('time horizon')
plt.ylabel('Mean state deviation')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('udev-T-{}-{}-{}_ss{}.jpeg'.format(name, epsilon, delta, j))
plt.close()


# --------------------------------------------------
# Cost vs gamma for fixed horizon T = 6
# --------------------------------------------------

# T_list = [4, 6, 8, 10]
# lambda_star = find_lambda_star(n, epsilon, 1-delta)

# all_gamma = {}
# all_cost = {}
# all_lambda = {}
# all_dev = {}
# all_cost1 = {}
# all_dev1 = {}

# for T in T_list:

#     A, B, NT = get_system(name, n, m, T, seed=0)

#     N_T = np.eye(n)
#     C_T = NT
#     for i in range(1, T):
#         N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

#     b = np.linalg.matrix_power(A, T) @ x_0 - x_f

#     sigma_w = np.eye(n*T) * 0.1
#     sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

#     Q_tilde = block_diag(*([Q] * T))

#     gamma_min = 0.0
#     gamma_max = float(b.T @ np.linalg.inv(sigma_z) @ b)

#     gamma_vals = np.linspace(gamma_min, gamma_max, 10)

#     cost_vals = []
#     lambda_vals = []
#     dev_vals = []
#     cost1_vals = []
#     dev1_vals = []

#     for gamma in gamma_vals:

#         u = cp.Variable(m*T)
#         u1 = cp.Variable(m*T)

#         objective = cp.Minimize(cp.quad_form(u, Q_tilde))
#         objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))

#         constraint = (
#             cp.quad_form(
#                 C_T @ u + b,
#                 np.linalg.inv(sigma_z)
#             ) <= gamma
#         )
#         constraint1 = (
#             cp.quad_form(C_T@u1 + b,
#                          np.identity(n)
#             ) <= gamma
#         )

#         prob = cp.Problem(objective, [constraint])
#         prob1 = cp.Problem(objective1, [constraint1])
#         prob.solve(solver=cp.SCS, verbose=False)
#         prob1.solve(solver=cp.SCS, verbose= False)

#         if u.value is None:
#             cost_vals.append(np.nan)
#             lambda_vals.append(np.nan)
#             dev_vals.append(np.nan)
#             continue

#         if u1.value is None:
#             cost1_vals.append(np.nan)
#             dev1_vals.append(np.nan)
#             continue

#         Jstar = float(u.value.T @ Q_tilde @ u.value)
#         J1star = float(u1.value.T @ Q_tilde @ u1.value)
#         cost_vals.append(Jstar)
#         cost1_vals.append(J1star)

#         # try:
#         #     lam = solve_nu(
#         #         C_T,
#         #         Q_tilde,
#         #         sigma_z,
#         #         b,
#         #         gamma
#         #     )
#         # except:
#         #     lam = np.nan

#         # lambda_vals.append(lam)

#         idev = []
#         idev1 = []
#         for _ in range(5000):
#             x_0 = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.2)
#             w = np.random.multivariate_normal(
#                 np.zeros(n*T),
#                 np.eye(n*T)*0.1
#             )

#             xT = (
#                 np.linalg.matrix_power(A, T) @ x_0
#                 + C_T @ u.value
#                 + N_T @ w
#             )
#             xT1 = (np.linalg.matrix_power(A, T) @ x_0
#                 + C_T @ u1.value
#                 + N_T @ w)
            
#             e = xT - x_f
#             e1 = xT1 - x_f

#             idev.append(
#                 e.T @ e
#             )
#             idev1.append(e1.T @ e1)

#         dev_vals.append(np.mean(idev))
#         dev1_vals.append(np.mean(idev1))

#     all_gamma[T] = gamma_vals
#     all_cost[T] = cost_vals
#     all_cost1[T] = cost1_vals
#     all_lambda[T] = lambda_vals
#     all_dev[T] = dev_vals
#     all_dev1[T] = dev1_vals

# plt.figure(figsize=(10,6))

# for T in T_list:
#     plt.plot(all_gamma[T],all_cost[T],marker='^',linewidth=2,label=f"T={T}")
#     plt.plot(all_gamma[T], all_cost1[T], marker = 'o', linewidth = 2, label = f"unweighted T = {T}")

# plt.xlabel(r'$\gamma$')
# plt.ylabel(r'$J^*(\gamma)$')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'cost_vs_gamma_allT_{name}.jpeg')
# plt.show()

# plt.figure(figsize=(10,6))


# for T in T_list:
#     plt.semilogy(all_gamma[T],all_dev[T],marker='^',linewidth=2,label=f"T={T}")
#     plt.semilogy(all_gamma[T], all_dev1[T], marker = 'o', linewidth=2, label =f"unweighted T = {T}")


# plt.xlabel(r'$\gamma$')
# plt.ylabel('Mean deviation')
# plt.legend()
# plt.grid(True)
# plt.savefig(f"deviation_vs_gamma_allT_{name}.jpeg")
# plt.show()

