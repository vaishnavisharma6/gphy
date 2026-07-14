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

n, m = 4, 4
x_0 = np.array([1.0, 2.0, 3.0, 4.0])
x_f = np.array([4.0, 5.0, 6.0, 7.0])
T_list = range(1, 25, 2)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
delta = 0.5
Q = np.diag([2.0, 3.0, 4.0, 5.0])
q = norm.ppf(1 - delta)
u_norm_diff_list = []
state_dev_diff_list = []
name = 'stable'
norm_folder = 'norms_{}'.format(name)
state_folder = 'states_{}'.format(name)
diff_folder = 'diffs_{}'.format(name)
delta_folder = 'traj_{}'.format(delta)
exact_folder = 'traj_{}'.format(delta)
epsilon = 2.4
os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)
os.makedirs(delta_folder, exist_ok = True)


secd = []
epd = []
firstd = []
modeld = []

secu = []
epu = []
exact = []
alleps = []

x1_dev = []
x3_dev = []
x5_dev = []

tdev1 = []
tdev3 = []
tdev5 = []

g1 = []
g2 = []
g3 = []
g4 = []
g5 = []


all_lam = []

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

ratios1 = []
ratios3 = []
ratios5 = []

bT = []


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
def solve_lambda(C_T, Qp, V, b, gamma, tol=1e-10, max_iter=1000):
    """
    Computes optimal lambda*.

    Parameters
    ----------
    C_T : matrix
    Qp : matrix (Q')
    V : matrix
    b : vector
    gamma : float
    """

    # V^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(V)
    V_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Q'^{-1}
    Qp_inv = np.linalg.inv(Qp)

    # build symmetric matrix
    M = V_inv_half @ C_T @ Qp_inv @ C_T.T @ V_inv_half

    # eigen decomposition
    delta, W = np.linalg.eigh(M)

    # compute b_tilde
    y = V_inv_half @ b
    b_tilde = W.T @ y

    # function to solve
    def f(lam):
        return np.sum((b_tilde**2) / (1 + lam * delta)**2) - gamma

    # bracket root
    lo = 0.0
    hi = 1.0
    while f(hi) > 0:
        hi *= 2

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)

        if abs(val) < tol:
            return mid

        if val > 0:
            lo = mid
        else:
            hi = mid

    return mid


for t in T_list:
    A,B, NT = get_system(name, n, m, t, seed=0)
    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = NT
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    sigma_w = np.eye(n*t) * 0.1
    sigma_z =  N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)

    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = (np.linalg.matrix_power(A, t) @ x_0) - x_f
    vals = np.linalg.eigvals(sigma_z)
    eig = np.max(vals)
    gamma = epsilon**2 - n
    dev = epsilon**2  #epsilon sd
    lambda_star = find_lambda_star(n, epsilon, 1-delta)
    marc = (epsilon - np.sqrt(-2 * math.log(1-delta)))**2
    Q_tilde = block_diag(*([Q] * t))


    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u3 = cp.Variable(m*t)
    u5 = cp.Variable(m*t)

    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective5 = cp.Minimize(cp.quad_form(u5, Q_tilde))

    f = cp.quad_form(C_T @ u + b, np.linalg.inv(sigma_z))
    f3 = cp.quad_form(C_T @ u3 + b, np.linalg.inv(sigma_z))
    f5 = cp.quad_form(C_T@ u5 + b, np.linalg.inv(sigma_z))


    prob = cp.Problem(objective, [f <= gamma]) #second order
    prob3 = cp.Problem(objective3, [f3 <= dev]) #mahalanobis
    prob5 = cp.Problem(objective5, [f5 <= lambda_star]) #non-central chi

    prob.solve(solver=cp.SCS, verbose=False)
    prob3.solve(solver = cp.SCS, verbose = False)
    prob5.solve(solver = cp.SCS, verbose = False)

    u_convex = u.value  #second order
    u_c3 = u3.value #epsilon sd
    u_c5 = u5.value

    if u_convex is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c3 is None:
        print(f'Convex solver failed at T={t}')
    if u_c5 is None:
        print(f'Convex solver failed at T={t}')    
      
    if t ==19 or t == 21:  
        print('t,:', t)
        print('u:', u_convex)

    bT.append(b.T @ np.linalg.inv(sigma_z) @ b)    
    # --- Simulate both systems ---
    np.random.seed(0)
    x1 = []
    x3 = []
    x5 = []

    s1 = []
    s3 = []
    s5 = []


    for j in range(0, 5000):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_convex + N_T @ w  # second order
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c3 + N_T @ w  # epsilon 
        x5T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c5 + N_T @ w #non-central
        
        x1.append((x1T-x_f).T @ np.linalg.inv(sigma_z) @ (x1T-x_f))
        x3.append((x3T-x_f))
        
        x5.append((x5T - x_f).T @ np.linalg.inv(sigma_z) @ (x5T - x_f))
        count5 = sum(d <= epsilon**2 for d in x5)
        
        s1.append(x1T)
        s3.append(x3T)
        s5.append(x5T)

    
    tdev1.append((C_T @ u_convex + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_convex + b)+n)
    tdev3.append((C_T @ u_c3 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c3 + b)+n )
    tdev5.append((C_T @ u_c5 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c5 + b)+n)


    count1 = np.mean(x1, axis = 0)  
    ratios1.append(count1)
    count3 = np.mean(x3, axis = 0) @ np.linalg.inv(sigma_z) @ np.mean(x3, axis = 0)
    ratios3.append(count3)
    ratios5.append(count5/5000)

    x1_dev.append((np.mean(x1, axis = 0)))
    x3_dev.append((np.mean(x3, axis = 0)))
    x5_dev.append((np.mean(x5, axis = 0)))

    s1 = np.array(s1)
    s3 = np.array(s3)
    s5 = np.array(s5)

    # --- Record metrics ---
    u_convex = u.value.reshape(-1, 1)
    u_c3     = u3.value.reshape(-1, 1)
    u_c5 = u5.value.reshape(-1, 1)

    ep_norm = float(u_c3.T @ Q_tilde @ u_c3)
    sec_norm = float(u_convex.T @ Q_tilde @ u_convex)
    exact_norm = float(u_c5.T @ Q_tilde @ u_c5)


    epu.append(ep_norm)
    secu.append(sec_norm)
    exact.append(exact_norm)


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
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('second order moment, horizon = {}, ratio = {}'.format(t, count1/5000))

    plt.savefig(filename1)


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
    plt.title('mahalanobis, horizon = {}, ratio = {}'.format(t, count3/5000))
    plt.legend()
    plt.savefig(filename3)

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
    plt.title('non-central chi, horizon = {}, ratio = {}'.format(t, count5/5000))
    plt.legend()
    plt.savefig(filename5)

plt.close()

N = np.arange(0, len(ratios1))
plt.figure(figsize = (10, 6))
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
plt.savefig('ratio.jpeg')
plt.close()


plt.figure(figsize = (10,6))

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
plt.savefig('dev-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()

plt.figure(figsize = (10,6))


plt.semilogy(T_list, epu, 'o-', label = 'mahalanobis')
plt.semilogy(T_list, secu, 'o-', label = 'second order moment')
plt.semilogy(T_list, exact, 'o-', label = 'non-central chi')
plt.xlabel('time horizon')
plt.ylabel('u* norm')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('norm-T-{}-{}_ss{}.jpeg'.format(epsilon, delta, j))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(T_list, bT)  # your curve
plt.axhline(y=lambda_star, color='r', linestyle='--', label=r'$\gamma$')

plt.xlabel('T')
plt.ylabel(r'$b^\top \Sigma_e^{-1}b$')
plt.legend()
plt.savefig('bT.jpeg')
plt.close()


# --------------------------------------------------
# Cost vs gamma for fixed horizon T = 6
# --------------------------------------------------

T_list = [4, 6, 8]
lambda_star = find_lambda_star(n, epsilon, 1-delta)

all_gamma = {}
all_cost = {}
all_lambda = {}
all_dev = {}

for T in T_list:

    A, B, NT = get_system(name, n, m, T, seed=0)

    N_T = np.eye(n)
    C_T = NT
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b = np.linalg.matrix_power(A, T) @ x_0 - x_f

    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T

    Q_tilde = block_diag(*([Q] * T))

    gamma_min = 0.0
    gamma_max = float(b.T @ np.linalg.inv(sigma_z) @ b)

    gamma_vals = np.linspace(gamma_min, gamma_max, 10)

    cost_vals = []
    lambda_vals = []
    dev_vals = []

    for gamma in gamma_vals:

        u = cp.Variable(m*T)

        objective = cp.Minimize(cp.quad_form(u, Q_tilde))

        constraint = (
            cp.quad_form(
                C_T @ u + b,
                np.linalg.inv(sigma_z)
            ) <= gamma
        )

        prob = cp.Problem(objective, [constraint])
        prob.solve(solver=cp.SCS, verbose=False)

        if u.value is None:
            cost_vals.append(np.nan)
            lambda_vals.append(np.nan)
            dev_vals.append(np.nan)
            continue

        Jstar = float(u.value.T @ Q_tilde @ u.value)
        cost_vals.append(Jstar)

        try:
            lam = solve_lambda(
                C_T,
                Q_tilde,
                sigma_z,
                b,
                gamma
            )
        except:
            lam = np.nan

        lambda_vals.append(lam)

        idev = []
        for _ in range(5000):
            w = np.random.multivariate_normal(
                np.zeros(n*T),
                np.eye(n*T)*0.1
            )

            xT = (
                np.linalg.matrix_power(A, T) @ x_0
                + C_T @ u.value
                + N_T @ w
            )

            e = xT - x_f

            idev.append(
                e.T @ np.linalg.inv(sigma_z) @ e
            )

        dev_vals.append(np.mean(idev))

    all_gamma[T] = gamma_vals
    all_cost[T] = cost_vals
    all_lambda[T] = lambda_vals
    all_dev[T] = dev_vals

plt.figure(figsize=(10,6))

for T in T_list:
    plt.plot(
        all_gamma[T],
        all_cost[T],
        marker='^',
        linewidth=2,
        label=f"T={T}"
    )
 

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.legend()
plt.grid(True)
plt.savefig('cost_vs_gamma_allT.jpeg')
plt.show()

plt.figure(figsize=(10,6))


for T in T_list:
    plt.plot(
        all_gamma[T],
        all_dev[T],
        marker='^',
        linewidth=2,
        label=f"T={T}"
    )


plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\mathbb{E}[e^\top \Sigma_e^{-1} e]$')
plt.legend()
plt.grid(True)
plt.savefig('deviation_vs_gamma_allT.jpeg')
plt.show()

