#results for stable case
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib as mpl
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
import GPy

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

mpl.rcParams.update({
    "figure.dpi": 150,        # display quality
    "savefig.dpi": 600,       # high-resolution saved raster images
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "text.usetex": False,     # avoids your current LaTeX-not-found error
})

#----------------------------------------------------------------------------------------
# fixed system parameters

n, m = 4, 4
sigma_0 = np.eye(n) * 2.0
x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
x_f = np.array([4.0, 5.0, 6.0, 7.0])
T_list = range(1, 30, 2)  
sigma = np.eye(n) * 0.1
delta = 0.4
Q = np.diag([2.0, 3.0, 4.0, 5.0])
q = norm.ppf(1 - delta)
name = 'stable'
delta_folder = 'traj_{}'.format(delta)
epsilon = 2.4
os.makedirs(delta_folder, exist_ok = True)

#normalized norm
secu = []
epu = []
exact = []

#absolute norm
abs_secu = []
abs_exactu = []
abs_chebyu = []

#normalized dev
x1_dev = []
x3_dev = []
x5_dev = []

#absolute dev
abs_secdev = []
abs_exactdev = []
abs_chebydev = []

tdev1 = []
tdev3 = []
tdev5 = []

abs_tsecdev = []
abs_texactdev = []
abs_tchebydev = []

ratios1 = []
ratios3 = []
ratios5 = []

abs_ratiosec = []
abs_ratioexact = []
abs_ratiocheby = []

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
    sigma_z =  N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T
    traceSigma = np.trace(sigma_z)

    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = -1*  x_f
    vals = np.linalg.eigvals(sigma_z)
    eig = np.max(vals)
    gamma = epsilon**2 - n
    dev = epsilon**2  #epsilon sd
    lambda_star = find_lambda_star(n, epsilon, 1-delta)
    marc = (epsilon - np.sqrt(-2 * math.log(1-delta)))**2

    cheby = (epsilon - np.sqrt(np.trace(sigma_z)/delta))**2
    sec = epsilon**2 - np.trace(sigma_z)
    Q_tilde = block_diag(*([Q] * t))


    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u3 = cp.Variable(m*t)
    u5 = cp.Variable(m*t)

    u1 = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    u4 = cp.Variable(m*t)

    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective5 = cp.Minimize(cp.quad_form(u5, Q_tilde))

    objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))

    f = cp.quad_form(C_T @ u + b, np.linalg.inv(sigma_z))
    f3 = cp.quad_form(C_T @ u3 + b, np.linalg.inv(sigma_z))
    f5 = cp.quad_form(C_T@ u5 + b, np.linalg.inv(sigma_z))

    f1 = cp.quad_form(C_T @ u1 + b, np.identity(n))
    f2 = cp.quad_form(C_T @ u2 + b, np.identity(n))
    f4 = cp.quad_form(C_T @ u4 + b, np.identity(n))


    prob = cp.Problem(objective, [f <= gamma]) #second order
    prob3 = cp.Problem(objective3, [f3 <= dev]) #mahalanobis
    prob5 = cp.Problem(objective5, [f5 <= lambda_star]) #non-central chi

    prob1 = cp.Problem(objective1, [f1 <= lambda_star])# absolute exact
    prob2 = cp.Problem(objective2, [f2 <= sec])# absolute second order
    prob4 = cp.Problem(objective4, [f4 <= cheby])# absolute chebyshev

    prob.solve(solver=cp.SCS, verbose=False)
    prob3.solve(solver = cp.SCS, verbose = False)
    prob5.solve(solver = cp.SCS, verbose = False)

    prob1.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver=cp.SCS, verbose=False)
    prob4.solve(solver=cp.SCS, verbose=False)

    u_convex = u.value  #second order
    u_c3 = u3.value #epsilon sd
    u_c5 = u5.value # non-central

    u_c1 = u1.value #exact absolute
    u_c2 = u2.value # second order absolute
    u_c4 = u4.value # cheyshev absolute

    if u_convex is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c3 is None:
        print(f'Convex solver failed at T={t}')
    if u_c5 is None:
        print(f'Convex solver failed at T={t}') 

    if u_c1 is None:
        print(f"Convex solver failed at T={t}")
    if u_c2 is None:
        print(f"Convex solver failed at T={t}")
    if u_c4 is None:
        print(f"Convex solver failed at T={t}")        

    bT.append(b.T @ np.linalg.inv(sigma_z) @ b)  
    abs_bT.append(b.T @ b)  
    # --- Simulate both systems ---
    np.random.seed(0)
    x1 = []
    x3 = []
    x5 = []
    xm = []

    x2 = [] #absolute exact
    x4 = [] # absolute second
    x6 = [] #absolute cheby

    s1 = []
    s3 = []
    s5 = []

    s2 = []
    s4 = []
    s6 = []

    for j in range(0, 5000):
        x_0 = np.random.multivariate_normal(np.zeros(n), sigma_0)
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_convex + N_T @ w  # second order
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c3 + N_T @ w  # epsilon 
        x5T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c5 + N_T @ w #non-central
        
        x2T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c1 + N_T @ w # exact absolute
        x4T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c2 + N_T @ w # second absolute
        x6T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c4 + N_T @ w # chebyshev absolute

        
        x1.append((x1T-x_f).T @ np.linalg.inv(sigma_z) @ (x1T-x_f))
        x3.append((x3T-x_f))
        xm.append((x3T-x_f) @ np.linalg.inv(sigma_z) @ (x3T-x_f))
        
        x5.append((x5T - x_f).T @ np.linalg.inv(sigma_z) @ (x5T - x_f))
        count5 = sum(d <= epsilon**2 for d in x5)

        x2.append((x2T - x_f))# exact absolute
        x4.append((x4T - x_f)) # second absolute
        x6.append((x6T - x_f)) # cheby absolute

    tdev1.append((C_T @ u_convex + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_convex + b)+n)
    tdev3.append((C_T @ u_c3 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c3 + b)+n )
    tdev5.append((C_T @ u_c5 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c5 + b)+n)

    abs_texactdev.append((C_T @ u_c1 + b).T @ (C_T @ u_c1 + b))
    abs_tsecdev.append((C_T @ u_c2 + b).T @ (C_T @ u_c2 + b))
    abs_tchebydev.append((C_T @ u_c4 + b).T @ (C_T @ u_c4 + b))
    
    count1 = np.mean(x1, axis = 0)  
    ratios1.append(count1)
    count3 = np.mean(x3, axis = 0) @ np.linalg.inv(sigma_z) @ np.mean(x3, axis = 0)
    ratios3.append(count3)
    ratios5.append(count5/5000)
    count6 = (x3T-x_f).T @np.linalg.inv(sigma_z) @ (x3T - x_f)

    x1_dev.append((np.mean(x1, axis = 0)))
    x3_dev.append(np.mean(xm, axis = 0))
    x5_dev.append((np.mean(x5, axis = 0)))
    abs_exactdev.append(np.mean(x2, axis = 0) @ np.mean(x2, axis = 0))# deviation for abs exact
    abs_secdev.append(np.mean(x4, axis = 0) @ np.mean(x4, axis = 0)) # deviation for abs second
    abs_chebydev.append(np.mean(x6, axis = 0) @ np.mean(x6, axis = 0)) # deviation for abs chebyshev

    # --- Record metrics ---
    u_convex = u.value.reshape(-1, 1)
    u_c3  = u3.value.reshape(-1, 1)
    u_c5 = u5.value.reshape(-1, 1)

    u_c1 = u1.value.reshape(-1, 1)
    u_c2 = u2.value.reshape(-1, 1)
    u_c4 = u4.value.reshape(-1, 1)

    ep_norm = float(u_c3.T @ Q_tilde @ u_c3)
    sec_norm = float(u_convex.T @ Q_tilde @ u_convex)
    exact_norm = float(u_c5.T @ Q_tilde @ u_c5)

    abs_exactnorm = float(u_c1.T @ Q_tilde @ u_c1)
    abs_secnorm = float(u_c2.T @ Q_tilde @ u_c2)
    abs_chebynorm = float(u_c4.T @ Q_tilde @ u_c4)

    epu.append(ep_norm)
    secu.append(sec_norm)
    exact.append(exact_norm)

    abs_exactu.append(abs_exactnorm)
    abs_secu.append(abs_secnorm)
    abs_chebyu.append(abs_chebynorm)

N = np.arange(0, len(ratios1))

plt.figure(figsize=(10, 6))

plt.plot(
    T_list,
    ratios3,
    '^-',
    label=r'$\mu_e^\top \Sigma_e^{-1}\mu_e$'
)

plt.plot(
    T_list,
    ratios1,
    '^-',
    label=r'$\mathbb{E}\!\left[e^\top \Sigma_e^{-1} e\right]$'
)

plt.plot(
    T_list,
    ratios5,
    '^-',
    label=r'$\mathbb{P}\!\left\{e^\top \Sigma_e^{-1} e \leq \epsilon^2\right\}$'
)

plt.axhline(
    y=epsilon**2,
    color='black',
    linestyle=':',
    linewidth=2,
    label=r'$\epsilon^2$'
)

plt.xlabel(r'$T$')
plt.ylabel('')

plt.legend()
plt.grid(True)
plt.xticks(T_list)

plt.tight_layout()
plt.savefig('ratio.pdf', bbox_inches='tight')
plt.close()

  # 1 row, 2 columns
# ---------------- Left plot: Deviation ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes[0]

ax.plot(T_list, x3_dev, 'o-', label=r'Mahalanobis')
ax.plot(T_list, tdev3, '^--', label=r'Mahalanobis theoretical')
ax.plot(T_list, x1_dev, 'o-', label=r'Second-order moment')
ax.plot(T_list, tdev1, '^--', label=r'Second-order theoretical')
ax.plot(T_list, x5_dev, 'o-', label=r'Non-central $\chi^2$')
ax.plot(T_list, tdev5, '^--', label=r'Non-central $\chi^2$ theoretical')

ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\mathbb{E}\!\left[e^\top \Sigma_e^{-1} e\right]$')
ax.set_xticks(T_list)
ax.legend()
# ax.grid(True)
# ax.set_title(r'(a) Deviation')

# ---------------- Right plot: Optimal cost ----------------
ax = axes[1]

ax.semilogy(T_list, epu, 'o-', label=r'Mahalanobis')
ax.semilogy(T_list, secu, 'o-', label=r'Second-order moment')
ax.semilogy(T_list, exact, 'o-', label=r'Non-central $\chi^2$')

ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$J^\star$')
ax.set_xticks(T_list)
# ax.grid(True)
# ax.set_title(r'(b) Optimal cost')

ax.legend()

plt.tight_layout()
plt.savefig('combined_T.pdf', bbox_inches='tight')
plt.show()
# ---------------- Right plot ----------------
# axs[1].semilogy(T_list, abs_exactu, 'o-', label='Absolute exact')
# axs[1].semilogy(T_list, abs_secu, 'o-', label='Absolute second')
# axs[1].semilogy(T_list, abs_chebyu, 'o-', label='Absolute Chebyshev')

# axs[1].set_xlabel('Time horizon')
# axs[1].set_ylabel('J*')
# # axs[1].set_xticks(T_list)
# # axs[1].grid(True, which='both')
# axs[1].legend()


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# ---------------- Left plot ----------------
axs[0].plot(T_list, bT, 'o-', label=r'$x_f^\top\Sigma_e^{-1}x_f$')
axs[0].axhline(y=lambda_star, color='r', linestyle='--',
               label=r'$\gamma$')

axs[0].set_xlabel('Time horizon')
axs[0].set_ylabel(r'$x_f^\top \Sigma_e^{-1} x_f$')
# axs[0].set_xticks(T_list)
# axs[0].grid(True)
axs[0].legend()

# ---------------- Right plot ----------------
axs[1].plot(T_list, abs_bT, 'o-', label=r'$x_f^\top\Sigma_e^{-1}x_f$')
axs[1].axhline(y=lambda_star, color='r', linestyle='--',
               label=r'$\gamma$')

axs[1].set_xlabel('Time horizon')
axs[1].set_ylabel(r'$x_f^\top x_f$')
# axs[1].set_xticks(T_list)
# axs[1].grid(True)
axs[1].legend()

plt.tight_layout()

plt.savefig(
    'combined_bT_{}_{}_{}.jpeg'.format(epsilon, delta, name),
    dpi=300,
    bbox_inches='tight'
)

# --------------------------------------------------
# Cost vs gamma for fixed horizon T = 6
# --------------------------------------------------

T_list = [4, 6, 8]
lambda_star = find_lambda_star(n, epsilon, 1-delta)

all_gamma = {}
all_cost = {}
all_lambda = {}
all_dev = {}
all_cost1 = {}
all_dev1 = {}

for T in T_list:

    A, B, NT = get_system(name, n, m, T, seed=0)

    N_T = np.eye(n)
    C_T = NT
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b = -1 *  x_f

    sigma_w = np.eye(n*T) * 0.1
    sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

    Q_tilde = block_diag(*([Q] * T))

    gamma_min = 0.0
    gamma_max = float(b.T @ np.linalg.inv(sigma_z) @ b)

    gamma_vals = np.linspace(gamma_min, gamma_max, 10)

    cost_vals = []
    lambda_vals = []
    dev_vals = []
    cost1_vals = []
    dev1_vals = []

    for gamma in gamma_vals:

        u = cp.Variable(m*T)
        u1 = cp.Variable(m*T)

        objective = cp.Minimize(cp.quad_form(u, Q_tilde))
        objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))

        constraint = (
            cp.quad_form(
                C_T @ u + b,
                np.linalg.inv(sigma_z)
            ) <= gamma
        )
        constraint1 = (
            cp.quad_form(C_T@u1 + b,
                         np.identity(n)
            ) <= gamma
        )

        prob = cp.Problem(objective, [constraint])
        prob1 = cp.Problem(objective1, [constraint1])
        prob.solve(solver=cp.SCS, verbose=False)
        prob1.solve(solver=cp.SCS, verbose= False)

        if u.value is None:
            cost_vals.append(np.nan)
            lambda_vals.append(np.nan)
            dev_vals.append(np.nan)
            continue

        if u1.value is None:
            cost1_vals.append(np.nan)
            dev1_vals.append(np.nan)
            continue

        Jstar = float(u.value.T @ Q_tilde @ u.value)
        J1star = float(u1.value.T @ Q_tilde @ u1.value)
        cost_vals.append(Jstar)
        cost1_vals.append(J1star)

        # try:
        #     lam = solve_nu(
        #         C_T,
        #         Q_tilde,
        #         sigma_z,
        #         b,
        #         gamma
        #     )
        # except:
        #     lam = np.nan

        # lambda_vals.append(lam)

        idev = []
        idev1 = []
        for _ in range(5000):
            x_0 = np.random.multivariate_normal(np.zeros(n), np.eye(n)*0.2)
            w = np.random.multivariate_normal(
                np.zeros(n*T),
                np.eye(n*T)*0.1
            )

            xT = (
                np.linalg.matrix_power(A, T) @ x_0
                + C_T @ u.value
                + N_T @ w
            )
            xT1 = (np.linalg.matrix_power(A, T) @ x_0
                + C_T @ u1.value
                + N_T @ w)
            
            e = xT - x_f
            e1 = xT1 - x_f

            idev.append(
                e.T @ e
            )
            idev1.append(e1.T @ e1)

        dev_vals.append(
        (C_T @ u.value + b).T
        @ np.linalg.inv(sigma_z)
        @ (C_T @ u.value + b)
)
        dev1_vals.append(np.mean(idev1))

    all_gamma[T] = gamma_vals
    all_cost[T] = cost_vals
    all_cost1[T] = cost1_vals
    all_lambda[T] = lambda_vals
    all_dev[T] = dev_vals
    all_dev1[T] = dev1_vals

# plt.figure(figsize=(10,6))

# for T in T_list:
#     plt.plot(all_gamma[T],all_cost[T],marker='^',linewidth=2,label=f"normalized T={T}")
#     plt.plot(all_gamma[T], all_cost1[T], marker = 'o', linewidth = 2, label = f"unweighted T = {T}")

# plt.xlabel(r'$\gamma$')
# plt.ylabel(r'$J^*(\gamma)$')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'cost_vs_gamma_allT_{name}.jpeg')
# plt.show()

# plt.figure(figsize=(10,6))


# for T in T_list:
#     plt.plot(all_gamma[T],all_dev[T],marker='^',linewidth=2,label=f"normalized T={T}")
#     plt.plot(all_gamma[T], all_dev1[T], marker = 'o', linewidth=2, label =f"unweighted T = {T}")


# plt.xlabel(r'$\gamma$')
# plt.ylabel('Mean deviation')
# plt.legend()
# plt.grid(True)
# plt.savefig(f"deviation_vs_gamma_allT_{name}.jpeg")
# plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# ---------------- Left plot: Cost vs gamma ----------------
for T in T_list:
    axs[0].semilogy(
        all_gamma[T], all_cost[T],
        linewidth=2,
        label=rf'$T={T}$'
    )

axs[0].set_xlabel(r'$\gamma$')
axs[0].set_ylabel(r'$J^\star(\gamma)$')
axs[0].legend(fontsize=8)

# ---------------- Right plot: Deviation vs gamma ----------------
for T in T_list:
    axs[1].semilogy(
        all_gamma[T], all_dev[T],
        linewidth=2,
        label=rf'$T={T}$'
    )

axs[1].set_xlabel(r'$\gamma$')
axs[1].set_ylabel(
    r'$\mathbb{E}\!\left[e^\top \Sigma_e^{-1} e\right]$'
)
axs[1].legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    f'combined_gamma_plots_{name}.pdf',
    bbox_inches='tight'
)

plt.show()