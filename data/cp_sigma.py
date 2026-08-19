import numpy as np
import matplotlib.pyplot as plt
from datagen import get_system
import os
import math
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.stats import ncx2
from scipy.optimize import brentq
import matplotlib.patches as patches

#system and parameters

n,m = 4,4
xf = np.array([4.0, 5.0, 6.0, 7.0])
delta = 0.5
Q = np.diag([2.0, 3.0, 4.0, 5.0])
name = 'unstable'
epsilon = 3.0
M = 100
k_est = 10

#to store optimal costs
sampleu = []
structu = []
cpu = []
originalu = []
originalnd = []

quant = []
empirical_coverage = {}
quantiles = {}
delta_list = [0.30, 0.20, 0.10, 0.05, 0.01]
num_runs = 20
N_test = 1000

#to store final state deviation
x1_dev = []
x2_dev = []
xk_dev = []
xkd_dev = []
x3_dev = []


def conformal_quantile(scores, delta):
    N_cal = len(scores)

    k = int(np.ceil((N_cal + 1) * (1 - delta)))
    k = min(k, N_cal)

    q = np.sort(scores)[k - 1]

    return q


def get_scores(U_cal, N_cal, CT_hat, A, NT, CT, t, w_cal_train, x_cal_train):
    scores = []
    for i in range(N_cal):
        w =  w_cal_train[i]
        x_0 = x_cal_train[i]
        # x_0 = np.random.multivariate_normal(np.zeros(n), sigma_0)
        x_true = np.linalg.matrix_power(A, t) @ x_0+ CT @ U_cal[:, i] + NT@w
        x_cal = CT_hat @ U_cal[:, i]
        scores.append(np.linalg.norm(x_true - x_cal))

    return(np.array(scores))


# def get_test_scores(U_test, CT_hat, A,  NT, CT, t, sigma, sigma1):

#     scores = []
#     It = np.eye(t)
#     sigmaw = np.kron(It, sigma1**2 * np.eye(n))
#     for i in range(U_test.shape[1]):

#         w = np.random.multivariate_normal(
#             np.zeros(n*t),
#             sigmaw
#         )
#         x_0 = np.random.multivariate_normal(
#             np.zeros(n),
#             np.eye(n)*sigma**2
#         )

#         x_true = np.linalg.matrix_power(A, t) @x_0 + CT @ U_test[:, i] + NT @ w
#         x_pred = CT_hat @ U_test[:, i]

#         score = np.linalg.norm(x_true - x_pred)

#         scores.append(score)

#     return np.array(scores)

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
    lam_upper = 1e11

    # Ensure bracket contains root
    if equation(lam_lower) * equation(lam_upper) > 0:
        raise ValueError("Root not bracketed. Increase lam_upper.")

    lambda_star = brentq(equation, lam_lower, lam_upper)
    
    return lambda_star

T = 3
all_sigma = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_known = {}
quant1 = []
sigma_1 = 0.2
conf_infeasible = []

A, B, CT = get_system(name, n, m, T, seed=0)
N_T = np.eye(n)
for i in range(1, T):
    N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

Q_tilde = block_diag(*([Q] * T))

N = m*T + 10
N_cal = 60

U = np.random.randn(m*T, N)
U_cal = np.random.randn(m*T, N_cal)

sigma_vals = np.linspace(0.3, 7, 40)

# ---------- initialize ONCE ----------
cost_struct = []
cost_conf = []
cost_known = []
cost_abs = []

dev_struct = []
dev_conf = []
dev_known = []
dev_abs = []

lambda_star = find_lambda_star(n, epsilon, 1-delta)
z_x_train = np.random.randn(N, M, n)
z_x_cal   = np.random.randn(N_cal, n)

x_train = sigma_1 * z_x_train
x_cal_train = sigma_1 * z_x_cal

z_w_train = np.random.randn(N, M, n*T)
z_w_cal   = np.random.randn(N_cal, n*T)
sigma_0 = np.eye(n) * sigma_1**2

q_sigma = []
It = np.eye(T)
# -------------------------------------
for sigma in sigma_vals:
    sigma_w = np.kron(It, sigma**2 * np.eye(n))
    sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))
    w_train = sigma * z_w_train
    w_cal_train = sigma * z_w_cal

    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w =  w_train[i, j]
            x_0 =  x_train[i, j]
            xT = np.linalg.matrix_power(A, T) @ x_0 + CT @ U[:, i] + N_T @ w  

            all_xT[:, i, j] = xT
            sumx += xT

        X_bar[:, i] = sumx / M

    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

    sample_sigma = np.zeros((n, n))
    struct_sigma = np.zeros((n, n))

    for i in range(N):
        for j in range(M):
            diff = all_xT[:, i, j] - X_bar[:, i]
            diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]

            sample_sigma += np.outer(diff, diff)
            struct_sigma += np.outer(diff1, diff1)

    sample_sigma /= (N*M)
    struct_sigma /= (N*M)

    scores = get_scores(
        U_cal,
        N_cal,
        CT_hat,
        A,
        N_T,
        CT,
        T,
        w_cal_train,
        x_cal_train       # after modifying get_scores
    )

    q = conformal_quantile(scores, delta)
    conf_infeasible.append(q >= epsilon)
    q_sigma.append(q)
    lambda_cp = (epsilon-q)**2

    u1 = cp.Variable(m*T)
    u2 = cp.Variable(m*T)
    u3 = cp.Variable(m*T)
    u4 = cp.Variable(m*T)

    objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))

    constraint1 = cp.quad_form(
        CT_hat @ u1 - xf,
        np.linalg.inv(struct_sigma)
    ) <= lambda_star

    constraint2 = cp.quad_form(
        CT_hat @ u2 - xf,
        np.eye(n)
    ) <= lambda_cp

    constraint3 = cp.quad_form(
        CT @ u3 - xf,
        np.linalg.inv(sigma_z)
    ) <= lambda_star

    constraint4 = cp.quad_form(
        CT @ u4 - xf,
        np.eye(n)
    )<= epsilon**2

    cp.Problem(objective1, [constraint1]).solve(solver=cp.SCS)
    cp.Problem(objective2, [constraint2]).solve(solver=cp.SCS)
    cp.Problem(objective3, [constraint3]).solve(solver=cp.SCS)
    cp.Problem(objective4, [constraint4]).solve(solver=cp.SCS)

    if u1 is None:
        print(f"1st Convex solver failed")
        continue

    if u2 is None:
        print(f"2nd Convex solver failed")
        continue

    if u3 is None:
        print(f"original Convex solver failed")
        continue

    if u4 is None:
        print(f'original Convex solver failed')

    cost_struct.append(float(u1.value.T @ Q_tilde @ u1.value))
    if q < epsilon:
        cost_conf.append(float(u2.value.T @ Q_tilde @ u2.value))
    else:
        cost_conf.append(np.nan)
    cost_known.append(float(u3.value.T @ Q_tilde @ u3.value))
    cost_abs.append(float(u4.value.T @ Q_tilde @ u4.value))

    idev_struct = []
    idev_conf = []
    idev_econf = []
    idev_known = []
    idev_abs = []
    N_test = 5000
    z_x_test = np.random.randn(N_test, n)
    z_w_test = np.random.randn(N_test, n*T)

    for i in range(5000):
        x0 = sigma * z_x_test[i]
        w = sigma_1 * z_w_test[i]

        xT1 = np.linalg.matrix_power(A, T) @ x0 + CT @ u1.value + N_T @ w
        xT2 = np.linalg.matrix_power(A, T) @ x0 + CT @ u2.value + N_T @ w
        xT3 = np.linalg.matrix_power(A, T) @ x0 + CT @ u3.value + N_T @ w
        xT4 = np.linalg.matrix_power(A, T) @ x0 + CT @ u4.value + N_T @ w

        e1 = xT1 - xf
        e2 = xT2 - xf
        e3 = xT3 - xf
        e4 = xT4 - xf

        idev_struct.append(e1) # structure
        idev_conf.append(e2) # conformal
        idev_known.append(e3) # known 
        idev_abs.append(e4) # known abs

    dev_struct.append(np.mean(idev_struct, axis = 0).T @ np.linalg.pinv(sigma_z) @ np.mean(idev_struct, axis = 0))
    if q < epsilon:
        dev_conf.append(np.mean(idev_conf, axis = 0).T  @ np.mean(idev_conf, axis = 0))
    else:
        dev_conf.append(np.nan)
  
    dev_known.append(np.mean(idev_known, axis = 0).T @ np.linalg.pinv(sigma_z) @ np.mean(idev_known, axis = 0))
    dev_abs.append(np.mean(idev_abs, axis = 0).T  @ np.mean(idev_abs, axis = 0))


plt.figure(figsize=(10,6))

plt.plot(sigma_vals, cost_struct,
        marker='o', linewidth=2,
        label='Structure')

plt.plot(sigma_vals, cost_known, color = 'black',
        marker='o', linewidth=2,
        label='Known normalized')

plt.plot(sigma_vals, cost_conf,
        marker='s', linewidth=2,
        label='Conformal')
conf_infeasible = np.array(conf_infeasible)

if np.any(conf_infeasible):
    idx = np.where(conf_infeasible)[0][0]
    sigma_crit = sigma_vals[idx]

    plt.axvspan(
        sigma_crit,
        sigma_vals[-1],
        alpha=0.2,
        label='Conformal infeasible'
    )        

plt.plot(sigma_vals, cost_abs,  '--',color = 'black',
        label='Known absolute')
plt.xlabel(r'$\sigma_1$')
plt.ylabel(r'$J^*(\sigma)$')
plt.grid(True)
plt.legend()
plt.savefig('cost_vs_sigma2_T8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))

plt.loglog(sigma_vals, dev_struct,
        marker='o', linewidth=2,
        label='Structure')
plt.loglog(sigma_vals, dev_known, color = 'black',
        marker='o', linewidth=2,
        label='Known normalized')

plt.loglog(sigma_vals, dev_conf,
        marker='s', linewidth=2,
        label='Conformal')

conf_infeasible = np.array(conf_infeasible)

if np.any(conf_infeasible):
    idx = np.where(conf_infeasible)[0][0]
    sigma_crit = sigma_vals[idx]

    plt.axvspan(
        sigma_crit,
        sigma_vals[-1],
        alpha=0.2,
        label='Conformal infeasible'
    )         

plt.loglog(sigma_vals, dev_abs, '--', color = 'black',
        label='Known absolute')

plt.xlabel(r'$\sigma_1$')
plt.ylabel('Mean deviation')
# plt.ylabel(r'$\mu_e^\top\Sigma_e^{-1}\mu_e$-$$')
plt.grid(True)
plt.legend()
plt.savefig('deviation_vs_sigma2_T8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(sigma_vals, q_sigma, marker = 'o')
plt.xlabel(r'$\sigma_1$')
plt.ylabel(r'$q_{1-\delta}$')
plt.grid(True, which = 'both')
plt.savefig('q_sigma.jpeg')

T = 3
all_sigma = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_known = {}
quant1 = []
sigma_1 = 0.3

A, B, CT = get_system(name, n, m, T, seed=0)
N_T = np.eye(n)
for i in range(1, T):
    N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

Q_tilde = block_diag(*([Q] * T))

N = m*T + 10
N_cal = 60

U = np.random.randn(m*T, N)
U_cal = np.random.randn(m*T, N_cal)

sigma_vals = np.linspace(0.3, 7, 40)
It = np.eye(T)
sigma_w = np.kron(It, sigma_1**2 * np.eye(n))

# ---------- initialize ONCE ----------
cost_struct = []
cost_conf = []
cost_known = []
cost_abs = []

dev_struct = []
dev_conf = []
dev_known = []
dev_abs = []

lambda_star = find_lambda_star(n, epsilon, 1-delta)

w_train = np.random.multivariate_normal(
    np.zeros(n*T),
    sigma_w,
    size=(N, M)
)

w_cal_train = np.random.multivariate_normal(
    np.zeros(n*T),
    sigma_w,
    size=N_cal
)

q_sigma = []
# Standard Gaussian samples
z_x_train = np.random.randn(N, M, n)
z_x_cal   = np.random.randn(N_cal, n)

z_w_train = np.random.randn(N, M, n*T)
z_w_cal   = np.random.randn(N_cal, n*T)

w_train = sigma_1 * z_w_train
w_cal_train = sigma_1 * z_w_cal
conf_infeasible=[]
# -------------------------------------
for sigma in sigma_vals:
    sigma_0 = np.eye(n) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))

    x_train = sigma * z_x_train
    
    x_cal_train = sigma * z_x_cal

    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w =  w_train[i, j]
            x0 = x_train[i,j]
            xT = np.linalg.matrix_power(A, T) @ x0 + CT @ U[:, i] + N_T @ w  

            all_xT[:, i, j] = xT
            sumx += xT

        X_bar[:, i] = sumx / M

    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

    sample_sigma = np.zeros((n, n))
    struct_sigma = np.zeros((n, n))

    for i in range(N):
        for j in range(M):
            diff = all_xT[:, i, j] - X_bar[:, i]
            diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]

            sample_sigma += np.outer(diff, diff)
            struct_sigma += np.outer(diff1, diff1)

    sample_sigma /= (N*M)
    struct_sigma /= (N*M)

    scores = get_scores(
        U_cal,
        N_cal,
        CT_hat,
        A,
        N_T,
        CT,
        T,
        w_cal_train,
        x_cal_train      # after modifying get_scores
    )

    q = conformal_quantile(scores, delta)
    conf_infeasible.append(q >= epsilon)
    if q >= epsilon:
       lambda_cp = 0.0
    else:
       lambda_cp = (epsilon - q)**2
    q_sigma.append(q)
    # lambda_cp = (epsilon - q)**2

    u1 = cp.Variable(m*T)
    u2 = cp.Variable(m*T)
    u3 = cp.Variable(m*T)
    u4 = cp.Variable(m*T)

    objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))

    constraint1 = cp.quad_form(
        CT_hat @ u1 - xf,
        np.linalg.inv(struct_sigma)
    ) <= lambda_star

    constraint2 = cp.quad_form(
        CT_hat @ u2 - xf,
        np.eye(n)
    ) <= lambda_cp

    constraint3 = cp.quad_form(
        CT @ u3 - xf,
        np.linalg.inv(sigma_z)
    ) <= lambda_star

    constraint4 = cp.quad_form(
        CT @ u4 - xf,
        np.eye(n)
    )<= epsilon**2

    cp.Problem(objective1, [constraint1]).solve(solver=cp.SCS)
    cp.Problem(objective2, [constraint2]).solve(solver=cp.SCS)
    cp.Problem(objective3, [constraint3]).solve(solver=cp.SCS)
    cp.Problem(objective4, [constraint4]).solve(solver=cp.SCS)

    if u1 is None:
        print(f"1st Convex solver failed")
        continue

    if u2 is None:
        print(f"2nd Convex solver failed")
        continue

    if u3 is None:
        print(f"original Convex solver failed")
        continue

    if u4 is None:
        print(f'original Convex solver failed')

    cost_struct.append(float(u1.value.T @ Q_tilde @ u1.value))
    if q < epsilon:
        cost_conf.append(float(u2.value.T @ Q_tilde @ u2.value))
    else:
        cost_conf.append(np.nan)
    cost_known.append(float(u3.value.T @ Q_tilde @ u3.value))
    cost_abs.append(float(u4.value.T @ Q_tilde @ u4.value))

    idev_struct = []
    idev_conf = []
    idev_econf = []
    idev_known = []
    idev_abs = []
    N_test = 5000
    z_x_test = np.random.randn(N_test, n)
    z_w_test = np.random.randn(N_test, n*T)
    for i in range(5000):
        x0 = sigma * z_x_test[i]
        w = sigma_1 * z_w_test[i]
        xT1 = np.linalg.matrix_power(A, T) @ x0 + CT @ u1.value + N_T @ w
        xT2 = np.linalg.matrix_power(A, T) @ x0 + CT @ u2.value + N_T @ w
        xT3 = np.linalg.matrix_power(A, T) @ x0 + CT @ u3.value + N_T @ w
        xT4 = np.linalg.matrix_power(A, T) @ x0 + CT @ u4.value + N_T @ w

        e1 = xT1 - xf
        e2 = xT2 - xf
        e3 = xT3 - xf
        e4 = xT4 - xf

        idev_struct.append(e1) # structure
        idev_conf.append(e2) # conformal
        idev_known.append(e3) # known 
        idev_abs.append(e4) # known abs

    dev_struct.append(np.mean(idev_struct, axis = 0).T @ np.linalg.pinv(sigma_z) @ np.mean(idev_struct, axis = 0))
    if q < epsilon:
        dev_conf.append(np.mean(idev_conf, axis = 0).T  @ np.mean(idev_conf, axis = 0))
    else:
        dev_conf.append(np.nan)
  
    dev_known.append(np.mean(idev_known, axis = 0).T @ np.linalg.pinv(sigma_z) @ np.mean(idev_known, axis = 0))
    dev_abs.append(np.mean(idev_abs, axis = 0).T  @ np.mean(idev_abs, axis = 0))


plt.figure(figsize=(10,6))

plt.plot(sigma_vals, cost_struct,
        marker='o', linewidth=2,
        label='Structure')

plt.plot(sigma_vals, cost_known, color = 'black',
        marker='o', linewidth=2,
        label='Known normalized')

plt.plot(sigma_vals, cost_conf,
        marker='s', linewidth=2,
        label='Conformal')

conf_infeasible = np.array(conf_infeasible)

if np.any(conf_infeasible):
    idx = np.where(conf_infeasible)[0][0]
    sigma_crit = sigma_vals[idx]

    plt.axvspan(
        sigma_crit,
        sigma_vals[-1],
        alpha=0.2,
        label='Conformal infeasible'
    )         

plt.plot(sigma_vals, cost_abs,  '--',color = 'black',
        label='Known absolute')
plt.xlabel(r'$\sigma_0$')
plt.ylabel(r'$J^*(\sigma)$')
plt.grid(True)
plt.legend()
plt.savefig('cost_vs_sigma1_T8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))

plt.loglog(sigma_vals, dev_struct,
        marker='o', linewidth=2,
        label='Structure')
plt.loglog(sigma_vals, dev_known, color = 'black',
        marker='o', linewidth=2,
        label='Known normalized')

plt.loglog(sigma_vals, dev_conf,
        marker='s', linewidth=2,
        label='Conformal')
conf_infeasible = np.array(conf_infeasible)

if np.any(conf_infeasible):
    idx = np.where(conf_infeasible)[0][0]
    sigma_crit = sigma_vals[idx]

    plt.axvspan(
        sigma_crit,
        sigma_vals[-1],
        alpha=0.2,
        label='Conformal infeasible'
    )          

plt.loglog(sigma_vals, dev_abs, '--', color = 'black',
        label='Known absolute')

plt.xlabel(r'$\sigma_0$')
plt.ylabel('Mean deviation')
# plt.ylabel(r'$\mu_e^\top\Sigma_e^{-1}\mu_e$-$$')
plt.grid(True)
plt.legend()
plt.savefig('deviation_vs_sigma1_T8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(sigma_vals, q_sigma, marker = 'o')
plt.xlabel(r'$\sigma_0$')
plt.ylabel(r'$q_{1-\delta}$')
plt.grid(True, which = 'both')
plt.savefig('q_sigma1.jpeg')
