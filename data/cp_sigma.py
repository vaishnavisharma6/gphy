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
T_list = range(1, 25, 8)
sigma_1 = 0.2
sigma_2 = 0.3
sigma_0 = np.eye(n) * sigma_1**2
x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
delta = 0.3
Q = np.diag([2.0, 3.0, 4.0, 5.0])
name = 'stable'
epsilon = 2.4
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

ratios1 = []
ratios2 = []
ratiosk = []
ratioskd = []
ratios3 = []

#to store final state deviation
x1_dev = []
x2_dev = []
xk_dev = []
xkd_dev = []
x3_dev = []
delta_folder = 'traj_{}'.format(delta)
os.makedirs(delta_folder, exist_ok = True)


def build_system_matrices(t):
    A, B, CT = get_system(name, n, m, t, seed=0)
    
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    sigmaw = np.eye(n*t) * sigma_2**2
    sigmaz = NT @ sigmaw @ NT.T + np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T

    return A, B, CT, NT, sigmaw, sigmaz


def generate_nested_data(t, N_max, M_max, seed=0, U_fixed=None):
 
    rng = np.random.default_rng(seed)
    A, B, CT, NT, sigmaw, sigmaz = build_system_matrices(t)

    if U_fixed is None:
        U_full = rng.standard_normal((m*t, N_max))
    else:
        U_full = U_fixed
        N_max = U_full.shape[1]

    all_xT_full = np.zeros((n, N_max, M_max))

    for i in range(N_max):
        for j in range(M_max):
            x_0 = rng.multivariate_normal(np.zeros(n), np.eye(n) * 0.2)
            w = rng.multivariate_normal(np.zeros(n*t), sigmaw)
            all_xT_full[:, i, j] = np.linalg.matrix_power(A, t) @ x_0 + CT @ U_full[:, i] + NT @ w

    return U_full, all_xT_full, CT, sigmaz


def estimates_from_nested_data(t, U_full, all_xT_full, N, M):
   
    A, B, CT, NT, sigmaw, sigmaz = build_system_matrices(t)

    U = U_full[:, :N]
    all_xT = all_xT_full[:, :N, :M]

    X_bar = np.mean(all_xT, axis=2)

    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

    sample_sigma = np.zeros((n, n))
    struct_sigma = np.zeros((n, n))

    for i in range(N):
        for j in range(M):
            diff_sample = all_xT[:, i, j] - X_bar[:, i]
            diff_struct = all_xT[:, i, j] - CT_hat @ U[:, i]
            sample_sigma += np.outer(diff_sample, diff_sample)
            struct_sigma += np.outer(diff_struct, diff_struct)

    sample_sigma /= (N * M)
    struct_sigma /= (N * M)

    mu_true = CT @ U
    mu_sample = X_bar
    mu_struct = CT_hat @ U

    return mu_sample, mu_struct, sample_sigma, struct_sigma, mu_true, sigmaz, CT_hat, CT


def estimate_errors_from_nested_data(t, U_full, all_xT_full, N, M):
    mu_sample, mu_struct, sample_sigma, struct_sigma, mu_true, sigmaz, CT_hat, CT = estimates_from_nested_data(
        t, U_full, all_xT_full, N, M
    )

    mu_sample_error = np.linalg.norm(mu_sample - mu_true, ord=2)
    mu_struct_error = np.linalg.norm(mu_struct - mu_true, ord=2)
    sigma_sample_error = np.linalg.norm(sample_sigma - sigmaz, ord=2)
    sigma_struct_error = np.linalg.norm(struct_sigma - sigmaz, ord=2)
    CT_error = np.linalg.norm(CT_hat - CT, ord=2)

    return mu_sample_error, mu_struct_error, sigma_sample_error, sigma_struct_error, CT_error


def average_errors_nested(t, N_list=None, M_list=None, N_fixed=None, M_fixed=None, K=30, seed=100):

    if M_list is not None:
        N_max = N_fixed
        M_max = max(M_list)
        keys = list(M_list)
    else:
        N_max = max(N_list)
        M_max = M_fixed
        keys = list(N_list)

    mu_samp = {key: [] for key in keys}
    mu_struct = {key: [] for key in keys}
    sig_samp = {key: [] for key in keys}
    sig_struct = {key: [] for key in keys}
    CT_err = {key: [] for key in keys}

    for k in range(K):
        U_full, all_xT_full, CT, sigmaz = generate_nested_data(
            t, N_max, M_max, seed=seed + k
        )

        for key in keys:
            if M_list is not None:
                N = N_fixed
                M = key
            else:
                N = key
                M = M_fixed

            e1, e2, e3, e4, e5 = estimate_errors_from_nested_data(
                t, U_full, all_xT_full, N, M
            )

            mu_samp[key].append(e1)
            mu_struct[key].append(e2)
            sig_samp[key].append(e3)
            sig_struct[key].append(e4)
            CT_err[key].append(e5)

    return (
        [np.mean(mu_samp[key]) for key in keys],
        [np.mean(mu_struct[key]) for key in keys],
        [np.mean(sig_samp[key]) for key in keys],
        [np.mean(sig_struct[key]) for key in keys],
        [np.mean(CT_err[key]) for key in keys]
    )


def compute_bias_variance_from_replicates(estimates, true_value):
    """
    Bias = || average estimator - true value ||_F
    Variance = average || estimator - average estimator ||_F^2
    """
    estimates = np.array(estimates)
    mean_estimate = np.mean(estimates, axis=0)

    bias = np.linalg.norm(mean_estimate - true_value, ord='fro')
    variance = np.mean([
        np.linalg.norm(est - mean_estimate, ord='fro')**2
        for est in estimates
    ])

    return bias, variance


def average_bias_variance_nested(t, N_list=None, M_list=None, N_fixed=None, M_fixed=None, K=30, seed=500):
    """
    Computes bias and variance over K independent datasets using nested slicing.
    """
    if M_list is not None:
        N_max = N_fixed
        M_max = max(M_list)
        keys = list(M_list)
    else:
        N_max = max(N_list)
        M_max = M_fixed
        keys = list(N_list)

    mu_s_bias, mu_st_bias = [], []
    mu_s_var, mu_st_var = [], []
    sig_s_bias, sig_st_bias = [], []
    sig_s_var, sig_st_var = [], []

    for key in keys:
        mu_sample_list = []
        mu_struct_list = []
        sigma_sample_list = []
        sigma_struct_list = []

        for k in range(K):
            U_full, all_xT_full, CT, sigmaz = generate_nested_data(
                t, N_max, M_max, seed=seed + k
            )

            if M_list is not None:
                N = N_fixed
                M = key
            else:
                N = key
                M = M_fixed

            mu_sample, mu_struct, sample_sigma, struct_sigma, mu_true, sig_true, CT_hat, CT_true = estimates_from_nested_data(
                t, U_full, all_xT_full, N, M
            )

            mu_sample_list.append(mu_sample)
            mu_struct_list.append(mu_struct)
            sigma_sample_list.append(sample_sigma)
            sigma_struct_list.append(struct_sigma)

        b1, v1 = compute_bias_variance_from_replicates(mu_sample_list, mu_true)
        b2, v2 = compute_bias_variance_from_replicates(mu_struct_list, mu_true)
        b3, v3 = compute_bias_variance_from_replicates(sigma_sample_list, sig_true)
        b4, v4 = compute_bias_variance_from_replicates(sigma_struct_list, sig_true)

        mu_s_bias.append(b1)
        mu_st_bias.append(b2)
        mu_s_var.append(v1)
        mu_st_var.append(v2)

        sig_s_bias.append(b3)
        sig_st_bias.append(b4)
        sig_s_var.append(v3)
        sig_st_var.append(v4)

    return (
        mu_s_bias, mu_st_bias, mu_s_var, mu_st_var,
        sig_s_bias, sig_st_bias, sig_s_var, sig_st_var
    )

def conformal_quantile(scores, delta):
    N_cal = len(scores)

    k = int(np.ceil((N_cal + 1) * (1 - delta)))
    k = min(k, N_cal)

    q = np.sort(scores)[k - 1]

    return q


def get_scores(U_cal, N_cal, CT_hat, A, NT, CT, t, sigma, sigma1, w_cal_train, x_cal_train):
    scores = []
    for i in range(N_cal):
        w =  sigma1 * w_cal_train[i]
        x_0 = sigma * x_cal_train[i]
        # x_0 = np.random.multivariate_normal(np.zeros(n), sigma_0)
        x_true = np.linalg.matrix_power(A, t) @ x_0+ CT @ U_cal[:, i] + NT@w
        x_cal = CT_hat @ U_cal[:, i]
        scores.append(np.linalg.norm(x_true - x_cal))

    return(np.array(scores))


def get_test_scores(U_test, CT_hat, A,  NT, CT, t, sigma, sigma1):

    scores = []

    for i in range(U_test.shape[1]):

        w = np.random.multivariate_normal(
            np.zeros(n*t),
            np.eye(n*t) * sigma1**2
        )
        x_0 = np.random.multivariate_normal(
            np.zeros(n),
            np.eye(n)*sigma**2
        )

        x_true = np.linalg.matrix_power(A, t) @x_0 + CT @ U_test[:, i] + NT @ w
        x_pred = CT_hat @ U_test[:, i]

        score = np.linalg.norm(x_true - x_pred)

        scores.append(score)

    return np.array(scores)

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

T = 8
all_sigma = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_known = {}
quant1 = []
sigma_1 = 0.2

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
dev_econf = []
dev_known = []
dev_abs = []

lambda_star = find_lambda_star(n, epsilon, 1-delta)

w_train = np.random.multivariate_normal(
    np.zeros(n*T),
    np.eye(n*T),
    size=(N, M)
)
x_train = np.random.multivariate_normal(
    np.zeros(n),
    np.eye(n),
    size=(N, M)
)
w_cal_train = np.random.multivariate_normal(
    np.zeros(n*T),
    np.eye(n*T),
    size=N_cal
)
x_cal_train = np.random.multivariate_normal(
    np.zeros(n),
    np.eye(n),
    size=N_cal
)

q_sigma = []
# -------------------------------------
for sigma in sigma_vals:

    sigma_w = np.eye(n*T) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))

    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = sigma * w_train[i, j]
            x_0 = sigma_1 * x_train[i, j]
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
        sigma_1,
        sigma,
        w_cal_train,
        x_cal_train       # after modifying get_scores
    )

    q = conformal_quantile(scores, delta)
    q_sigma.append(q)
    lambda_cp = (epsilon - q)**2

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
    )<= lambda_star

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
    cost_conf.append(float(u2.value.T @ Q_tilde @ u2.value))
    cost_known.append(float(u3.value.T @ Q_tilde @ u3.value))
    cost_abs.append(float(u4.value.T @ Q_tilde @ u4.value))

    idev_struct = []
    idev_conf = []
    idev_econf = []
    idev_known = []
    idev_abs = []

    for _ in range(5000):
        x0 = np.random.multivariate_normal(
            np.zeros(n),
            sigma_0)
        w = np.random.multivariate_normal(
            np.zeros(n*T),
            sigma_w
        )

        xT1 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u1.value + N_T @ w
        xT2 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u2.value + N_T @ w
        xT3 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u3.value + N_T @ w
        xT4 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u4.value + N_T @ w

        e1 = xT1 - xf
        e2 = xT2 - xf
        e3 = xT3 - xf
        e4 = xT4 - xf

        idev_struct.append(e1.T @ np.linalg.inv(struct_sigma) @ e1)
        idev_conf.append(e2.T @ np.linalg.inv(struct_sigma) @ e2)
        idev_econf.append(e2.T @ e2)
        idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)
        idev_abs.append(e4.T @ e4)

    dev_struct.append(np.mean(idev_struct))
    dev_econf.append(np.mean(idev_econf))
    dev_known.append(np.mean(idev_known))
    dev_abs.append(np.mean(idev_abs))


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
plt.loglog(sigma_vals, dev_known,
        marker='o', linewidth=2,
        label='Known normalized')

plt.loglog(sigma_vals, dev_econf,
        marker='s', linewidth=2,
        label='Conformal')

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

T = 8
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

# ---------- initialize ONCE ----------
cost_struct = []
cost_conf = []
cost_known = []
cost_abs = []

dev_struct = []
dev_econf = []
dev_known = []
dev_abs = []

lambda_star = find_lambda_star(n, epsilon, 1-delta)

w_train = np.random.multivariate_normal(
    np.zeros(n*T),
    np.eye(n*T),
    size=(N, M)
)
x_train = np.random.multivariate_normal(
    np.zeros(n),
    np.eye(n),
    size=(N, M)
)
w_cal_train = np.random.multivariate_normal(
    np.zeros(n*T),
    np.eye(n*T),
    size=N_cal
)
x_cal_train = np.random.multivariate_normal(
    np.zeros(n),
    np.eye(n),
    size=N_cal
)
q_sigma = []
# -------------------------------------
for sigma in sigma_vals:

    sigma_w = np.eye(n*T) * sigma_1**2
    sigma_0 = np.eye(n) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))

    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = sigma_1 * w_train[i, j]
            x_0 = sigma * x_train[i,j]
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
        sigma,
        sigma_1,
        w_cal_train,
        x_cal_train      # after modifying get_scores
    )

    q = conformal_quantile(scores, delta)
    q_sigma.append(q)
    lambda_cp = (epsilon - q)**2

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
    )<= lambda_star

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
    cost_conf.append(float(u2.value.T @ Q_tilde @ u2.value))
    cost_known.append(float(u3.value.T @ Q_tilde @ u3.value))
    cost_abs.append(float(u4.value.T @ Q_tilde @ u4.value))

    idev_struct = []
    idev_conf = []
    idev_econf = []
    idev_known = []
    idev_abs = []

    for _ in range(5000):
        x0 = np.random.multivariate_normal(
            np.zeros(n),
            sigma_0)
        w = np.random.multivariate_normal(
            np.zeros(n*T),
            sigma_w
        )

        xT1 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u1.value + N_T @ w
        xT2 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u2.value + N_T @ w
        xT3 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u3.value + N_T @ w
        xT4 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u4.value + N_T @ w

        e1 = xT1 - xf
        e2 = xT2 - xf
        e3 = xT3 - xf
        e4 = xT4 - xf

        idev_struct.append(e1.T @ np.linalg.inv(struct_sigma) @ e1)
        idev_conf.append(e2.T @ np.linalg.inv(struct_sigma) @ e2)
        idev_econf.append(e2.T @ e2)
        idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)
        idev_abs.append(e4.T @ e4)

    dev_struct.append(np.mean(idev_struct))
    dev_econf.append(np.mean(idev_econf))
    dev_known.append(np.mean(idev_known))
    dev_abs.append(np.mean(idev_abs))


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
plt.loglog(sigma_vals, dev_known,
        marker='o', linewidth=2,
        label='Known normalized')

plt.loglog(sigma_vals, dev_econf,
        marker='s', linewidth=2,
        label='Conformal')

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
