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

T_list = [6, 8]

all_gamma = {}

all_lambda = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_cost_abs = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_econf = {}
all_dev_known = {}
all_dev_eknown = {}
all_dev_estruct = {}
all_dev_abs = {}

for T in T_list:

    # Build system matrices
    A, B, CT = get_system(name, n, m, T, seed=0)
    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b =  -1 *  xf

    sigma_w = np.eye(n*T) * sigma_2**2
    sigma_z = N_T @ sigma_w @ N_T.T

    Q_tilde = block_diag(*([Q] * T))

    gamma_min = 0.0
    gamma_max = float(b.T @ np.linalg.inv(sigma_z) @ b)

    gamma_vals = np.linspace(gamma_min, gamma_max, 10)

    cost_struct = []
    cost_conf = []
    cost_known = []
    cost_abs = []

    dev_struct = []
    dev_conf = []
    dev_known = []
    dev_econf = []
    dev_estruct = []
    dev_eknown = []
    dev_abs = []
    N = m*T +10
    N_cal = 60
    
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
    q_gamma = []
    for gamma in gamma_vals:

        # ---------- averages over datasets ----------
        cost_struct_runs = []
        cost_conf_runs = []
        cost_known_runs = []
        cost_abs_runs = []

        dev_struct_runs = []
        dev_conf_runs = []
        dev_known_runs = []
        dev_econf_runs = []
        dev_estruct_runs = []
        dev_eknown_runs = []
        dev_abs_runs = []

        for dataset in range(k_est):

            np.random.seed(dataset)
            U = np.random.randn(m*T, N)
            U_cal = np.random.randn(m*T, N_cal)

            X_bar = np.zeros((n, N))
            all_xT = np.zeros((n, N, M))

            for i in range(N):

                sumx = np.zeros(n)

                for j in range(M):
                    x0 = np.random.multivariate_normal(np.zeros(n),sigma_0)

                    w = np.random.multivariate_normal(
                        np.zeros(n*T),
                        np.eye(n*T) * sigma_2**2
                    )

                    xT = np.linalg.matrix_power(A, T) @ x_0 + CT @ U[:, i] + N_T @ w

                    all_xT[:, i, j] = xT
                    sumx += xT

                X_bar[:, i] = sumx / M

            # ======================================================
            # Estimate CT
            # ======================================================

            CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

            sample_sigma = np.zeros((n, n))
            struct_sigma = np.zeros((n, n))

            for i in range(N):
                for j in range(M):

                    diff = all_xT[:, i, j] - X_bar[:, i]
                    diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]

                    sample_sigma += np.outer(diff, diff)
                    struct_sigma += np.outer(diff1, diff1)

            sample_sigma /= (N * M)
            struct_sigma /= (N * M)

            scores = get_scores(
                U_cal,
                N_cal,
                CT_hat,
                A,
                N_T,
                CT,
                T,
                sigma_1,
                sigma_2,
                w_cal_train,
                x_cal_train       # after modifying get_scores
            )
            q= conformal_quantile(scores, delta)
            q_gamma.append(q)
            lambda_cp = (gamma - q)**2
            # ======================================================
            # Optimization
            # ======================================================

            u1 = cp.Variable(m*T)
            u2 = cp.Variable(m*T)
            u3 = cp.Variable(m*T)
            u4 = cp.Variable(m*T)

            prob1 = cp.Problem(
                cp.Minimize(cp.quad_form(u1, Q_tilde)),
                [cp.quad_form(CT_hat @ u1 - xf,
                              np.linalg.inv(struct_sigma)) <= gamma]
            )

            prob2 = cp.Problem(
                cp.Minimize(cp.quad_form(u2, Q_tilde)),
                [cp.quad_form(CT_hat @ u2 - xf,
                              np.eye(n)) <= lambda_cp]
            )

            prob3 = cp.Problem(
                cp.Minimize(cp.quad_form(u3, Q_tilde)),
                [cp.quad_form(CT @ u3 - xf,
                              np.linalg.inv(sigma_z)) <= gamma]
            )

            prob4 = cp.Problem(
                cp.Minimize(cp.quad_form(u4, Q_tilde)),
                [cp.quad_form(CT @ u4 - xf,
                              np.eye(n)) <= gamma]
            )

            prob1.solve(solver=cp.SCS, verbose=False)
            prob2.solve(solver=cp.SCS, verbose=False)
            prob3.solve(solver=cp.SCS, verbose=False)
            prob4.solve(solver =cp.SCS, verbose = False)

            if (u1.value is None or
                u2.value is None or
                u3.value is None or
                u4.value is None):
                continue

            # ======================================================
            # Store costs
            # ======================================================

            cost_struct_runs.append(float(u1.value.T @ Q_tilde @ u1.value))
            cost_conf_runs.append(float(u2.value.T @ Q_tilde @ u2.value))
            cost_known_runs.append(float(u3.value.T @ Q_tilde @ u3.value))
            cost_abs_runs.append(float(u4.value.T @ Q_tilde @ u4.value))

            # ======================================================
            # Monte-Carlo evaluation
            # ======================================================

            idev_struct = []
            idev_conf = []
            idev_known = []
            idev_econf = []
            idev_estruct = []
            idev_eknown = []
            idev_abs = []

            for _ in range(5000):
                x0 = np.random.multivariate_normal(
                np.zeros(n),
                sigma_0)

                w = np.random.multivariate_normal(
                    np.zeros(n*T),
                    np.eye(n*T) * sigma_2**2
                )

                x1 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u1.value + N_T @ w
                x2 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u2.value + N_T @ w
                x3 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u3.value + N_T @ w
                x4 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u4.value + N_T @ w

                e1 = x1 - xf
                e2 = x2 - xf
                e3 = x3 - xf
                e4 = x4 - xf

                idev_struct.append(e1.T @ np.linalg.inv(struct_sigma) @ e1)
                idev_conf.append(e2.T @ np.linalg.inv(struct_sigma) @ e2)
                idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)
               

                idev_econf.append(e2.T @ e2)
                idev_estruct.append(e1.T @ e1)
                idev_eknown.append(e3.T @ e3)
                idev_abs.append(e4.T @e4)

            dev_struct_runs.append(np.mean(idev_struct))
            dev_conf_runs.append(np.mean(idev_conf))
            dev_known_runs.append(np.mean(idev_known))

            dev_econf_runs.append(np.mean(idev_econf))
            dev_estruct_runs.append(np.mean(idev_estruct))
            dev_eknown_runs.append(np.mean(idev_eknown))
            dev_abs_runs.append(np.mean(idev_abs))

        # ==========================================================
        # Average over datasets for THIS gamma
        # ==========================================================

        cost_struct.append(np.mean(cost_struct_runs))
        cost_conf.append(np.mean(cost_conf_runs))
        cost_known.append(np.mean(cost_known_runs))
        cost_abs.append(np.mean(cost_abs_runs))

        dev_struct.append(np.mean(dev_struct_runs))
        dev_conf.append(np.mean(dev_conf_runs))
        dev_known.append(np.mean(dev_known_runs))

        dev_econf.append(np.mean(dev_econf_runs))
        dev_estruct.append(np.mean(dev_estruct_runs))
        dev_eknown.append(np.mean(dev_eknown_runs))
        dev_abs.append(np.mean(dev_abs_runs))

    all_gamma[T] = gamma_vals

    all_cost_struct[T] = cost_struct
    all_cost_conf[T] = cost_conf
    all_cost_known[T] = cost_known
    all_cost_abs[T] = cost_abs

    all_dev_struct[T] = dev_struct
    all_dev_conf[T] = dev_conf
    all_dev_known[T] = dev_known

    all_dev_econf[T] = dev_econf
    all_dev_estruct[T] = dev_estruct
    all_dev_eknown[T] = dev_eknown
    all_dev_abs[T] = dev_abs

plt.figure(figsize=(10,6))

colors = ['tab:blue', 'tab:orange', 'tab:green']

for  T in T_list:

    plt.plot(all_gamma[T], all_cost_struct[T], marker = 'o',
             label=f'Structure, T={T}')

    plt.plot(all_gamma[T], all_cost_conf[T],marker = '^',
             label=f'Conformal, T={T}')

    plt.plot(all_gamma[T], all_cost_known[T],'--', color = 'black',
             label=f'Known normalized, T={T}')

    plt.plot(all_gamma[T], all_cost_abs[T],'--', color = 'brown', marker = 'o',
             label=f'Known absolute, T={T}')         

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.legend()
plt.grid(True)
 
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.legend()
plt.grid(True)
plt.savefig(f'cost_vs_gamma_allTdata_{name}.jpeg')
plt.show()

plt.figure(figsize=(10,6))

for T in T_list:
    plt.plot(all_gamma[T], all_dev_struct[T], marker='o', label=f"Structure, T={T}")
    plt.plot(all_gamma[T], all_dev_econf[T], marker='s', label=f"Conformal, T={T}")
    plt.plot(all_gamma[T], all_dev_known[T], '--', color = 'black', label=f"Known normalized, T={T}")
    plt.plot(all_gamma[T], all_dev_abs[T], '--', color = 'brown', marker = 'o', label =f"Known absolute, T={T}" )


plt.xlabel(r'$\gamma$')
plt.ylabel('Mean deviation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'deviation_vs_gamma_allTdata_{name}.jpeg')
