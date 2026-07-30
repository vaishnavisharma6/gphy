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
T_list = range(1, 25, 2)

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

num_runs = 20
N_test = 1000


#to store final state deviation
def build_system_matrices(t):
    A, B, CT = get_system(name, n, m, t, seed=0)
    sigma = 0.3**2 * np.eye(n) 
    It = np.eye(t)
    sigmaw = np.kron(It, sigma)
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    # sigmaw = np.eye(n*t) * 0.3**2
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
            x_0 = rng.multivariate_normal(np.zeros(n), np.eye(n) * 0.2**2)
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


n, m = 4, 4
Q = np.diag([2.0, 3.0, 4.0, 5.0])
t_fixed = 5
N_fixed = 80
M_fixed = 80

M_list = [5, 10, 20, 40, 80, 120, 140, 180, 220, 260]
N_list = [10, 20, 40, 80, 160, 320, 640, 700, 720, 900]

K_errors = 40          # number of independent estimates for averaged error curves
K_bias_var = 40        # number of independent estimates for bias-variance curves

# =========================
# Average Error vs M
# =========================

mu_samp_M, mu_struct_M, sig_samp_M, sig_struct_M, CT_err_M = average_errors_nested(
    t=t_fixed,
    M_list=M_list,
    N_fixed=N_fixed,
    K=K_errors,
    seed=100
)

# =========================
# Average Error vs N
# =========================

mu_samp_N, mu_struct_N, sig_samp_N, sig_struct_N, CT_err_N = average_errors_nested(
    t=t_fixed,
    N_list=N_list,
    M_fixed=M_fixed,
    K=K_errors,
    seed=200
)

# =========================
# Plot average estimation errors
# =========================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].loglog(M_list, mu_samp_M, '^-', label='sample')
axes[0, 0].loglog(M_list, mu_struct_M, '^-', label='structure')
axes[0, 0].set_xlabel(r'$N_r$')
axes[0, 0].set_ylabel(r'Average $\|\hat{\mu}_X - \mu_X\|_2$')
axes[0, 0].legend()
axes[0, 0].grid(True, which="both", alpha=0.3)

axes[0, 1].loglog(N_list, mu_samp_N, '^-', label='sample')
axes[0, 1].loglog(N_list, mu_struct_N, '^-', label='structure')
axes[0, 1].set_xlabel(r'$N_e$')
axes[0, 1].set_ylabel(r'Average $\|\hat{\mu}_X - \mu_X\|_2$')
axes[0, 1].legend()
axes[0, 1].grid(True, which="both", alpha=0.3)

axes[1, 0].loglog(M_list, sig_samp_M, '^-', label='sample')
axes[1, 0].loglog(M_list, sig_struct_M, '^-', label='structure')
axes[1, 0].set_xlabel(r'$N_r$')
axes[1, 0].set_ylabel(r'Average $\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_2$')
axes[1, 0].legend()
axes[1, 0].grid(True, which="both", alpha=0.3)


axes[1, 1].loglog(N_list, sig_samp_N, '^-', label='sample')
axes[1, 1].loglog(N_list, sig_struct_N, '^-', label='structure')
axes[1, 1].set_xlabel(r'$N_e$')
axes[1, 1].set_ylabel(r'Average $\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_2$')
axes[1, 1].legend()
axes[1, 1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('average_estimation_errors_combined.jpeg')
plt.close()

# =========================
# Plot average CT error
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].loglog(M_list, CT_err_M, '^-')
axes[0].set_xlabel(r'$N_r$')
axes[0].set_ylabel(r'Average $\|\hat{C}_T - C_T\|_2$')
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)

axes[1].loglog(N_list, CT_err_N, '^-')
axes[1].set_xlabel(r'$N_e$')
axes[1].set_ylabel(r'Average $\|\hat{C}_T - C_T\|_2$')
axes[1].legend()
axes[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('average_CT_error_combined.jpeg')
plt.close()

