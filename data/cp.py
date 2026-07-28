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
sigma_0 = np.eye(n) * 0.2
x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
xf = np.array([4.0, 5.0, 6.0, 7.0])
T_list = range(1, 20, 2)
sigma = 0.3
delta = 0.3
Q = np.diag([2.0, 3.0, 4.0, 5.0])
name = 'stable'
epsilon = 2.4
M = 50
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
    sigmaw = np.eye(n*t) * sigma**2
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


def get_scores(U_cal, N_cal, CT_hat, NT, CT, t, sigma):
    scores = []
    for i in range(N_cal):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*sigma**2)
        # x_0 = np.random.multivariate_normal(np.zeros(n), sigma_0)
        x_true = CT @ U_cal[:, i] + NT@w
        x_cal = CT_hat @ U_cal[:, i]
        scores.append(np.linalg.norm(x_true - x_cal))

    return(np.array(scores))


def get_test_scores(U_test, CT_hat, NT, CT, t, sigma):

    scores = []

    for i in range(U_test.shape[1]):

        w = np.random.multivariate_normal(
            np.zeros(n*t),
            np.eye(n*t) * sigma**2
        )

        x_true = CT @ U_test[:, i] + NT @ w
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

#end of all function definitons--------

# for t in T_list:
#     sample_cost_runs = []
#     struct_cost_runs = []
#     cp_cost_runs = []
#     known_cost_runs = []

#     sample_dev_runs = []
#     struct_dev_runs = []
#     cp_dev_runs = []
#     known_dev_runs = []

#     ratio1_runs = []
#     ratio2_runs = []
#     ratio3_runs = []
#     ratiok_runs = []
#     A, B, CT = get_system(name, n, m, t, seed=0)
#     print(t)
#     Q_tilde = block_diag(*([Q] * t))

#     NT = np.eye(n)
#     for i in range(1, t):
#         NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

#     sigmaw = np.eye(n*t) * sigma**2
#     sigmaz = NT @ sigmaw @ NT.T + np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T

#     #estimation
#     N = m*t +5
#     N_cal = 20
#     for dataset in range(k_est):
#         np.random.seed(dataset)
#         U = np.random.randn(m*t, N) # input matrix
#         U_cal = np.random.randn(m*t, N_cal)
#         #create bar{X}
#         X_bar = np.zeros((n, N))
#         all_xT = np.zeros((n, N, M))
#         for i in range(N):
#             sumx = np.zeros(n)
#             for j in range(M):
#                 x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
#                 w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*sigma**2)
#                 xT = np.linalg.matrix_power(A, t) @ x_0 + CT @ U[:,i] + NT@ w
#                 all_xT[:, i, j] = xT
#                 sumx = sumx + xT
#             x_bar = sumx/M     
#             X_bar[:, i] = x_bar
        
#         CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)
#         sample_sigma = np.zeros((n, n))
#         struct_sigma = np.zeros((n, n))

#         for i in range(N):
#             for j in range(M):
#                 diff = all_xT[:, i, j] - X_bar[:, i]
#                 diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]
#                 sample_sigma += np.outer(diff, diff)
#                 struct_sigma += np.outer(diff1, diff1)

#         sample_sigma = sample_sigma / (N*M)
#         struct_sigma = struct_sigma/ (N*M)
        
#         quantiles[t] = []
#         empirical_coverage[t] = {}

#         for delta in delta_list:

#             coverages = []
#             q_runs = []

#             for run in range(num_runs):

#                 np.random.seed(run)

#                 scores = get_scores(
#                     U_cal,
#                     N_cal,
#                     CT_hat,
#                     NT,
#                     CT,
#                     t,
#                     sigma
#                 )

#                 q = conformal_quantile(scores, delta)

#                 q_runs.append(q)

#                 U_test = np.random.randn(m*t, N_test)

#                 scores_test = get_test_scores(
#                     U_test,
#                     CT_hat,
#                     NT,
#                     CT,
#                     t,
#                     sigma
#                 )

#                 coverages.append(
#                     np.mean(scores_test <= q)
#                 )

#             empirical_coverage[t][delta] = (
#                 np.mean(coverages),
#                 np.std(coverages)
#         )

#             if delta == 0.30:
#                 quant.append(np.mean(q_runs))

#             quantiles[t].append(np.mean(q_runs))

#         delta = 0.30

#         scores = get_scores(
#             U_cal,
#             N_cal,
#             CT_hat,
#             NT,
#             CT,
#             t,
#             sigma
#         )

#         q = conformal_quantile(scores, delta)

#         #rhs of all inequalities
#         lambda_star = find_lambda_star(n, epsilon, 1-delta)
#         lambda_cp = (epsilon - q)**2    

#         alpha1 = cp.Variable(N)
#         alpha2 = cp.Variable(m*t)
#         alphak = cp.Variable(m*t)
#         u = cp.Variable(m*t)
        

#         objective1 = cp.Minimize(cp.quad_form((U @ alpha1), Q_tilde))
#         objective2 = cp.Minimize(cp.quad_form((alpha2), Q_tilde))
#         objectivek = cp.Minimize(cp.quad_form((alphak), Q_tilde))
#         objective3 = cp.Minimize(cp.quad_form(u, Q_tilde))

#         f1 = cp.quad_form(X_bar @ alpha1 - xf, np.linalg.inv(sample_sigma))
#         f2 = cp.quad_form(CT_hat @ alpha2 - xf, np.linalg.inv(struct_sigma))
#         fk = cp.quad_form(CT @ alphak - xf, np.linalg.inv(sigmaz))
#         f3 = cp.quad_form(CT_hat @ u - xf, np.eye(n))

#         prob1 = cp.Problem(objective1, [f1 <= lambda_star])
#         prob2 = cp.Problem(objective2, [f2 <= lambda_star])
#         probk = cp.Problem(objectivek, [fk <= lambda_star])
#         prob3 = cp.Problem(objective3, [f3 <= lambda_cp])

#         prob1.solve(solver = cp.SCS, verbose = False)
#         prob2.solve(solver = cp.SCS, verbose = False)
#         probk.solve(solver = cp.SCS, verbose = False)
#         prob3.solve(solver = cp.SCS, verbose = False)

#         a1 = alpha1.value
#         a2 = alpha2.value
#         ak = alphak.value
#         u3 = u.value

#         u1 = U @ a1
#         u2 = a2
#         uk = ak

#         if a1 is None:
#             print(f"1st Convex solver failed at T={t}")
#             continue

#         if a2 is None:
#             print(f"2nd Convex solver failed at T={t}")
#             continue

#         if ak is None:
#             print(f"original Convex solver failed at T={t}")
#             continue

#         if u3 is None:
#             print(f"cp Convex solver failed at T={t}")
#             continue

#         np.random.seed(0)

#         x1 = []
#         x2 = []
#         xk = []
#         x3 = []
        

#         s1 = []
#         s2 = []
#         sk = []
#         s3 = []

#         for j in range(0, 2000):
#             x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
#             w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*sigma**2)
#             x1T = np.linalg.matrix_power(A, t) @ x_0 + CT @ u1 + NT@w
#             x2T =  np.linalg.matrix_power(A, t) @ x_0+ CT_hat @ u2 + NT @ w  
#             xkT = np.linalg.matrix_power(A, t) @ x_0+ CT @ uk + NT @ w  
#             x3T = np.linalg.matrix_power(A, t) @ x_0+ CT_hat @ u3 + NT @ w
#             s1.append(x1T)
#             s2.append(x2T)
#             sk.append(xkT)
#             s3.append(x3T)
        
#             x1.append(X_bar @ a1-xf)
#             x2.append(x2T-xf)
#             xk.append(xkT-xf)
#             x3.append(x3T-xf)

#         count1 = np.mean(x1, axis = 0).T @ np.linalg.inv(sample_sigma) @ np.mean(x1, axis = 0)  
#         count2 =  np.mean(x2, axis = 0).T @ np.linalg.inv(struct_sigma) @ np.mean(x2, axis = 0)
#         countk = np.mean(xk, axis = 0).T @  np.linalg.inv(sigmaz) @ np.mean(xk, axis = 0) 
#         count3 = np.mean(x3, axis = 0).T @ np.mean(x3, axis = 0)

#         ratio1_runs.append(count1)
#         ratio2_runs.append(count2)
#         ratiok_runs.append(countk)
#         ratio3_runs.append(count3)

#         sample_dev_runs.append(
#             np.mean(x1, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(x1, axis=0)
#         )

#         struct_dev_runs.append(
#             np.mean(x2, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(x2, axis=0)
#         )

#         known_dev_runs.append(
#             np.mean(xk, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(xk, axis=0)
#         )

#         cp_dev_runs.append(
#             np.mean(x3, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(x3, axis=0)
#         )

#         s1 = np.array(s1)
#         s2 = np.array(s2)
#         sk = np.array(sk)
#         s3 = np.array(s3)

#         u1 = u1.reshape(-1, 1)
#         u2 = u2.reshape(-1, 1)
#         uk = uk.reshape(-1, 1)
#         u3 = u3.reshape(-1, 1)

#         sample_cost_runs.append(float(u1.T @ Q_tilde @ u1))
#         struct_cost_runs.append(float(u2.T @ Q_tilde @ u2))
#         known_cost_runs.append(float(uk.T @ Q_tilde @ uk))
#         cp_cost_runs.append(float(u3.T @ Q_tilde @ u3))

#     sampleu.append(np.mean(sample_cost_runs))
#     structu.append(np.mean(struct_cost_runs))
#     originalu.append(np.mean(known_cost_runs))
#     cpu.append(np.mean(cp_cost_runs))

#     x1_dev.append(np.mean(sample_dev_runs))
#     x2_dev.append(np.mean(struct_dev_runs))
#     xk_dev.append(np.mean(known_dev_runs))
#     x3_dev.append(np.mean(cp_dev_runs))

#     ratios1.append(np.mean(ratio1_runs))
#     ratios2.append(np.mean(ratio2_runs))
#     ratiosk.append(np.mean(ratiok_runs))
#     ratios3.append(np.mean(ratio3_runs))
  
# plt.figure(figsize = (10, 6))
# plt.semilogy(T_list, x1_dev, 'o-', label = 'sample based MLE')
# plt.semilogy(T_list, x2_dev, 'o-', label = 'structure MLE')
# plt.semilogy(T_list, xk_dev, 'o--', color = 'black', label = 'known dynamics')
# plt.semilogy(T_list, x3_dev, 'o-', label = 'conformal')
# plt.xlabel('time horizon')
# plt.ylabel('average deviation in terminal state')
# plt.legend()
# plt.grid(True)
# plt.xticks(T_list)
# plt.savefig('dev_{}_{}_{}_{}.jpeg'.format(epsilon, delta, M, name))

# plt.figure(figsize = (10, 6))
# plt.semilogy(T_list, sampleu, 'o-', label = 'sample based MLE')
# plt.semilogy(T_list, structu, 'o-', label = 'structure MLE')
# plt.semilogy(T_list, originalu, 'o--', color = 'black', label = 'known dynamics')
# plt.semilogy(T_list, cpu, 'o-', label = 'conformal')
# plt.xlabel('time horizon')
# plt.ylabel('optimal cost')
# plt.legend()
# plt.grid(True)
# plt.xticks(T_list)
# plt.savefig('norm_{}_{}_{}_{}.jpeg'.format(epsilon, delta, M, name))

# lambda_star = find_lambda_star(n, epsilon, 1-delta)
# lambda_cp = (epsilon - q)**2  
# print(lambda_star)
# print(lambda_cp)

# plt.figure(figsize = (10, 6))
# plt.yscale('log')
# plt.plot(
#     T_list, ratios1, 'o-',
#     label=r'sample-based MLE, $\hat{\mu}_e^\top \hat{\Sigma}_e^{-1}\hat{\mu}_e$'
# )

# plt.plot(
#     T_list, ratios2, 'o-',
#     label=r'structure-aware MLE, $\hat{\mu}_e^\top \hat{\Sigma}_e^{-1}\hat{\mu}_e$'
# )

# plt.plot(
#     T_list, ratiosk, 'o--',
#     color='black',
#     label=r'known dynamics, $\mu_e^\top \Sigma_e^{-1}\mu_e$'
# )

# plt.plot(
#     T_list, ratios3, 'o-',
#     label=r'conformal, $\hat{\mu}_e^\top \hat{\Sigma}_e^{-1}\hat{\mu}_e$'
# )
# plt.axhline(y=lambda_star,
#             color='black',
#             linestyle=':',
#             linewidth=2,
#             label=r'$\lambda^*$')
# plt.axhline(y=lambda_cp,
#             linestyle=':',
#             linewidth=2,
#             label=r'$\lambda_{cp}$')
# plt.xlabel('time horizon')
# # plt.ylabel(r'$\mu_e^\top \Sigma_e^{-1}\mu_e$')
# plt.legend()
# plt.grid(True)
# plt.xticks(T_list)
# plt.savefig('ratios_{}.jpeg'.format(M))



# n, m = 4, 4
# Q = np.diag([2.0, 3.0, 4.0, 5.0])
# t_fixed = 5
# N_fixed = 80
# M_fixed = 80

# M_list = [5, 10, 20, 40, 80, 120, 140, 180]
# N_list = [10, 20, 40, 80, 160, 320, 640]

# K_errors = 40          # number of independent estimates for averaged error curves
# K_bias_var = 40        # number of independent estimates for bias-variance curves

# # =========================
# # Average Error vs M
# # =========================

# mu_samp_M, mu_struct_M, sig_samp_M, sig_struct_M, CT_err_M = average_errors_nested(
#     t=t_fixed,
#     M_list=M_list,
#     N_fixed=N_fixed,
#     K=K_errors,
#     seed=100
# )

# # =========================
# # Average Error vs N
# # =========================

# mu_samp_N, mu_struct_N, sig_samp_N, sig_struct_N, CT_err_N = average_errors_nested(
#     t=t_fixed,
#     N_list=N_list,
#     M_fixed=M_fixed,
#     K=K_errors,
#     seed=200
# )

# # =========================
# # Plot average estimation errors
# # =========================

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# axes[0, 0].loglog(M_list, mu_samp_M, '^-', label='sample')
# axes[0, 0].loglog(M_list, mu_struct_M, '^-', label='structure')
# axes[0, 0].set_xlabel('M')
# axes[0, 0].set_ylabel(r'Average $\|\hat{\mu}_X - \mu_X\|_2$')
# axes[0, 0].legend()
# axes[0, 0].grid(True, which="both", alpha=0.3)

# axes[0, 1].loglog(M_list, sig_samp_M, '^-', label='sample')
# axes[0, 1].loglog(M_list, sig_struct_M, '^-', label='structure')
# axes[0, 1].set_xlabel('M')
# axes[0, 1].set_ylabel(r'Average $\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_2$')
# axes[0, 1].legend()
# axes[0, 1].grid(True, which="both", alpha=0.3)

# axes[1, 0].loglog(N_list, mu_samp_N, '^-', label='sample')
# axes[1, 0].loglog(N_list, mu_struct_N, '^-', label='structure')
# axes[1, 0].set_xlabel('N')
# axes[1, 0].set_ylabel(r'Average $\|\hat{\mu}_X - \mu_X\|_2$')
# axes[1, 0].legend()
# axes[1, 0].grid(True, which="both", alpha=0.3)

# axes[1, 1].loglog(N_list, sig_samp_N, '^-', label='sample')
# axes[1, 1].loglog(N_list, sig_struct_N, '^-', label='structure')
# axes[1, 1].set_xlabel('N')
# axes[1, 1].set_ylabel(r'Average $\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_2$')
# axes[1, 1].legend()
# axes[1, 1].grid(True, which="both", alpha=0.3)

# plt.tight_layout()
# plt.savefig('average_estimation_errors_combined.jpeg')
# plt.close()

# # =========================
# # Plot average CT error
# # =========================

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].loglog(M_list, CT_err_M, '^-', label=r'$\hat{C}_T$ error')
# axes[0].set_xlabel('M')
# axes[0].set_ylabel(r'Average $\|\hat{C}_T - C_T\|_2$')
# axes[0].legend()
# axes[0].grid(True, which="both", alpha=0.3)

# axes[1].loglog(N_list, CT_err_N, '^-', label=r'$\hat{C}_T$ error')
# axes[1].set_xlabel('N')
# axes[1].set_ylabel(r'Average $\|\hat{C}_T - C_T\|_2$')
# axes[1].legend()
# axes[1].grid(True, which="both", alpha=0.3)

# plt.tight_layout()
# plt.savefig('average_CT_error_combined.jpeg')
# plt.close()

T_list = [6, 8]

all_gamma = {}

all_lambda = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_econf = {}
all_dev_known = {}
all_dev_eknown = {}
all_dev_estruct = {}

for T in T_list:

    # Build system matrices
    A, B, CT = get_system(name, n, m, T, seed=0)
    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b =  -1 *  xf

    sigma_w = np.eye(n*T) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T

    Q_tilde = block_diag(*([Q] * T))

    gamma_min = 0.0
    gamma_max = float(b.T @ np.linalg.inv(sigma_z) @ b)

    gamma_vals = np.linspace(gamma_min, gamma_max, 10)

    cost_struct = []
    cost_conf = []
    cost_known = []

    dev_struct = []
    dev_conf = []
    dev_known = []
    dev_econf = []
    dev_estruct = []
    dev_eknown = []
    N = m*T +10
    N_cal = 60


    for gamma in gamma_vals:

        # ---------- averages over datasets ----------
        cost_struct_runs = []
        cost_conf_runs = []
        cost_known_runs = []

        dev_struct_runs = []
        dev_conf_runs = []
        dev_known_runs = []
        dev_econf_runs = []
        dev_estruct_runs = []
        dev_eknown_runs = []

        for dataset in range(k_est):

            np.random.seed(dataset)
            U = np.random.randn(m*T, N)

            X_bar = np.zeros((n, N))
            all_xT = np.zeros((n, N, M))

            for i in range(N):

                sumx = np.zeros(n)

                for j in range(M):
                    x0 = np.random.multivariate_normal(np.zeros(n),sigma_0)

                    w = np.random.multivariate_normal(
                        np.zeros(n*T),
                        np.eye(n*T) * sigma**2
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

            # ======================================================
            # Optimization
            # ======================================================

            u1 = cp.Variable(m*T)
            u2 = cp.Variable(m*T)
            u3 = cp.Variable(m*T)

            prob1 = cp.Problem(
                cp.Minimize(cp.quad_form(u1, Q_tilde)),
                [cp.quad_form(CT_hat @ u1 - xf,
                              np.linalg.inv(struct_sigma)) <= gamma]
            )

            prob2 = cp.Problem(
                cp.Minimize(cp.quad_form(u2, Q_tilde)),
                [cp.quad_form(CT_hat @ u2 - xf,
                              np.eye(n)) <= gamma]
            )

            prob3 = cp.Problem(
                cp.Minimize(cp.quad_form(u3, Q_tilde)),
                [cp.quad_form(CT @ u3 - xf,
                              np.linalg.inv(sigma_z)) <= gamma]
            )

            prob1.solve(solver=cp.SCS, verbose=False)
            prob2.solve(solver=cp.SCS, verbose=False)
            prob3.solve(solver=cp.SCS, verbose=False)

            if (u1.value is None or
                u2.value is None or
                u3.value is None):
                continue

            # ======================================================
            # Store costs
            # ======================================================

            cost_struct_runs.append(float(u1.value.T @ Q_tilde @ u1.value))
            cost_conf_runs.append(float(u2.value.T @ Q_tilde @ u2.value))
            cost_known_runs.append(float(u3.value.T @ Q_tilde @ u3.value))

            # ======================================================
            # Monte-Carlo evaluation
            # ======================================================

            idev_struct = []
            idev_conf = []
            idev_known = []
            idev_econf = []
            idev_estruct = []
            idev_eknown = []

            for _ in range(5000):
                x0 = np.random.multivariate_normal(
                np.zeros(n),
                sigma_0)

                w = np.random.multivariate_normal(
                    np.zeros(n*T),
                    np.eye(n*T) * sigma**2
                )

                x1 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u1.value + N_T @ w
                x2 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u2.value + N_T @ w
                x3 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u3.value + N_T @ w

                e1 = x1 - xf
                e2 = x2 - xf
                e3 = x3 - xf

                idev_struct.append(e1.T @ np.linalg.inv(struct_sigma) @ e1)
                idev_conf.append(e2.T @ np.linalg.inv(struct_sigma) @ e2)
                idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)

                idev_econf.append(e2.T @ e2)
                idev_estruct.append(e1.T @ e1)
                idev_eknown.append(e3.T @ e3)

            dev_struct_runs.append(np.mean(idev_struct))
            dev_conf_runs.append(np.mean(idev_conf))
            dev_known_runs.append(np.mean(idev_known))

            dev_econf_runs.append(np.mean(idev_econf))
            dev_estruct_runs.append(np.mean(idev_estruct))
            dev_eknown_runs.append(np.mean(idev_eknown))

        # ==========================================================
        # Average over datasets for THIS gamma
        # ==========================================================

        cost_struct.append(np.mean(cost_struct_runs))
        cost_conf.append(np.mean(cost_conf_runs))
        cost_known.append(np.mean(cost_known_runs))

        dev_struct.append(np.mean(dev_struct_runs))
        dev_conf.append(np.mean(dev_conf_runs))
        dev_known.append(np.mean(dev_known_runs))

        dev_econf.append(np.mean(dev_econf_runs))
        dev_estruct.append(np.mean(dev_estruct_runs))
        dev_eknown.append(np.mean(dev_eknown_runs))

    all_gamma[T] = gamma_vals

    all_cost_struct[T] = cost_struct
    all_cost_conf[T] = cost_conf
    all_cost_known[T] = cost_known

    all_dev_struct[T] = dev_struct
    all_dev_conf[T] = dev_conf
    all_dev_known[T] = dev_known

    all_dev_econf[T] = dev_econf
    all_dev_estruct[T] = dev_estruct
    all_dev_eknown[T] = dev_eknown

plt.figure(figsize=(10,6))

colors = ['tab:blue', 'tab:orange', 'tab:green']

for  T in T_list:

    plt.plot(all_gamma[T], all_cost_struct[T], marker = 'o',
             label=f'Structure, T={T}')

    plt.plot(all_gamma[T], all_cost_conf[T],marker = '^',
             label=f'Conformal, T={T}')

    plt.plot(all_gamma[T], all_cost_known[T],marker = 's',
             label=f'Known, T={T}')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.legend()
plt.grid(True)
 

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.legend()
plt.grid(True)
plt.savefig('cost_vs_gamma_allT.jpeg')
plt.show()

plt.figure(figsize=(10,6))

for T in T_list:
    plt.plot(all_gamma[T], all_dev_struct[T], marker='o', label=f"Structure, T={T}")
    plt.plot(all_gamma[T], all_dev_conf[T], marker='s', label=f"Conformal, T={T}")
    plt.plot(all_gamma[T], all_dev_known[T], '--', marker='^', label=f"Known, T={T}")


plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\mu_e^\top \Sigma_e^{-1} \mu_e$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('deviation_vs_gamma_allT.jpeg')

plt.figure(figsize=(10,6))
for T in T_list:
    plt.plot(all_gamma[T], all_dev_econf[T], marker = '^', label = f'Conformal, T = {T}')
    plt.plot(all_gamma[T], all_dev_estruct[T], marker='o', label=f"Structure, T={T}")
    plt.plot(all_gamma[T], all_dev_eknown[T], '--', marker='^', label=f"Known, T={T}")


plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\mu_e^\top \mu_e$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('edeviation_vs_gamma_allT.jpeg')


# # T = 8

# # all_sigma = {}
# # all_cost_struct = {}
# # all_cost_conf = {}
# # all_cost_known = {}
# # all_dev_struct = {}
# # all_dev_conf = {}
# # all_dev_known = {}
# # quant1 = []


# # A, B, CT = get_system(name, n, m, T, seed=0)

# # N_T = np.eye(n)
# # for i in range(1, T):
# #     N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

# # Q_tilde = block_diag(*([Q] * T))

# # N = m*T + 10
# # N_cal = 60

# # U = np.random.randn(m*T, N)
# # U_cal = np.random.randn(m*T, N_cal)

# sigma_vals = np.linspace(0.3, 7, 40)

# # ---------- initialize ONCE ----------
# cost_struct = []
# cost_conf = []
# cost_known = []

# dev_struct = []
# dev_conf = []
# dev_econf = []
# dev_known = []

# lambda_star = find_lambda_star(n, epsilon, 1-delta)

# w_train = np.random.multivariate_normal(
#     np.zeros(n*T),
#     np.eye(n*T),
#     size=(N, M)
# )
# q_sigma = []
# # -------------------------------------
# for sigma in sigma_vals:

#     sigma_w = np.eye(n*T) * sigma**2
#     sigma_z = N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

#     X_bar = np.zeros((n, N))
#     all_xT = np.zeros((n, N, M))

#     for i in range(N):
#         sumx = np.zeros(n)
#         for j in range(M):
#             w = sigma * w_train[i, j]
#             x_0 = np.random.multivariate_normal(np.zeros(n), sigma_0)
#             xT = np.linalg.matrix_power(A, T) @ x_0 + CT @ U[:, i] + N_T @ w  

#             all_xT[:, i, j] = xT
#             sumx += xT

#         X_bar[:, i] = sumx / M

#     CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

#     sample_sigma = np.zeros((n, n))
#     struct_sigma = np.zeros((n, n))

#     for i in range(N):
#         for j in range(M):
#             diff = all_xT[:, i, j] - X_bar[:, i]
#             diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]

#             sample_sigma += np.outer(diff, diff)
#             struct_sigma += np.outer(diff1, diff1)

#     sample_sigma /= (N*M)
#     struct_sigma /= (N*M)

#     scores = get_scores(
#         U_cal,
#         N_cal,
#         CT_hat,
#         N_T,
#         CT,
#         T,
#         sigma       # after modifying get_scores
#     )

#     q = conformal_quantile(scores, delta)
#     q_sigma.append(q)
#     lambda_cp = (epsilon - q)**2

#     u1 = cp.Variable(m*T)
#     u2 = cp.Variable(m*T)
#     u3 = cp.Variable(m*T)

#     objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
#     objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
#     objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))

#     constraint1 = cp.quad_form(
#         CT_hat @ u1 - xf,
#         np.linalg.inv(struct_sigma)
#     ) <= lambda_star

#     constraint2 = cp.quad_form(
#         CT_hat @ u2 - xf,
#         np.eye(n)
#     ) <= lambda_cp

#     constraint3 = cp.quad_form(
#         CT @ u3 - xf,
#         np.linalg.inv(sigma_z)
#     ) <= lambda_star

#     cp.Problem(objective1, [constraint1]).solve(solver=cp.SCS)
#     cp.Problem(objective2, [constraint2]).solve(solver=cp.SCS)
#     cp.Problem(objective3, [constraint3]).solve(solver=cp.SCS)

#     if u1 is None:
#         print(f"1st Convex solver failed at T={t}")
#         continue

#     if u2 is None:
#         print(f"2nd Convex solver failed at T={t}")
#         continue

#     if u3 is None:
#         print(f"original Convex solver failed at T={t}")
#         continue

#     cost_struct.append(float(u1.value.T @ Q_tilde @ u1.value))
#     cost_conf.append(float(u2.value.T @ Q_tilde @ u2.value))
#     cost_known.append(float(u3.value.T @ Q_tilde @ u3.value))

#     idev_struct = []
#     idev_conf = []
#     idev_econf = []
#     idev_known = []

#     for _ in range(5000):
#         x0 = np.random.multivariate_normal(
#             np.zeros(n),
#             sigma_0)
#         w = np.random.multivariate_normal(
#             np.zeros(n*T),
#             sigma_w
#         )

#         xT1 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u1.value + N_T @ w
#         xT2 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u2.value + N_T @ w
#         xT3 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u3.value + N_T @ w

#         e1 = xT1 - xf
#         e2 = xT2 - xf
#         e3 = xT3 - xf

#         idev_struct.append(e1.T @ np.linalg.inv(struct_sigma) @ e1)
#         idev_conf.append(e2.T @ np.linalg.inv(struct_sigma) @ e2)
#         idev_econf.append(e2.T @ e2)
#         idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)

#     dev_struct.append(np.mean(idev_struct))
#     dev_conf.append(np.mean(idev_conf))
#     dev_econf.append(np.mean(idev_econf))
#     dev_known.append(np.mean(idev_known))


# plt.figure(figsize=(10,6))

# plt.plot(sigma_vals, cost_struct,
#         marker='o', linewidth=2,
#         label='Structure')

# plt.plot(sigma_vals, cost_known,
#         marker='o', linewidth=2,
#         label='Known')

# plt.plot(sigma_vals, cost_conf,
#         marker='s', linewidth=2,
#         label='Conformal')
# plt.xlabel(r'$\sigma$')
# plt.ylabel(r'$J^*(\sigma)$')
# plt.grid(True)
# plt.legend()
# plt.savefig('cost_vs_sigma_T8.jpeg', dpi=300, bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(10,6))

# plt.plot(sigma_vals, dev_struct,
#         marker='o', linewidth=2,
#         label='Structure')
# plt.plot(sigma_vals, dev_known,
#         marker='o', linewidth=2,
#         label='Known')

# plt.plot(sigma_vals, dev_conf,
#         marker='s', linewidth=2,
#         label='Conformal')

# plt.xlabel(r'$\sigma$')
# plt.ylabel(r'$\hat{\mu}_e^\top\hat{\Sigma}_e^{-1}\hat{\mu}_e$')
# # plt.ylabel(r'$\mu_e^\top\Sigma_e^{-1}\mu_e$-$$')
# plt.grid(True)
# plt.legend()
# plt.savefig('deviation_vs_sigma_T8.jpeg', dpi=300, bbox_inches='tight')
# plt.show()

# plt.figure()
# plt.plot(sigma_vals, q_sigma, marker = 'o')
# plt.xlabel(r'$\sigma$')
# plt.ylabel(r'$q_{1-\delta}$')
# plt.grid(True, which = 'both')
# plt.savefig('q_sigma.jpeg')

# plt.figure(figsize=(10,6))

# nominal = [1-d for d in delta_list]
# T_list = range(1, 25, 2)
# T_list_cov = range(1, 25, 8)

# for t in T_list_cov:

#     cover = [
#         empirical_coverage[t][d][0]
#         for d in delta_list
#     ]

#     plt.plot(
#         nominal,
#         cover,
#         marker='o',
#         label=f"T={t}"
#     )

# plt.plot([0.75,1],[0.75,1],'k--',label='Ideal')

# plt.xlabel("Nominal Coverage")

# plt.ylabel("Empirical Coverage")

# plt.grid(True)

# plt.legend()

# plt.savefig("coverage_vs_nominal.jpeg")

# plt.figure(figsize=(10,6))

# plt.plot(
#     T_list,
#     quant,
#     marker='o',
#     linewidth=2
# )

# plt.xlabel("T")

# plt.ylabel(r"$q_{1-\delta}$")

# plt.grid(True)

# plt.savefig("quantile_vs_T.jpeg")

# #q vs n
# # -----------------------------
# # Parameters
# # -----------------------------
# T = 8
# m = 4
# N = m*T + 10
# N_cal = 100
# M = 100

# sigma = 0.3
# delta = 0.3       # Fixed miscoverage level

# n_list = [2,4,6,8,10,12,16,20]

# q_values = []

# for n in n_list:

#     print(f"Running n = {n}")

#     # -----------------------------
#     # Build system
#     # -----------------------------
#     A, B, CT = get_system("stable", n, m, T, seed=0)

#     NT = np.eye(n)
#     for k in range(1, T):
#         NT = np.hstack((NT, np.linalg.matrix_power(A, k)))

#     # -----------------------------
#     # Generate training data
#     # -----------------------------
#     np.random.seed(0)

#     U = np.random.randn(m*T, N)
#     U_cal = np.random.randn(m*T, N_cal)

#     X_bar = np.zeros((n, N))

#     for i in range(N):

#         x_avg = np.zeros(n)

#         for _ in range(M):
#             x0 = np.random.multivariate_normal(
#             np.zeros(n),
#             sigma_0)
#             w = np.random.multivariate_normal(
#                 np.zeros(n*T),
#                 sigma**2 * np.eye(n*T)
#             )

#             xT = np.linalg.matrix_power(A, T)@x_0 +  CT @ U[:, i] + NT @ w
#             x_avg += xT

#         X_bar[:, i] = x_avg / M

#     # -----------------------------
#     # Estimate C_T
#     # -----------------------------
#     CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

#     # -----------------------------
#     # Calibration scores
#     # -----------------------------
#     scores = get_scores(
#         U_cal,
#         N_cal,
#         CT_hat,
#         NT,
#         CT,
#         T,
#         sigma
#     )

#     q = conformal_quantile(scores, delta)
#     q_values.append(q)

# # -----------------------------
# # Plot
# # -----------------------------
# plt.figure(figsize=(7,5))
# plt.plot(n_list, q_values, 'o-', linewidth=2, label = 'sigma = {}'.format(sigma))
# plt.xlabel(r"State dimension $n$", fontsize=13)
# plt.ylabel(r"$q_{1-\delta}$", fontsize=13)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('q_n.jpeg')