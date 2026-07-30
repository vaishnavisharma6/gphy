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
T_list = range(1, 20, 2)
sigma_1 = 0.2
sigma_2 = 0.3
sigma_0 = np.eye(n) * sigma_1**2
x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
delta = 0.5
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
originalabsu = []

quant = []
empirical_coverage = {}
quantiles = {}
delta_list = [0.30, 0.20, 0.10, 0.05, 0.01]
num_runs = 20
N_test = 1000

ratios1 = []
ratios2 = []
ratiosk = []
ratiosabs = []
ratios3 = []

#to store final state deviation
x1_dev = []
x2_dev = []
xk_dev = []
xabs_dev = []
x3_dev = []
delta_folder = 'traj_{}'.format(delta)
os.makedirs(delta_folder, exist_ok = True)


def build_system_matrices(t):
    A, B, CT = get_system(name, n, m, t, seed=0)
    sigma = 0.3**2 * np.eye(n*t)
    It = np.eye(t)
    sigmaw = np.kron(It, sigma)
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    # sigmaw = np.eye(n*t) * sigma_2**2
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
            x_0 = rng.multivariate_normal(np.zeros(n), np.eye(n) * (0.2)**2)
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

#end of all function definitons--------

for t in T_list:
    sample_cost_runs = []
    struct_cost_runs = []
    cp_cost_runs = []
    abs_cost_runs = []
    known_cost_runs = []

    sample_dev_runs = []
    struct_dev_runs = []
    cp_dev_runs = []
    known_dev_runs = []
    abs_dev_runs = []

    ratio1_runs = []
    ratio2_runs = []
    ratio3_runs = []
    ratiok_runs = []
    ratioabs_runs = []
    quant_runs = []
    A, B, CT = get_system(name, n, m, t, seed=0)
    print(t)
    Q_tilde = block_diag(*([Q] * t))
    sigma = 0.3**2 * np.eye(n)
    It = np.eye(t)
    sigmaw = np.kron(It, sigma)
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    # sigmaw = np.eye(n*t) * sigma_2**2
    sigmaz = NT @ sigmaw @ NT.T + np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T

    #estimation
    N = m*t +10
    N_cal = 60
    for dataset in range(k_est):
        np.random.seed(dataset)
        U = np.random.randn(m*t, N) # input matrix
        U_cal = np.random.randn(m*t, N_cal)
        w_cal_train = np.random.multivariate_normal(
        np.zeros(n*t),
        np.eye(n*t),
        size=N_cal
        )

        x_cal_train = np.random.multivariate_normal(
        np.zeros(n),
        np.eye(n),
        size=N_cal
        )
        #create bar{X}
        X_bar = np.zeros((n, N))
        all_xT = np.zeros((n, N, M))
        for i in range(N):
            sumx = np.zeros(n)
            for j in range(M):
                x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
                w = np.random.multivariate_normal(np.zeros(n*t), sigmaw)
                xT = np.linalg.matrix_power(A, t) @ x_0 + CT @ U[:,i] + NT@ w
                all_xT[:, i, j] = xT
                sumx = sumx + xT
            x_bar = sumx/M     
            X_bar[:, i] = x_bar
        
        CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)
        sample_sigma = np.zeros((n, n))
        struct_sigma = np.zeros((n, n))
        
        for i in range(N):
            for j in range(M):
                diff = all_xT[:, i, j] - X_bar[:, i]
                diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]
                sample_sigma += np.outer(diff, diff)
                struct_sigma += np.outer(diff1, diff1)

        sample_sigma = sample_sigma / (N*M)
        struct_sigma = struct_sigma/ (N*M)
        
        quantiles[t] = []
        empirical_coverage[t] = {}

        for delta in delta_list:

            coverages = []
            q_runs = []

            for run in range(num_runs):

                np.random.seed(run)

                scores = get_scores(
                    U_cal,
                    N_cal,
                    CT_hat,
                    A,
                    NT,
                    CT,
                    t,
                    sigma_1,
                    sigma_2,
                    w_cal_train,
                    x_cal_train
                )

                q = conformal_quantile(scores, delta)
                quant_runs.append(q)
                q_runs.append(q)

                U_test = np.random.randn(m*t, N_test)

                scores_test = get_test_scores(
                    U_test,
                    CT_hat,
                    A,
                    NT,
                    CT,
                    t,
                    sigma_1,
                    sigma_2,
                )

                coverages.append(
                    np.mean(scores_test <= q)
                )

            empirical_coverage[t][delta] = (
                np.mean(coverages),
                np.std(coverages)
        )

            # if delta == 0.30:
            #     quant.append(np.mean(q_runs))

            quantiles[t].append(np.mean(q_runs))

        delta = 0.50

        scores = get_scores(
            U_cal,
            N_cal,
            CT_hat,
            A,
            NT,
            CT,
            t,
            sigma_1,
            sigma_2,
            w_cal_train,
            x_cal_train
        )

        q = conformal_quantile(scores, delta)

        #rhs of all inequalities
        lambda_star = find_lambda_star(n, epsilon, 1-delta)
        lambda_cp = (epsilon - q)**2    

        alpha1 = cp.Variable(N)
        alpha2 = cp.Variable(m*t)
        alphak = cp.Variable(m*t)
        u = cp.Variable(m*t)
        u4 = cp.Variable(m*t)
        objective1 = cp.Minimize(cp.quad_form((U @ alpha1), Q_tilde))
        objective2 = cp.Minimize(cp.quad_form((alpha2), Q_tilde))
        objectivek = cp.Minimize(cp.quad_form((alphak), Q_tilde))
        objective3 = cp.Minimize(cp.quad_form(u, Q_tilde))
        objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))

        f1 = cp.quad_form(X_bar @ alpha1 - xf, np.linalg.inv(sample_sigma))
        f2 = cp.quad_form(CT_hat @ alpha2 - xf, np.linalg.inv(struct_sigma))
        fk = cp.quad_form(CT @ alphak - xf, np.linalg.inv(sigmaz))
        f3 = cp.quad_form(CT_hat @ u - xf, np.eye(n))
        f4 = cp.quad_form(CT @ u4 - xf, np.eye(n))

        prob1 = cp.Problem(objective1, [f1 <= lambda_star])
        prob2 = cp.Problem(objective2, [f2 <= lambda_star])
        probk = cp.Problem(objectivek, [fk <= lambda_star])
        prob3 = cp.Problem(objective3, [f3 <= lambda_cp])
        prob4 = cp.Problem(objective4, [f4 <= lambda_star])

        prob1.solve(solver = cp.SCS, verbose = False)
        prob2.solve(solver = cp.SCS, verbose = False)
        probk.solve(solver = cp.SCS, verbose = False)
        prob3.solve(solver = cp.SCS, verbose = False)
        prob4.solve(solver = cp.SCS, verbose = False)

        a1 = alpha1.value
        a2 = alpha2.value
        ak = alphak.value
        u3 = u.value
        u4 = u4.value

        u1 = U @ a1
        u2 = a2
        uk = ak
        u4 = u4
        
        if a1 is None:
            print(f"1st Convex solver failed at T={t}")
            continue

        if a2 is None:
            print(f"2nd Convex solver failed at T={t}")
            continue

        if ak is None:
            print(f"original Convex solver failed at T={t}")
            continue

        if u3 is None:
            print(f"cp Convex solver failed at T={t}")
            continue
        if u4 is None:
            print(f"cp Convex solver failed at T={t}")
            continue

        np.random.seed(0)

        x1 = []
        x2 = []
        xk = []
        x3 = []
        x4 = []
        
        for j in range(0, 5000):
            x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
            w = np.random.multivariate_normal(np.zeros(n*t), sigmaw)
            x1T = np.linalg.matrix_power(A, t) @ x_0 + CT @ u1 + NT@w
            x2T =  np.linalg.matrix_power(A, t) @ x_0+ CT_hat @ u2 + NT @ w  
            xkT = np.linalg.matrix_power(A, t) @ x_0+ CT @ uk + NT @ w  
            x3T = np.linalg.matrix_power(A, t) @ x_0+ CT_hat @ u3 + NT @ w
            x4T = np.linalg.matrix_power(A, t) @ x_0 + CT @ u4 + NT @ w
        
            x1.append(X_bar @ a1-xf)
            x2.append(x2T-xf)
            xk.append(xkT-xf)
            x3.append(x3T-xf)
            x4.append(x4T -xf)

        count1 = np.mean(x1, axis = 0).T @ np.linalg.inv(sample_sigma) @ np.mean(x1, axis = 0)  
        count2 =  np.mean(x2, axis = 0).T @ np.linalg.inv(struct_sigma) @ np.mean(x2, axis = 0)
        countk = np.mean(xk, axis = 0).T @  np.linalg.inv(sigmaz) @ np.mean(xk, axis = 0) 
        count3 = np.mean(x3, axis = 0).T @ np.mean(x3, axis = 0)

        ratio1_runs.append(count1)
        ratio2_runs.append(count2)
        ratiok_runs.append(countk)
        ratio3_runs.append(count3)

        sample_dev_runs.append(
            np.mean(x1, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(x1, axis=0)
        )

        struct_dev_runs.append(
            np.mean(x2, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(x2, axis=0)
        )

        known_dev_runs.append(
            np.mean(xk, axis=0).T @ np.linalg.pinv(sigmaz) @ np.mean(xk, axis=0)
        )

        cp_dev_runs.append(
            np.mean(x3, axis=0).T @ np.mean(x3, axis=0)
        )
        abs_dev_runs.append(np.mean(x4, axis = 0).T @ np.mean(x4, axis = 0))

        u1 = u1.reshape(-1, 1)
        u2 = u2.reshape(-1, 1)
        uk = uk.reshape(-1, 1)
        u3 = u3.reshape(-1, 1)
        u4 = u4.reshape(-1, 1)

        sample_cost_runs.append(float(u1.T @ Q_tilde @ u1))
        struct_cost_runs.append(float(u2.T @ Q_tilde @ u2))
        known_cost_runs.append(float(uk.T @ Q_tilde @ uk))
        cp_cost_runs.append(float(u3.T @ Q_tilde @ u3))
        abs_cost_runs.append(float(u4.T @ Q_tilde @ u4))
    quant.append(np.mean(quant_runs))
    sampleu.append(np.mean(sample_cost_runs))
    structu.append(np.mean(struct_cost_runs))
    originalu.append(np.mean(known_cost_runs))
    cpu.append(np.mean(cp_cost_runs))
    originalabsu.append(np.mean(abs_cost_runs))

    x1_dev.append(np.mean(sample_dev_runs))
    x2_dev.append(np.mean(struct_dev_runs))
    xk_dev.append(np.mean(known_dev_runs))
    x3_dev.append(np.mean(cp_dev_runs))
    xabs_dev.append(np.mean(abs_dev_runs))
 
plt.figure(figsize = (10, 6))
# plt.semilogy(T_list, x1_dev, 'o-', label = 'sample based MLE')
plt.semilogy(T_list, x2_dev, 'o-', label = 'structure MLE')
plt.semilogy(T_list, xk_dev, '--', color = 'black', label = 'normalized known')
plt.semilogy(T_list, x3_dev, 'o-', label = 'conformal')
plt.semilogy(T_list, xabs_dev, 'o--', color = 'red', label = 'absolute known')
plt.xlabel('T')
plt.ylabel('mean deviation')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('dev_{}_{}_{}_{}.jpeg'.format(epsilon, delta, M, name))

plt.figure(figsize = (10, 6))
# plt.semilogy(T_list, sampleu, 'o-', label = 'sample based MLE')
plt.semilogy(T_list, structu, 'o-', label = 'structure MLE')
plt.semilogy(T_list, originalu, '--', color = 'black', label = 'normalized known')
plt.semilogy(T_list, cpu, 'o-', label = 'conformal')
plt.semilogy(T_list, originalabsu, '--', color = 'red', label = 'absolute known')
plt.xlabel('T')
plt.ylabel('optimal cost')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('norm_{}_{}_{}_{}.jpeg'.format(epsilon, delta, M, name))

lambda_star = find_lambda_star(n, epsilon, 1-delta)
lambda_cp = (epsilon - q)**2  
print(lambda_star)
print(lambda_cp)
plt.figure(figsize=(10,6))

nominal = [1-d for d in delta_list]
T_list = range(1, 20, 2)
T_list_cov = range(1, 20, 2)

for t in T_list_cov:
    cover = [
        empirical_coverage[t][d][0]
        for d in delta_list
    ]
    plt.plot(
        nominal,
        cover,
        marker='o',
        label=f"T={t}"
    )

plt.plot([0.75,1],[0.75,1],'k--',label='Ideal')

plt.xlabel("Nominal Coverage")

plt.ylabel("Empirical Coverage")

plt.grid(True)

plt.legend()

plt.savefig("coverage_vs_nominal.jpeg")

plt.figure(figsize=(10,6))

plt.plot(
    T_list,
    quant,
    marker='o',
    linewidth=2
)

plt.xlabel("T")

plt.ylabel(r"$q_{1-\delta}$")

plt.grid(True)

plt.savefig("quantile_vs_T.jpeg")

#q vs n
# -----------------------------
# Parameters
# -----------------------------
T = 8
m = 4
N = m*T + 10
N_cal = 100
M = 100

sigma_1 = 0.2
sigma_2 = 0.3
delta = 0.3       # Fixed miscoverage level

n_list = [2,4,6,8,10,12,16,20]

q_values = []

for n in n_list:

    print(f"Running n = {n}")

    # -----------------------------
    # Build system
    # -----------------------------
    A, B, CT = get_system("stable", n, m, T, seed=0)
    w_cal_train = np.random.multivariate_normal(np.zeros(n*T), 
        np.kron(np.eye(T),np.eye(n)*sigma_2**2),
        size=N_cal
    )
    x_cal_train = np.random.multivariate_normal(
        np.zeros(n),
        np.eye(n),
        size=N_cal
    )
    NT = np.eye(n)
    for k in range(1, T):
        NT = np.hstack((NT, np.linalg.matrix_power(A, k)))
    # -----------------------------
    # Generate training data
    # -----------------------------
    np.random.seed(0)

    U = np.random.randn(m*T, N)
    U_cal = np.random.randn(m*T, N_cal)

    X_bar = np.zeros((n, N))

    for i in range(N):

        x_avg = np.zeros(n)

        for _ in range(M):
            x_0 = np.random.multivariate_normal(
            np.zeros(n),
            sigma_1**2 * np.eye(n))
            w = np.random.multivariate_normal(np.zeros(n*T), np.kron(np.eye(T),np.eye(n)*sigma_2**2))

            xT = np.linalg.matrix_power(A, T)@x_0 +  CT @ U[:, i] + NT @ w
            x_avg += xT

        X_bar[:, i] = x_avg / M

    # -----------------------------
    # Estimate C_T
    # -----------------------------
    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

    # -----------------------------
    # Calibration scores
    # -----------------------------
    scores = get_scores(
        U_cal,
        N_cal,
        CT_hat,
        A,
        NT,
        CT,
        T,
        sigma_1,
        sigma_2,
        w_cal_train,
        x_cal_train
    )

    q = conformal_quantile(scores, delta)
    q_values.append(q)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(n_list, q_values, 'o-', linewidth=2)
plt.xlabel(r"State dimension $n$", fontsize=13)
plt.ylabel(r"$q_{1-\delta}$", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig('q_n.jpeg')