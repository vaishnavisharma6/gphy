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

n,m = 2,2
x_0 = np.array([0.0, 0.0])
xf = np.array([3.0, 7.0])
T_list = range(1, 50, 2)
sigma_0 = np.eye(n) * 0.1
delta = 0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 3.0
M = 60

sampleu = []
structu = []
cpu = []
originalu = []
originalnd = []

quant = []

ratios1 = []
ratios2 = []
ratiosk = []
ratioskd = []
ratios3 = []

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

    sigmaw = np.eye(n*t) * 0.1
    sigmaz = NT @ sigmaw @ NT.T

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
            w = rng.multivariate_normal(np.zeros(n*t), sigmaw)
            all_xT_full[:, i, j] = CT @ U_full[:, i] + NT @ w

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

    mu_sample_error = np.linalg.norm(mu_sample - mu_true, ord='fro')
    mu_struct_error = np.linalg.norm(mu_struct - mu_true, ord='fro')
    sigma_sample_error = np.linalg.norm(sample_sigma - sigmaz, ord='fro')
    sigma_struct_error = np.linalg.norm(struct_sigma - sigmaz, ord='fro')
    CT_error = np.linalg.norm(CT_hat - CT, ord='fro')

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


def get_scores(U_cal, N_cal, CT_hat, NT, CT, t):
    scores = []
    for i in range(N_cal):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x_true = CT @ U_cal[:, i] + NT@w
        x_cal = CT_hat @ U_cal[:, i]
        scores.append(np.linalg.norm(x_true - x_cal))

    return(np.array(scores))

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

for t in T_list:
    np.random.seed(0)
    A, B, CT = get_system(name, n, m, t, seed=0)
    # uf = np.random.randn(m*t, 1)
    # xf = CT@ uf
    # xf = xf.reshape(-1)
    Q_tilde = block_diag(*([Q] * t))
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

    sigmaw = np.eye(n*t) * 0.1
    sigmaz = NT @ sigmaw @ NT.T
    
    #estimation
    N = m*t +3
    N_cal = 30

    U = np.random.randn(m*t, N) # input matrix
    U_cal = np.random.randn(m*t, N_cal)
    #create bar{X}
    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))
    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
            xT = CT @ U[:,i] + NT@ w
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
    
    #find 1-delta quantile of scores
    scores = get_scores(U_cal, N_cal, CT_hat, NT, CT, t)
    q = conformal_quantile(scores, delta)
    quant.append(q)

    #rhs of all inequalities
    lambda_star = find_lambda_star(n, epsilon, delta)
    lambda_cp = (epsilon - q)**2    

    alpha1 = cp.Variable(N)
    alpha2 = cp.Variable(N)
    alphak = cp.Variable(N)
    ukd = cp.Variable(m*t)
    u = cp.Variable(m*t)
    


    objective1 = cp.Minimize(cp.quad_form((U @ alpha1), Q_tilde))
    objective2 = cp.Minimize(cp.quad_form((U @ alpha2), Q_tilde))
    objectivek = cp.Minimize(cp.quad_form((U @ alphak), Q_tilde))
    objectivekd = cp.Minimize(cp.quad_form(ukd, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u, Q_tilde))

    f1 = cp.quad_form(X_bar @ alpha1 - xf, np.linalg.inv(sample_sigma))
    f2 = cp.quad_form(CT_hat @ U @ alpha2 - xf, np.linalg.inv(struct_sigma))
    fk = cp.quad_form(CT @ U @ alphak - xf, np.linalg.inv(sigmaz))
    fkd = cp.quad_form(CT @ ukd - xf, np.linalg.inv(sigmaz))
    f3 = cp.quad_form(CT_hat @ u - xf, np.eye(n))

    prob1 = cp.Problem(objective1, [f1 <= lambda_star])
    prob2 = cp.Problem(objective2, [f2 <= lambda_star])
    probk = cp.Problem(objectivek, [fk <= lambda_star])
    probkd = cp.Problem(objectivekd, [fkd <= lambda_star])
    prob3 = cp.Problem(objective3, [f3 <= lambda_cp])

    prob1.solve(solver = cp.SCS, verbose = False)
    prob2.solve(solver = cp.SCS, verbose = False)
    probk.solve(solver = cp.SCS, verbose = False)
    probkd.solve(solver = cp.SCS, verbose = False)
    prob3.solve(solver = cp.SCS, verbose = False)

    a1 = alpha1.value
    a2 = alpha2.value
    ak = alphak.value
    ud = ukd.value
    u3 = u.value

    u1 = U @ a1
    u2 = U @ a2
    uk = U @ ak

    w = np.random.randn(n*t)
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

    np.random.seed(0)

    x1 = []
    x2 = []
    xk = []
    xkd = []
    x3 = []
    

    s1 = []
    s2 = []
    sk = []
    skd = []
    s3 = []

    for j in range(0, 2000):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = CT @ u1 + NT @ w  
        x2T =  CT @ u2 + NT @ w  
        xkT = CT @ uk + NT @ w  
        xkdT = CT@ ud + NT@ w
        x3T = CT @ u3 + NT @ w
        s1.append(x1T)
        s2.append(x2T)
        sk.append(xkT)
        skd.append(xkdT)
        s3.append(x3T)
      
        x1.append(np.linalg.norm(x1T-xf))

        x2.append(np.linalg.norm(x2T-xf))
        xk.append(np.linalg.norm(xkT-xf))
        xkd.append(np.linalg.norm(xkdT - xf))
        x3.append(np.linalg.norm(x3T-xf))

    count1 = sum(d <= epsilon for d in x1)   
    count2 = sum(d <= epsilon for d in x2)   
    countk = sum(d <= epsilon for d in xk)   
    countkd = sum(d<= epsilon for d in xkd)
    count3 = sum(d <= epsilon for d in x3)   

    

    ratios1.append(count1/2000)
    ratios2.append(count2/2000)
    ratiosk.append(countk/2000)
    ratioskd.append(countkd/2000)
    ratios3.append(count3/2000)

    x1_dev.append(np.average(x1))
    x2_dev.append(np.average(x2))
    xk_dev.append(np.average(xk))
    xkd_dev.append(np.average(xkd))
    x3_dev.append(np.average(x3))

    s1 = np.array(s1)
    s2 = np.array(s2)
    sk = np.array(sk)
    skd = np.array(skd)
    s3 = np.array(s3)

    u1 = u1.reshape(-1, 1)
    u2 = u2.reshape(-1, 1)
    uk = uk.reshape(-1, 1)
    ud = ud.reshape(-1, 1)
    u3 = u3.reshape(-1, 1)

    sampleu.append(float(u1.T @ Q_tilde @ u1))
    structu.append(float(u2.T @ Q_tilde @ u2))
    originalu.append(float(uk.T @ Q_tilde @ uk))
    originalnd.append(float(ud.T @ Q_tilde @ ud))
    cpu.append(float(u3.T @ Q_tilde @ u3))

    center = (xf[0], xf[1])

    theta = np.linspace(0, 2*np.pi, 400)
    x = center[0] + epsilon * np.cos(theta)
    y = center[1] + epsilon * np.sin(theta)




    filename1 = os.path.join(delta_folder, 'sa-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('sample based MLE, horizon = {}, ratio = {}'.format(t, count1/2000))
    plt.savefig(filename1)

    filename2 = os.path.join(delta_folder, 'st-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s2[:, 0], s2[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('structure based MLE, horizon = {}, ratio = {}'.format(t, count2/2000))

    plt.savefig(filename2)

    filename3 = os.path.join(delta_folder, 'known-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(sk[:, 0], sk[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('known dynamics, horizon = {}, ratio = {}'.format(t, countk/2000))

    plt.savefig(filename3)

    filename4 = os.path.join(delta_folder, 'cp-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s3[:, 0], s3[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('conformal, horizon = {}, ratio = {}'.format(t, count3/2000))

    plt.savefig(filename4)

plt.close()    
plt.figure(figsize = (10, 6))
plt.semilogy(T_list, x1_dev, 'o-', label = 'sample based MLE')
plt.semilogy(T_list, x2_dev, 'o-', label = 'structure MLE')
plt.semilogy(T_list, xk_dev, 'o--', color = 'black', label = 'known dynamics with data')
# plt.plot(T_list, xkd_dev, 'o--', color = 'black', label = 'known dynamics')
plt.semilogy(T_list, x3_dev, 'o-', label = 'conformal')
plt.xlabel('time horizon')
plt.ylabel('average deviation in terminal state')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('dev_{}_{}_{}.jpeg'.format(epsilon, delta, M))

plt.figure(figsize = (10, 6))
plt.semilogy(T_list, sampleu, 'o-', label = 'sample based MLE')
plt.semilogy(T_list, structu, 'o-', label = 'structure MLE')
plt.semilogy(T_list, originalu, 'o--', color = 'black', label = 'known dynamics with data QCQP')
# plt.plot(T_list, originalnd, 'o--', color = 'black', label = 'known dynamics')
plt.semilogy(T_list, cpu, 'o-', label = 'conformal')
plt.xlabel('time horizon')
plt.ylabel('optimal cost')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('norm_{}_{}_{}.jpeg'.format(epsilon, delta, M))


plt.figure(figsize = (10, 6))
plt.plot(T_list, ratios1, 'o-', label = 'sample based MLE')
plt.plot(T_list, ratios2, 'o-', label = 'structure MLE')
plt.plot(T_list, ratiosk, 'o--', color = 'black', label = 'known dynamics')
plt.plot(T_list, ratios3, 'o-', label = 'conformal')
plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('ratio of terminal states inside epsilon radius ball')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('ratios_{}.jpeg'.format(M))



n, m = 2, 2
Q = np.diag([2.0, 3.0])
t_fixed = 5
N_fixed = 80
M_fixed = 80

M_list = [5, 10, 20, 40, 80, 120, 140, 180]
N_list = [10, 20, 40, 80, 160, 320, 640]

K_errors = 30          # number of independent estimates for averaged error curves
K_bias_var = 30        # number of independent estimates for bias-variance curves

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
axes[0, 0].set_xlabel('M')
axes[0, 0].set_ylabel(r'Average $\|\hat{\mu}_X - \mu_X\|_F$')
axes[0, 0].set_title(f'Mean error vs M (N={N_fixed})')
axes[0, 0].legend()
axes[0, 0].grid(True, which="both", alpha=0.3)

axes[0, 1].loglog(M_list, sig_samp_M, '^-', label='sample')
axes[0, 1].loglog(M_list, sig_struct_M, '^-', label='structure')
axes[0, 1].set_xlabel('M')
axes[0, 1].set_ylabel(r'Average $\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_F$')
axes[0, 1].set_title(f'Covariance error vs M (N={N_fixed})')
axes[0, 1].legend()
axes[0, 1].grid(True, which="both", alpha=0.3)

axes[1, 0].loglog(N_list, mu_samp_N, '^-', label='sample')
axes[1, 0].loglog(N_list, mu_struct_N, '^-', label='structure')
axes[1, 0].set_xlabel('N')
axes[1, 0].set_ylabel(r'Average $\|\hat{\mu}_X - \mu_X\|_F$')
axes[1, 0].set_title(f'Mean error vs N (M={M_fixed})')
axes[1, 0].legend()
axes[1, 0].grid(True, which="both", alpha=0.3)

axes[1, 1].loglog(N_list, sig_samp_N, '^-', label='sample')
axes[1, 1].loglog(N_list, sig_struct_N, '^-', label='structure')
axes[1, 1].set_xlabel('N')
axes[1, 1].set_ylabel(r'Average $\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_F$')
axes[1, 1].set_title(f'Covariance error vs N (M={M_fixed})')
axes[1, 1].legend()
axes[1, 1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('average_estimation_errors_combined.jpeg')
plt.close()

# =========================
# Plot average CT error
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].loglog(M_list, CT_err_M, '^-', label=r'$\hat{C}_T$ error')
axes[0].set_xlabel('M')
axes[0].set_ylabel(r'Average $\|\hat{C}_T - C_T\|_F$')
axes[0].set_title('CT error vs M, N = {}'.format(N_fixed))
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)

axes[1].loglog(N_list, CT_err_N, '^-', label=r'$\hat{C}_T$ error')
axes[1].set_xlabel('N')
axes[1].set_ylabel(r'Average $\|\hat{C}_T - C_T\|_F$')
axes[1].set_title('CT error vs N, M = {}'.format(M_fixed))
axes[1].legend()
axes[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('average_CT_error_combined.jpeg')
plt.close()

# =========================
# Bias-Variance vs M
# =========================

(
    mu_s_bias_M, mu_st_bias_M,
    mu_s_var_M, mu_st_var_M,
    sig_s_bias_M, sig_st_bias_M,
    sig_s_var_M, sig_st_var_M
) = average_bias_variance_nested(
    t=t_fixed,
    M_list=M_list,
    N_fixed=N_fixed,
    K=K_bias_var,
    seed=300
)

# =========================
# Bias-Variance vs N
# =========================

(
    mu_s_bias_N, mu_st_bias_N,
    mu_s_var_N, mu_st_var_N,
    sig_s_bias_N, sig_st_bias_N,
    sig_s_var_N, sig_st_var_N
) = average_bias_variance_nested(
    t=t_fixed,
    N_list=N_list,
    M_fixed=M_fixed,
    K=K_bias_var,
    seed=400
)

# =========================
# Plot: Mu Bias-Variance
# =========================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].loglog(M_list, mu_s_bias_M, '^-', label='sample')
axes[0, 0].loglog(M_list, mu_st_bias_M, '^-', label='structure')
axes[0, 0].set_xlabel('M')
axes[0, 0].set_ylabel(r'Bias of $\hat{\mu}_X$')
axes[0, 0].set_title(f'Mean bias vs M (N={N_fixed})')
axes[0, 0].legend()
axes[0, 0].grid(True, which="both", alpha=0.3)

axes[0, 1].loglog(M_list, mu_s_var_M, '^-', label='sample')
axes[0, 1].loglog(M_list, mu_st_var_M, '^-', label='structure')
axes[0, 1].set_xlabel('M')
axes[0, 1].set_ylabel(r'Variance of $\hat{\mu}_X$')
axes[0, 1].set_title(f'Mean variance vs M (N={N_fixed})')
axes[0, 1].legend()
axes[0, 1].grid(True, which="both", alpha=0.3)

axes[1, 0].loglog(N_list, mu_s_bias_N, '^-', label='sample')
axes[1, 0].loglog(N_list, mu_st_bias_N, '^-', label='structure')
axes[1, 0].set_xlabel('N')
axes[1, 0].set_ylabel(r'Bias of $\hat{\mu}_X$')
axes[1, 0].set_title(f'Mean bias vs N (M={M_fixed})')
axes[1, 0].legend()
axes[1, 0].grid(True, which="both", alpha=0.3)

axes[1, 1].loglog(N_list, mu_s_var_N, '^-', label='sample')
axes[1, 1].loglog(N_list, mu_st_var_N, '^-', label='structure')
axes[1, 1].set_xlabel('N')
axes[1, 1].set_ylabel(r'Variance of $\hat{\mu}_X$')
axes[1, 1].set_title(f'Mean variance vs N (M={M_fixed})')
axes[1, 1].legend()
axes[1, 1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('mu_bias_variance_combined.jpeg')
plt.close()

# =========================
# Plot: Sigma Bias-Variance
# =========================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].loglog(M_list, sig_s_bias_M, '^-', label='sample')
axes[0, 0].loglog(M_list, sig_st_bias_M, '^-', label='structure')
axes[0, 0].set_xlabel('M')
axes[0, 0].set_ylabel(r'Bias of $\hat{\Sigma}_{x_T}$')
axes[0, 0].set_title(f'Covariance bias vs M (N={N_fixed})')
axes[0, 0].legend()
axes[0, 0].grid(True, which="both", alpha=0.3)

axes[0, 1].loglog(M_list, sig_s_var_M, '^-', label='sample')
axes[0, 1].loglog(M_list, sig_st_var_M, '^-', label='structure')
axes[0, 1].set_xlabel('M')
axes[0, 1].set_ylabel(r'Variance of $\hat{\Sigma}_{x_T}$')
axes[0, 1].set_title(f'Covariance variance vs M (N={N_fixed})')
axes[0, 1].legend()
axes[0, 1].grid(True, which="both", alpha=0.3)

axes[1, 0].loglog(N_list, sig_s_bias_N, '^-', label='sample')
axes[1, 0].loglog(N_list, sig_st_bias_N, '^-', label='structure')
axes[1, 0].set_xlabel('N')
axes[1, 0].set_ylabel(r'Bias of $\hat{\Sigma}_{x_T}$')
axes[1, 0].set_title(f'Covariance bias vs N (M={M_fixed})')
axes[1, 0].legend()
axes[1, 0].grid(True, which="both", alpha=0.3)

axes[1, 1].loglog(N_list, sig_s_var_N, '^-', label='sample')
axes[1, 1].loglog(N_list, sig_st_var_N, '^-', label='structure')
axes[1, 1].set_xlabel('N')
axes[1, 1].set_ylabel(r'Variance of $\hat{\Sigma}_{x_T}$')
axes[1, 1].set_title(f'Covariance variance vs N (M={M_fixed})')
axes[1, 1].legend()
axes[1, 1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('sigma_bias_variance_combined.jpeg')
plt.close()

# =========================
# Optimal cost vs dimension n
# =========================

n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]

T_fixed_dim = 5
m_fixed_dim = 2
M_fixed_dim = 60
N_cal = 30

sampleu_n = []
structu_n = []
originalu_n = []
cpu_n = []

for n_val in n_list:
    print(n_val)

    n = n_val
    m = m_fixed_dim
    t = T_fixed_dim

    A, B, CT = get_system(name, n, m, t, seed=0)

    x_0 = np.zeros(n)
    uf = np.random.randn(m*t, 1)
    xf = CT @ uf
    xf = xf.reshape(-1)

    Q = np.diag([2.0, 3.0])
    Q_tilde = block_diag(*([Q] * t))

    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

    sigmaw = np.eye(n*t) * 0.1
    sigmaz = NT @ sigmaw @ NT.T

    N = m*t + 3

    U = np.random.randn(m*t, N)
    U_cal = np.random.randn(m*t, N_cal)

    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M_fixed_dim))

    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M_fixed_dim):
            w = np.random.multivariate_normal(np.zeros(n*t), sigmaw)
            xT = CT @ U[:, i] + NT @ w
            all_xT[:, i, j] = xT
            sumx += xT
        X_bar[:, i] = sumx / M_fixed_dim

    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)

    sample_sigma = np.zeros((n, n))
    struct_sigma = np.zeros((n, n))

    for i in range(N):
        for j in range(M_fixed_dim):
            diff_sample = all_xT[:, i, j] - X_bar[:, i]
            diff_struct = all_xT[:, i, j] - CT_hat @ U[:, i]

            sample_sigma += np.outer(diff_sample, diff_sample)
            struct_sigma += np.outer(diff_struct, diff_struct)

    sample_sigma /= (N * M_fixed_dim)
    struct_sigma /= (N * M_fixed_dim)

    scores = get_scores(U_cal, N_cal, CT_hat, NT, CT, t)
    q = conformal_quantile(scores, delta)

    lambda_star = find_lambda_star(n, epsilon, delta)
    lambda_cp = (epsilon - q)**2

    alpha1 = cp.Variable(N)
    alpha2 = cp.Variable(N)
    alphak = cp.Variable(N)
    u3 = cp.Variable(m*t)

    objective1 = cp.Minimize(cp.quad_form(U @ alpha1, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(U @ alpha2, Q_tilde))
    objectivek = cp.Minimize(cp.quad_form(U @ alphak, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))

    f1 = cp.quad_form(X_bar @ alpha1 - xf, np.linalg.pinv(sample_sigma))
    f2 = cp.quad_form(CT_hat @ U @ alpha2 - xf, np.linalg.pinv(struct_sigma))
    fk = cp.quad_form(CT @ U @ alphak - xf, np.linalg.pinv(sigmaz))
    f3 = cp.quad_form(CT_hat @ u3 - xf, np.eye(n))

    prob1 = cp.Problem(objective1, [f1 <= lambda_star])
    prob2 = cp.Problem(objective2, [f2 <= lambda_star])
    probk = cp.Problem(objectivek, [fk <= lambda_star])
    prob3 = cp.Problem(objective3, [f3 <= lambda_cp])

    prob1.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver=cp.SCS, verbose=False)
    probk.solve(solver=cp.SCS, verbose=False)
    prob3.solve(solver=cp.SCS, verbose=False)

    if alpha1.value is None or alpha2.value is None or alphak.value is None or u3.value is None:
        print(f"Solver failed for n = {n}")
        sampleu_n.append(np.nan)
        structu_n.append(np.nan)
        originalu_n.append(np.nan)
        cpu_n.append(np.nan)
        continue

    u1 = U @ alpha1.value
    u2 = U @ alpha2.value
    uk = U @ alphak.value
    uc = u3.value

    sampleu_n.append(float(u1.T @ Q_tilde @ u1))
    structu_n.append(float(u2.T @ Q_tilde @ u2))
    originalu_n.append(float(uk.T @ Q_tilde @ uk))
    cpu_n.append(float(uc.T @ Q_tilde @ uc))


plt.figure(figsize=(10, 6))
plt.semilogy(n_list, sampleu_n, 'o-', label='sample based MLE')
plt.semilogy(n_list, structu_n, 'o-', label='structure MLE')
plt.semilogy(n_list, originalu_n, 'o--', color='black', label='known dynamics with data QCQP')
plt.semilogy(n_list, cpu_n, 'o-', label='conformal')
plt.xlabel('state dimension n')
plt.ylabel('optimal cost')
plt.title(f'Optimal cost vs dimension n, T={T_fixed_dim}, M={M_fixed_dim}')
plt.legend()
plt.grid(True)
plt.xticks(n_list)
plt.tight_layout()
plt.savefig('optimal_cost_vs_n.jpeg')