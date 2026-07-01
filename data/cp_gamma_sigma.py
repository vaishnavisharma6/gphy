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

n,m = 4,4
x_0 = np.array([1.0, 2.0, 3.0, 4.0])
xf = np.array([4.0, 5.0, 6.0, 7.0])
T_list = range(1, 25, 2)
sigma = 0.1
sigma_0 = np.eye(n) * sigma
delta = 0.3
Q = np.diag([2.0, 3.0, 4.0, 5.0])
name = 'stable'
epsilon = 2.4
M = 100

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

def build_system_matrices(t):
    A, B, CT = get_system(name, n, m, t, seed=0)

    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    sigmaw = np.eye(n*t) * sigma**2
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


T_list = [6, 8]

all_gamma = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_lambda = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_econf = {}
all_dev_known = {}
all_dev_eknown = {}
all_dev_estruct = {}

for T in T_list:

    A, B, CT = get_system(name, n, m, T, seed=0)
    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b = np.linalg.matrix_power(A, T) @ x_0 - xf

    sigma_w = np.eye(n*T) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T

    Q_tilde = block_diag(*([Q] * T))
    N = m*T +10
    N_cal = 60

    U = np.random.randn(m*T, N) # input matrix
    U_cal = np.random.randn(m*T, N_cal)
    #create bar{X}
    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))
    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = np.random.multivariate_normal(
                np.zeros(n*T),
                np.eye(n*T)*sigma**2)
            xT = CT @ U[:,i] + N_T@ w
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


    gamma_min = 0.0
    gamma_max = float(b.T @ np.linalg.inv(sigma_z) @ b)

    gamma_vals = np.linspace(gamma_min, gamma_max, 10)

    cost_struct = []
    cost_conf = []
    cost_known = []

    dev_struct = []
    dev_conf = []
    dev_econf = []
    dev_known = []
    dev_estruct = []
    dev_eknown = []

    for gamma in gamma_vals:

        u1 = cp.Variable(m*T) #structure
        u2 = cp.Variable(m*T) #conformal
        u3 = cp.Variable(m*T) #known dynamics

        objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
        objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
        objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
        
        constraint1 = (
            cp.quad_form(
                (CT_hat @ u1 -xf),
                np.linalg.inv(struct_sigma)
            ) <= gamma
        )
        constraint2 = (
            cp.quad_form(
                CT_hat @ u2 -xf,
                np.eye(n)
            ) <= gamma
        )
        constraint3 = (
            cp.quad_form(
                CT @ u3 -xf,
                np.linalg.inv(sigma_z)
            ) <= gamma
        )

        prob1 = cp.Problem(objective1, [constraint1])
        prob1.solve(solver=cp.SCS, verbose=False)

        prob2 = cp.Problem(objective2, [constraint2])
        prob2.solve(solver=cp.SCS, verbose=False)

        prob3 = cp.Problem(objective3, [constraint3])
        prob3.solve(solver=cp.SCS, verbose=False)

        Jstar1= float(u1.value.T @ Q_tilde @ u1.value)
        cost_struct.append(Jstar1)

        Jstar2 = float(u2.value.T @ Q_tilde @ u2.value)
        cost_conf.append(Jstar2)

        Jstar3 = float(u3.value.T @ Q_tilde @ u3.value)
        cost_known.append(Jstar3)

        idev_struct = []
        idev_conf = []
        idev_econf = []
        idev_estruct = []
        idev_eknown = []
        idev_known = []

        for _ in range(5000):
            w = np.random.multivariate_normal(
                np.zeros(n*T),
                np.eye(n*T)*sigma**2
            )

            xT1 = (
                np.linalg.matrix_power(A, T) @ x_0
                + CT @ u1.value
                + N_T @ w
            )
            
            xT2 = (
                np.linalg.matrix_power(A, T) @ x_0
                + CT @ u2.value
                + N_T @ w
            )

            xT3 = (
                np.linalg.matrix_power(A, T) @ x_0
                + CT @ u3.value
                + N_T @ w
            )

            e1 = xT1 - xf
            e2 = xT2 - xf
            e3 = xT3 - xf
            

            idev_struct.append(e1.T @ np.linalg.inv(sigma_z) @ e1)
            idev_conf.append(e2.T @ np.linalg.inv(sigma_z) @ e2)
            idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)
            idev_econf.append(e2.T @ e2)
            idev_estruct.append(e1.T @ e1)
            idev_eknown.append(e3.T @ e3)

        dev_struct.append(np.mean(idev_struct))
        dev_conf.append(np.mean(idev_conf))
        dev_econf.append(np.mean(idev_econf))
        dev_known.append(np.mean(idev_known))
        dev_estruct.append(np.mean(idev_estruct))
        dev_eknown.append(np.mean(idev_eknown))

    all_gamma[T] = gamma_vals
    all_cost_struct[T] = cost_struct
    all_cost_conf[T] = cost_conf
    all_cost_known[T] = cost_known

    all_dev_struct[T] = dev_struct
    all_dev_conf[T] = dev_conf
    all_dev_econf[T] = dev_econf
    all_dev_estruct[T] = dev_estruct
    all_dev_eknown[T] = dev_eknown
    all_dev_known[T] = dev_known

plt.figure(figsize=(10,6))

colors = ['tab:blue', 'tab:orange', 'tab:green']

for color, T in zip(colors, T_list):

    plt.plot(all_gamma[T], all_cost_struct[T],
             color=color, linestyle='-',
             label=f'Structure, T={T}')

    plt.plot(all_gamma[T], all_cost_conf[T],
             color=color, linestyle=':',
             label=f'Conformal, T={T}')

    plt.plot(all_gamma[T], all_cost_known[T],
             color=color, linestyle='--',
             label=f'Known, T={T}')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.legend()
plt.grid(True)
 

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$J^*(\gamma)$')
plt.title(r'Optimal Cost vs $\gamma$')
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
plt.ylabel(r'$\mu_e^\top \Sigma_e^{-1} \mu_e]$')
plt.title(r'Empirical deviation vs $\gamma$')
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
plt.ylabel(r'$\mu_e^\top \mu_e]$')
plt.title(r'Empirical deviation vs $\gamma$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('edeviation_vs_gamma_allT.jpeg')

#------------------------------
#cost and deviation vs sigma
#------------------------------
T = 8

all_sigma = {}
all_cost_struct = {}
all_cost_conf = {}
all_cost_known = {}
all_dev_struct = {}
all_dev_conf = {}
all_dev_known = {}
quant1 = []


A, B, CT = get_system(name, n, m, T, seed=0)

N_T = np.eye(n)
for i in range(1, T):
    N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

Q_tilde = block_diag(*([Q] * T))

N = m*T + 10
N_cal = 60

U = np.random.randn(m*T, N)
U_cal = np.random.randn(m*T, N_cal)

sigma_vals = np.linspace(0.1, 2, 20)

# ---------- initialize ONCE ----------
cost_struct = []
cost_conf = []
cost_known = []

dev_struct = []
dev_conf = []
dev_econf = []
dev_known = []

lambda_star = find_lambda_star(n, epsilon, 1-delta)

w_train = np.random.multivariate_normal(
    np.zeros(n*T),
    np.eye(n*T),
    size=(N, M)
)
# -------------------------------------
for sigma in sigma_vals:

    sigma_w = np.eye(n*T) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T

    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))

    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = sigma**2 * w_train[i, j]
            xT = CT @ U[:, i] + N_T @ w  

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
        N_T,
        CT,
        T,
        sigma       # after modifying get_scores
    )

    q = conformal_quantile(scores, delta)

    lambda_cp = (epsilon - q)**2

    u1 = cp.Variable(m*T)
    u2 = cp.Variable(m*T)
    u3 = cp.Variable(m*T)

    objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))

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

    cp.Problem(objective1, [constraint1]).solve(solver=cp.SCS)
    cp.Problem(objective2, [constraint2]).solve(solver=cp.SCS)
    cp.Problem(objective3, [constraint3]).solve(solver=cp.SCS)

    cost_struct.append(float(u1.value.T @ Q_tilde @ u1.value))
    cost_conf.append(float(u2.value.T @ Q_tilde @ u2.value))
    cost_known.append(float(u3.value.T @ Q_tilde @ u3.value))

    idev_struct = []
    idev_conf = []
    idev_econf = []
    idev_known = []

    for _ in range(5000):

        w = np.random.multivariate_normal(
            np.zeros(n*T),
            sigma_w
        )

        xT1 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u1.value + N_T @ w
        xT2 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u2.value + N_T @ w
        xT3 = np.linalg.matrix_power(A, T) @ x_0 + CT @ u3.value + N_T @ w

        e1 = xT1 - xf
        e2 = xT2 - xf
        e3 = xT3 - xf

        idev_struct.append(e1.T @ np.linalg.inv(sigma_z) @ e1)
        idev_conf.append(e2.T @ np.linalg.inv(sigma_z) @ e2)
        idev_econf.append(e2.T @ e2)
        idev_known.append(e3.T @ np.linalg.inv(sigma_z) @ e3)

    dev_struct.append(np.mean(idev_struct))
    dev_conf.append(np.mean(idev_conf))
    dev_econf.append(np.mean(idev_econf))
    dev_known.append(np.mean(idev_known))


plt.figure(figsize=(10,6))

plt.plot(sigma_vals, cost_struct,
         marker='o', linewidth=2,
         label='Structure')

plt.plot(sigma_vals, cost_conf,
         marker='s', linewidth=2,
         label='Conformal')

plt.plot(sigma_vals, cost_known,
         marker='^', linewidth=2,
         label='Known')

plt.xlabel(r'$\sigma$')
plt.ylabel(r'$J^*(\sigma)$')
plt.title(r'Optimal Cost vs $\sigma$ ($T=8$)')
plt.grid(True)
plt.legend()
plt.savefig('cost_vs_sigma_T8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))

plt.plot(sigma_vals, dev_struct,
         marker='o', linewidth=2,
         label='Structure')

plt.plot(sigma_vals, dev_conf,
         marker='s', linewidth=2,
         label='Conformal')

plt.plot(sigma_vals, dev_econf, '^--', linewidth = 2, label = r'Conformal $E[e^\top e]$')

plt.plot(sigma_vals, dev_known,
         marker='^', linewidth=2,
         linestyle='--',
         label='Known')

plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\mu_e^\top\Sigma_e^{-1}\mu_e]$')
plt.title(r'Empirical Deviation vs $\sigma$ ($T=8$)')
plt.grid(True)
plt.legend()
plt.savefig('deviation_vs_sigma_T8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#--------------------------------------------------------------
#conformal q_{1-\delta} vs T and empirical vs nominal coverage
#--------------------------------------------------------------
plt.figure(figsize=(10,6))

nominal = [1-d for d in delta_list]
T_list = range(1, 25, 8)

for t in T_list:
    A, B, CT = get_system(name, n, m, T, seed=0)
    N_T = np.eye(n)
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b = np.linalg.matrix_power(A, T) @ x_0 - xf

    sigma_w = np.eye(n*T) * sigma**2
    sigma_z = N_T @ sigma_w @ N_T.T

    Q_tilde = block_diag(*([Q] * T))
    N = m*T +10
    N_cal = 60
    quantiles[t] = []
    empirical_coverage[t] = {}
    U = np.random.randn(m*t, N) # input matrix
    U_cal = np.random.randn(m*t, N_cal)
    #create bar{X}
    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))
    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*sigma)
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

    for delta in delta_list:

        coverages = []
        q_runs = []

        for run in range(num_runs):

            np.random.seed(run)

            scores = get_scores(
                U_cal,
                N_cal,
                CT_hat,
                N_T,
                CT,
                t,
                sigma
            )

            q = conformal_quantile(scores, delta)

            q_runs.append(q)

            U_test = np.random.randn(m*t, N_test)

            scores_test = get_test_scores(
                U_test,
                CT_hat,
                N_T,
                CT,
                t,
                sigma
            )

            coverages.append(
                np.mean(scores_test <= q)
            )

        empirical_coverage[t][delta] = (
            np.mean(coverages),
            np.std(coverages)
    )

        if delta == 0.30:
            quant.append(np.mean(q_runs))

        quantiles[t].append(np.mean(q_runs))

    delta = 0.30

    scores = get_scores(
        U_cal,
        N_cal,
        CT_hat,
        N_T,
        CT,
        t,
        sigma
    )

    q = conformal_quantile(scores, delta)

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

plt.title("Empirical vs Nominal Coverage")

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

plt.xlabel("Prediction Horizon T")

plt.ylabel(r"$q_{1-\delta}$")

plt.title(r"Scaling of $q_{1-\delta}$ with Horizon")

plt.grid(True)

plt.savefig("quantile_vs_T.jpeg")