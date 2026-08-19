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


# ============================================================
# Parameters
# ============================================================

n, m = 4, 4
T = 8

name = "stable"

xf = np.array([4.0,5.0,6.0,7.0])

sigma_1 = 0.2
sigma_2 = 0.3

sigma_0 = sigma_1**2*np.eye(n)

Q = np.diag([2,3,4,5])

epsilon = 2.4
delta = 0.5

N_list = [20,40,80,120,200]
M_list = [5,10,20,50,100]

K_runs = 30



# ============================================================
# System matrices
# ============================================================

A,B,CT = get_system(name,n,m,T,seed=0)

Q_tilde = block_diag(*([Q]*T))


sigma = sigma_2**2*np.eye(n)

sigmaw = np.kron(
    np.eye(T),
    sigma
)


NT = np.eye(n)

for i in range(1,T):
    NT = np.hstack(
        (NT,np.linalg.matrix_power(A,i))
    )


sigmaz = (
    NT@sigmaw@NT.T
    +
    np.linalg.matrix_power(A,T)
    @sigma_0
    @np.linalg.matrix_power(A,T).T
)



# ============================================================
# Replace with your function
# ============================================================

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



# ============================================================
# Known dynamics optimal cost
# ============================================================

lambda_star = find_lambda_star(
    n,
    epsilon,
    1-delta
)


u_star = cp.Variable(m*T)


problem = cp.Problem(
    cp.Minimize(
        cp.quad_form(u_star,Q_tilde)
    ),
    [
        cp.quad_form(
            CT@u_star-xf,
            np.linalg.inv(sigmaz)
        )
        <= lambda_star
    ]
)


problem.solve(
    solver=cp.SCS
)


J_known = float(
    u_star.value.T
    @Q_tilde
    @u_star.value
)


print("Known optimal cost:",J_known)



# ============================================================
# Data driven cost
# ============================================================

def data_driven_cost(N,M):

    costs=[]


    for run in range(K_runs):

        rng=np.random.default_rng(run)


        U=rng.standard_normal(
            (m*T,N)
        )


        X_bar=np.zeros((n,N))

        all_xT=np.zeros(
            (n,N,M)
        )


        for i in range(N):

            samples=[]

            for j in range(M):

                x0=rng.multivariate_normal(
                    np.zeros(n),
                    sigma_0
                )

                w=rng.multivariate_normal(
                    np.zeros(n*T),
                    sigmaw
                )


                xT = (
                    np.linalg.matrix_power(A,T)
                    @x0
                    +
                    CT@U[:,i]
                    +
                    NT@w
                )


                all_xT[:,i,j]=xT

                samples.append(xT)


            X_bar[:,i]=np.mean(
                samples,
                axis=0
            )



        # Estimate CT

        CT_hat = (
            X_bar
            @U.T
            @np.linalg.pinv(U@U.T)
        )


        # Estimate covariance

        Sigma_hat=np.zeros(
            (n,n)
        )


        for i in range(N):

            for j in range(M):

                e = (
                    all_xT[:,i,j]
                    -
                    CT_hat@U[:,i]
                )

                Sigma_hat += np.outer(
                    e,e
                )


        Sigma_hat /= N*M



        # Optimization

        u=cp.Variable(m*T)


        prob=cp.Problem(
            cp.Minimize(
                cp.quad_form(u,Q_tilde)
            ),
            [
                cp.quad_form(
                    CT_hat@u-xf,
                    np.linalg.inv(Sigma_hat)
                )
                <= lambda_star
            ]
        )


        prob.solve(
            solver=cp.SCS,
            verbose=False
        )


        if u.value is not None:

            J=float(
                u.value.T
                @Q_tilde
                @u.value
            )

            costs.append(J)



    return np.mean(costs)



# ============================================================
# Error vs N
# ============================================================

M_fixed=100

error_N=[]


for N in N_list:

    print("Running N =",N)

    J=data_driven_cost(
        N,
        M_fixed
    )

    error_N.append(
        (J-J_known)/J_known
    )



plt.figure(figsize=(7,5))

plt.plot(
    N_list,
    error_N,
    'o-'
)

plt.xlabel("$N$")
plt.ylabel(
    r"$\frac{J^*_{data}-J^*_{known}}{J^*_{known}}$"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "cost_error_vs_N.png",
    dpi=300
)



# ============================================================
# Error vs M
# ============================================================

N_fixed=100

error_M=[]


for M in M_list:

    print("Running M =",M)

    J=data_driven_cost(
        N_fixed,
        M
    )


    error_M.append(
        (J-J_known)/J_known
    )



plt.figure(figsize=(7,5))

plt.plot(
    M_list,
    error_M,
    'o-'
)

plt.xlabel("$M$")
plt.ylabel(
    r"$\frac{J^*_{data}-J^*_{known}}{J^*_{known}}$"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "cost_error_vs_M.png",
    dpi=300
)