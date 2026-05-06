import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.stats import ncx2
from scipy.optimize import brentq
import os
from datagen import get_system

# ------------------ PARAMETERS ------------------
m = 2
delta = 0.9
epsilon = 4.0
Q = np.diag([2.0, 3.0])
T = 10

N_list = range(2*T + 1, 100, 3)
n_list = range(2, 5)

M = 1000
name = 'stable'

data_folder = f'data_{name}'
os.makedirs(data_folder, exist_ok=True)

# ------------------ FUNCTION ------------------
def find_lambda_star(n, epsilon, delta):
    x = epsilon**2

    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - delta

    return brentq(equation, 0.0, 1e4)

# ------------------ PLOTTING ------------------
plt.figure(figsize=(10, 6))

for n in n_list:
    print(f"Running for n = {n}")

    gamma = find_lambda_star(n, epsilon, delta)

    errors = []

    sigma_w_base = np.eye(n*T) * 0.1
    sigma_0 = np.eye(n) * 0.1

    for N in N_list:

        # ----- System -----
        A, B, CT = get_system(name, n, m, T, seed=0)

        NT = np.eye(n)
        for i in range(1, T):
            NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

        Q_tilde = block_diag(*([Q] * T))

        # ----- Random target -----
        w = np.random.multivariate_normal(np.zeros(n*T), sigma_w_base)
        x_f = (NT @ w).reshape(-1, 1)

        # ----- Basis -----
        U = np.random.randn(m*T, N)

        # ----- Estimate mean -----
        mu = np.empty((n, 0))

        for i in range(N):
            samples = []

            for _ in range(M):
                w = np.random.multivariate_normal(np.zeros(n*T), sigma_w_base)
                xT = CT @ U[:, i] + NT @ w
                samples.append(xT)

            samples = np.array(samples)              # (M, n)
            mu_i = np.mean(samples, axis=0)          # (n,)
            mu = np.hstack((mu, mu_i.reshape(-1, 1)))

            if i == N - 1:
                X_last = samples                     # store for covariance

        # ----- Covariance -----
        Sigma = np.cov(X_last.T)
        Sigma += 1e-6 * np.eye(n)
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)

        # ----- True covariance -----
        sigma_x = NT @ sigma_w_base @ NT.T
        sigma_x_inv = np.linalg.inv(sigma_x)

        # ----- CVXPY -----
        alpha = cp.Variable((N, 1))
        u = cp.Variable((m*T, 1))

        P = U.T @ Q_tilde @ U
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(N)

        objective = cp.Minimize(cp.quad_form(alpha, P))
        objective_true = cp.Minimize(cp.quad_form(u, Q_tilde))

        mu_eff = mu
        x_f_col = x_f

        f = cp.quad_form(mu_eff @ alpha - x_f_col, Sigma_inv)
        f_true = cp.quad_form(CT @ u - x_f_col, sigma_x_inv)

        prob = cp.Problem(objective, [f <= gamma])
        prob_true = cp.Problem(objective_true, [f_true <= gamma])

        prob.solve(solver=cp.SCS, verbose=False)
        prob_true.solve(solver=cp.SCS, verbose=False)

        alpha_hat = alpha.value
        u_star = u.value

        if alpha_hat is None or u_star is None:
            print(f"Solver failed at n={n}, N={N}")
            continue

        # ----- Error -----
        err = np.linalg.norm(U @ alpha_hat - u_star)
        errors.append(err)

    # ----- Plot -----
    plt.plot(list(N_list)[:len(errors)], errors, label=f'n = {n}')

# ------------------ FINALIZE PLOT ------------------
plt.xlabel('N')
plt.ylabel(r'$\|U\hat{\alpha} - u^*\|$')
plt.title('Error vs N for different dimensions')
plt.legend()
plt.grid(True)

plt.savefig('error_vs_N.jpeg')
plt.show()