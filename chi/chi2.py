
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
import os
from datagen import get_system
from scipy.stats import ncx2
from scipy.optimize import brentq

# ================= PARAMETERS =================
m = 2
x_0 = np.array([0.0, 0.0])

delta = 0.9
epsilon = 8.0
Q = np.diag([2.0, 3.0])
T = 10
name = 'stable'

N_list = range(2*T+1, 100, 3)
M = 50   # reduced for speed (you can increase later)

data_folder = 'data_{}'.format(name)
os.makedirs(data_folder, exist_ok=True)

n_list = range(2, 5, 1)

# ================= FUNCTION =================
def find_lambda_star(n, epsilon, delta):
    x = epsilon**2

    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - delta

    lam_lower = 0.0
    lam_upper = 1e4

    if equation(lam_lower) * equation(lam_upper) > 0:
        raise ValueError("Root not bracketed. Increase lam_upper.")

    return brentq(equation, lam_lower, lam_upper)

# ================= MAIN =================
plt.figure(figsize=(10, 6))

for n in n_list:

    errors_sample = []
    errors_ct = []

    sigma_0 = np.eye(n) * 0.1
    gamma = find_lambda_star(n, epsilon, delta)

    for N in N_list:

        # ===== SYSTEM =====
        A, B, CT = get_system(name, n, m, T, seed=0)

        NT = np.eye(n)
        for i in range(1, T):
            NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

        Q_tilde = block_diag(*([Q] * T))

        # ===== TARGET STATE =====
        w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
        x_f = (NT @ w).reshape(-1, 1)

        # ===== INPUTS =====
        U = np.random.randn(m*T, N)

        # ===== DATA GENERATION =====
        mu = np.zeros((n, 0))   # FIXED
        X_samples = np.zeros((M, n, N))

        for i in range(N):
            sum_x = np.zeros(n)

            for j in range(M):
                w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
                xT = CT @ U[:, i] + NT @ w

                X_samples[j, :, i] = xT
                sum_x += xT

            mu_i = (1/M) * sum_x
            mu = np.hstack((mu, mu_i.reshape(-1, 1)))

        # ===== TRUE MEAN =====
        mu_true = CT @ U

        # ===== METHOD 1: SAMPLE MEAN =====
        mu_sample = mu

        # ===== METHOD 2: CT-BASED =====
        UUT_inv = np.linalg.inv(U @ U.T + 1e-8*np.eye(m*T))
        C_hat = mu_sample @ U.T @ UUT_inv
        mu_ct = C_hat @ U

        # ===== ERRORS =====
        err_sample = np.linalg.norm(mu_sample - mu_true, 'fro')
        err_ct = np.linalg.norm(mu_ct - mu_true, 'fro')

        errors_sample.append(err_sample)
        errors_ct.append(err_ct)

    # ===== PLOT =====
    plt.semilogy(N_list, errors_sample, '--', label=f'Sample mean (n={n})')
    plt.semilogy(N_list, errors_ct, '-', label=f'CT-based (n={n})')

# ===== FINAL PLOT SETTINGS =====
plt.xlabel('Number of Inputs (N)')
plt.ylabel('Frobenius Error')
plt.title('Comparison: Sample Mean vs CT-based Estimation')
plt.legend()
plt.grid()
plt.savefig('comparison_mu_error.jpeg')
plt.show()