import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
import os

# Assuming datagen and other custom imports are in your path
from datagen import get_system 

# Parameters
n, m = 2, 2
x_0 = np.array([0.0, 0.0])
delta_conf = 0.9 # Confidence level for CP (1-beta)
beta = 0.1       # Risk for concentration bound
Q = np.diag([2.0, 3.0])
T = 10
name = 'stable'
gamma = 2.3
N = 21
M_list = range(100, 500, 50) # Increased step for speed

# Result containers
results = {k: [] for k in ['true_cost', 'data_cost', 'mle_cost', 'cp_cost', 
                           'true_dev', 'data_dev', 'mle_dev', 'cp_dev']}

for M in M_list:
    # 1. System Setup
    A, B, CT = get_system(name, n, m, T, seed=0)
    NT = np.eye(n)
    for i in range(1, T):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
    
    Q_tilde = block_diag(*([Q] * T))
    u_rand = np.random.randn(m * T, 1)
    x_f = np.linalg.matrix_power(A, T) @ x_0.reshape(-1, 1) + NT @ u_rand
    U = np.random.rand(m*T, N)
    
    # 2. Data Generation & Estimation
    # We collect M samples for one specific input to estimate Sigma
    # And N inputs (U) to estimate the mean matrix mu
    mu_matrix = []
    all_samples_for_sigma = []
    
    for i in range(N):
        samples = []
        for j in range(M):
            w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
            xT = CT @ U[:, i] + NT @ w
            samples.append(xT)
        mu_matrix.append(np.mean(samples, axis=0))
        if i == 0: all_samples_for_sigma = np.array(samples)

    mu_eff = np.array(mu_matrix).T # n x N
    Sigma = np.cov(all_samples_for_sigma.T)
    Sigma_reg = Sigma + 1e-6 * np.eye(n)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    
    # --- METHOD 1: CONCENTRATION DELTA ---
    # Simplified Bernstein-style delta based on your derivation
    # Delta scales with 1/sqrt(M)
    delta_val = 0.5 * np.sqrt(n * np.log(1/beta) / M)
    
    # --- METHOD 2: CONFORMAL PREDICTION ---
    # Use half of the samples from the first input as a calibration set
    cal_set = all_samples_for_sigma[M//2:]
    # Non-conformity scores: Mahalanobis distance to the empirical mean
    scores = np.array([ (s - mu_eff[:, 0]).T @ Sigma_inv @ (s - mu_eff[:, 0]) for s in cal_set])
    gamma_cp = np.quantile(scores, 1 - beta)

    # 3. Optimization Setup
    alpha = cp.Variable((N, 1))
    P = U.T @ Q_tilde @ U
    P = 0.5 * (P + P.T) + 1e-8 * np.eye(N)
    
    # Define constraints for 3 variants: Original, Concentration, and CP
    f_mahal = cp.quad_form(mu_eff @ alpha - x_f, Sigma_inv)
    
    # Problem solving helper
    def solve_alpha(rhs):
        prob = cp.Problem(cp.Minimize(cp.quad_form(alpha, P)), [f_mahal <= rhs])
        prob.solve(solver=cp.SCS)
        return alpha.value if prob.status == 'optimal' else None

    # Solve all cases
    a_data = solve_alpha(gamma)            # Standard Plug-in
    a_mle  = solve_alpha(gamma - delta_val)# Concentration Shrunk
    a_cp   = solve_alpha(gamma_cp)         # Conformal Adjusted

    # True System Case (Oracle)
    u_cvx = cp.Variable((m*T, 1))
    sigma_x_true = NT @ (np.eye(n*T)*0.1) @ NT.T
    f_true = cp.quad_form(CT @ u_cvx - x_f, np.linalg.inv(sigma_x_true))
    prob_true = cp.Problem(cp.Minimize(cp.quad_form(u_cvx, Q_tilde)), [f_true <= gamma])
    prob_true.solve()
    u_star = u_cvx.value

    # 4. Evaluation via Monte Carlo
    def get_metrics(sol_alpha, is_u_star=False):
        if sol_alpha is None: return np.nan, np.nan
        u_actual = sol_alpha if is_u_star else U @ sol_alpha
        costs = float(u_actual.T @ Q_tilde @ u_actual)
        
        # Test deviation over 100 noise realizations
        devs = []
        for _ in range(100):
            w = np.random.multivariate_normal(np.zeros(n*T), np.eye(n*T)*0.1)
            xT_real = CT @ u_actual.flatten() + NT @ w
            devs.append(np.linalg.norm(xT_real - x_f.flatten()))
        return costs, np.mean(devs)

    # Store metrics
    for label, sol, is_u in [('true', u_star, True), ('data', a_data, False), 
                             ('mle', a_mle, False), ('cp', a_cp, False)]:
        c, d = get_metrics(sol, is_u)
        results[f'{label}_cost'].append(c)
        results[f'{label}_dev'].append(d)

# 5. Final Plotting
plt.figure(figsize=(12, 5))

# Cost Comparison
plt.subplot(1, 2, 1)
plt.plot(M_list, results['true_cost'], 'k-', label='True (Oracle)')
plt.plot(M_list, results['data_cost'], 'g--', label='Standard Data')
plt.plot(M_list, results['mle_cost'], 'r-o', label='Concentration ($\Delta$)')
plt.plot(M_list, results['cp_cost'], 'b-s', label='Conformal')
plt.xlabel('Number of Samples (M)')
plt.ylabel('Control Cost')
plt.title('Optimal Energy Cost')
plt.legend()

# Deviation Comparison
plt.subplot(1, 2, 2)
plt.plot(M_list, results['true_dev'], 'k-', label='True (Oracle)')
plt.plot(M_list, results['data_dev'], 'g--', label='Standard Data')
plt.plot(M_list, results['mle_dev'], 'r-o', label='Concentration ($\Delta$)')
plt.plot(M_list, results['cp_dev'], 'b-s', label='Conformal')
plt.axhline(y=np.sqrt(gamma), color='gray', linestyle=':', label='Threshold')
plt.xlabel('Number of Samples (M)')
plt.ylabel('Avg Deviation $\|x_T - x_f\|$')
plt.title('State Deviation')
plt.legend()

plt.tight_layout()
plt.show()