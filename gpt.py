import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Stable 2D system: compare Method 1 vs Method 2
#
# Method 1:
#   For each input u(i), estimate mu(i) by sample mean over M rollouts.
#   Estimate Sigma by pooling within-input centered samples.
#
# Method 2:
#   First estimate C_T from averaged data X_bar using least squares:
#       C_hat = X_bar U^T (U U^T)^(-1)
#   Then estimate mu = C_hat U.
#   Estimate Sigma from pooled residuals around C_hat U.
#
# Plots produced:
#   1) Mean error vs N
#   2) Covariance error vs N
#   3) C_T error vs N
#   4) Mean error vs M
#   5) Covariance error vs M
#   6) C_T error vs M
# ============================================================

np.random.seed(0)

# ----------------------------
# Parameters
# ----------------------------
n = 2
m = 2
T = 5
mT = m * T

sigma_w = 0.1
num_trials = 50

# Vary N with M fixed
N_list = [10, 20, 40, 80, 160, 320, 640]
M_fixed = 20

# Vary M with N fixed
M_list = [2, 5, 10, 20, 50, 100]
N_fixed = 80


# ----------------------------
# System generation
# ----------------------------
def get_stable_system(n, m, T):
    # Stable 2D system
    A = np.array([[0.85, 0.10],
                  [0.00, 0.75]])
    B = np.array([[1.0, 0.2],
                  [0.1, 1.0]])

    # T-step controllability matrix:
    # C_T = [A^{T-1}B, A^{T-2}B, ..., B]
    CT_blocks = [np.linalg.matrix_power(A, T - 1 - i) @ B for i in range(T)]
    C_T = np.hstack(CT_blocks)

    # Noise propagation matrix:
    # N_T = [A^{T-1}, A^{T-2}, ..., I]
    NT_blocks = [np.linalg.matrix_power(A, T - 1 - i) for i in range(T)]
    N_T = np.hstack(NT_blocks)

    return A, B, C_T, N_T


A, B, C_T, N_T = get_stable_system(n, m, T)

# True terminal covariance:
# x_T = C_T u + N_T w_stack
# with w_stack ~ N(0, sigma_w I_{nT})
Sigma_w_big = sigma_w * np.eye(n * T)
Sigma_true = N_T @ Sigma_w_big @ N_T.T


# ----------------------------
# Utility functions
# ----------------------------
def sample_full_row_rank_U(mT, N):
    while True:
        U = np.random.randn(mT, N)
        if np.linalg.matrix_rank(U) == mT:
            return U


def generate_rollout_samples(C_T, N_T, U, M, sigma_w):
    """
    Returns:
        X_all: shape (N, M, n)
               X_all[i, j, :] = j-th terminal sample for input u(i)
        mu_true: shape (n, N)
    """
    N = U.shape[1]
    n = C_T.shape[0]
    mu_true = C_T @ U

    X_all = np.zeros((N, M, n))
    for i in range(N):
        noise = N_T @ np.random.normal(0, np.sqrt(sigma_w), size=(n * T, M))
        samples = mu_true[:, [i]] + noise   # shape (n, M)
        X_all[i] = samples.T

    return X_all, mu_true


def method1_estimate(X_all):
    """
    Method 1:
      - estimate mu(i) independently by sample mean
      - estimate Sigma by pooled within-input covariance
    """
    N, M, n = X_all.shape

    # Mean estimate
    mu_hat = X_all.mean(axis=1).T   # shape (n, N)

    # Pooled covariance estimate
    Sigma_hat = np.zeros((n, n))
    for i in range(N):
        diff = X_all[i].T - mu_hat[:, [i]]   # shape (n, M)
        Sigma_hat += (diff @ diff.T) / ((M - 1) * N)

    return mu_hat, Sigma_hat


def method2_estimate(U, X_all):
    """
    Method 2:
      - average repeated samples first
      - estimate C_T by least squares
      - mu_hat = C_hat U
      - estimate Sigma from pooled residuals
    """
    N, M, n = X_all.shape
    mT = U.shape[0]

    # Averaged terminal data
    X_bar = X_all.mean(axis=1).T  # shape (n, N)

    # Least-squares estimate of C_T
    C_hat = X_bar @ U.T @ np.linalg.inv(U @ U.T)

    # Mean estimate
    mu_hat = C_hat @ U

    # Residual covariance
    residual_list = []
    for i in range(N):
        pred = mu_hat[:, [i]]
        diff = X_all[i].T - pred    # shape (n, M)
        residual_list.append(diff)

    R = np.hstack(residual_list)    # shape (n, N*M)

    # Unbiased-ish denominator for regression residual covariance
    dof = N * M - mT
    if dof <= 0:
        dof = N * M
    Sigma_hat = (R @ R.T) / dof

    return C_hat, mu_hat, Sigma_hat


def frob_error(A, B):
    return np.linalg.norm(A - B, ord='fro')


# ----------------------------
# Experiment 1: vary N, keep M fixed
# ----------------------------
err_mu_m1_vsN = []
err_mu_m2_vsN = []
err_sig_m1_vsN = []
err_sig_m2_vsN = []
err_C_vsN = []

for N in N_list:
    mu1_trials = []
    mu2_trials = []
    sig1_trials = []
    sig2_trials = []
    C_trials = []

    for _ in range(num_trials):
        U = sample_full_row_rank_U(mT, N)
        X_all, mu_true = generate_rollout_samples(C_T, N_T, U, M_fixed, sigma_w)

        mu1_hat, sig1_hat = method1_estimate(X_all)
        C2_hat, mu2_hat, sig2_hat = method2_estimate(U, X_all)

        mu1_trials.append(frob_error(mu1_hat, mu_true) / np.sqrt(N))
        mu2_trials.append(frob_error(mu2_hat, mu_true) / np.sqrt(N))
        sig1_trials.append(frob_error(sig1_hat, Sigma_true))
        sig2_trials.append(frob_error(sig2_hat, Sigma_true))
        C_trials.append(frob_error(C2_hat, C_T))

    err_mu_m1_vsN.append(np.mean(mu1_trials))
    err_mu_m2_vsN.append(np.mean(mu2_trials))
    err_sig_m1_vsN.append(np.mean(sig1_trials))
    err_sig_m2_vsN.append(np.mean(sig2_trials))
    err_C_vsN.append(np.mean(C_trials))


# ----------------------------
# Experiment 2: vary M, keep N fixed
# ----------------------------
err_mu_m1_vsM = []
err_mu_m2_vsM = []
err_sig_m1_vsM = []
err_sig_m2_vsM = []
err_C_vsM = []

for M in M_list:
    mu1_trials = []
    mu2_trials = []
    sig1_trials = []
    sig2_trials = []
    C_trials = []

    for _ in range(num_trials):
        U = sample_full_row_rank_U(mT, N_fixed)
        X_all, mu_true = generate_rollout_samples(C_T, N_T, U, M, sigma_w)

        mu1_hat, sig1_hat = method1_estimate(X_all)
        C2_hat, mu2_hat, sig2_hat = method2_estimate(U, X_all)

        mu1_trials.append(frob_error(mu1_hat, mu_true) / np.sqrt(N_fixed))
        mu2_trials.append(frob_error(mu2_hat, mu_true) / np.sqrt(N_fixed))
        sig1_trials.append(frob_error(sig1_hat, Sigma_true))
        sig2_trials.append(frob_error(sig2_hat, Sigma_true))
        C_trials.append(frob_error(C2_hat, C_T))

    err_mu_m1_vsM.append(np.mean(mu1_trials))
    err_mu_m2_vsM.append(np.mean(mu2_trials))
    err_sig_m1_vsM.append(np.mean(sig1_trials))
    err_sig_m2_vsM.append(np.mean(sig2_trials))
    err_C_vsM.append(np.mean(C_trials))


# ============================================================
# Plotting
# ============================================================

# 1) Mean error vs N
plt.figure(figsize=(7, 5))
plt.loglog(N_list, err_mu_m1_vsN, 'o-', label='Method 1: pointwise mean')
plt.loglog(N_list, err_mu_m2_vsN, 's-', label='Method 2: structural mean')
plt.xlabel('N (number of distinct inputs)')
plt.ylabel(r'$\|\hat{\mu}_X - \mu_X\|_F / \sqrt{N}$')
plt.title(f'Mean estimation error vs N (M={M_fixed})')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('mean_error_vs_N.png', dpi=300)
plt.show()

# 2) Covariance error vs N
plt.figure(figsize=(7, 5))
plt.loglog(N_list, err_sig_m1_vsN, 'o-', label='Method 1: pooled covariance')
plt.loglog(N_list, err_sig_m2_vsN, 's-', label='Method 2: residual covariance')
plt.xlabel('N (number of distinct inputs)')
plt.ylabel(r'$\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_F$')
plt.title(f'Covariance estimation error vs N (M={M_fixed})')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('covariance_error_vs_N.png', dpi=300)
plt.show()

# 3) C_T error vs N
plt.figure(figsize=(7, 5))
plt.loglog(N_list, err_C_vsN, '^-', label=r'Method 2: $\hat{C}_T$')
plt.xlabel('N (number of distinct inputs)')
plt.ylabel(r'$\|\hat{C}_T - C_T\|_F$')
plt.title(f'Direct $C_T$ estimation error vs N (M={M_fixed})')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('CT_error_vs_N.png', dpi=300)
plt.show()

# 4) Mean error vs M
plt.figure(figsize=(7, 5))
plt.loglog(M_list, err_mu_m1_vsM, 'o-', label='Method 1: pointwise mean')
plt.loglog(M_list, err_mu_m2_vsM, 's-', label='Method 2: structural mean')
plt.xlabel('M (number of rollouts per input)')
plt.ylabel(r'$\|\hat{\mu}_X - \mu_X\|_F / \sqrt{N}$')
plt.title(f'Mean estimation error vs M (N={N_fixed})')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('mean_error_vs_M.png', dpi=300)
plt.show()

# 5) Covariance error vs M
plt.figure(figsize=(7, 5))
plt.loglog(M_list, err_sig_m1_vsM, 'o-', label='Method 1: pooled covariance')
plt.loglog(M_list, err_sig_m2_vsM, 's-', label='Method 2: residual covariance')
plt.xlabel('M (number of rollouts per input)')
plt.ylabel(r'$\|\hat{\Sigma}_{x_T} - \Sigma_{x_T}\|_F$')
plt.title(f'Covariance estimation error vs M (N={N_fixed})')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('covariance_error_vs_M.png', dpi=300)
plt.show()

# 6) C_T error vs M
plt.figure(figsize=(7, 5))
plt.loglog(M_list, err_C_vsM, '^-', label=r'Method 2: $\hat{C}_T$')
plt.xlabel('M (number of rollouts per input)')
plt.ylabel(r'$\|\hat{C}_T - C_T\|_F$')
plt.title(f'Direct $C_T$ estimation error vs M (N={N_fixed})')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('CT_error_vs_M.png', dpi=300)
plt.show()

# 7) Optional: combined 2x3 panel for paper
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Row 1: vs N
axes[0, 0].loglog(N_list, err_mu_m1_vsN, 'o-', label='Method 1')
axes[0, 0].loglog(N_list, err_mu_m2_vsN, 's-', label='Method 2')
axes[0, 0].set_title(f'Mean error vs N (M={M_fixed})')
axes[0, 0].set_xlabel('N')
axes[0, 0].set_ylabel(r'$\|\hat{\mu}-\mu\|_F/\sqrt{N}$')
axes[0, 0].grid(True, which='both', alpha=0.3)
axes[0, 0].legend()

axes[0, 1].loglog(N_list, err_sig_m1_vsN, 'o-', label='Method 1')
axes[0, 1].loglog(N_list, err_sig_m2_vsN, 's-', label='Method 2')
axes[0, 1].set_title(f'Covariance error vs N (M={M_fixed})')
axes[0, 1].set_xlabel('N')
axes[0, 1].set_ylabel(r'$\|\hat{\Sigma}-\Sigma\|_F$')
axes[0, 1].grid(True, which='both', alpha=0.3)
axes[0, 1].legend()

axes[0, 2].loglog(N_list, err_C_vsN, '^-', label=r'$\hat{C}_T$')
axes[0, 2].set_title(f'$C_T$ error vs N (M={M_fixed})')
axes[0, 2].set_xlabel('N')
axes[0, 2].set_ylabel(r'$\|\hat{C}_T-C_T\|_F$')
axes[0, 2].grid(True, which='both', alpha=0.3)
axes[0, 2].legend()

# Row 2: vs M
axes[1, 0].loglog(M_list, err_mu_m1_vsM, 'o-', label='Method 1')
axes[1, 0].loglog(M_list, err_mu_m2_vsM, 's-', label='Method 2')
axes[1, 0].set_title(f'Mean error vs M (N={N_fixed})')
axes[1, 0].set_xlabel('M')
axes[1, 0].set_ylabel(r'$\|\hat{\mu}-\mu\|_F/\sqrt{N}$')
axes[1, 0].grid(True, which='both', alpha=0.3)
axes[1, 0].legend()

axes[1, 1].loglog(M_list, err_sig_m1_vsM, 'o-', label='Method 1')
axes[1, 1].loglog(M_list, err_sig_m2_vsM, 's-', label='Method 2')
axes[1, 1].set_title(f'Covariance error vs M (N={N_fixed})')
axes[1, 1].set_xlabel('M')
axes[1, 1].set_ylabel(r'$\|\hat{\Sigma}-\Sigma\|_F$')
axes[1, 1].grid(True, which='both', alpha=0.3)
axes[1, 1].legend()

axes[1, 2].loglog(M_list, err_C_vsM, '^-', label=r'$\hat{C}_T$')
axes[1, 2].set_title(f'$C_T$ error vs M (N={N_fixed})')
axes[1, 2].set_xlabel('M')
axes[1, 2].set_ylabel(r'$\|\hat{C}_T-C_T\|_F$')
axes[1, 2].grid(True, which='both', alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('all_comparison_plots.png', dpi=300)
plt.show()

print("Saved figures:")
print("  mean_error_vs_N.png")
print("  covariance_error_vs_N.png")
print("  CT_error_vs_N.png")
print("  mean_error_vs_M.png")
print("  covariance_error_vs_M.png")
print("  CT_error_vs_M.png")
print("  all_comparison_plots.png")