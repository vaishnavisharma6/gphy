"""
Comparison of Data-Driven Minimum Energy Control Methods
=========================================================
Method 1: MLE of mu_X and Sigma_{x_T}
Method 2: Direct Estimation of C_T via MLE
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import solve

np.random.seed(42)

# ─────────────────────────────────────────────
# System Parameters
# ─────────────────────────────────────────────
n = 3       # state dimension
m = 2       # input dimension
T = 2       # horizon
mT = m * T  # total input dimension

# True system matrices (unknown to the data-driven methods)
A = np.array([[0.9, 0.1, 0.0],
              [0.0, 0.8, 0.1],
              [0.0, 0.0, 0.7]])

B = np.array([[1.0, 0.0],
              [0.5, 1.0],
              [0.0, 0.5]])

# Process noise covariance
Sigma_w = 0.1 * np.eye(n)

# Cost weight matrix Q (on the input u)
Q = np.eye(mT)

# Target terminal state
x_f = np.array([1.0, 0.5, -0.5])

# Constraint parameter
gamma = 5.0

# Number of data columns (experiments) N and samples per experiment M
N_data = 50   # number of distinct input sequences
M_samples = 30  # samples per input for Method 1

# ─────────────────────────────────────────────
# Compute True C_T and N_T (for reference)
# ─────────────────────────────────────────────
def compute_CT_NT(A, B, T):
    """
    For x_T = C_T u + N_T w  (x_0 = 0),
    C_T = [A^{T-1}B | A^{T-2}B | ... | B]  (controllability-like)
    N_T = [A^{T-1} | A^{T-2} | ... | I]
    """
    # C_T: n x mT
    CT_cols = []
    for k in range(T):
        Ak = np.linalg.matrix_power(A, T - 1 - k)
        CT_cols.append(Ak @ B)
    CT = np.hstack(CT_cols)  # n x mT

    # N_T: n x nT  (noise enters at each step)
    NT_cols = []
    for k in range(T):
        Ak = np.linalg.matrix_power(A, T - 1 - k)
        NT_cols.append(Ak)
    NT = np.hstack(NT_cols)  # n x nT

    return CT, NT

CT_true, NT_true = compute_CT_NT(A, B, T)
Sigma_xT_true = NT_true @ np.kron(np.eye(T), Sigma_w) @ NT_true.T

# ─────────────────────────────────────────────
# Generate Data Matrices U and X
# ─────────────────────────────────────────────
# U: mT x N_data  (random input sequences, full row rank)
U = np.random.randn(mT, N_data)
# Ensure full row rank by augmenting
U[:, :mT] = np.eye(mT) + 0.1 * np.random.randn(mT, mT)

def simulate_xT(CT, NT, u_col, Sigma_w, T, n, M=1):
    """
    Simulate terminal states for input u_col (x_0=0).
    Returns shape (n, M).
    """
    samples = []
    for _ in range(M):
        w = np.random.multivariate_normal(np.zeros(n * T),
                                          np.kron(np.eye(T), Sigma_w))
        x = CT @ u_col + NT @ w
        samples.append(x)
    return np.array(samples).T  # (n, M)

# X_mean: n x N_data  (one sample per column for Method 2)
X_single = np.column_stack([
    simulate_xT(CT_true, NT_true, U[:, i], Sigma_w, T, n, M=1)[:, 0]
    for i in range(N_data)
])

# X_multi: list of (n, M) arrays, one per column of U (for Method 1)
X_multi = [
    simulate_xT(CT_true, NT_true, U[:, i], Sigma_w, T, n, M=M_samples)
    for i in range(N_data)
]

# ─────────────────────────────────────────────
# QCQP Solver (shared by both methods)
# ─────────────────────────────────────────────
def solve_qcqp(mu_X, Sigma_xT, U, Q, x_f, gamma, nu_grid=None):
    """
    Solve:
      min_{alpha in R^N}  (U alpha)^T Q (U alpha)
      s.t.  (mu_X alpha - x_f)^T Sigma_xT^{-1} (mu_X alpha - x_f) <= gamma

    Since U is mT x N with mT << N, we reparametrize u = U alpha and note
    the minimum-norm alpha = U^+ u, so effectively minimize over u in range(U).
    Using u = U U^+ u, the problem becomes (setting v = U alpha, v in col-span U):

      min_u  u^T Q u   s.t.  (mu_X U^+ u - x_f)^T Sigma^{-1}(...) <= gamma

    Equivalently, let F = mu_X U^+  (n x mT after pseudoinverse):
      (F u - x_f)^T Sigma^{-1} (F u - x_f) <= gamma

    KKT:  (Q + nu F^T Sigma^{-1} F) u = nu F^T Sigma^{-1} x_f
    Then alpha = U^+ u_opt  (minimum norm representative).
    """
    Upinv = np.linalg.pinv(U)          # N x mT  (pseudoinverse)
    F = mu_X @ Upinv                   # n x mT
    Sigma_inv = np.linalg.inv(Sigma_xT)
    FtSF = F.T @ Sigma_inv @ F         # mT x mT
    FtSxf = F.T @ Sigma_inv @ x_f     # mT

    if nu_grid is None:
        nu_grid = np.logspace(-6, 6, 100000)

    best_u = None
    best_nu = None
    best_violation = np.inf

    for nu in nu_grid:
        mat = Q + nu * FtSF
        try:
            u = np.linalg.solve(mat, nu * FtSxf)
        except np.linalg.LinAlgError:
            continue
        residual = F @ u - x_f
        constr_val = residual @ Sigma_inv @ residual
        violation = abs(constr_val - gamma)
        if violation < best_violation:
            best_violation = violation
            best_u = u
            best_nu = nu

    alpha_opt = Upinv @ best_u
    residual = F @ best_u - x_f
    constr_val = residual @ Sigma_inv @ residual
    return alpha_opt, best_u, best_nu, constr_val

# ─────────────────────────────────────────────
# Method 1: MLE of mu_X(i) and Sigma_{x_T}
# ─────────────────────────────────────────────
def method1(U, X_multi, N_data, n):
    # Estimate mu_X(i): sample mean per column
    mu_X_hat = np.column_stack([X_multi[i].mean(axis=1) for i in range(N_data)])  # n x N_data

    # Pooled MLE covariance: (1/(N*M)) sum_i sum_j (x_{T,j}(i) - mu_hat(i))(...)^T
    M = X_multi[0].shape[1]
    Sigma_hat = np.zeros((n, n))
    for i in range(N_data):
        diff = X_multi[i] - mu_X_hat[:, i:i+1]   # n x M
        Sigma_hat += diff @ diff.T
    Sigma_hat /= (N_data * M)

    return mu_X_hat, Sigma_hat

mu_X_hat, Sigma_hat_m1 = method1(U, X_multi, N_data, n)

alpha_m1, u_m1, nu_m1, cv_m1 = solve_qcqp(
    mu_X_hat, Sigma_hat_m1, U, Q, x_f, gamma
)

# Predicted terminal state mean under Method 1 solution
mu_xT_m1 = mu_X_hat @ np.linalg.pinv(U) @ u_m1

# ─────────────────────────────────────────────
# Method 2: Direct MLE estimation of C_T
# ─────────────────────────────────────────────
def method2(U, X_single, n, mT):
    # CT_hat = X U^T (U U^T)^{-1}
    CT_hat = X_single @ U.T @ np.linalg.inv(U @ U.T)  # n x mT

    # Sigma_xT: residual covariance
    residuals = X_single - CT_hat @ U   # n x N_data
    Sigma_hat = (residuals @ residuals.T) / X_single.shape[1]

    return CT_hat, Sigma_hat

CT_hat, Sigma_hat_m2 = method2(U, X_single, n, mT)

# mu_X under Method 2 is CT_hat U
mu_X_m2 = CT_hat @ U   # n x N_data

alpha_m2, u_m2, nu_m2, cv_m2 = solve_qcqp(
    mu_X_m2, Sigma_hat_m2, U, Q, x_f, gamma
)

mu_xT_m2 = mu_X_m2 @ np.linalg.pinv(U) @ u_m2

# ─────────────────────────────────────────────
# True Optimal (model-based, for reference)
# ─────────────────────────────────────────────
alpha_true, u_true, nu_true, cv_true = solve_qcqp(
    CT_true @ U, Sigma_xT_true, U, Q, x_f, gamma
)
mu_xT_true = CT_true @ u_true

# ─────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────
def compute_metrics(u_opt, u_ref, mu_xT, x_f, CT, Sigma_xT, gamma):
    energy = u_opt @ u_opt
    energy_ref = u_ref @ u_ref
    energy_ratio = energy / energy_ref if energy_ref > 0 else float('inf')

    # True constraint value at the solution
    mu_e = CT @ u_opt - x_f
    Sigma_inv = np.linalg.inv(Sigma_xT)
    true_constr = mu_e @ Sigma_inv @ mu_e

    # Terminal state error
    terminal_err = np.linalg.norm(mu_xT - x_f)
    return {
        'energy': energy,
        'energy_ratio': energy_ratio,
        'true_constraint': true_constr,
        'terminal_error': terminal_err,
    }

metrics_m1 = compute_metrics(u_m1, u_true, mu_xT_m1, x_f, CT_true, Sigma_xT_true, gamma)
metrics_m2 = compute_metrics(u_m2, u_true, mu_xT_m2, x_f, CT_true, Sigma_xT_true, gamma)
metrics_ref = compute_metrics(u_true, u_true, mu_xT_true, x_f, CT_true, Sigma_xT_true, gamma)

# CT estimation error
CT_err_m1 = np.linalg.norm(mu_X_hat - CT_true @ U, 'fro') / np.linalg.norm(CT_true @ U, 'fro')
CT_err_m2 = np.linalg.norm(CT_hat @ U - CT_true @ U, 'fro') / np.linalg.norm(CT_true @ U, 'fro')
Sigma_err_m1 = np.linalg.norm(Sigma_hat_m1 - Sigma_xT_true, 'fro') / np.linalg.norm(Sigma_xT_true, 'fro')
Sigma_err_m2 = np.linalg.norm(Sigma_hat_m2 - Sigma_xT_true, 'fro') / np.linalg.norm(Sigma_xT_true, 'fro')

# ─────────────────────────────────────────────
# Sample-Complexity Analysis
# ─────────────────────────────────────────────
M_range = [5, 10, 20, 50, 100]
N_range = [10, 20, 30, 50, 80, 100]

energy_m1_vs_M = []
energy_m2_vs_N = []

for M in M_range:
    X_m = [simulate_xT(CT_true, NT_true, U[:, i], Sigma_w, T, n, M=M)
           for i in range(N_data)]
    mu_X_m, Sigma_m = method1(U, X_m, N_data, n)
    _, u_m, _, _ = solve_qcqp(mu_X_m, Sigma_m, U, Q, x_f, gamma)
    energy_m1_vs_M.append(u_m @ u_m)

for N in N_range:
    U_n = np.random.randn(mT, N)
    # Ensure full row rank: embed identity in first mT columns (if N >= mT)
    if N >= mT:
        U_n[:, :mT] = np.eye(mT) + 0.1 * np.random.randn(mT, mT)
    X_n = np.column_stack([
        simulate_xT(CT_true, NT_true, U_n[:, i], Sigma_w, T, n, M=1)[:, 0]
        for i in range(N)
    ])
    CT_n, Sigma_n = method2(U_n, X_n, n, mT)
    mu_X_n = CT_n @ U_n
    _, u_n, _, _ = solve_qcqp(mu_X_n, Sigma_n, U_n, Q, x_f, gamma)
    energy_m2_vs_N.append(u_n @ u_n)

ref_energy = metrics_ref['energy']

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
})

fig = plt.figure(figsize=(15, 11))
fig.suptitle("Data-Driven Minimum Energy Control: Method 1 vs Method 2",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel 1: Optimal control sequences comparison ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
t_idx = np.arange(mT)
w = 0.25
ax1.bar(t_idx - w, u_true, width=w, label='True Optimal', color='#2c5f8a', alpha=0.85)
ax1.bar(t_idx,     u_m1,   width=w, label='Method 1 (MLE μ,Σ)', color='#e07b39', alpha=0.85)
ax1.bar(t_idx + w, u_m2,   width=w, label='Method 2 (MLE $C_T$)', color='#3a9e6f', alpha=0.85)
ax1.set_xticks(t_idx)
ax1.set_xticklabels([f'$u_{{{i+1}}}$' for i in range(mT)])
ax1.set_title("Optimal control sequence $u^*$")
ax1.set_ylabel("Input value")
ax1.legend(fontsize=9)
ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')

# ── Panel 2: Summary Metrics Table ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
rows = ['Energy $u^\\top u$',
        'Energy ratio\n(vs true)',
        'True constraint\n$\\mu_e^\\top\\Sigma^{-1}\\mu_e$',
        'Terminal error\n$\\|\\hat{\\mu}_{x_T}-x_f\\|$']
col1 = [f"{metrics_ref['energy']:.4f}", "1.000",
        f"{metrics_ref['true_constraint']:.4f}", f"{metrics_ref['terminal_error']:.4f}"]
col2 = [f"{metrics_m1['energy']:.4f}", f"{metrics_m1['energy_ratio']:.4f}",
        f"{metrics_m1['true_constraint']:.4f}", f"{metrics_m1['terminal_error']:.4f}"]
col3 = [f"{metrics_m2['energy']:.4f}", f"{metrics_m2['energy_ratio']:.4f}",
        f"{metrics_m2['true_constraint']:.4f}", f"{metrics_m2['terminal_error']:.4f}"]
table = ax2.table(
    cellText=list(zip(col1, col2, col3)),
    rowLabels=rows,
    colLabels=['True', 'M1', 'M2'],
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (r, c), cell in table.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor('#dde8f0')
        cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('#bbbbbb')
ax2.set_title("Performance metrics", pad=12, fontsize=11)

# ── Panel 3: Sigma estimation error ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
methods_lbl = ['Method 1', 'Method 2']
sigma_errs = [Sigma_err_m1, Sigma_err_m2]
colors_bar = ['#e07b39', '#3a9e6f']
bars = ax3.bar(methods_lbl, sigma_errs, color=colors_bar, alpha=0.85, width=0.5)
for bar, val in zip(bars, sigma_errs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax3.set_title("$\\hat{\\Sigma}_{x_T}$ relative error (Frobenius)")
ax3.set_ylabel("Relative error")

# ── Panel 4: mu_X / C_T U estimation error ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ct_errs = [CT_err_m1, CT_err_m2]
bars2 = ax4.bar(methods_lbl, ct_errs, color=colors_bar, alpha=0.85, width=0.5)
for bar, val in zip(bars2, ct_errs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax4.set_title("$\\hat{\\mu}_X$ (or $\\hat{C}_T U$) relative error")
ax4.set_ylabel("Relative error")

# ── Panel 5: True vs predicted terminal state ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
dims = np.arange(n)
ax5.plot(dims, x_f, 'k--o', label='$x_f$ (target)', linewidth=1.5, markersize=6)
ax5.plot(dims, mu_xT_true, 's-', color='#2c5f8a', label='True optimal', linewidth=1.5, markersize=6)
ax5.plot(dims, mu_xT_m1, '^-', color='#e07b39', label='Method 1', linewidth=1.5, markersize=6)
ax5.plot(dims, mu_xT_m2, 'D-', color='#3a9e6f', label='Method 2', linewidth=1.5, markersize=6)
ax5.set_xticks(dims)
ax5.set_xticklabels([f'$x_{{{i+1}}}$' for i in range(n)])
ax5.set_title("Predicted terminal state $\\hat{\\mu}_{x_T}$")
ax5.legend(fontsize=8)

# ── Panel 6: Method 1 energy vs number of samples M ──────────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
ax6.plot(M_range, energy_m1_vs_M, 'o-', color='#e07b39',
         label='Method 1 energy', linewidth=2, markersize=7)
ax6.axhline(ref_energy, color='#2c5f8a', linestyle='--', linewidth=1.5,
            label=f'True optimal energy ({ref_energy:.3f})')
ax6.set_xlabel("Samples per experiment $M$")
ax6.set_ylabel("Control energy $u^{\\top}u$")
ax6.set_title("Method 1: Energy vs. samples per experiment $M$")
ax6.legend(fontsize=9)
ax6.set_xscale('log')

# ── Panel 7: Method 2 energy vs N (number of data columns) ───────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(N_range, energy_m2_vs_N, 's-', color='#3a9e6f',
         label='Method 2 energy', linewidth=2, markersize=7)
ax7.axhline(ref_energy, color='#2c5f8a', linestyle='--', linewidth=1.5,
            label='True optimal')
ax7.set_xlabel("Number of experiments $N$")
ax7.set_title("Method 2: Energy vs. $N$")
ax7.legend(fontsize=8)

plt.savefig("/mnt/user-data/outputs/comparison_plot.png",
            bbox_inches='tight', dpi=130)
plt.close()

# ─────────────────────────────────────────────
# Console Summary
# ─────────────────────────────────────────────
sep = "─" * 60
print(sep)
print("  Data-Driven Minimum Energy Control: Method Comparison")
print(sep)
print(f"\n  System:  n={n}, m={m}, T={T}, mT={mT}")
print(f"  Data:    N={N_data} experiments, M={M_samples} samples/exp (Method 1)")
print(f"  gamma   = {gamma}   (constraint threshold)")
print()
print(f"  {'Metric':<36} {'True':>10} {'Method 1':>10} {'Method 2':>10}")
print(f"  {'-'*36} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'Control energy  u^T u':<36} {metrics_ref['energy']:>10.4f} {metrics_m1['energy']:>10.4f} {metrics_m2['energy']:>10.4f}")
print(f"  {'Energy ratio  (vs true)':<36} {'1.0000':>10} {metrics_m1['energy_ratio']:>10.4f} {metrics_m2['energy_ratio']:>10.4f}")
print(f"  {'True constraint value':<36} {metrics_ref['true_constraint']:>10.4f} {metrics_m1['true_constraint']:>10.4f} {metrics_m2['true_constraint']:>10.4f}")
print(f"  {'Terminal state error  ||mu-xf||':<36} {metrics_ref['terminal_error']:>10.4f} {metrics_m1['terminal_error']:>10.4f} {metrics_m2['terminal_error']:>10.4f}")
print()
print(f"  {'mu_X / C_T U relative error':<36} {'N/A':>10} {CT_err_m1:>10.4f} {CT_err_m2:>10.4f}")
print(f"  {'Sigma relative error':<36} {'N/A':>10} {Sigma_err_m1:>10.4f} {Sigma_err_m2:>10.4f}")
print()
print(f"  Optimal nu*:   True={nu_true:.4f}   M1={nu_m1:.4f}   M2={nu_m2:.4f}")
print(sep)
print("\n  Saved plot to: comparison_plot.png")
print(sep)