import os
import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.stats import ncx2
from scipy.optimize import brentq
from datagen import get_system
# ============================================================
# IEEE / OVERLEAF PUBLICATION SETTINGS
# ============================================================
mpl.rcParams.update({
    "text.usetex": False,
    # IEEE-compatible serif text + Times-like math
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    # IEEE-friendly sizes
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 6.4,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    # Thin publication-style lines
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "lines.linewidth": 0.65,
    "axes.linewidth": 0.65,
    # Compact ticks
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    # Keep vector text editable/searchable in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # High-quality raster fallback
    "savefig.dpi": 600,
})
def style_ieee_axis(ax):
    """Apply the same compact IEEE style used in the earlier figures."""
    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.45,
        alpha=0.45,
    )
    ax.tick_params(direction="in", top=True, right=True, pad=2)
    ax.margins(x=0.02)
def save_ieee_figure(fig, stem):
    """Save vector PDF and high-resolution PNG."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
# ============================================================
# GLOBAL PARAMETERS
# ============================================================
n, m = 4, 4
xf = np.array([4.0, 5.0, 6.0, 7.0])
T_cost_list = list(range(1, 20, 2))
T_gamma_list = [6]
sigma_1 = 0.2
sigma_2 = 0.5
sigma_0 = (sigma_1 ** 2) * np.eye(n)
Q = np.diag([2.0, 3.0, 4.0, 5.0])
name = "stable"
epsilon = 2.4
delta = 0.50
M = 100
k_est = 10
N_cal = 60
# Number of gamma samples for right/left panel sweep
num_gamma = 10
# Output directory
OUTDIR = "ieee_cost_plots"
os.makedirs(OUTDIR, exist_ok=True)
# ============================================================
# HELPERS
# ============================================================
def find_lambda_star(n, epsilon, probability_level):
    """
    Solve
        F_{chi^2_n(lambda)}(epsilon^2) = probability_level
    for lambda >= 0.
    """
    x = epsilon ** 2
    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - probability_level
    lo, hi = 0.0, 1e11
    if equation(lo) * equation(hi) > 0:
        raise ValueError("Root not bracketed. Increase upper bound.")
    return brentq(equation, lo, hi)
def build_system_matrices(T):
    """Return A, B, C_T, N_T, Sigma_w, Sigma_e, Q_T."""
    A, B, CT = get_system(name, n, m, T, seed=0)
    sigma_w = np.kron(np.eye(T), (sigma_2 ** 2) * np.eye(n))
    NT = np.eye(n)
    for k in range(1, T):
        NT = np.hstack((NT, np.linalg.matrix_power(A, k)))
    Sigma_e = (
        NT @ sigma_w @ NT.T
        + np.linalg.matrix_power(A, T)
        @ sigma_0
        @ np.linalg.matrix_power(A, T).T
    )
    Q_T = block_diag(*([Q] * T))
    return A, B, CT, NT, sigma_w, Sigma_e, Q_T
def generate_structure_mle(T, dataset_seed):
    """
    Structure MLE used in your code:
        C_T_hat = X_bar U^T (U U^T)^dagger,
    with covariance estimated using residuals
        x_T^{(i,j)} - C_T_hat u^{(i)}.
    """
    A, _, CT, NT, sigma_w, _, _ = build_system_matrices(T)
    N = m * T + 10
    rng = np.random.default_rng(dataset_seed)
    U = rng.standard_normal((m * T, N))
    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))
    for i in range(N):
        for j in range(M):
            x0 = rng.multivariate_normal(np.zeros(n), sigma_0)
            w = rng.multivariate_normal(np.zeros(n * T), sigma_w)
            xT = np.linalg.matrix_power(A, T) @ x0 + CT @ U[:, i] + NT @ w
            all_xT[:, i, j] = xT
        X_bar[:, i] = np.mean(all_xT[:, i, :], axis=1)
    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)
    Sigma_hat = np.zeros((n, n))
    for i in range(N):
        for j in range(M):
            residual = all_xT[:, i, j] - CT_hat @ U[:, i]
            Sigma_hat += np.outer(residual, residual)
    Sigma_hat /= (N * M)
    return CT_hat, Sigma_hat
def conformal_quantile(scores, delta):
    N_cal_local = len(scores)
    k = int(np.ceil((N_cal_local + 1) * (1 - delta)))
    k = min(k, N_cal_local)
    return np.sort(scores)[k - 1]
def compute_conformal_q(T, CT_hat, dataset_seed, delta_used=None):
    """Calibration radius for ||x_T - C_T_hat u||_2."""
    A, _, CT, NT, sigma_w, _, _ = build_system_matrices(T)
    rng = np.random.default_rng(10000 + dataset_seed)
    U_cal = rng.standard_normal((m * T, N_cal))
    scores = []
    for i in range(N_cal):
        x0 = rng.multivariate_normal(np.zeros(n), sigma_0)
        w = rng.multivariate_normal(np.zeros(n * T), sigma_w)
        x_true = np.linalg.matrix_power(A, T) @ x0 + CT @ U_cal[:, i] + NT @ w
        x_pred = CT_hat @ U_cal[:, i]
        scores.append(np.linalg.norm(x_true - x_pred, ord=2))
    if delta_used is None:
        delta_used = delta
    return conformal_quantile(np.asarray(scores), delta_used)
def solve_mec(CT_used, Q_T, rhs, metric=None, return_u=False):
    """
    Solve
        min_u  u^T Q_T u
        s.t.  (C_T u - x_f)^T metric (C_T u - x_f) <= rhs.
    If metric=None, metric=I, i.e. the absolute Euclidean-norm case.
    Returns np.nan if infeasible / solver fails.
    If return_u=True, returns (cost, u_star).
    """
    if rhs < 0:
        return (np.nan, None) if return_u else np.nan
    if metric is None:
        metric = np.eye(n)
    # Numerical symmetrization is useful for estimated inverse covariance.
    metric = 0.5 * (metric + metric.T)
    u = cp.Variable(CT_used.shape[1])
    objective = cp.Minimize(cp.quad_form(u, Q_T))
    constraint = cp.quad_form(CT_used @ u - xf, metric) <= rhs
    prob = cp.Problem(objective, [constraint])
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=50000)
    if u.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return (np.nan, None) if return_u else np.nan
    uval = np.asarray(u.value).reshape(-1)
    cost = float(uval.T @ Q_T @ uval)
    return (cost, uval) if return_u else cost
def nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(values)
# ============================================================
# PROBLEMS INCLUDED
# ============================================================
# 1) Structure MLE, normalized / Mahalanobis norm:
#
#       min_u      u^T Q_T u
#       s.t.      (C_T_hat u - x_f)^T Sigma_hat^{-1}
#                 (C_T_hat u - x_f) <= gamma.
#
# 2) Structure MLE, absolute norm  [NEW]:
#
#       min_u      u^T Q_T u
#       s.t.      ||C_T_hat u - x_f||_2^2 <= gamma.
#
#    For the epsilon-probability experiment vs T, gamma is replaced by
#    lambda_star so that it is directly comparable with the known-dynamics
#    absolute-norm formulation already present in your code.
#
# 3) Conformal absolute-norm formulation:
#
#       min_u      u^T Q_T u
#       s.t.      ||C_T_hat u - x_f||_2^2 <= (epsilon-q)^2.
#
# 4) Known dynamics, normalized norm:
#
#       min_u      u^T Q_T u
#       s.t.      (C_T u - x_f)^T Sigma_e^{-1}
#                 (C_T u - x_f) <= gamma.
#
# 5) Known dynamics, absolute norm:
#
#       min_u      u^T Q_T u
#       s.t.      ||C_T u - x_f||_2^2 <= gamma.
#
# The same convex-program solution routine solve_mec() solves all five
# cases by changing C_T, the metric matrix, and the right-hand side.
# ============================================================
# A. COST VS T
# ============================================================
lambda_star = find_lambda_star(n, epsilon, 1 - delta)
cost_T_struct_norm = []
cost_T_struct_abs = []
cost_T_conf = []
cost_T_known_norm = []
cost_T_known_abs = []
for T in T_cost_list:
    print(f"Cost vs T: T={T}")
    _, _, CT, _, _, Sigma_e, Q_T = build_system_matrices(T)
    # Known-dynamics curves do not depend on the dataset.
    known_norm = solve_mec(
        CT_used=CT,
        Q_T=Q_T,
        rhs=lambda_star,
        metric=np.linalg.pinv(Sigma_e),
    )
    known_abs = solve_mec(
        CT_used=CT,
        Q_T=Q_T,
        rhs=lambda_star,
        metric=None,
    )
    struct_norm_runs = []
    struct_abs_runs = []
    conf_runs = []
    for dataset in range(k_est):
        CT_hat, Sigma_hat = generate_structure_mle(T, dataset)
        struct_norm_runs.append(
            solve_mec(
                CT_used=CT_hat,
                Q_T=Q_T,
                rhs=lambda_star,
                metric=np.linalg.pinv(Sigma_hat),
            )
        )
        struct_abs_runs.append(
            solve_mec(
                CT_used=CT_hat,
                Q_T=Q_T,
                rhs=lambda_star,
                metric=None,
            )
        )
        q = compute_conformal_q(T, CT_hat, dataset)
        lambda_cp = (epsilon - q) ** 2 if epsilon >= q else -1.0
        conf_runs.append(
            solve_mec(
                CT_used=CT_hat,
                Q_T=Q_T,
                rhs=lambda_cp,
                metric=None,
            )
        )
    cost_T_struct_norm.append(nanmean_or_nan(struct_norm_runs))
    cost_T_struct_abs.append(nanmean_or_nan(struct_abs_runs))
    cost_T_conf.append(nanmean_or_nan(conf_runs))
    cost_T_known_norm.append(known_norm)
    cost_T_known_abs.append(known_abs)
# ============================================================
# B. COST VS GAMMA
# ============================================================
all_gamma = {}
all_cost_struct_norm = {}
all_cost_struct_abs = {}
all_cost_conf = {}
all_cost_known_norm = {}
all_cost_known_abs = {}
for T in T_gamma_list:
    print(f"Cost vs gamma: T={T}")
    _, _, CT, _, _, Sigma_e, Q_T = build_system_matrices(T)
    # Same normalized gamma upper limit as in your original script.
    b = -xf
    gamma_max = float(b.T @ np.linalg.pinv(Sigma_e) @ b)
    gamma_vals = np.linspace(0.0, gamma_max, num_gamma)
    struct_norm = []
    struct_abs = []
    conf = []
    known_norm = []
    known_abs = []
    # Generate the MLE datasets once per T and reuse them for all gamma values.
    # This gives a cleaner comparison because only gamma changes across a curve.
    mle_datasets = []
    for dataset in range(k_est):
        CT_hat, Sigma_hat = generate_structure_mle(T, dataset)
        q = compute_conformal_q(T, CT_hat, dataset)
        mle_datasets.append((CT_hat, Sigma_hat, q))
    for gamma in gamma_vals:
        s_norm_runs = []
        s_abs_runs = []
        conf_runs = []
        for CT_hat, Sigma_hat, q in mle_datasets:
            s_norm_runs.append(
                solve_mec(
                    CT_used=CT_hat,
                    Q_T=Q_T,
                    rhs=gamma,
                    metric=np.linalg.pinv(Sigma_hat),
                )
            )
            # NEW: structure MLE + absolute Euclidean norm
            s_abs_runs.append(
                solve_mec(
                    CT_used=CT_hat,
                    Q_T=Q_T,
                    rhs=gamma,
                    metric=None,
                )
            )
            # Existing conformal curve from your original gamma sweep.
            # Here gamma is used as the terminal radius, as in the original code.
            lambda_cp = (gamma - q) ** 2
            conf_runs.append(
                solve_mec(
                    CT_used=CT_hat,
                    Q_T=Q_T,
                    rhs=lambda_cp,
                    metric=None,
                )
            )
        struct_norm.append(nanmean_or_nan(s_norm_runs))
        struct_abs.append(nanmean_or_nan(s_abs_runs))
        conf.append(nanmean_or_nan(conf_runs))
        known_norm.append(
            solve_mec(
                CT_used=CT,
                Q_T=Q_T,
                rhs=gamma,
                metric=np.linalg.pinv(Sigma_e),
            )
        )
        known_abs.append(
            solve_mec(
                CT_used=CT,
                Q_T=Q_T,
                rhs=gamma,
                metric=None,
            )
        )
    all_gamma[T] = gamma_vals
    all_cost_struct_norm[T] = struct_norm
    all_cost_struct_abs[T] = struct_abs
    all_cost_conf[T] = conf
    all_cost_known_norm[T] = known_norm
    all_cost_known_abs[T] = known_abs
# ============================================================
# C. IEEE TWO-PANEL FIGURE
# Same compact sizing/style as the earlier IEEE figures:
# two side-by-side panels inside one 3.48-in column.
# ============================================================
fig, ax = plt.subplots(
    1, 2,
    figsize=(3.48, 2.12)
)
# Reserve the upper part of the figure for legends so that
# no legend overlaps either plotting region.
fig.subplots_adjust(
    left=0.13,
    right=0.98,
    bottom=0.22,
    top=0.72,
    wspace=0.42
)
# ------------------------------------------------------------
# Visual encoding
#   color      -> method
#   gamma panel uses only T = 6
# No markers.
# ------------------------------------------------------------
method_colors = {
    "struct_norm": "C0",
    "struct_abs":  "C1",
    "conformal":   "C2",
    "known_norm":  "C3",
    "known_abs":   "C4",
}
# Use exactly the same line style for each method in both cost panels and the legend.
method_linestyles = {
    "struct_norm": "-",
    "struct_abs": "-",
    "conformal": "-",
    "known_norm": "--",
    "known_abs": "--",
}
# ---------------- Cost vs gamma ----------------
for T in T_gamma_list:
    ax[0].plot(
        all_gamma[T], all_cost_struct_norm[T],
        color=method_colors["struct_norm"],
        linestyle=method_linestyles["struct_norm"],
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_struct_abs[T],
        color=method_colors["struct_abs"],
        linestyle=method_linestyles["struct_abs"],
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_conf[T],
        color=method_colors["conformal"],
        linestyle=method_linestyles["conformal"],
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_known_norm[T],
        color=method_colors["known_norm"],
        linestyle=method_linestyles["known_norm"],
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_known_abs[T],
        color=method_colors["known_abs"],
        linestyle=method_linestyles["known_abs"],
        linewidth=0.70
    )
# ---------------- Cost vs T ----------------
ax[1].plot(
    T_cost_list, cost_T_struct_norm,
    color=method_colors["struct_norm"],
    linestyle=method_linestyles["struct_norm"],
    linewidth=0.70,
    label="MLE normalised norm"
)
ax[1].plot(
    T_cost_list, cost_T_struct_abs,
    color=method_colors["struct_abs"],
    linestyle=method_linestyles["struct_abs"],
    linewidth=0.70,
    label="MLE absolute norm"
)
ax[1].plot(
    T_cost_list, cost_T_conf,
    color=method_colors["conformal"],
    linestyle=method_linestyles["conformal"],
    linewidth=0.70,
    label="Conformal"
)
ax[1].plot(
    T_cost_list, cost_T_known_norm,
    color=method_colors["known_norm"],
    linestyle=method_linestyles["known_norm"],
    linewidth=0.70,
    label="Known normalised norm"
)
ax[1].plot(
    T_cost_list, cost_T_known_abs,
    color=method_colors["known_abs"],
    linestyle=method_linestyles["known_abs"],
    linewidth=0.70,
    label="Known absolute norm"
)
for a in ax:
    style_ieee_axis(a)
ax[0].set_xlabel(r"$\gamma$")
ax[0].set_ylabel(r"$J_T^*$")
ax[1].set_xlabel(r"$T$")
ax[1].set_ylabel(r"$J_T^*$", labelpad=0)
ax[1].set_xticks(T_cost_list)
ax[1].yaxis.set_label_coords(-0.13, 0.5)
# Same small label/tick sizes as the earlier two-panel figures.
for a in ax:
    a.xaxis.label.set_size(6.5)
    a.yaxis.label.set_size(6.5)
    a.tick_params(labelsize=5.8)
# ------------------------------------------------------------
# ------------------------------------------------------------
from matplotlib.lines import Line2D
method_handles = [
    Line2D([0], [0], color=method_colors["struct_norm"], lw=0.70,
           linestyle=method_linestyles["struct_norm"], label="MLE normalised norm"),
    Line2D([0], [0], color=method_colors["struct_abs"], lw=0.70,
           linestyle=method_linestyles["struct_abs"], label="MLE absolute norm"),
    Line2D([0], [0], color=method_colors["conformal"], lw=0.70,
           linestyle=method_linestyles["conformal"], label="Conformal"),
    Line2D([0], [0], color=method_colors["known_norm"], lw=0.70,
           linestyle=method_linestyles["known_norm"], label="Known normalised norm"),
    Line2D([0], [0], color=method_colors["known_abs"], lw=0.70,
           linestyle=method_linestyles["known_abs"], label="Known absolute norm"),
]
# One shared method legend centered above BOTH cost panels.
fig.legend(
    handles=method_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=3,
    frameon=False,
    fontsize=4.25,
    handlelength=1.45,
    handletextpad=0.28,
    columnspacing=0.65,
    labelspacing=0.18,
    borderaxespad=0.0,
)
stem = os.path.join(OUTDIR, f"cost_gamma_T_{name}")
save_ieee_figure(fig, stem)
plt.show()
print(f"Saved: {stem}.pdf")
print(f"Saved: {stem}.png")
# ============================================================
# D. EMPIRICAL PROBABILITY VS DESIRED PROBABILITY 1-delta
# ============================================================
#
# This section is appended to the ORIGINAL cost-vs-gamma and
# cost-vs-T code above. Nothing in those original plots is removed.
#
# x-axis:
#       desired probability = 1-delta
#
# y-axis:
#       empirical probability under the TRUE system
#
# Three horizons only:
#       T = 4, 6, 8
#
# Left panel:
#       P{ e^T Sigma_e^{-1} e <= epsilon^2 }
#
# Right panel:
#       P{ ||e||_2^2 <= epsilon^2 }
#
# The reference line y=x shows exact calibration.
# ============================================================
# Three horizons requested for the probability experiment.
T_prob_list = [4, 6]
# For the current n=4 and epsilon=2.4, the normalized Gaussian
# probability at zero noncentrality is about 0.782. Therefore the
# desired probabilities below are chosen inside the feasible range
# for find_lambda_star().
desired_prob_list = np.array([
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
])
# Number of independent true-system samples used to estimate each
# terminal probability.
N_mc = 2000
# ------------------------------------------------------------
# Probability-validation helpers
# ------------------------------------------------------------
def generate_terminal_noise_samples(T, N_mc, seed):
    """
    Generate independent samples of the random terminal component
        eta_T = A^T x_0 + N_T w.
    Hence, for any fixed controller u,
        e_T = x_T - x_f
            = C_T u - x_f + eta_T.
    The TRUE A, C_T, N_T and noise covariances are used here.
    """
    A, _, CT, NT, sigma_w, Sigma_e, _ = build_system_matrices(T)
    rng = np.random.default_rng(seed)
    x0_samples = rng.multivariate_normal(
        np.zeros(n),
        sigma_0,
        size=N_mc,
    )
    w_samples = rng.multivariate_normal(
        np.zeros(n * T),
        sigma_w,
        size=N_mc,
    )
    AT = np.linalg.matrix_power(A, T)
    # Each row is one independent realization of
    # A^T x_0 + N_T w.
    eta_samples = (
        x0_samples @ AT.T
        + w_samples @ NT.T
    )
    return eta_samples, CT, Sigma_e
def empirical_terminal_probabilities(
    u_star,
    CT_true,
    Sigma_e_true,
    eta_samples,
    epsilon,
):
    """
    Estimate, under the TRUE stochastic system,
        P{ e^T Sigma_e^{-1} e <= epsilon^2 }
    and
        P{ ||e||_2^2 <= epsilon^2 }.
    """
    if u_star is None:
        return np.nan, np.nan
    mean_error = CT_true @ u_star - xf
    # rows = independent terminal-error samples
    errors = eta_samples + mean_error[None, :]
    Sigma_e_inv = np.linalg.pinv(Sigma_e_true)
    mahal_sq = np.einsum(
        "bi,ij,bj->b",
        errors,
        Sigma_e_inv,
        errors,
    )
    euclidean_sq = np.sum(errors ** 2, axis=1)
    p_norm = np.mean(mahal_sq <= epsilon ** 2)
    p_abs = np.mean(euclidean_sq <= epsilon ** 2)
    return p_norm, p_abs
# ------------------------------------------------------------
# Method names / probability-plot line styles
# ------------------------------------------------------------
prob_method_names = [
    "struct_norm",
    "struct_abs",
    "conformal",
    "known_norm",
    "known_abs",
]
prob_method_labels = {
    "struct_norm": "MLE normalised norm",
    "struct_abs": "MLE absolute norm",
    "conformal": "Conformal",
    "known_norm": "Known normalised norm",
    "known_abs": "Known absolute norm",
}
# Reuse the same method colors already used in the original cost plot.
# Line style now represents T.
T_prob_linestyles = {
    4: "-",
    6: "--"
}
# ------------------------------------------------------------
# Storage
#
# prob_norm[T][method][j]
# prob_abs[T][method][j]
#
# j indexes desired_prob_list.
# ------------------------------------------------------------
prob_norm = {
    T: {method: [] for method in prob_method_names}
    for T in T_prob_list
}
prob_abs = {
    T: {method: [] for method in prob_method_names}
    for T in T_prob_list
}
# ============================================================
# D1. RUN PROBABILITY EXPERIMENT
# ============================================================
for T in T_prob_list:
    print("\n" + "=" * 72)
    print(f"Empirical probability experiment: T={T}")
    print("=" * 72)
    _, _, CT, _, _, Sigma_e, Q_T = build_system_matrices(T)
    # --------------------------------------------------------
    # Generate one set of TRUE Monte-Carlo disturbances for T.
    # The same realizations are reused for every method and every
    # desired probability. This reduces comparison noise.
    # --------------------------------------------------------
    eta_samples, CT_true, Sigma_e_true = generate_terminal_noise_samples(
        T=T,
        N_mc=N_mc,
        seed=30000 + T,
    )
    # --------------------------------------------------------
    # Generate the estimation datasets once for this T and reuse
    # them across all values of 1-delta.
    # --------------------------------------------------------
    probability_datasets = []
    for dataset in range(k_est):
        CT_hat, Sigma_hat = generate_structure_mle(T, dataset)
        probability_datasets.append((
            dataset,
            CT_hat,
            Sigma_hat,
        ))
    # ========================================================
    # Sweep desired probability 1-delta
    # ========================================================
    for desired_prob in desired_prob_list:
        delta_current = 1.0 - desired_prob
        # ----------------------------------------------------
        # For the normalized Gaussian chance constraint, find
        # lambda_star satisfying
        #
        # F_{chi_n^2(lambda_star)}(epsilon^2) = 1-delta.
        # ----------------------------------------------------
        lambda_star_current = find_lambda_star(
            n=n,
            epsilon=epsilon,
            probability_level=desired_prob,
        )
        # ====================================================
        # 1) KNOWN DYNAMICS, NORMALIZED NORM
        # ====================================================
        _, u_known_norm = solve_mec(
            CT_used=CT,
            Q_T=Q_T,
            rhs=lambda_star_current,
            metric=np.linalg.pinv(Sigma_e),
            return_u=True,
        )
        p_norm, p_abs = empirical_terminal_probabilities(
            u_star=u_known_norm,
            CT_true=CT_true,
            Sigma_e_true=Sigma_e_true,
            eta_samples=eta_samples,
            epsilon=epsilon,
        )
        prob_norm[T]["known_norm"].append(p_norm)
        prob_abs[T]["known_norm"].append(p_abs)
        # ====================================================
        # 2) KNOWN DYNAMICS, ABSOLUTE NORM
        # ====================================================
        _, u_known_abs = solve_mec(
            CT_used=CT,
            Q_T=Q_T,
            rhs=lambda_star_current,
            metric=None,
            return_u=True,
        )
        p_norm, p_abs = empirical_terminal_probabilities(
            u_star=u_known_abs,
            CT_true=CT_true,
            Sigma_e_true=Sigma_e_true,
            eta_samples=eta_samples,
            epsilon=epsilon,
        )
        prob_norm[T]["known_abs"].append(p_norm)
        prob_abs[T]["known_abs"].append(p_abs)
        # ====================================================
        # DATA-DRIVEN METHODS: collect one probability for each
        # independently estimated model, then average them.
        # ====================================================
        temp_prob_norm = {
            "struct_norm": [],
            "struct_abs": [],
            "conformal": [],
        }
        temp_prob_abs = {
            "struct_norm": [],
            "struct_abs": [],
            "conformal": [],
        }
        for dataset, CT_hat, Sigma_hat in probability_datasets:
            # ------------------------------------------------
            # 3) STRUCTURE MLE, NORMALIZED NORM
            # ------------------------------------------------
            _, u_struct_norm = solve_mec(
                CT_used=CT_hat,
                Q_T=Q_T,
                rhs=lambda_star_current,
                metric=np.linalg.pinv(Sigma_hat),
                return_u=True,
            )
            p_norm, p_abs = empirical_terminal_probabilities(
                u_star=u_struct_norm,
                CT_true=CT_true,
                Sigma_e_true=Sigma_e_true,
                eta_samples=eta_samples,
                epsilon=epsilon,
            )
            temp_prob_norm["struct_norm"].append(p_norm)
            temp_prob_abs["struct_norm"].append(p_abs)
            # ------------------------------------------------
            # 4) STRUCTURE MLE, ABSOLUTE NORM
            # ------------------------------------------------
            _, u_struct_abs = solve_mec(
                CT_used=CT_hat,
                Q_T=Q_T,
                rhs=lambda_star_current,
                metric=None,
                return_u=True,
            )
            p_norm, p_abs = empirical_terminal_probabilities(
                u_star=u_struct_abs,
                CT_true=CT_true,
                Sigma_e_true=Sigma_e_true,
                eta_samples=eta_samples,
                epsilon=epsilon,
            )
            temp_prob_norm["struct_abs"].append(p_norm)
            temp_prob_abs["struct_abs"].append(p_abs)
            # ------------------------------------------------
            # 5) CONFORMAL ABSOLUTE-NORM FORMULATION
            #
            # Recompute q at the CURRENT delta while retaining the
            # same estimation/calibration dataset seed.
            # ------------------------------------------------
            q_current = compute_conformal_q(
                T,
                CT_hat,
                dataset,
                delta_used=delta_current,
            )
            lambda_cp_current = (
                (np.sqrt(epsilon) - q_current) ** 2
                if np.sqrt(epsilon) >= q_current
                else -1.0
            )
            _, u_conf = solve_mec(
                CT_used=CT_hat,
                Q_T=Q_T,
                rhs=lambda_cp_current,
                metric=None,
                return_u=True,
            )
            p_norm, p_abs = empirical_terminal_probabilities(
                u_star=u_conf,
                CT_true=CT_true,
                Sigma_e_true=Sigma_e_true,
                eta_samples=eta_samples,
                epsilon=epsilon,
            )
            temp_prob_norm["conformal"].append(p_norm)
            temp_prob_abs["conformal"].append(p_abs)
        # ----------------------------------------------------
        # Average probabilities over independently estimated
        # datasets for the three data-driven methods.
        # ----------------------------------------------------
        for method in ["struct_norm", "struct_abs", "conformal"]:
            prob_norm[T][method].append(
                nanmean_or_nan(temp_prob_norm[method])
            )
            prob_abs[T][method].append(
                nanmean_or_nan(temp_prob_abs[method])
            )
        # ----------------------------------------------------
        # Small progress printout
        # ----------------------------------------------------
        print(
            f"T={T:2d}, desired 1-delta={desired_prob:.2f}, "
            f"known-norm empirical={prob_norm[T]['known_norm'][-1]:.4f}"
        )
# ============================================================
# D2. IEEE TWO-PANEL PROBABILITY FIGURE
# ============================================================
fig_prob, ax_prob = plt.subplots(
    1, 2,
    figsize=(3.48, 2.22),
)
fig_prob.subplots_adjust(
    left=0.15,
    right=0.98,
    bottom=0.22,
    top=0.68,
    wspace=0.52,
)
# ------------------------------------------------------------
# Plot only matching formulations for T = 4, 6.
#
# color      -> method
# line style -> T
# ------------------------------------------------------------
normalized_prob_methods = [
    "struct_norm",
    "known_norm",
]
absolute_prob_methods = [
    "struct_abs",
    "conformal",
    "known_abs",
]
# Left panel: normalized formulations only.
for T in T_prob_list:
    ls = T_prob_linestyles[T]
    for method in normalized_prob_methods:
        ax_prob[0].plot(
            desired_prob_list,
            prob_norm[T][method],
            color=method_colors[method],
            linestyle=ls,
            linewidth=0.75,
        )
# Right panel: absolute-norm formulations only.
for T in T_prob_list:
    ls = T_prob_linestyles[T]
    for method in absolute_prob_methods:
        ax_prob[1].plot(
            desired_prob_list,
            prob_abs[T][method],
            color=method_colors[method],
            linestyle=ls,
            linewidth=0.75,
        )
# ------------------------------------------------------------
# Ideal calibration / feasibility boundary y=x.
#
# Above y=x: empirical probability >= requested probability.
# Below y=x: empirical probability < requested probability.
# ------------------------------------------------------------
x_reference = np.linspace(
    desired_prob_list.min(),
    desired_prob_list.max(),
    200,
)
for a in ax_prob:
    a.plot(
        x_reference,
        x_reference,
        color="black",
        linestyle=":",
        linewidth=0.90,
    )
    style_ieee_axis(a)
    a.set_xlabel(r"$1-\delta$")
    a.set_xlim(
        desired_prob_list.min() - 0.01,
        desired_prob_list.max() + 0.01,
    )
    a.set_ylim(0.0, 1.02)
    a.set_xticks(desired_prob_list)
    a.xaxis.label.set_size(6.5)
    a.yaxis.label.set_size(6.5)
    a.tick_params(labelsize=5.6)
# No panel titles. The event itself is written on the y-axis.
ax_prob[0].set_ylabel(
    r"$\hat{P}(e^\top \Sigma_e^{-1}e \leq \epsilon^2)$"
)
ax_prob[1].set_ylabel(
    r"$\hat{P}(\|e\|_2^2 \leq \epsilon^2)$"
)
ax_prob[0].yaxis.set_label_coords(-0.16, 0.5)
ax_prob[1].yaxis.set_label_coords(-0.18, 0.5)
# ------------------------------------------------------------
# Probability-figure legends
# ------------------------------------------------------------
# Legends are kept only at the top. Each panel has one compact
# combined legend containing the methods, T line styles, and y=x.
# ------------------------------------------------------------
norm_prob_handles = [
    Line2D(
        [0], [0],
        color=method_colors[method],
        lw=0.75,
        linestyle="-",
        label=prob_method_labels[method],
    )
    for method in normalized_prob_methods
]
abs_prob_handles = [
    Line2D(
        [0], [0],
        color=method_colors[method],
        lw=0.75,
        linestyle="-",
        label=prob_method_labels[method],
    )
    for method in absolute_prob_methods
]
prob_T_handles = [
    Line2D(
        [0], [0],
        color="black",
        lw=0.75,
        linestyle=T_prob_linestyles[T],
        label=rf"$T={T}$",
    )
    for T in T_prob_list
]
prob_reference_handle = Line2D(
    [0], [0],
    color="black",
    lw=0.90,
    linestyle=":",
    label=r"$y=x$",
)
# Top legend for normalized panel.
ax_prob[0].legend(
    handles=norm_prob_handles + prob_T_handles + [prob_reference_handle],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.07),
    ncol=3,
    frameon=False,
    fontsize=3.95,
    handlelength=1.40,
    handletextpad=0.24,
    columnspacing=0.45,
    labelspacing=0.14,
    borderaxespad=0.0,
)
# Top legend for absolute-norm panel.
ax_prob[1].legend(
    handles=abs_prob_handles + prob_T_handles + [prob_reference_handle],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.07),
    ncol=3,
    frameon=False,
    fontsize=3.95,
    handlelength=1.40,
    handletextpad=0.24,
    columnspacing=0.45,
    labelspacing=0.14,
    borderaxespad=0.0,
)
# ------------------------------------------------------------
# Save probability figure
# ------------------------------------------------------------
prob_stem = os.path.join(
    OUTDIR,
    f"empirical_vs_desired_probability_{name}",
)
save_ieee_figure(fig_prob, prob_stem)
plt.show()
print(f"Saved: {prob_stem}.pdf")
print(f"Saved: {prob_stem}.png")
# ============================================================
# D3. PRINT NUMERICAL PROBABILITY TABLES
# ============================================================
for T in T_prob_list:
    print("\n" + "=" * 78)
    print(f"EMPIRICAL PROBABILITIES, T={T}")
    print("=" * 78)
    for j, desired_prob in enumerate(desired_prob_list):
        print(f"\nDesired probability 1-delta = {desired_prob:.2f}")
        print("  Normalized event:")
        for method in normalized_prob_methods:
            print(
                f"    {prob_method_labels[method]:15s}"
                f"{prob_norm[T][method][j]:10.4f}"
            )
        print("  Absolute-norm event:")
        for method in absolute_prob_methods:
            print(
                f"    {prob_method_labels[method]:15s}"
                f"{prob_abs[T][method][j]:10.4f}"
            )
# ============================================================
# E. TRUE VS ESTIMATED QUADRATIC FORM VS T
# ============================================================

T_quad_list = list(range(2, 20, 2))

quad_true_mean = []
quad_est_mean = []

quad_true_std = []
quad_est_std = []

lambda_star_quad = find_lambda_star(
    n=n,
    epsilon=epsilon,
    probability_level=1 - delta,
)

for T in T_quad_list:

    print(f"True vs estimated quadratic form: T={T}")

    _, _, CT, _, _, Sigma_e, Q_T = build_system_matrices(T)

    Sigma_e_inv = np.linalg.pinv(Sigma_e)

    q_true_runs = []
    q_est_runs = []

    for dataset in range(k_est):

        # Estimate C_T and Sigma_e
        CT_hat, Sigma_hat = generate_structure_mle(
            T=T,
            dataset_seed=dataset,
        )

        Sigma_hat_inv = np.linalg.pinv(Sigma_hat)

        # Solve the ESTIMATED normalized-norm problem
        _, u_hat = solve_mec(
            CT_used=CT_hat,
            Q_T=Q_T,
            rhs=lambda_star_quad,
            metric=Sigma_hat_inv,
            return_u=True,
        )

        if u_hat is None:
            continue

        # ----------------------------------------------------
        # TRUE quadratic form evaluated at estimated optimizer
        # ----------------------------------------------------
        e_true = CT @ u_hat - xf

        q_true = float(
            e_true.T
            @ Sigma_e_inv
            @ e_true
        )

        # ----------------------------------------------------
        # ESTIMATED quadratic form
        # ----------------------------------------------------
        e_hat = CT_hat @ u_hat - xf

        q_est = float(
            e_hat.T
            @ Sigma_hat_inv
            @ e_hat
        )

        q_true_runs.append(q_true)
        q_est_runs.append(q_est)

    quad_true_mean.append(np.mean(q_true_runs))
    quad_est_mean.append(np.mean(q_est_runs))

    quad_true_std.append(np.std(q_true_runs))
    quad_est_std.append(np.std(q_est_runs))


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(3.48, 2.2)
)

ax.plot(
    T_quad_list,
    quad_true_mean,
    linewidth=0.8,
    label=r"True quadratic form",
)

ax.plot(
    T_quad_list,
    quad_est_mean,
    linewidth=0.8,
    linestyle="--",
    label=r"Estimated quadratic form",
)

# Constraint threshold
ax.axhline(
    lambda_star_quad,
    linestyle=":",
    linewidth=0.9,
    label=r"$\lambda^\star$",
)

style_ieee_axis(ax)

ax.set_xlabel(r"$T$")

ax.set_ylabel(
    r"Quadratic form value"
)

ax.set_xticks(T_quad_list)

ax.legend(
    frameon=False,
    fontsize=5.5,
)

fig.tight_layout()

stem = os.path.join(
    OUTDIR,
    f"true_vs_estimated_quadratic_{name}",
)

save_ieee_figure(fig, stem)

plt.show()