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
T_gamma_list = [6, 8]

sigma_1 = 0.2
sigma_2 = 0.3
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


def compute_conformal_q(T, CT_hat, dataset_seed):
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

    return conformal_quantile(np.asarray(scores), delta)


def solve_mec(CT_used, Q_T, rhs, metric=None):
    """
    Solve
        min_u  u^T Q_T u
        s.t.  (C_T u - x_f)^T metric (C_T u - x_f) <= rhs.

    If metric=None, metric=I, i.e. the absolute Euclidean-norm case.

    Returns np.nan if infeasible / solver fails.
    """
    if rhs < 0:
        return np.nan

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
        return np.nan

    uval = np.asarray(u.value).reshape(-1)
    return float(uval.T @ Q_T @ uval)


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
    top=0.58,
    wspace=0.42
)

# ------------------------------------------------------------
# Visual encoding
#   color      -> method
#   line style -> prediction horizon T (gamma panel only)
# No markers.
# ------------------------------------------------------------
method_colors = {
    "struct_norm": "C0",
    "struct_abs":  "C1",
    "conformal":   "C2",
    "known_norm":  "C3",
    "known_abs":   "C4",
}

# For the gamma panel, distinguish the two horizons by line style.
T_linestyles = {
    T_gamma_list[0]: "-",
    T_gamma_list[1]: "--",
}

# ---------------- Cost vs gamma ----------------
for T in T_gamma_list:
    ls = T_linestyles[T]

    ax[0].plot(
        all_gamma[T], all_cost_struct_norm[T],
        color=method_colors["struct_norm"],
        linestyle=ls,
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_struct_abs[T],
        color=method_colors["struct_abs"],
        linestyle=ls,
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_conf[T],
        color=method_colors["conformal"],
        linestyle=ls,
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_known_norm[T],
        color=method_colors["known_norm"],
        linestyle=ls,
        linewidth=0.70
    )
    ax[0].plot(
        all_gamma[T], all_cost_known_abs[T],
        color=method_colors["known_abs"],
        linestyle=ls,
        linewidth=0.70
    )

# ---------------- Cost vs T ----------------
ax[1].plot(
    T_cost_list, cost_T_struct_norm,
    color=method_colors["struct_norm"],
    linestyle="-",
    linewidth=0.70,
    label="Struct. norm."
)
ax[1].plot(
    T_cost_list, cost_T_struct_abs,
    color=method_colors["struct_abs"],
    linestyle="-",
    linewidth=0.70,
    label="Struct. abs."
)
ax[1].plot(
    T_cost_list, cost_T_conf,
    color=method_colors["conformal"],
    linestyle="-",
    linewidth=0.70,
    label="Conformal"
)
ax[1].plot(
    T_cost_list, cost_T_known_norm,
    color=method_colors["known_norm"],
    linestyle="--",
    linewidth=0.70,
    label="Known norm."
)
ax[1].plot(
    T_cost_list, cost_T_known_abs,
    color=method_colors["known_abs"],
    linestyle="--",
    linewidth=0.70,
    label="Known abs."
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
# Separate legends ABOVE each panel.
# Left: method colors + T line-style key.
# Right: method colors only; all curves are solid.
# ------------------------------------------------------------
from matplotlib.lines import Line2D

method_handles = [
    Line2D([0], [0], color=method_colors["struct_norm"], lw=0.70,
           linestyle="-", label="Struct. norm."),
    Line2D([0], [0], color=method_colors["struct_abs"], lw=0.70,
           linestyle="-", label="Struct. abs."),
    Line2D([0], [0], color=method_colors["conformal"], lw=0.70,
           linestyle="-", label="Conformal"),
    Line2D([0], [0], color=method_colors["known_norm"], lw=0.70,
           linestyle="--", label="Known norm."),
    Line2D([0], [0], color=method_colors["known_abs"], lw=0.70,
           linestyle="--", label="Known abs."),
]

T_handles = [
    Line2D([0], [0], color="black", lw=0.70,
           linestyle=T_linestyles[T], label=rf"$T={T}$")
    for T in T_gamma_list
]

# Method legend above the gamma panel.
legend_gamma_methods = ax[0].legend(
    handles=method_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=2,
    frameon=False,
    fontsize=4.25,
    handlelength=1.45,
    handletextpad=0.28,
    columnspacing=0.55,
    labelspacing=0.18,
    borderaxespad=0.0,
)
ax[0].add_artist(legend_gamma_methods)

# Horizon line-style key immediately above the gamma axes.
ax[0].legend(
    handles=T_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=len(T_handles),
    frameon=False,
    fontsize=4.35,
    handlelength=1.65,
    handletextpad=0.30,
    columnspacing=0.75,
    borderaxespad=0.0,
)

# Separate method legend above the T panel. All right-panel curves are solid.
ax[1].legend(
    handles=method_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=2,
    frameon=False,
    fontsize=4.25,
    handlelength=1.45,
    handletextpad=0.28,
    columnspacing=0.55,
    labelspacing=0.18,
    borderaxespad=0.0,
)

stem = os.path.join(OUTDIR, f"cost_gamma_T_{name}")
save_ieee_figure(fig, stem)
plt.show()

print(f"Saved: {stem}.pdf")
print(f"Saved: {stem}.png")
