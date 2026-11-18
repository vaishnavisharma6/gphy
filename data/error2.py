import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datagen import get_system


# ============================================================
# IEEE / OVERLEAF PUBLICATION SETTINGS
# ============================================================

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "lines.linewidth": 0.90,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


# ============================================================
# PLOTTING HELPERS
# ============================================================

def style_ieee_axis(ax):
    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.45,
        alpha=0.45,
    )
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        pad=2,
    )
    ax.margins(x=0.02)


def save_ieee_figure(fig, stem):
    fig.savefig(
        f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )


def plain_log_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{x:g}"


def set_plain_log_ticks(ax):
    # x-axis
    ax.xaxis.set_major_locator(
        mpl.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
    )
    ax.xaxis.set_major_formatter(FuncFormatter(plain_log_formatter))
    ax.xaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # y-axis
    ax.yaxis.set_major_locator(
        mpl.ticker.LogLocator(base=10, subs=(1.0,))
    )
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_formatter))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

n = 4
m = 4
name = "stable"

sigma_1 = 0.2
sigma_2 = 0.3

sigma_0 = sigma_1**2 * np.eye(n)


# ============================================================
# BUILD SYSTEM MATRICES
# ============================================================

def build_system_matrices(t):
    A, B, CT = get_system(
        name,
        n,
        m,
        t,
        seed=0,
    )

    # Process-noise covariance at one time step
    sigma = sigma_2**2 * np.eye(n)

    # Full process-noise covariance over horizon
    sigmaw = np.kron(np.eye(t), sigma)

    # Noise propagation matrix
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack(
            (
                NT,
                np.linalg.matrix_power(A, i),
            )
        )

    # True terminal-state covariance Sigma_{e_T}
    A_t = np.linalg.matrix_power(A, t)
    sigmaz = (
        NT @ sigmaw @ NT.T
        + A_t @ sigma_0 @ A_t.T
    )

    return A, B, CT, NT, sigmaw, sigmaz


# ============================================================
# GENERATE NESTED DATA
# ============================================================

def generate_nested_data(
    t,
    N_max,
    M_max,
    seed=0,
    U_fixed=None,
):
    rng = np.random.default_rng(seed)

    A, B, CT, NT, sigmaw, sigmaz = build_system_matrices(t)

    if U_fixed is None:
        U_full = rng.standard_normal((m * t, N_max))
    else:
        U_full = U_fixed
        N_max = U_full.shape[1]

    all_xT_full = np.zeros((n, N_max, M_max))
    A_t = np.linalg.matrix_power(A, t)

    for i in range(N_max):
        for j in range(M_max):
            x0 = rng.multivariate_normal(
                np.zeros(n),
                sigma_0,
            )

            w = rng.multivariate_normal(
                np.zeros(n * t),
                sigmaw,
            )

            all_xT_full[:, i, j] = (
                A_t @ x0
                + CT @ U_full[:, i]
                + NT @ w
            )

    return U_full, all_xT_full, CT, sigmaz


# ============================================================
# ESTIMATE C_T AND Sigma_{e_T}
# ============================================================
# Only the structure/MLE covariance estimator is retained.
# The sample-based covariance estimator has been removed.
# ============================================================

def estimates_from_nested_data(
    t,
    U_full,
    all_xT_full,
    N,
    M,
):
    _, _, CT, _, _, sigmaz = build_system_matrices(t)

    # Nested subset
    U = U_full[:, :N]
    all_xT = all_xT_full[:, :N, :M]

    # Mean terminal state for each excitation experiment
    X_bar = np.mean(all_xT, axis=2)

    # Structure/MLE estimator of C_T
    CT_hat = (
        X_bar
        @ U.T
        @ np.linalg.pinv(U @ U.T)
    )

    # Structure/MLE covariance estimator
    sigma_hat = np.zeros((n, n))

    for i in range(N):
        for j in range(M):
            diff = (
                all_xT[:, i, j]
                - CT_hat @ U[:, i]
            )
            sigma_hat += np.outer(diff, diff)

    sigma_hat /= N * M

    return sigma_hat, sigmaz, CT_hat, CT


# ============================================================
# ESTIMATION ERRORS
# ============================================================

def estimate_errors_from_nested_data(
    t,
    U_full,
    all_xT_full,
    N,
    M,
):
    sigma_hat, sigmaz, CT_hat, CT = estimates_from_nested_data(
        t,
        U_full,
        all_xT_full,
        N,
        M,
    )

    sigma_error = np.linalg.norm(
        sigma_hat - sigmaz,
        ord=2,
    )

    CT_error = np.linalg.norm(
        CT_hat - CT,
        ord=2,
    )

    return sigma_error, CT_error


# ============================================================
# AVERAGE ERRORS OVER K INDEPENDENT DATASETS
# ============================================================

def average_errors_nested(
    t,
    N_list=None,
    M_list=None,
    N_fixed=None,
    M_fixed=None,
    K=30,
    seed=100,
):
    if M_list is not None:
        N_max = N_fixed
        M_max = max(M_list)
        keys = list(M_list)
    else:
        N_max = max(N_list)
        M_max = M_fixed
        keys = list(N_list)

    sigma_errors = {key: [] for key in keys}
    CT_errors = {key: [] for key in keys}

    for k in range(K):
        U_full, all_xT_full, _, _ = generate_nested_data(
            t,
            N_max,
            M_max,
            seed=seed + k,
        )

        for key in keys:
            if M_list is not None:
                N = N_fixed
                M = key
            else:
                N = key
                M = M_fixed

            e_sigma, e_CT = estimate_errors_from_nested_data(
                t,
                U_full,
                all_xT_full,
                N,
                M,
            )

            sigma_errors[key].append(e_sigma)
            CT_errors[key].append(e_CT)

    mean_sigma_error = [
        np.mean(sigma_errors[key])
        for key in keys
    ]

    mean_CT_error = [
        np.mean(CT_errors[key])
        for key in keys
    ]

    return mean_sigma_error, mean_CT_error


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

t_fixed = 5
N_fixed = 100
M_fixed = 100

# Number of repeated experiments
M_list = [
    5,
    10,
    20,
    40,
    80,
    120,
    140,
    180,
    220,
    260,
]

# Number of distinct excitation experiments
N_list = [
    10,
    20,
    40,
    80,
    160,
    320,
    640,
    700,
    720,
    900,
]

# Number of independent datasets
K_errors = 40


# ============================================================
# ERRORS VS N_r = M
# ============================================================

print("Computing errors versus N_r...")

sigma_error_M, CT_error_M = average_errors_nested(
    t=t_fixed,
    M_list=M_list,
    N_fixed=N_fixed,
    K=K_errors,
    seed=100,
)


# ============================================================
# ERRORS VS N_e = N
# ============================================================

print("Computing errors versus N_e...")

sigma_error_N, CT_error_N = average_errors_nested(
    t=t_fixed,
    N_list=N_list,
    M_fixed=M_fixed,
    K=K_errors,
    seed=200,
)


# ============================================================
# COMBINED FIGURE:
# C_T ERROR AND Sigma_{e_T} ERROR IN THE SAME PANELS
# ============================================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(3.48, 1.85),
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.25,
    top=0.76,
    wspace=0.42,
)

# ------------------------------------------------------------
# LEFT: errors vs N_r
# ------------------------------------------------------------

ax[0].loglog(
    M_list,
    CT_error_M,
    linestyle="-",
    linewidth=0.9,
    label=r"$\mathbb{\hat{E}}|\widehat{C}_T-C_T\|_2$",
)

ax[0].loglog(
    M_list,
    sigma_error_M,
    linestyle="--",
    linewidth=0.9,
    label=r"$\mathbb{\hat{E}}|\widehat{\Sigma}_{e_T}-\Sigma_{e_T}\|_2$",
)

ax[0].set_xlabel(r"$N_r$")
ax[0].set_ylabel("Estimation error", labelpad=1)

# ------------------------------------------------------------
# RIGHT: errors vs N_e
# ------------------------------------------------------------

ax[1].loglog(
    N_list,
    CT_error_N,
    linestyle="-",
    linewidth=0.9,
)

ax[1].loglog(
    N_list,
    sigma_error_N,
    linestyle="--",
    linewidth=0.9,
)

ax[1].set_xlabel(r"$N_e$")
ax[1].set_ylabel("Estimation error", labelpad=1)

# ------------------------------------------------------------
# IEEE axis formatting
# ------------------------------------------------------------

for a in ax:
    style_ieee_axis(a)
    set_plain_log_ticks(a)
    a.xaxis.label.set_size(6.5)
    a.yaxis.label.set_size(6.5)
    a.tick_params(labelsize=5.8)

ax[0].yaxis.set_label_coords(-0.18, 0.5)

# ------------------------------------------------------------
# Shared legend above both panels
# ------------------------------------------------------------

handles, labels = ax[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    frameon=False,
    fontsize=8.0,
    handlelength=2.0,
    handletextpad=0.4,
    columnspacing=0.9,
    borderaxespad=0.0,
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

save_ieee_figure(
    fig,
    "combined_estimation_errors",
)

plt.show()
