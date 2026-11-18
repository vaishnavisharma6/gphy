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

    # IEEE-compatible serif text
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif"
    ],

    "mathtext.fontset": "stix",

    # Compact IEEE sizes
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 5.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,

    # Thin publication-style lines
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "lines.linewidth": 0.90,
    "axes.linewidth": 0.65,

    # Compact ticks
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    # Keep vector text editable/searchable
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
})


# ============================================================
# PLOTTING HELPERS
# ============================================================

def style_ieee_axis(ax):
    """
    Compact IEEE-style axis.
    """

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.45,
        alpha=0.45
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        pad=2
    )

    ax.margins(x=0.02)


def save_ieee_figure(fig, stem):
    """
    Save vector PDF and high-resolution PNG.
    """

    fig.savefig(
        f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig.savefig(
        f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02
    )


def plain_log_formatter(x, pos):
    """
    Display log ticks as normal numbers rather than 10^k
    wherever possible.
    """

    if x == 0:
        return "0"

    if x >= 1:
        return f"{x:g}"

    return f"{x:g}"


def set_plain_log_ticks(ax):
    """
    IEEE-friendly logarithmic ticks.
    """

    # x-axis
    ax.xaxis.set_major_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=(1.0, 2.0, 5.0)
        )
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            plain_log_formatter
        )
    )

    ax.xaxis.set_minor_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=np.arange(2, 10) * 0.1
        )
    )

    ax.xaxis.set_minor_formatter(
        mpl.ticker.NullFormatter()
    )

    # y-axis
    ax.yaxis.set_major_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=(1.0,)
        )
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            plain_log_formatter
        )
    )

    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=np.arange(2, 10) * 0.1
        )
    )

    ax.yaxis.set_minor_formatter(
        mpl.ticker.NullFormatter()
    )


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

n = 4
m = 4

name = "stable"

sigma_1 = 0.2
sigma_2 = 0.3

sigma_0 = (
    sigma_1**2
    * np.eye(n)
)


# ============================================================
# BUILD SYSTEM MATRICES
# ============================================================

def build_system_matrices(t):

    A, B, CT = get_system(
        name,
        n,
        m,
        t,
        seed=0
    )

    # Process-noise covariance at one time step
    sigma = (
        sigma_2**2
        * np.eye(n)
    )

    # Full process-noise covariance over horizon
    sigmaw = np.kron(
        np.eye(t),
        sigma
    )

    # Noise propagation matrix
    NT = np.eye(n)

    for i in range(1, t):

        NT = np.hstack(
            (
                NT,
                np.linalg.matrix_power(
                    A,
                    i
                )
            )
        )

    # True terminal-state covariance
    sigmaz = (
        NT
        @ sigmaw
        @ NT.T

        + np.linalg.matrix_power(
            A,
            t
        )
        @ sigma_0
        @ np.linalg.matrix_power(
            A,
            t
        ).T
    )

    return (
        A,
        B,
        CT,
        NT,
        sigmaw,
        sigmaz
    )


# ============================================================
# GENERATE NESTED DATA
# ============================================================

def generate_nested_data(
    t,
    N_max,
    M_max,
    seed=0,
    U_fixed=None
):

    rng = np.random.default_rng(
        seed
    )

    (
        A,
        B,
        CT,
        NT,
        sigmaw,
        sigmaz
    ) = build_system_matrices(t)


    # --------------------------------------------------------
    # Generate input matrix
    # --------------------------------------------------------

    if U_fixed is None:

        U_full = rng.standard_normal(
            (m*t, N_max)
        )

    else:

        U_full = U_fixed

        N_max = U_full.shape[1]


    # --------------------------------------------------------
    # Terminal-state samples
    # --------------------------------------------------------

    all_xT_full = np.zeros(
        (
            n,
            N_max,
            M_max
        )
    )


    A_t = np.linalg.matrix_power(
        A,
        t
    )


    for i in range(N_max):

        for j in range(M_max):

            # Initial state
            x0 = rng.multivariate_normal(
                np.zeros(n),
                sigma_0
            )

            # Process noise
            w = rng.multivariate_normal(
                np.zeros(n*t),
                sigmaw
            )

            # Terminal state
            all_xT_full[:, i, j] = (
                A_t @ x0
                + CT @ U_full[:, i]
                + NT @ w
            )


    return (
        U_full,
        all_xT_full,
        CT,
        sigmaz
    )


# ============================================================
# ESTIMATE C_T AND COVARIANCE
# ============================================================

def estimates_from_nested_data(
    t,
    U_full,
    all_xT_full,
    N,
    M
):

    (
        A,
        B,
        CT,
        NT,
        sigmaw,
        sigmaz
    ) = build_system_matrices(t)


    # --------------------------------------------------------
    # Nested subset
    # --------------------------------------------------------

    U = U_full[:, :N]

    all_xT = all_xT_full[
        :,
        :N,
        :M
    ]


    # --------------------------------------------------------
    # Mean terminal state for each experiment
    # --------------------------------------------------------

    X_bar = np.mean(
        all_xT,
        axis=2
    )


    # --------------------------------------------------------
    # Structure estimator of C_T
    # --------------------------------------------------------

    CT_hat = (
        X_bar
        @ U.T
        @ np.linalg.pinv(
            U @ U.T
        )
    )


    # ========================================================
    # SAMPLE COVARIANCE
    # ========================================================

    sample_sigma = np.zeros(
        (n, n)
    )


    # ========================================================
    # STRUCTURE-MLE COVARIANCE
    # ========================================================

    struct_sigma = np.zeros(
        (n, n)
    )


    for i in range(N):

        for j in range(M):

            # Sample-mean residual
            diff_sample = (
                all_xT[:, i, j]
                - X_bar[:, i]
            )

            # Structure residual
            diff_struct = (
                all_xT[:, i, j]
                - CT_hat @ U[:, i]
            )


            sample_sigma += np.outer(
                diff_sample,
                diff_sample
            )


            struct_sigma += np.outer(
                diff_struct,
                diff_struct
            )


    sample_sigma /= (
        N * M
    )


    struct_sigma /= (
        N * M
    )


    return (
        sample_sigma,
        struct_sigma,
        sigmaz,
        CT_hat,
        CT
    )


# ============================================================
# ESTIMATION ERRORS
#
# No mu - mu_hat errors are computed here.
# ============================================================

def estimate_errors_from_nested_data(
    t,
    U_full,
    all_xT_full,
    N,
    M
):

    (
        sample_sigma,
        struct_sigma,
        sigmaz,
        CT_hat,
        CT
    ) = estimates_from_nested_data(
        t,
        U_full,
        all_xT_full,
        N,
        M
    )


    # --------------------------------------------------------
    # Covariance errors
    # --------------------------------------------------------

    sigma_sample_error = np.linalg.norm(
        sample_sigma - sigmaz,
        ord=2
    )


    sigma_struct_error = np.linalg.norm(
        struct_sigma - sigmaz,
        ord=2
    )


    # --------------------------------------------------------
    # C_T estimation error
    # --------------------------------------------------------

    CT_error = np.linalg.norm(
        CT_hat - CT,
        ord=2
    )


    return (
        sigma_sample_error,
        sigma_struct_error,
        CT_error
    )


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
    seed=100
):

    # ========================================================
    # DETERMINE MAXIMUM DATASET SIZE
    # ========================================================

    if M_list is not None:

        N_max = N_fixed

        M_max = max(
            M_list
        )

        keys = list(
            M_list
        )

    else:

        N_max = max(
            N_list
        )

        M_max = M_fixed

        keys = list(
            N_list
        )


    # ========================================================
    # STORAGE
    # ========================================================

    sig_sample = {
        key: []
        for key in keys
    }


    sig_struct = {
        key: []
        for key in keys
    }


    CT_error = {
        key: []
        for key in keys
    }


    # ========================================================
    # INDEPENDENT RUNS
    # ========================================================

    for k in range(K):

        (
            U_full,
            all_xT_full,
            CT,
            sigmaz
        ) = generate_nested_data(
            t,
            N_max,
            M_max,
            seed=seed + k
        )


        # ----------------------------------------------------
        # Nested subsets
        # ----------------------------------------------------

        for key in keys:

            if M_list is not None:

                N = N_fixed

                M = key

            else:

                N = key

                M = M_fixed


            (
                e_sigma_sample,
                e_sigma_struct,
                e_CT
            ) = estimate_errors_from_nested_data(
                t,
                U_full,
                all_xT_full,
                N,
                M
            )


            sig_sample[key].append(
                e_sigma_sample
            )


            sig_struct[key].append(
                e_sigma_struct
            )


            CT_error[key].append(
                e_CT
            )


    # ========================================================
    # AVERAGE ACROSS RUNS
    # ========================================================

    mean_sigma_sample = [
        np.mean(
            sig_sample[key]
        )
        for key in keys
    ]


    mean_sigma_struct = [
        np.mean(
            sig_struct[key]
        )
        for key in keys
    ]


    mean_CT_error = [
        np.mean(
            CT_error[key]
        )
        for key in keys
    ]


    return (
        mean_sigma_sample,
        mean_sigma_struct,
        mean_CT_error
    )


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

t_fixed = 5

N_fixed = 80

M_fixed = 80


# ------------------------------------------------------------
# Number of repeated experiments
# ------------------------------------------------------------

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
    260
]


# ------------------------------------------------------------
# Number of distinct excitation experiments
# ------------------------------------------------------------

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
    900
]


# ------------------------------------------------------------
# Number of independent datasets
# ------------------------------------------------------------

K_errors = 40


# ============================================================
# ERRORS VS N_r = M
# ============================================================

print(
    "Computing errors versus N_r..."
)


(
    sig_sample_M,
    sig_struct_M,
    CT_error_M
) = average_errors_nested(
    t=t_fixed,
    M_list=M_list,
    N_fixed=N_fixed,
    K=K_errors,
    seed=100
)


# ============================================================
# ERRORS VS N_e = N
# ============================================================

print(
    "Computing errors versus N_e..."
)


(
    sig_sample_N,
    sig_struct_N,
    CT_error_N
) = average_errors_nested(
    t=t_fixed,
    N_list=N_list,
    M_fixed=M_fixed,
    K=K_errors,
    seed=200
)


# ============================================================
# ============================================================
#
# FIGURE 1:
# COVARIANCE ESTIMATION ERROR
#
# ============================================================
# ============================================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(3.48, 1.85)
)


fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.25,
    top=0.76,
    wspace=0.42
)


# ============================================================
# LEFT:
# ERROR VS N_r
# ============================================================

ax[0].loglog(
    M_list,
    sig_sample_M,
    linestyle="-",
    linewidth=0.9,
    label="Sample"
)


ax[0].loglog(
    M_list,
    sig_struct_M,
    linestyle="--",
    linewidth=0.9,
    label="Structure"
)


ax[0].set_xlabel(
    r"$N_r$"
)


ax[0].set_ylabel(
    r"$\|\widehat{\Sigma}_{e_T}"
    r"-\Sigma_{e_T}\|_2$",
    labelpad=1
)


# ============================================================
# RIGHT:
# ERROR VS N_e
# ============================================================

ax[1].loglog(
    N_list,
    sig_sample_N,
    linestyle="-",
    linewidth=0.9
)


ax[1].loglog(
    N_list,
    sig_struct_N,
    linestyle="--",
    linewidth=0.9
)


ax[1].set_xlabel(
    r"$N_e$"
)

ax[1].set_ylabel(
    r"$\|\widehat{\Sigma}_{e_T}"
    r"-\Sigma_{e_T}\|_2$",
    labelpad=1
)


# ============================================================
# IEEE AXIS FORMATTING
# ============================================================

for a in ax:

    style_ieee_axis(a)

    set_plain_log_ticks(a)

    a.xaxis.label.set_size(
        6.5
    )

    a.yaxis.label.set_size(
        6.5
    )

    a.tick_params(
        labelsize=5.8
    )


# Pull ylabel slightly closer
ax[0].yaxis.set_label_coords(
    -0.18,
    0.5
)


# ============================================================
# SHARED LEGEND ABOVE BOTH PANELS
# ============================================================

handles, labels = (
    ax[0].get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.98
    ),

    ncol=2,

    frameon=False,

    fontsize=5.8,

    handlelength=2.0,

    handletextpad=0.4,

    columnspacing=0.9,

    borderaxespad=0.0
)


# ============================================================
# SAVE
# ============================================================

save_ieee_figure(
    fig,
    "covariance_estimation_error"
)


plt.show()


# ============================================================
# ============================================================
#
# FIGURE 2:
# C_T ESTIMATION ERROR
#
# ============================================================
# ============================================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(3.48, 1.85)
)


fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.25,
    top=0.94,
    wspace=0.42
)


# ============================================================
# LEFT:
# C_T ERROR VS N_r
# ============================================================

ax[0].loglog(
    M_list,
    CT_error_M,
    linestyle="-",
    linewidth=0.9
)


ax[0].set_xlabel(
    r"$N_r$"
)


ax[0].set_ylabel(
    r"$\|\widehat{C}_T-C_T\|_2$",
    labelpad=1
)


# ============================================================
# RIGHT:
# C_T ERROR VS N_e
# ============================================================

ax[1].loglog(
    N_list,
    CT_error_N,
    linestyle="-",
    linewidth=0.9
)


ax[1].set_xlabel(
    r"$N_e$"
)

ax[1].set_ylabel(
    r"$\|\widehat{C}_T-C_T\|_2$",
    labelpad=1
)


# ============================================================
# IEEE AXIS FORMATTING
# ============================================================

for a in ax:

    style_ieee_axis(a)

    set_plain_log_ticks(a)

    a.xaxis.label.set_size(
        6.5
    )

    a.yaxis.label.set_size(
        6.5
    )

    a.tick_params(
        labelsize=5.8
    )


ax[0].yaxis.set_label_coords(
    -0.18,
    0.5
)


# ============================================================
# SAVE
# ============================================================

save_ieee_figure(
    fig,
    "CT_estimation_error"
)


plt.show()

