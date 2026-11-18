import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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


# ============================================================
# IEEE PLOT HELPERS
# ============================================================

def style_ieee_axis(ax):
    """Apply compact IEEE publication style."""

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


def set_compact_font_sizes(ax):
    """Same compact label/tick sizes used in the reference code."""

    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)

    ax.tick_params(
        labelsize=5.8
    )


def save_ieee_figure(fig, stem):
    """Save vector PDF and high-resolution PNG."""

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


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

name = "stable"

# Default state/input dimensions
n_default = 4
m = 4

# Default horizon used when another parameter is varied
T_default = 8


# ============================================================
# UNCERTAINTY PARAMETERS
#
# Sigma_0 = sigma_0^2 I
# Sigma_w = sigma^2 I
# ============================================================

sigma0_default = 0.2
sigma_default = 0.3


# ============================================================
# DATA PARAMETERS
# ============================================================

# Number of repeated experiments per training input
M = 100

# Number of calibration points
N_cal = 100

# Number of independent test points
N_test = 2000

# Number of independent datasets / experiments
num_runs = 20


# ============================================================
# CONFORMAL PARAMETERS
# ============================================================

# Fixed miscoverage level for all q plots
delta_q = 0.30

# Miscoverage levels for coverage plot
delta_list = [
    0.30,
    0.20,
    0.10,
    0.05,
    0.01,
]


# ============================================================
# PARAMETER SWEEPS
# ============================================================

# Only THREE horizons in nominal-vs-empirical coverage
T_list_cov = [
    2,
    6,
    10,
]

# q vs T
T_list = list(
    range(1, 20, 2)
)

# q vs n
n_list = [
    2,
    4,
    6,
    8,
    10,
    12,
    16,
    20,
]

# q vs sigma_0
sigma0_list = np.linspace(
    0.0,
    0.5,
    11,
)

# q vs sigma
sigma_list = np.linspace(
    0.0,
    0.5,
    11,
)


# ============================================================
# RANDOM SEED
# ============================================================

BASE_SEED = 1000


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTDIR = "ieee_conformal_plots"

os.makedirs(
    OUTDIR,
    exist_ok=True,
)


# ============================================================
# CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(
    scores,
    delta,
):
    """
    Split-conformal finite-sample quantile

        q_{1-delta}.

    k = ceil(
        (N_cal + 1)(1-delta)
    ).
    """

    N_cal_local = len(scores)

    k = int(
        np.ceil(
            (N_cal_local + 1)
            *
            (1 - delta)
        )
    )

    k = min(
        k,
        N_cal_local,
    )

    return np.sort(scores)[
        k - 1
    ]


# ============================================================
# BUILD SYSTEM
# ============================================================

def build_system(
    n,
    m,
    T,
):
    """
    Build

        A,
        B,
        C_T,
        N_T

    such that

        x_T = A^T x_0 + C_T u + N_T w.
    """

    A, B, CT = get_system(
        name,
        n,
        m,
        T,
        seed=0,
    )

    NT = np.eye(n)

    for k in range(1, T):

        NT = np.hstack((
            NT,
            np.linalg.matrix_power(
                A,
                k,
            ),
        ))

    return (
        A,
        B,
        CT,
        NT,
    )


# ============================================================
# STRUCTURE-BASED ESTIMATE OF C_T
# ============================================================

def estimate_CT(
    n,
    m,
    T,
    sigma0_std,
    sigma_std,
    M,
    rng,
):
    """
    Structure-based estimate

        C_T_hat
        =
        X_bar U^T (U U^T)^dagger.

    Noise models:

        x_0 ~ N(0, sigma_0^2 I),

        w ~ N(0, sigma^2 I).
    """

    (
        A,
        _,
        CT,
        NT,
    ) = build_system(
        n,
        m,
        T,
    )

    # Number of training inputs
    N = m * T + 10

    # --------------------------------------------------------
    # Input matrix
    # --------------------------------------------------------

    U = rng.standard_normal(
        (m * T, N)
    )

    # --------------------------------------------------------
    # Initial-state samples
    #
    # shape = N x M x n
    # --------------------------------------------------------

    x0 = (
        sigma0_std
        *
        rng.standard_normal(
            (N, M, n)
        )
    )

    # --------------------------------------------------------
    # Process-noise samples
    #
    # shape = N x M x nT
    # --------------------------------------------------------

    w = (
        sigma_std
        *
        rng.standard_normal(
            (N, M, n * T)
        )
    )

    AT = np.linalg.matrix_power(
        A,
        T,
    )

    # --------------------------------------------------------
    # A^T x_0
    # --------------------------------------------------------

    x0_term = np.einsum(
        "ab,ijb->ija",
        AT,
        x0,
    )

    # --------------------------------------------------------
    # N_T w
    # --------------------------------------------------------

    w_term = np.einsum(
        "ab,ijb->ija",
        NT,
        w,
    )

    # --------------------------------------------------------
    # C_T u
    #
    # N x 1 x n
    # --------------------------------------------------------

    control_term = (
        CT
        @ U
    ).T[:, None, :]

    # --------------------------------------------------------
    # Terminal-state samples
    # --------------------------------------------------------

    xT = (
        x0_term
        +
        control_term
        +
        w_term
    )

    # --------------------------------------------------------
    # Sample mean for each input
    #
    # X_bar in R^{n x N}
    # --------------------------------------------------------

    X_bar = np.mean(
        xT,
        axis=1,
    ).T

    # --------------------------------------------------------
    # Structure-based estimator
    # --------------------------------------------------------

    CT_hat = (
        X_bar
        @ U.T
        @ np.linalg.pinv(
            U @ U.T
        )
    )

    return (
        A,
        CT,
        NT,
        CT_hat,
    )


# ============================================================
# CALIBRATION SCORES
# ============================================================

def calibration_scores(
    A,
    CT,
    NT,
    CT_hat,
    n,
    m,
    T,
    sigma0_std,
    sigma_std,
    N_cal,
    rng,
):
    """
    Calibration nonconformity scores

        S_i
        =
        ||x_T^(i) - C_T_hat u_i||_2.
    """

    # --------------------------------------------------------
    # Calibration inputs
    # --------------------------------------------------------

    U_cal = rng.standard_normal(
        (m * T, N_cal)
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    x0 = (
        sigma0_std
        *
        rng.standard_normal(
            (N_cal, n)
        )
    )

    # --------------------------------------------------------
    # Process noise
    # --------------------------------------------------------

    w = (
        sigma_std
        *
        rng.standard_normal(
            (N_cal, n * T)
        )
    )

    AT = np.linalg.matrix_power(
        A,
        T,
    )

    # --------------------------------------------------------
    # Error due to estimating C_T
    # --------------------------------------------------------

    model_error = (
        (CT - CT_hat)
        @ U_cal
    ).T

    # --------------------------------------------------------
    # Initial-state uncertainty
    # --------------------------------------------------------

    x0_error = (
        x0
        @ AT.T
    )

    # --------------------------------------------------------
    # Process noise
    # --------------------------------------------------------

    w_error = (
        w
        @ NT.T
    )

    # --------------------------------------------------------
    # Total prediction error
    # --------------------------------------------------------

    error = (
        model_error
        +
        x0_error
        +
        w_error
    )

    return np.linalg.norm(
        error,
        axis=1,
    )


# ============================================================
# TEST SCORES
# ============================================================

def test_scores(
    A,
    CT,
    NT,
    CT_hat,
    n,
    m,
    T,
    sigma0_std,
    sigma_std,
    N_test,
    rng,
):
    """
    Independent test scores used to compute
    empirical coverage.
    """

    U_test = rng.standard_normal(
        (m * T, N_test)
    )

    x0 = (
        sigma0_std
        *
        rng.standard_normal(
            (N_test, n)
        )
    )

    w = (
        sigma_std
        *
        rng.standard_normal(
            (N_test, n * T)
        )
    )

    AT = np.linalg.matrix_power(
        A,
        T,
    )

    model_error = (
        (CT - CT_hat)
        @ U_test
    ).T

    x0_error = (
        x0
        @ AT.T
    )

    w_error = (
        w
        @ NT.T
    )

    error = (
        model_error
        +
        x0_error
        +
        w_error
    )

    return np.linalg.norm(
        error,
        axis=1,
    )


# ============================================================
# COMPUTE q FOR ONE DATASET
# ============================================================

def compute_q_once(
    n,
    m,
    T,
    sigma0_std,
    sigma_std,
    delta,
    seed,
):
    """
    Estimate C_T and compute q_{1-delta}
    from an independent calibration dataset.
    """

    rng = np.random.default_rng(
        seed
    )

    (
        A,
        CT,
        NT,
        CT_hat,
    ) = estimate_CT(
        n=n,
        m=m,
        T=T,
        sigma0_std=sigma0_std,
        sigma_std=sigma_std,
        M=M,
        rng=rng,
    )

    scores = calibration_scores(
        A=A,
        CT=CT,
        NT=NT,
        CT_hat=CT_hat,
        n=n,
        m=m,
        T=T,
        sigma0_std=sigma0_std,
        sigma_std=sigma_std,
        N_cal=N_cal,
        rng=rng,
    )

    return conformal_quantile(
        scores,
        delta,
    )


# ============================================================
# AVERAGE q ACROSS INDEPENDENT DATASETS
# ============================================================

def average_q(
    n,
    m,
    T,
    sigma0_std,
    sigma_std,
    delta,
    num_runs,
):
    """
    Average q_{1-delta} over multiple
    independently generated datasets.
    """

    q_runs = []

    for run in range(num_runs):

        q_runs.append(
            compute_q_once(
                n=n,
                m=m,
                T=T,
                sigma0_std=sigma0_std,
                sigma_std=sigma_std,
                delta=delta,
                seed=BASE_SEED + run,
            )
        )

    return (
        np.mean(q_runs),
        np.std(q_runs),
    )


# ============================================================
# A. NOMINAL VS EMPIRICAL COVERAGE
# ============================================================

coverage_results = {

    T: {
        delta: []
        for delta in delta_list
    }

    for T in T_list_cov
}


for T in T_list_cov:

    print(
        f"Coverage: T={T}"
    )

    for run in range(num_runs):

        rng = np.random.default_rng(
            BASE_SEED
            + 10000 * T
            + run
        )

        (
            A,
            CT,
            NT,
            CT_hat,
        ) = estimate_CT(
            n=n_default,
            m=m,
            T=T,
            sigma0_std=sigma0_default,
            sigma_std=sigma_default,
            M=M,
            rng=rng,
        )

        # ----------------------------------------------------
        # One calibration set per independent experiment
        # ----------------------------------------------------

        scores_cal = calibration_scores(
            A=A,
            CT=CT,
            NT=NT,
            CT_hat=CT_hat,
            n=n_default,
            m=m,
            T=T,
            sigma0_std=sigma0_default,
            sigma_std=sigma_default,
            N_cal=N_cal,
            rng=rng,
        )

        # ----------------------------------------------------
        # Independent test samples
        # ----------------------------------------------------

        scores_test = test_scores(
            A=A,
            CT=CT,
            NT=NT,
            CT_hat=CT_hat,
            n=n_default,
            m=m,
            T=T,
            sigma0_std=sigma0_default,
            sigma_std=sigma_default,
            N_test=N_test,
            rng=rng,
        )

        for delta in delta_list:

            q = conformal_quantile(
                scores_cal,
                delta,
            )

            empirical = np.mean(
                scores_test <= q
            )

            coverage_results[
                T
            ][
                delta
            ].append(
                empirical
            )


# Nominal coverage = 1-delta
nominal = np.asarray([
    1 - delta
    for delta in delta_list
])


# ============================================================
# COVERAGE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(3.48, 2.05),
)


# Enough room above for legend
fig.subplots_adjust(
    left=0.16,
    right=0.98,
    bottom=0.22,
    top=0.78,
)


# ------------------------------------------------------------
# Visual encoding
# ------------------------------------------------------------

coverage_colors = {
    2: "C0",
    6: "C1",
    10: "C2",
}


for T in T_list_cov:

    empirical = np.asarray([
        np.mean(
            coverage_results[T][delta]
        )
        for delta in delta_list
    ])

    ax.plot(
        nominal,
        empirical,
        color=coverage_colors[T],
        linestyle="-",
        linewidth=0.70,
        label=rf"$T={T}$",
    )


# ------------------------------------------------------------
# Ideal empirical = nominal line
# ------------------------------------------------------------

ideal_x = np.linspace(
    np.min(nominal),
    1.0,
    100,
)

ax.plot(
    ideal_x,
    ideal_x,
    color="black",
    linestyle=":",
    linewidth=0.70,
    label="Ideal",
)


# ------------------------------------------------------------
# Axis formatting
# ------------------------------------------------------------

style_ieee_axis(ax)

ax.set_xlabel(
    "Nominal coverage"
)

ax.set_ylabel(
    "Empirical coverage"
)

ax.set_xlim(
    np.min(nominal) - 0.01,
    1.005,
)

ax.set_ylim(
    np.min(nominal) - 0.03,
    1.01,
)

set_compact_font_sizes(ax)


# ------------------------------------------------------------
# Legend ABOVE figure
# ------------------------------------------------------------

ax.legend(
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        1.015,
    ),
    ncol=4,
    frameon=False,
    fontsize=4.7,
    handlelength=1.7,
    handletextpad=0.35,
    columnspacing=0.8,
    labelspacing=0.25,
    borderaxespad=0.0,
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

coverage_stem = os.path.join(
    OUTDIR,
    "coverage_vs_nominal",
)

save_ieee_figure(
    fig,
    coverage_stem,
)

plt.show()


# ============================================================
# B. q_{1-delta} VS T
# ============================================================

q_T = []
q_T_std = []


for T in T_list:

    print(
        f"q vs T: T={T}"
    )

    q_mean, q_std = average_q(
        n=n_default,
        m=m,
        T=T,
        sigma0_std=sigma0_default,
        sigma_std=sigma_default,
        delta=delta_q,
        num_runs=num_runs,
    )

    q_T.append(
        q_mean
    )

    q_T_std.append(
        q_std
    )


# ============================================================
# C. q_{1-delta} VS n
# ============================================================

q_n = []
q_n_std = []


for n_value in n_list:

    print(
        f"q vs n: n={n_value}"
    )

    q_mean, q_std = average_q(
        n=n_value,
        m=m,
        T=T_default,
        sigma0_std=sigma0_default,
        sigma_std=sigma_default,
        delta=delta_q,
        num_runs=num_runs,
    )

    q_n.append(
        q_mean
    )

    q_n_std.append(
        q_std
    )


# ============================================================
# D. q_{1-delta} VS sigma_0
# ============================================================

q_sigma0 = []
q_sigma0_std = []


for sigma0_value in sigma0_list:

    print(
        "q vs sigma_0: "
        f"sigma_0={sigma0_value:.2f}"
    )

    q_mean, q_std = average_q(
        n=n_default,
        m=m,
        T=T_default,
        sigma0_std=sigma0_value,
        sigma_std=sigma_default,
        delta=delta_q,
        num_runs=num_runs,
    )

    q_sigma0.append(
        q_mean
    )

    q_sigma0_std.append(
        q_std
    )


# ============================================================
# E. q_{1-delta} VS sigma
# ============================================================

q_sigma = []
q_sigma_std = []


for sigma_value in sigma_list:

    print(
        "q vs sigma: "
        f"sigma={sigma_value:.2f}"
    )

    q_mean, q_std = average_q(
        n=n_default,
        m=m,
        T=T_default,
        sigma0_std=sigma0_default,
        sigma_std=sigma_value,
        delta=delta_q,
        num_runs=num_runs,
    )

    q_sigma.append(
        q_mean
    )

    q_sigma_std.append(
        q_std
    )


# ============================================================
# F. COMBINED q_{1-delta} FIGURE
#
# Same compact IEEE format as your cost figure.
#
#       q vs T          q vs n
#
#       q vs sigma_0    q vs sigma
#
# ============================================================

fig, ax = plt.subplots(
    2,
    2,
    figsize=(3.48, 3.35),
)


# Compact spacing
fig.subplots_adjust(
    left=0.15,
    right=0.98,
    bottom=0.12,
    top=0.98,
    wspace=0.44,
    hspace=0.48,
)


# ============================================================
# TOP LEFT: q vs T
# ============================================================

ax[0, 0].plot(
    T_list,
    q_T,
    color="C0",
    linestyle="-",
    linewidth=0.70,
)

ax[0, 0].set_xlabel(
    r"$T$"
)

ax[0, 0].set_ylabel(
    r"$q_{1-\delta}$"
)

ax[0, 0].set_xticks(
    T_list
)


# ============================================================
# TOP RIGHT: q vs n
# ============================================================

ax[0, 1].plot(
    n_list,
    q_n,
    color="C0",
    linestyle="-",
    linewidth=0.70,
)

ax[0, 1].set_xlabel(
    r"$n$"
)

ax[0, 1].set_ylabel(
    r"$q_{1-\delta}$"
)

ax[0, 1].set_xticks(
    n_list
)


# ============================================================
# BOTTOM LEFT: q vs sigma_0
# ============================================================

ax[1, 0].plot(
    sigma0_list,
    q_sigma0,
    color="C0",
    linestyle="-",
    linewidth=0.70,
)

ax[1, 0].set_xlabel(
    r"$\sigma_0$"
)

ax[1, 0].set_ylabel(
    r"$q_{1-\delta}$"
)


# ============================================================
# BOTTOM RIGHT: q vs sigma
# ============================================================

ax[1, 1].plot(
    sigma_list,
    q_sigma,
    color="C0",
    linestyle="-",
    linewidth=0.70,
)

ax[1, 1].set_xlabel(
    r"$\sigma$"
)

ax[1, 1].set_ylabel(
    r"$q_{1-\delta}$"
)


# ============================================================
# COMMON AXIS FORMATTING
# ============================================================

for a in ax.flat:

    style_ieee_axis(a)

    set_compact_font_sizes(a)


# Move right-column y labels slightly closer
ax[0, 1].yaxis.set_label_coords(
    -0.14,
    0.5,
)

ax[1, 1].yaxis.set_label_coords(
    -0.14,
    0.5,
)


# ============================================================
# SAVE COMBINED QUANTILE FIGURE
# ============================================================

quantile_stem = os.path.join(
    OUTDIR,
    "conformal_quantiles",
)

save_ieee_figure(
    fig,
    quantile_stem,
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "Finished."
)

print(
    "========================================"
)

print(
    f"Saved: {coverage_stem}.pdf"
)

print(
    f"Saved: {coverage_stem}.png"
)

print(
    f"Saved: {quantile_stem}.pdf"
)

print(
    f"Saved: {quantile_stem}.png"
)

print(
    f"All q plots use delta = {delta_q}"
)