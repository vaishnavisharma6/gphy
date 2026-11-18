import os

import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.linalg import block_diag
from scipy.stats import ncx2, chi2
from scipy.optimize import brentq

from datagen import get_system


# ============================================================
# IEEE / OVERLEAF SETTINGS
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
    "legend.fontsize": 6.4,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,

    "font.weight": "normal",
    "axes.labelweight": "normal",
    "lines.linewidth": 0.65,
    "axes.linewidth": 0.65,

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
})


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
        pad=2
    )

    ax.margins(x=0.02)


def save_ieee_figure(fig, stem):
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


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

n = 4
m = 4

xf = np.array([
    4.0,
    5.0,
    6.0,
    7.0
])


# ------------------------------------------------------------
# Only these three T values
# ------------------------------------------------------------

T_prob_list = [4, 6, 8]


# ------------------------------------------------------------
# Noise
# ------------------------------------------------------------

sigma_1 = 0.2
sigma_2 = 0.3

sigma_0 = (
    sigma_1 ** 2
) * np.eye(n)


# ------------------------------------------------------------
# Control cost
# ------------------------------------------------------------

Q = np.diag([
    2.0,
    3.0,
    4.0,
    5.0
])


# ------------------------------------------------------------
# System
# ------------------------------------------------------------

name = "stable"


# ------------------------------------------------------------
# Terminal radius
# ------------------------------------------------------------

epsilon = 2.4


# ------------------------------------------------------------
# Structure-MLE / conformal parameters
# ------------------------------------------------------------

M = 100

k_est = 10

N_cal = 60


# ------------------------------------------------------------
# Monte-Carlo samples used to estimate probability
# ------------------------------------------------------------

N_mc = 20000


# ------------------------------------------------------------
# Desired probability values
#
# x-axis = 1-delta
#
# For n=4, epsilon=2.4, maximum normalized probability at
# lambda=0 is about 0.782.
#
# Therefore use values below this.
# ------------------------------------------------------------

desired_prob_list = np.array([
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
])


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTDIR = "ieee_probability_plots"

os.makedirs(
    OUTDIR,
    exist_ok=True
)


# ============================================================
# 1. SYSTEM MATRICES
# ============================================================

def build_system_matrices(T):

    A, B, CT = get_system(
        name,
        n,
        m,
        T,
        seed=0
    )

    # --------------------------------------------------------
    # Process-noise covariance over T steps
    # --------------------------------------------------------

    sigma_w = np.kron(
        np.eye(T),
        (sigma_2 ** 2) * np.eye(n)
    )


    # --------------------------------------------------------
    # Noise propagation
    #
    # N_T = [I, A, A^2, ..., A^(T-1)]
    # --------------------------------------------------------

    NT = np.eye(n)

    for k in range(1, T):

        NT = np.hstack((
            NT,
            np.linalg.matrix_power(A, k)
        ))


    AT = np.linalg.matrix_power(
        A,
        T
    )


    # --------------------------------------------------------
    # True terminal covariance
    # --------------------------------------------------------

    Sigma_e = (
        NT
        @ sigma_w
        @ NT.T

        +

        AT
        @ sigma_0
        @ AT.T
    )


    # --------------------------------------------------------
    # Horizon control-cost matrix
    # --------------------------------------------------------

    Q_T = block_diag(
        *([Q] * T)
    )


    return (
        A,
        B,
        CT,
        NT,
        sigma_w,
        Sigma_e,
        Q_T
    )


# ============================================================
# 2. STRUCTURE-AWARE MLE
# ============================================================

def generate_structure_mle(
    T,
    dataset_seed
):

    (
        A,
        _,
        CT,
        NT,
        sigma_w,
        _,
        _
    ) = build_system_matrices(T)


    # Number of experiments
    N = m * T + 10


    rng = np.random.default_rng(
        dataset_seed
    )


    # Inputs
    U = rng.standard_normal(
        (m * T, N)
    )


    X_bar = np.zeros(
        (n, N)
    )


    all_xT = np.zeros(
        (n, N, M)
    )


    AT = np.linalg.matrix_power(
        A,
        T
    )


    # --------------------------------------------------------
    # Generate data
    # --------------------------------------------------------

    for i in range(N):

        for j in range(M):

            x0 = rng.multivariate_normal(
                np.zeros(n),
                sigma_0
            )


            w = rng.multivariate_normal(
                np.zeros(n * T),
                sigma_w
            )


            xT = (
                AT @ x0
                +
                CT @ U[:, i]
                +
                NT @ w
            )


            all_xT[:, i, j] = xT


        X_bar[:, i] = np.mean(
            all_xT[:, i, :],
            axis=1
        )


    # --------------------------------------------------------
    # Estimate C_T
    # --------------------------------------------------------

    CT_hat = (
        X_bar
        @ U.T
        @ np.linalg.pinv(
            U @ U.T
        )
    )


    # --------------------------------------------------------
    # Estimate Sigma_e
    # --------------------------------------------------------

    Sigma_hat = np.zeros(
        (n, n)
    )


    for i in range(N):

        for j in range(M):

            residual = (
                all_xT[:, i, j]
                -
                CT_hat @ U[:, i]
            )


            Sigma_hat += np.outer(
                residual,
                residual
            )


    Sigma_hat /= (
        N * M
    )


    # numerical symmetrization
    Sigma_hat = 0.5 * (
        Sigma_hat
        +
        Sigma_hat.T
    )


    return (
        CT_hat,
        Sigma_hat
    )


# ============================================================
# 3. CONFORMAL CALIBRATION SCORES
# ============================================================

def generate_conformal_scores(
    T,
    CT_hat,
    dataset_seed
):
    """
    Generate calibration scores

        S_i =
        ||x_T^(i) - C_hat_T u_i||_2.

    Scores themselves do not depend on delta.

    Therefore we generate them once and then take different
    quantiles q_{1-delta} for different desired probabilities.
    """

    (
        A,
        _,
        CT,
        NT,
        sigma_w,
        _,
        _
    ) = build_system_matrices(T)


    rng = np.random.default_rng(
        10000 + dataset_seed
    )


    U_cal = rng.standard_normal(
        (m * T, N_cal)
    )


    AT = np.linalg.matrix_power(
        A,
        T
    )


    scores = []


    for i in range(N_cal):

        x0 = rng.multivariate_normal(
            np.zeros(n),
            sigma_0
        )


        w = rng.multivariate_normal(
            np.zeros(n * T),
            sigma_w
        )


        x_true = (
            AT @ x0
            +
            CT @ U_cal[:, i]
            +
            NT @ w
        )


        x_pred = (
            CT_hat
            @ U_cal[:, i]
        )


        score = np.linalg.norm(
            x_true - x_pred,
            ord=2
        )


        scores.append(
            score
        )


    return np.asarray(
        scores
    )


# ============================================================
# 4. CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(
    scores,
    delta
):

    N_local = len(scores)


    k = int(
        np.ceil(
            (N_local + 1)
            *
            (1 - delta)
        )
    )


    k = min(
        k,
        N_local
    )


    return np.sort(
        scores
    )[k - 1]


# ============================================================
# 5. LAMBDA STAR
# ============================================================

def find_lambda_star(
    n,
    epsilon,
    probability_level
):
    """
    Solve

        F_{chi^2_n(lambda)}(epsilon^2)
        =
        probability_level

    for lambda >= 0.
    """

    x = epsilon ** 2


    # --------------------------------------------------------
    # Maximum possible probability occurs for lambda=0
    # --------------------------------------------------------

    p_max = chi2.cdf(
        x,
        df=n
    )


    if probability_level > p_max + 1e-12:

        raise ValueError(
            f"Desired probability {probability_level:.4f} "
            f"is infeasible for n={n}, epsilon={epsilon}. "
            f"Maximum possible probability is {p_max:.4f}."
        )


    if np.isclose(
        probability_level,
        p_max,
        atol=1e-12
    ):

        return 0.0


    def equation(lam):

        return (
            ncx2.cdf(
                x,
                df=n,
                nc=lam
            )
            -
            probability_level
        )


    lo = 0.0
    hi = 1.0


    # Increase upper bracket until CDF falls below target
    while equation(hi) > 0:

        hi *= 2.0


        if hi > 1e12:

            raise ValueError(
                "Could not bracket lambda_star."
            )


    return brentq(
        equation,
        lo,
        hi
    )


# ============================================================
# 6. SOLVE MEC
# ============================================================

def solve_mec(
    CT_used,
    Q_T,
    rhs,
    metric=None
):
    """
    Solve

        min_u  u^T Q_T u

        s.t.

        (C u-x_f)^T metric (C u-x_f) <= rhs.

    If metric=None:

        ||C u-x_f||_2^2 <= rhs.

    Returns
    -------
    cost
    u_star
    """

    if rhs < 0:

        return (
            np.nan,
            None
        )


    if metric is None:

        metric = np.eye(n)


    metric = 0.5 * (
        metric + metric.T
    )


    u = cp.Variable(
        CT_used.shape[1]
    )


    objective = cp.Minimize(
        cp.quad_form(
            u,
            Q_T
        )
    )


    constraint = (
        cp.quad_form(
            CT_used @ u - xf,
            metric
        )
        <= rhs
    )


    problem = cp.Problem(
        objective,
        [constraint]
    )


    try:

        problem.solve(
            solver=cp.CLARABEL,
            verbose=False
        )


    except Exception:

        problem.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-6,
            max_iters=50000
        )


    if (
        u.value is None
        or
        problem.status not in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE
        )
    ):

        return (
            np.nan,
            None
        )


    u_star = np.asarray(
        u.value
    ).reshape(-1)


    cost = float(
        u_star.T
        @ Q_T
        @ u_star
    )


    return (
        cost,
        u_star
    )


# ============================================================
# 7. GENERATE TRUE TERMINAL NOISE
# ============================================================

def generate_terminal_noise_samples(
    T,
    N_mc,
    seed
):
    """
    Generate samples of

        eta_T = A^T x_0 + N_T w.

    Then for a particular controller u,

        e_T =
        C_T u-x_f + eta_T.
    """

    (
        A,
        _,
        CT,
        NT,
        sigma_w,
        Sigma_e,
        _
    ) = build_system_matrices(T)


    rng = np.random.default_rng(
        seed
    )


    x0_samples = rng.multivariate_normal(
        np.zeros(n),
        sigma_0,
        size=N_mc
    )


    w_samples = rng.multivariate_normal(
        np.zeros(n * T),
        sigma_w,
        size=N_mc
    )


    AT = np.linalg.matrix_power(
        A,
        T
    )


    eta_samples = (
        x0_samples @ AT.T
        +
        w_samples @ NT.T
    )


    return (
        eta_samples,
        CT,
        Sigma_e
    )


# ============================================================
# 8. EMPIRICAL PROBABILITIES
# ============================================================

def empirical_terminal_probabilities(
    u_star,
    CT_true,
    Sigma_e_true,
    eta_samples,
    epsilon
):
    """
    Return

    p_norm =
        empirical P(
          e^T Sigma_e^{-1} e <= epsilon^2
        )

    p_abs =
        empirical P(
          ||e||_2^2 <= epsilon^2
        ).
    """

    if u_star is None:

        return (
            np.nan,
            np.nan
        )


    # --------------------------------------------------------
    # Mean error
    # --------------------------------------------------------

    mean_error = (
        CT_true @ u_star
        -
        xf
    )


    # --------------------------------------------------------
    # Actual terminal-error samples
    # --------------------------------------------------------

    errors = (
        eta_samples
        +
        mean_error[None, :]
    )


    # --------------------------------------------------------
    # Normalized event
    # --------------------------------------------------------

    Sigma_inv = np.linalg.pinv(
        Sigma_e_true
    )


    mahal_sq = np.einsum(
        "bi,ij,bj->b",
        errors,
        Sigma_inv,
        errors
    )


    p_norm = np.mean(
        mahal_sq
        <=
        epsilon ** 2
    )


    # --------------------------------------------------------
    # Absolute event
    # --------------------------------------------------------

    euclidean_sq = np.sum(
        errors ** 2,
        axis=1
    )


    p_abs = np.mean(
        euclidean_sq
        <=
        epsilon ** 2
    )


    return (
        p_norm,
        p_abs
    )


# ============================================================
# 9. SAFE MEAN
# ============================================================

def nanmean_or_nan(values):

    values = np.asarray(
        values,
        dtype=float
    )


    if np.all(
        np.isnan(values)
    ):

        return np.nan


    return np.nanmean(
        values
    )


# ============================================================
# MAXIMUM POSSIBLE NORMALIZED PROBABILITY
# ============================================================

p_max_normalized = chi2.cdf(
    epsilon ** 2,
    df=n
)


print(
    "\nMaximum possible normalized probability "
    f"for epsilon={epsilon}: "
    f"{p_max_normalized:.6f}"
)


# ============================================================
# CHECK DESIRED PROBABILITIES
# ============================================================

if np.any(
    desired_prob_list
    >
    p_max_normalized
):

    raise ValueError(
        "\nSome desired probability levels are infeasible.\n"
        f"For n={n}, epsilon={epsilon}, maximum possible "
        f"normalized probability is {p_max_normalized:.4f}.\n"
        "Either lower desired_prob_list or increase epsilon."
    )


# ============================================================
# METHODS
# ============================================================

method_names = [
    "struct_norm",
    "struct_abs",
    "conformal",
    "known_norm",
    "known_abs",
]


method_labels = {
    "struct_norm": "Struct. norm.",
    "struct_abs": "Struct. abs.",
    "conformal": "Conformal",
    "known_norm": "Known norm.",
    "known_abs": "Known abs.",
}


method_colors = {
    "struct_norm": "C0",
    "struct_abs": "C1",
    "conformal": "C2",
    "known_norm": "C3",
    "known_abs": "C4",
}


# ------------------------------------------------------------
# Line style represents T
# ------------------------------------------------------------

T_linestyles = {
    4: "-",
    6: "--",
    8: "-.",
}


# ============================================================
# STORAGE
# ============================================================

prob_norm = {

    T: {

        method: []

        for method in method_names
    }

    for T in T_prob_list
}


prob_abs = {

    T: {

        method: []

        for method in method_names
    }

    for T in T_prob_list
}


# ============================================================
# MAIN EXPERIMENT
# ============================================================

for T in T_prob_list:

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"T = {T}"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # True matrices
    # --------------------------------------------------------

    (
        _,
        _,
        CT,
        _,
        _,
        Sigma_e,
        Q_T
    ) = build_system_matrices(T)


    # --------------------------------------------------------
    # True MC noise samples
    #
    # Generate only once per T and reuse them everywhere.
    # --------------------------------------------------------

    (
        eta_samples,
        CT_true,
        Sigma_e_true
    ) = generate_terminal_noise_samples(

        T=T,

        N_mc=N_mc,

        seed=30000 + T
    )


    # --------------------------------------------------------
    # Generate MLE datasets only once per T.
    #
    # Calibration scores are also generated only once.
    # Different delta values simply select different
    # quantiles from the SAME scores.
    # --------------------------------------------------------

    datasets = []


    for dataset in range(k_est):

        print(
            f"Generating dataset "
            f"{dataset + 1}/{k_est}"
        )


        (
            CT_hat,
            Sigma_hat
        ) = generate_structure_mle(

            T,
            dataset
        )


        calibration_scores = (
            generate_conformal_scores(

                T,

                CT_hat,

                dataset
            )
        )


        datasets.append({
            "dataset": dataset,
            "CT_hat": CT_hat,
            "Sigma_hat": Sigma_hat,
            "scores": calibration_scores,
        })


    # ========================================================
    # SWEEP 1-delta
    # ========================================================

    for desired_prob in desired_prob_list:

        delta_current = (
            1.0
            -
            desired_prob
        )


        print(
            "\n"
            f"T = {T}, "
            f"1-delta = {desired_prob:.2f}"
        )


        # ----------------------------------------------------
        # Normalized deterministic threshold
        #
        # F_ncx2(
        #     epsilon^2;
        #     n,
        #     lambda_star
        # ) = 1-delta
        # ----------------------------------------------------

        lambda_star_current = (
            find_lambda_star(

                n=n,

                epsilon=epsilon,

                probability_level=desired_prob
            )
        )


        print(
            f"lambda_star = "
            f"{lambda_star_current:.6f}"
        )


        # ====================================================
        # KNOWN DYNAMICS — NORMALIZED
        # ====================================================

        (
            _,
            u_known_norm
        ) = solve_mec(

            CT_used=CT,

            Q_T=Q_T,

            rhs=lambda_star_current,

            metric=np.linalg.pinv(
                Sigma_e
            )
        )


        (
            p_norm,
            p_abs
        ) = empirical_terminal_probabilities(

            u_star=u_known_norm,

            CT_true=CT_true,

            Sigma_e_true=Sigma_e_true,

            eta_samples=eta_samples,

            epsilon=epsilon
        )


        prob_norm[T][
            "known_norm"
        ].append(
            p_norm
        )


        prob_abs[T][
            "known_norm"
        ].append(
            p_abs
        )


        # ====================================================
        # KNOWN DYNAMICS — ABSOLUTE
        #
        # This preserves your original absolute formulation:
        #
        # ||C_T u-x_f||^2 <= lambda_star.
        #
        # Note: lambda_star is exact for the normalized
        # Gaussian event, not for a general Euclidean event.
        # The empirical plot shows the actual coverage attained.
        # ====================================================

        (
            _,
            u_known_abs
        ) = solve_mec(

            CT_used=CT,

            Q_T=Q_T,

            rhs=lambda_star_current,

            metric=None
        )


        (
            p_norm,
            p_abs
        ) = empirical_terminal_probabilities(

            u_star=u_known_abs,

            CT_true=CT_true,

            Sigma_e_true=Sigma_e_true,

            eta_samples=eta_samples,

            epsilon=epsilon
        )


        prob_norm[T][
            "known_abs"
        ].append(
            p_norm
        )


        prob_abs[T][
            "known_abs"
        ].append(
            p_abs
        )


        # ====================================================
        # DATA-DRIVEN TEMPORARY STORAGE
        # ====================================================

        temp_norm = {
            "struct_norm": [],
            "struct_abs": [],
            "conformal": [],
        }


        temp_abs = {
            "struct_norm": [],
            "struct_abs": [],
            "conformal": [],
        }


        # ====================================================
        # LOOP OVER ESTIMATED DATASETS
        # ====================================================

        for data in datasets:

            CT_hat = data[
                "CT_hat"
            ]

            Sigma_hat = data[
                "Sigma_hat"
            ]

            scores = data[
                "scores"
            ]


            # =================================================
            # STRUCTURE MLE — NORMALIZED
            # =================================================

            (
                _,
                u_struct_norm
            ) = solve_mec(

                CT_used=CT_hat,

                Q_T=Q_T,

                rhs=lambda_star_current,

                metric=np.linalg.pinv(
                    Sigma_hat
                )
            )


            (
                p_norm,
                p_abs
            ) = empirical_terminal_probabilities(

                u_star=u_struct_norm,

                CT_true=CT_true,

                Sigma_e_true=Sigma_e_true,

                eta_samples=eta_samples,

                epsilon=epsilon
            )


            temp_norm[
                "struct_norm"
            ].append(
                p_norm
            )


            temp_abs[
                "struct_norm"
            ].append(
                p_abs
            )


            # =================================================
            # STRUCTURE MLE — ABSOLUTE
            # =================================================

            (
                _,
                u_struct_abs
            ) = solve_mec(

                CT_used=CT_hat,

                Q_T=Q_T,

                rhs=lambda_star_current,

                metric=None
            )


            (
                p_norm,
                p_abs
            ) = empirical_terminal_probabilities(

                u_star=u_struct_abs,

                CT_true=CT_true,

                Sigma_e_true=Sigma_e_true,

                eta_samples=eta_samples,

                epsilon=epsilon
            )


            temp_norm[
                "struct_abs"
            ].append(
                p_norm
            )


            temp_abs[
                "struct_abs"
            ].append(
                p_abs
            )


            # =================================================
            # CONFORMAL
            # =================================================

            q_current = (
                conformal_quantile(

                    scores,

                    delta_current
                )
            )


            # -------------------------------------------------
            # ||C_hat u-x_f|| <= epsilon-q
            # -------------------------------------------------

            if epsilon >= q_current:

                rhs_conf = (
                    epsilon
                    -
                    q_current
                ) ** 2

            else:

                rhs_conf = -1.0


            (
                _,
                u_conf
            ) = solve_mec(

                CT_used=CT_hat,

                Q_T=Q_T,

                rhs=rhs_conf,

                metric=None
            )


            (
                p_norm,
                p_abs
            ) = empirical_terminal_probabilities(

                u_star=u_conf,

                CT_true=CT_true,

                Sigma_e_true=Sigma_e_true,

                eta_samples=eta_samples,

                epsilon=epsilon
            )


            temp_norm[
                "conformal"
            ].append(
                p_norm
            )


            temp_abs[
                "conformal"
            ].append(
                p_abs
            )


        # ====================================================
        # AVERAGE DATA-DRIVEN RESULTS OVER k_est DATASETS
        # ====================================================

        for method in [
            "struct_norm",
            "struct_abs",
            "conformal"
        ]:

            prob_norm[T][method].append(

                nanmean_or_nan(
                    temp_norm[
                        method
                    ]
                )
            )


            prob_abs[T][method].append(

                nanmean_or_nan(
                    temp_abs[
                        method
                    ]
                )
            )


        # ----------------------------------------------------
        # Print current results
        # ----------------------------------------------------

        print(
            "Normalized event:"
        )


        for method in method_names:

            print(
                f"  {method:15s} "
                f"{prob_norm[T][method][-1]:.4f}"
            )


        print(
            "Absolute event:"
        )


        for method in method_names:

            print(
                f"  {method:15s} "
                f"{prob_abs[T][method][-1]:.4f}"
            )


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(3.48, 2.25)
)


fig.subplots_adjust(
    left=0.13,
    right=0.98,
    bottom=0.22,
    top=0.61,
    wspace=0.42
)


# ============================================================
# CURVES
#
# color      = method
# line style = T
# ============================================================

for T in T_prob_list:

    ls = T_linestyles[T]


    for method in method_names:

        # ----------------------------------------------------
        # Normalized probability
        # ----------------------------------------------------

        ax[0].plot(
            desired_prob_list,
            prob_norm[T][method],

            color=method_colors[
                method
            ],

            linestyle=ls,

            linewidth=0.75
        )


        # ----------------------------------------------------
        # Absolute probability
        # ----------------------------------------------------

        ax[1].plot(
            desired_prob_list,
            prob_abs[T][method],

            color=method_colors[
                method
            ],

            linestyle=ls,

            linewidth=0.75
        )


# ============================================================
# REFERENCE LINE y=x
# ============================================================

x_ref = np.linspace(
    desired_prob_list.min(),
    desired_prob_list.max(),
    200
)


for a in ax:

    a.plot(
        x_ref,
        x_ref,
        color="black",
        linestyle=":",
        linewidth=0.90
    )


# ============================================================
# AXES
# ============================================================

for a in ax:

    style_ieee_axis(a)


    a.set_xlabel(
        r"Desired probability $1-\delta$"
    )


    a.set_ylabel(
        "Empirical probability"
    )


    # --------------------------------------------------------
    # Show a little space around the desired probability range
    # --------------------------------------------------------

    a.set_xlim(
        desired_prob_list.min() - 0.01,
        desired_prob_list.max() + 0.01
    )


    # --------------------------------------------------------
    # Since both quantities are probabilities, a focused
    # range makes y=x much easier to interpret.
    # --------------------------------------------------------

    a.set_ylim(
        0.40,
        1.00
    )


    a.set_xticks(
        desired_prob_list
    )


    a.xaxis.label.set_size(
        6.5
    )


    a.yaxis.label.set_size(
        6.5
    )


    a.tick_params(
        labelsize=5.5
    )


# ============================================================
# TITLES
# ============================================================

ax[0].set_title(
    r"$e^\top\Sigma_e^{-1}e"
    r"\leq\epsilon^2$",
    fontsize=6.5
)


ax[1].set_title(
    r"$\|e\|_2^2"
    r"\leq\epsilon^2$",
    fontsize=6.5
)


# ============================================================
# LEGENDS
# ============================================================

method_handles = [

    Line2D(
        [0],
        [0],
        color=method_colors[
            "struct_norm"
        ],
        lw=0.75,
        linestyle="-",
        label="Struct. norm."
    ),

    Line2D(
        [0],
        [0],
        color=method_colors[
            "struct_abs"
        ],
        lw=0.75,
        linestyle="-",
        label="Struct. abs."
    ),

    Line2D(
        [0],
        [0],
        color=method_colors[
            "conformal"
        ],
        lw=0.75,
        linestyle="-",
        label="Conformal"
    ),

    Line2D(
        [0],
        [0],
        color=method_colors[
            "known_norm"
        ],
        lw=0.75,
        linestyle="-",
        label="Known norm."
    ),

    Line2D(
        [0],
        [0],
        color=method_colors[
            "known_abs"
        ],
        lw=0.75,
        linestyle="-",
        label="Known abs."
    ),
]


T_handles = [

    Line2D(
        [0],
        [0],
        color="black",
        lw=0.75,
        linestyle=T_linestyles[T],
        label=rf"$T={T}$"
    )

    for T in T_prob_list
]


reference_handle = Line2D(
    [0],
    [0],
    color="black",
    lw=0.90,
    linestyle=":",
    label=r"$y=x$"
)


# ------------------------------------------------------------
# Methods
# ------------------------------------------------------------

fig.legend(
    handles=method_handles,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.995
    ),

    ncol=3,

    frameon=False,

    fontsize=4.2,

    handlelength=1.5,

    handletextpad=0.3,

    columnspacing=0.65
)


# ------------------------------------------------------------
# T values + y=x
# ------------------------------------------------------------

fig.legend(
    handles=(
        T_handles
        +
        [reference_handle]
    ),

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.875
    ),

    ncol=4,

    frameon=False,

    fontsize=4.3,

    handlelength=1.6,

    handletextpad=0.3,

    columnspacing=0.75
)


# ============================================================
# SAVE
# ============================================================

stem = os.path.join(
    OUTDIR,
    f"empirical_vs_desired_probability_{name}"
)


save_ieee_figure(
    fig,
    stem
)


plt.show()


print(
    f"\nSaved: {stem}.pdf"
)

print(
    f"Saved: {stem}.png"
)


# ============================================================
# FINAL NUMERICAL TABLES
# ============================================================

print(
    "\n"
    + "=" * 85
)

print(
    "FINAL EMPIRICAL PROBABILITY RESULTS"
)

print(
    "=" * 85
)


for T in T_prob_list:

    print(
        "\n"
        + "=" * 85
    )

    print(
        f"T = {T}"
    )

    print(
        "=" * 85
    )


    for j, desired_prob in enumerate(
        desired_prob_list
    ):

        print(
            f"\nDesired probability "
            f"1-delta = {desired_prob:.2f}"
        )


        print(
            f"{'Method':15s}"
            f"{'Normalized':>15s}"
            f"{'Absolute':>15s}"
        )


        print(
            "-" * 45
        )


        for method in method_names:

            p1 = prob_norm[
                T
            ][
                method
            ][j]


            p2 = prob_abs[
                T
            ][
                method
            ][j]


            print(
                f"{method:15s}"
                f"{p1:15.4f}"
                f"{p2:15.4f}"
            )