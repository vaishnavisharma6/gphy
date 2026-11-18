import numpy as np
import cvxpy as cp

from scipy.linalg import block_diag
from scipy.stats import ncx2
from scipy.optimize import brentq

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from datagen import get_system


# ============================================================
# IEEE / OVERLEAF PLOTTING SETTINGS
# ============================================================

mpl.rcParams.update({
    "text.usetex": False,

    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif"
    ],

    "mathtext.fontset": "stix",

    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 5.5,
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
# FIGURE HELPERS
# ============================================================

def style_ieee_axis(ax):

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


def shade_infeasible_regions(
    ax,
    x_vals,
    infeasible
):
    """
    Shade contiguous regions in which conformal prediction
    is infeasible, i.e. q >= epsilon.
    """

    x_vals = np.asarray(x_vals)

    infeasible = np.asarray(
        infeasible,
        dtype=bool
    )

    if not np.any(infeasible):
        return

    # Boundaries halfway between sampled sigma values
    edges = np.empty(
        len(x_vals) + 1
    )

    edges[1:-1] = (
        0.5
        * (
            x_vals[:-1]
            + x_vals[1:]
        )
    )

    edges[0] = (
        x_vals[0]
        - 0.5
        * (
            x_vals[1]
            - x_vals[0]
        )
    )

    edges[-1] = (
        x_vals[-1]
        + 0.5
        * (
            x_vals[-1]
            - x_vals[-2]
        )
    )

    # Needed because deviation panel uses log x-axis
    edges[0] = max(
        edges[0],
        1e-8
    )

    i = 0

    while i < len(infeasible):

        if not infeasible[i]:
            i += 1
            continue

        start = i

        while (
            i + 1 < len(infeasible)
            and infeasible[i + 1]
        ):
            i += 1

        end = i

        ax.axvspan(
            edges[start],
            edges[end + 1],
            facecolor="0.82",
            alpha=0.45,
            edgecolor="none",
            zorder=0
        )

        i += 1


# ============================================================
# CONFORMAL FUNCTIONS
# ============================================================

def conformal_quantile(
    scores,
    delta
):

    N_cal = len(scores)

    k = int(
        np.ceil(
            (N_cal + 1)
            * (1 - delta)
        )
    )

    k = min(
        k,
        N_cal
    )

    return np.sort(scores)[k - 1]


def get_scores(
    U_cal,
    N_cal,
    CT_hat,
    A,
    NT,
    CT,
    T,
    w_cal_train,
    x_cal_train
):

    scores = []

    A_T = np.linalg.matrix_power(
        A,
        T
    )

    for i in range(N_cal):

        w = w_cal_train[i]
        x0 = x_cal_train[i]

        x_true = (
            A_T @ x0
            + CT @ U_cal[:, i]
            + NT @ w
        )

        x_pred = (
            CT_hat
            @ U_cal[:, i]
        )

        scores.append(
            np.linalg.norm(
                x_true
                - x_pred
            )
        )

    return np.asarray(scores)


def find_lambda_star(
    n,
    epsilon,
    probability
):
    """
    Solve

    F_{chi_n^2(lambda)}(epsilon^2)
        = probability

    for lambda >= 0.
    """

    x = epsilon**2

    def equation(lam):

        return (
            ncx2.cdf(
                x,
                df=n,
                nc=lam
            )
            - probability
        )

    return brentq(
        equation,
        0.0,
        1e11
    )


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

np.random.seed(0)

n = 4
m = 4

T = 3

xf = np.array(
    [4.0, 5.0, 6.0, 7.0]
)

delta = 0.5

epsilon = 3.0

Q = np.diag(
    [2.0, 3.0, 4.0, 5.0]
)

name = "stable"

M = 100

N = m*T + 10

N_cal = 60

N_test = 5000


sigma_vals = np.linspace(
    0.3,
    7.0,
    40
)


# ============================================================
# SYSTEM MATRICES
# ============================================================

A, B, CT = get_system(
    name,
    n,
    m,
    T,
    seed=0
)


A_T = np.linalg.matrix_power(
    A,
    T
)


# ------------------------------------------------------------
# Noise controllability-type matrix N_T
# ------------------------------------------------------------

N_T = np.eye(n)

for i in range(1, T):

    N_T = np.hstack(
        (
            N_T,
            np.linalg.matrix_power(A, i)
        )
    )


# ------------------------------------------------------------
# Block control-cost matrix
# ------------------------------------------------------------

Q_tilde = block_diag(
    *([Q] * T)
)


I_T = np.eye(T)


# ------------------------------------------------------------
# Noncentral chi-square threshold
# ------------------------------------------------------------

lambda_star = find_lambda_star(
    n,
    epsilon,
    1 - delta
)


# ============================================================
# COMMON TRAINING INPUTS
# ============================================================

U = np.random.randn(
    m*T,
    N
)

U_cal = np.random.randn(
    m*T,
    N_cal
)


# ============================================================
# OPTIMIZATION PROBLEMS
# ============================================================

def solve_main_methods(
    CT_hat,
    struct_sigma,
    sigma_z
):
    """
    Solve four MEC formulations:

    1. Structure MLE -- normalized
    2. Structure MLE -- absolute
    3. Known dynamics -- normalized
    4. Known dynamics -- absolute

    Conformal is handled separately.
    """

    # ========================================================
    # VARIABLES
    # ========================================================

    u_struct = cp.Variable(
        m*T
    )

    u_struct_abs = cp.Variable(
        m*T
    )

    u_known = cp.Variable(
        m*T
    )

    u_known_abs = cp.Variable(
        m*T
    )


    # ========================================================
    # 1. STRUCTURE MLE -- NORMALIZED
    #
    # min u' Q u
    #
    # s.t.
    # (C_hat u - xf)' Sigma_hat^{-1}
    # (C_hat u - xf)
    # <= lambda_star
    # ========================================================

    objective_struct = cp.Minimize(
        cp.quad_form(
            u_struct,
            Q_tilde
        )
    )

    constraint_struct = (
        cp.quad_form(
            CT_hat @ u_struct - xf,
            np.linalg.pinv(struct_sigma)
        )
        <= lambda_star
    )

    problem_struct = cp.Problem(
        objective_struct,
        [constraint_struct]
    )

    problem_struct.solve(
        solver=cp.SCS,
        verbose=False
    )


    # ========================================================
    # 2. STRUCTURE MLE -- ABSOLUTE
    #
    # min u' Q u
    #
    # s.t.
    # ||C_hat u - xf||_2^2
    # <= epsilon^2
    # ========================================================

    objective_struct_abs = cp.Minimize(
        cp.quad_form(
            u_struct_abs,
            Q_tilde
        )
    )

    constraint_struct_abs = (
        cp.sum_squares(
            CT_hat @ u_struct_abs
            - xf
        )
        <= epsilon**2
    )

    problem_struct_abs = cp.Problem(
        objective_struct_abs,
        [constraint_struct_abs]
    )

    problem_struct_abs.solve(
        solver=cp.SCS,
        verbose=False
    )


    # ========================================================
    # 3. KNOWN DYNAMICS -- NORMALIZED
    #
    # min u' Q u
    #
    # s.t.
    # (C u - xf)' Sigma_z^{-1}
    # (C u - xf)
    # <= lambda_star
    # ========================================================

    objective_known = cp.Minimize(
        cp.quad_form(
            u_known,
            Q_tilde
        )
    )

    constraint_known = (
        cp.quad_form(
            CT @ u_known - xf,
            np.linalg.pinv(sigma_z)
        )
        <= lambda_star
    )

    problem_known = cp.Problem(
        objective_known,
        [constraint_known]
    )

    problem_known.solve(
        solver=cp.SCS,
        verbose=False
    )


    # ========================================================
    # 4. KNOWN DYNAMICS -- ABSOLUTE
    #
    # min u' Q u
    #
    # s.t.
    # ||C u - xf||_2^2
    # <= epsilon^2
    # ========================================================

    objective_known_abs = cp.Minimize(
        cp.quad_form(
            u_known_abs,
            Q_tilde
        )
    )

    constraint_known_abs = (
        cp.sum_squares(
            CT @ u_known_abs
            - xf
        )
        <= epsilon**2
    )

    problem_known_abs = cp.Problem(
        objective_known_abs,
        [constraint_known_abs]
    )

    problem_known_abs.solve(
        solver=cp.SCS,
        verbose=False
    )


    # ========================================================
    # CHECK SOLUTIONS
    # ========================================================

    if u_struct.value is None:
        raise RuntimeError(
            "Structure normalized problem failed."
        )

    if u_struct_abs.value is None:
        raise RuntimeError(
            "Structure absolute problem failed."
        )

    if u_known.value is None:
        raise RuntimeError(
            "Known normalized problem failed."
        )

    if u_known_abs.value is None:
        raise RuntimeError(
            "Known absolute problem failed."
        )


    return (
        np.asarray(u_struct.value).reshape(-1),
        np.asarray(u_struct_abs.value).reshape(-1),
        np.asarray(u_known.value).reshape(-1),
        np.asarray(u_known_abs.value).reshape(-1)
    )


# ============================================================
# CONFORMAL PROBLEM
# ============================================================

def solve_conformal(
    CT_hat,
    q
):
    """
    Solve

    min u' Q u

    s.t.
    ||C_hat u - xf||_2
        <= epsilon - q.

    If q >= epsilon, the problem is marked infeasible
    and no optimization problem is solved.
    """

    if q >= epsilon:
        return None

    radius = (
        epsilon
        - q
    )

    u_conf = cp.Variable(
        m*T
    )

    objective = cp.Minimize(
        cp.quad_form(
            u_conf,
            Q_tilde
        )
    )

    constraint = (
        cp.sum_squares(
            CT_hat @ u_conf
            - xf
        )
        <= radius**2
    )

    problem = cp.Problem(
        objective,
        [constraint]
    )

    problem.solve(
        solver=cp.SCS,
        verbose=False
    )

    if u_conf.value is None:
        return None

    return np.asarray(
        u_conf.value
    ).reshape(-1)


# ============================================================
# STRUCTURE-MLE ESTIMATION
# ============================================================

def estimate_structure(
    x_train,
    w_train
):
    """
    Generate terminal-state training data and estimate

        C_T

    using the structured least-squares / MLE estimator.

    Returns:
        CT_hat
        struct_sigma
    """

    X_bar = np.zeros(
        (n, N)
    )

    all_xT = np.zeros(
        (n, N, M)
    )


    # --------------------------------------------------------
    # Generate repeated experiments
    # --------------------------------------------------------

    for i in range(N):

        for j in range(M):

            xT = (
                A_T @ x_train[i, j]
                + CT @ U[:, i]
                + N_T @ w_train[i, j]
            )

            all_xT[:, i, j] = xT


        X_bar[:, i] = np.mean(
            all_xT[:, i, :],
            axis=1
        )


    # --------------------------------------------------------
    # Structured estimator of C_T
    # --------------------------------------------------------

    CT_hat = (
        X_bar
        @ U.T
        @ np.linalg.pinv(
            U @ U.T
        )
    )


    # --------------------------------------------------------
    # Structured covariance estimate
    # --------------------------------------------------------

    struct_sigma = np.zeros(
        (n, n)
    )


    for i in range(N):

        for j in range(M):

            residual = (
                all_xT[:, i, j]
                - CT_hat @ U[:, i]
            )

            struct_sigma += np.outer(
                residual,
                residual
            )


    struct_sigma /= (
        N * M
    )


    return (
        CT_hat,
        struct_sigma
    )


# ============================================================
# COST
# ============================================================

def control_cost(u):

    return float(
        u.T
        @ Q_tilde
        @ u
    )


# ============================================================
# PLOT FUNCTION
# ============================================================

def make_ieee_figure(
    sigma_vals,

    cost_struct,
    cost_struct_abs,
    cost_conf,
    cost_known,
    cost_known_abs,

    dev_struct,
    dev_struct_abs,
    dev_conf,
    dev_known,
    dev_known_abs,

    conf_infeasible,

    xlabel,
    filename
):

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(3.48, 2.10)
    )


    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.22,
        top=0.65,
        wspace=0.42
    )


    # ========================================================
    # LEFT: OPTIMAL COST
    # ========================================================

    # Structure normalized
    ax[0].plot(
        sigma_vals,
        cost_struct,
        linestyle="-",
        linewidth=0.9,
        label="MLE normalised norm.",
        zorder=3
    )


    # Structure absolute
    ax[0].plot(
        sigma_vals,
        cost_struct_abs,
        linestyle="-",
        linewidth=0.9,
        label="MLE absolute norm.",
        zorder=3
    )


    # Conformal
    ax[0].plot(
        sigma_vals,
        cost_conf,
        linestyle="-",
        linewidth=0.9,
        label="Conformal",
        zorder=3
    )


    # Known normalized -- dotted
    ax[0].plot(
        sigma_vals,
        cost_known,
        linestyle=":",
        linewidth=0.9,
        label="Known normalised norm.",
        zorder=3
    )


    # Known absolute -- dotted
    ax[0].plot(
        sigma_vals,
        cost_known_abs,
        linestyle=":",
        linewidth=0.9,
        label="Known absolute norm.",
        zorder=3
    )


    shade_infeasible_regions(
        ax[0],
        sigma_vals,
        conf_infeasible
    )


    # ========================================================
    # RIGHT: DEVIATION
    # ========================================================

    ax[1].loglog(
        sigma_vals,
        dev_struct,
        linestyle="-",
        linewidth=0.9,
        zorder=3
    )


    ax[1].loglog(
        sigma_vals,
        dev_struct_abs,
        linestyle="-",
        linewidth=0.9,
        zorder=3
    )


    ax[1].loglog(
        sigma_vals,
        dev_conf,
        linestyle="-",
        linewidth=0.9,
        zorder=3
    )


    ax[1].loglog(
        sigma_vals,
        dev_known,
        linestyle=":",
        linewidth=0.9,
        zorder=3
    )


    ax[1].loglog(
        sigma_vals,
        dev_known_abs,
        linestyle=":",
        linewidth=0.9,
        zorder=3
    )


    shade_infeasible_regions(
        ax[1],
        sigma_vals,
        conf_infeasible
    )


    # ========================================================
    # AXIS STYLE
    # ========================================================

    for a in ax:

        style_ieee_axis(a)

        a.xaxis.label.set_size(
            6.5
        )

        a.yaxis.label.set_size(
            6.5
        )

        a.tick_params(
            labelsize=5.8
        )


    ax[0].set_xlabel(
        xlabel
    )

    ax[0].set_ylabel(
        r"$J_T^*$"
    )


    ax[1].set_xlabel(
        xlabel
    )

    ax[1].set_ylabel(
        "Mean deviation",
        labelpad=0
    )


    ax[1].yaxis.set_label_coords(
        -0.13,
        0.5
    )


    # ========================================================
    # LOG Y-AXIS:
    # plain numbers instead of 10^k where possible
    # ========================================================

    ax[1].yaxis.set_major_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=(1.0,)
        )
    )

    ax[1].yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, pos: f"{y:g}"
        )
    )

    ax[1].yaxis.set_minor_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=np.arange(2, 10) * 0.1
        )
    )

    ax[1].yaxis.set_minor_formatter(
        mpl.ticker.NullFormatter()
    )


    # ========================================================
    # LOG X-AXIS:
    # plain labels 0.5, 1, 2, 5, ...
    # ========================================================

    ax[1].xaxis.set_major_locator(
        mpl.ticker.LogLocator(
            base=10,
            subs=(1.0, 2.0, 5.0)
        )
    )

    ax[1].xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: f"{x:g}"
        )
    )

    ax[1].xaxis.set_minor_formatter(
        mpl.ticker.NullFormatter()
    )


    # ========================================================
    # SHARED LEGEND
    # ========================================================

    handles, labels = (
        ax[0].get_legend_handles_labels()
    )


    infeasible_patch = Patch(
        facecolor="0.82",
        alpha=0.45,
        edgecolor="none",
        label="CP infeasible"
    )


    handles.append(
        infeasible_patch
    )

    labels.append(
        "CP infeasible"
    )


    fig.legend(
        handles,
        labels,

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            0.985
        ),

        ncol=3,

        frameon=False,

        fontsize=5.0,

        handlelength=2.0,

        handletextpad=0.35,

        columnspacing=0.65,

        borderaxespad=0.0
    )


    save_ieee_figure(
        fig,
        filename
    )

    plt.show()


# ============================================================
# ============================================================
#
# EXPERIMENT 1:
# VARY PROCESS-NOISE STANDARD DEVIATION sigma
#
# ============================================================
# ============================================================


# ------------------------------------------------------------
# Fixed initial-state standard deviation
# ------------------------------------------------------------

sigma_0_fixed = 0.2


sigma_0 = (
    sigma_0_fixed**2
    * np.eye(n)
)


# ============================================================
# COMMON RANDOM NUMBERS
# ============================================================

z_x_train = np.random.randn(
    N,
    M,
    n
)

z_x_cal = np.random.randn(
    N_cal,
    n
)

z_w_train = np.random.randn(
    N,
    M,
    n*T
)

z_w_cal = np.random.randn(
    N_cal,
    n*T
)

z_x_test = np.random.randn(
    N_test,
    n
)

z_w_test = np.random.randn(
    N_test,
    n*T
)


# ============================================================
# STORAGE
# ============================================================

cost_struct = []
cost_struct_abs = []
cost_conf = []
cost_known = []
cost_known_abs = []


dev_struct = []
dev_struct_abs = []
dev_conf = []
dev_known = []
dev_known_abs = []


conf_infeasible = []

q_sigma = []


# ============================================================
# PROCESS-NOISE SWEEP
# ============================================================

for sigma in sigma_vals:

    # ========================================================
    # TRUE COVARIANCE
    # ========================================================

    sigma_w = np.kron(
        I_T,
        sigma**2
        * np.eye(n)
    )


    sigma_z = (
        N_T
        @ sigma_w
        @ N_T.T

        + A_T
        @ sigma_0
        @ A_T.T
    )


    tr_sigma_z = np.trace(
        sigma_z
    )


    # ========================================================
    # TRAINING / CALIBRATION DATA
    #
    # x0 uncertainty is fixed
    # process noise varies with sigma
    # ========================================================

    x_train = (
        sigma_0_fixed
        * z_x_train
    )


    x_cal_train = (
        sigma_0_fixed
        * z_x_cal
    )


    w_train = (
        sigma
        * z_w_train
    )


    w_cal_train = (
        sigma
        * z_w_cal
    )


    # ========================================================
    # STRUCTURE MLE
    # ========================================================

    (
        CT_hat,
        struct_sigma
    ) = estimate_structure(
        x_train,
        w_train
    )


    # ========================================================
    # CONFORMAL QUANTILE
    # ========================================================

    scores = get_scores(
        U_cal,
        N_cal,
        CT_hat,
        A,
        N_T,
        CT,
        T,
        w_cal_train,
        x_cal_train
    )


    q = conformal_quantile(
        scores,
        delta
    )


    q_sigma.append(
        q
    )


    cp_infeasible = (
        q >= epsilon
    )


    conf_infeasible.append(
        cp_infeasible
    )


    # ========================================================
    # SOLVE FOUR MEC METHODS
    # ========================================================

    (
        u_struct,
        u_struct_abs,
        u_known,
        u_known_abs
    ) = solve_main_methods(
        CT_hat,
        struct_sigma,
        sigma_z
    )


    # ========================================================
    # CONFORMAL
    # ========================================================

    u_conf = solve_conformal(
        CT_hat,
        q
    )


    # ========================================================
    # COSTS
    # ========================================================

    cost_struct.append(
        control_cost(
            u_struct
        )
    )


    cost_struct_abs.append(
        control_cost(
            u_struct_abs
        )
    )


    cost_known.append(
        control_cost(
            u_known
        )
    )


    cost_known_abs.append(
        control_cost(
            u_known_abs
        )
    )


    if u_conf is None:

        cost_conf.append(
            np.nan
        )

    else:

        cost_conf.append(
            control_cost(
                u_conf
            )
        )


    # ========================================================
    # MONTE-CARLO EVALUATION
    # ========================================================

    errors_struct = []
    errors_struct_abs = []
    errors_known = []
    errors_known_abs = []
    errors_conf = []
    tr_struct = np.trace(struct_sigma)

    for i in range(N_test):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # process-noise experiment:
        #
        # x0 uses fixed sigma_0
        # w uses varying sigma
        # ----------------------------------------------------

        x0 = (
            sigma_0_fixed
            * z_x_test[i]
        )


        w = (
            sigma
            * z_w_test[i]
        )


        # ----------------------------------------------------
        # Structure normalized
        # ----------------------------------------------------

        xT_struct = (
            A_T @ x0
            + CT @ u_struct
            + N_T @ w
        )


        # ----------------------------------------------------
        # Structure absolute
        # ----------------------------------------------------

        xT_struct_abs = (
            A_T @ x0
            + CT @ u_struct_abs
            + N_T @ w
        )


        # ----------------------------------------------------
        # Known normalized
        # ----------------------------------------------------

        xT_known = (
            A_T @ x0
            + CT @ u_known
            + N_T @ w
        )


        # ----------------------------------------------------
        # Known absolute
        # ----------------------------------------------------

        xT_known_abs = (
            A_T @ x0
            + CT @ u_known_abs
            + N_T @ w
        )


        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        errors_struct.append(
            xT_struct
            - xf
        )


        errors_struct_abs.append(
            xT_struct_abs
            - xf
        )


        errors_known.append(
            xT_known
            - xf
        )


        errors_known_abs.append(
            xT_known_abs
            - xf
        )


        # ----------------------------------------------------
        # Conformal only if feasible
        # ----------------------------------------------------

        if u_conf is not None:

            xT_conf = (
                A_T @ x0
                + CT @ u_conf
                + N_T @ w
            )


            errors_conf.append(
                xT_conf
                - xf
            )


    # ========================================================
    # MEAN ERRORS
    # ========================================================

    mu_struct = np.mean(
        errors_struct,
        axis=0
    )


    mu_struct_abs = np.mean(
        errors_struct_abs,
        axis=0
    )


    mu_known = np.mean(
        errors_known,
        axis=0
    )


    mu_known_abs = np.mean(
        errors_known_abs,
        axis=0
    )


    # ========================================================
    # DEVIATION:
    # STRUCTURE NORMALIZED
    # ========================================================

    dev_struct.append(
        float(
            mu_struct.T
            @ np.linalg.pinv(
                sigma_z
            )
            @ mu_struct
        )
    )


    # ========================================================
    # DEVIATION:
    # STRUCTURE ABSOLUTE
    #
    # E ||e||^2
    # =
    # ||E[e]||^2
    # + trace(Sigma_z)
    # ========================================================

    dev_struct_abs.append(
        float(
            mu_struct_abs.T
            @ mu_struct_abs
            + tr_struct
        )
    )


    # ========================================================
    # DEVIATION:
    # KNOWN NORMALIZED
    # ========================================================

    dev_known.append(
        float(
            mu_known.T
            @ np.linalg.pinv(
                sigma_z
            )
            @ mu_known
        )
    )


    # ========================================================
    # DEVIATION:
    # KNOWN ABSOLUTE
    #
    # E ||e||^2
    # =
    # ||E[e]||^2
    # + trace(Sigma_z)
    # ========================================================

    dev_known_abs.append(
        float(
            (mu_known_abs.T
            @ mu_known_abs)
            + tr_sigma_z
        )
    )


    # ========================================================
    # CONFORMAL DEVIATION
    # ========================================================

    if u_conf is None:

        dev_conf.append(
            np.nan
        )

    else:

        mu_conf = np.mean(
            errors_conf,
            axis=0
        )


        dev_conf.append(
            float(
                mu_conf.T
                @ mu_conf
            )
        )


# ============================================================
# PROCESS-NOISE FIGURE
# ============================================================

make_ieee_figure(
    sigma_vals=sigma_vals,

    cost_struct=cost_struct,
    cost_struct_abs=cost_struct_abs,
    cost_conf=cost_conf,
    cost_known=cost_known,
    cost_known_abs=cost_known_abs,

    dev_struct=dev_struct,
    dev_struct_abs=dev_struct_abs,
    dev_conf=dev_conf,
    dev_known=dev_known,
    dev_known_abs=dev_known_abs,

    conf_infeasible=conf_infeasible,

    xlabel=r"$\sigma$",

    filename="all_methods_cost_dev_vs_sigma_w"
)


# ============================================================
# OPTIONAL:
# PRINT CONFORMAL THRESHOLD INFORMATION
# ============================================================

print("\nProcess-noise sweep:")
print("--------------------")

for sigma, q, infeasible in zip(
    sigma_vals,
    q_sigma,
    conf_infeasible
):

    if infeasible:

        print(
            f"sigma = {sigma:.3f}, "
            f"q = {q:.3f}  --> CP infeasible"
        )


# ============================================================
# ============================================================
#
# EXPERIMENT 2:
# VARY INITIAL-STATE STANDARD DEVIATION sigma_0
#
# ============================================================
# ============================================================


# ------------------------------------------------------------
# Fixed process-noise standard deviation
# ------------------------------------------------------------

sigma_w_fixed = 0.3


sigma_w = np.kron(
    I_T,
    sigma_w_fixed**2
    * np.eye(n)
)


# ============================================================
# COMMON RANDOM NUMBERS
# ============================================================

z_x_train = np.random.randn(
    N,
    M,
    n
)

z_x_cal = np.random.randn(
    N_cal,
    n
)

z_w_train = np.random.randn(
    N,
    M,
    n*T
)

z_w_cal = np.random.randn(
    N_cal,
    n*T
)

z_x_test = np.random.randn(
    N_test,
    n
)

z_w_test = np.random.randn(
    N_test,
    n*T
)


# ------------------------------------------------------------
# Process-noise samples remain fixed across sigma_0 sweep
# ------------------------------------------------------------

w_train = (
    sigma_w_fixed
    * z_w_train
)


w_cal_train = (
    sigma_w_fixed
    * z_w_cal
)


# ============================================================
# STORAGE
# ============================================================

cost_struct = []
cost_struct_abs = []
cost_conf = []
cost_known = []
cost_known_abs = []


dev_struct = []
dev_struct_abs = []
dev_conf = []
dev_known = []
dev_known_abs = []


conf_infeasible = []

q_sigma = []


# ============================================================
# INITIAL-STATE UNCERTAINTY SWEEP
# ============================================================

for sigma in sigma_vals:

    # ========================================================
    # TRUE INITIAL-STATE COVARIANCE
    # ========================================================

    sigma_0 = (
        sigma**2
        * np.eye(n)
    )


    # ========================================================
    # TRUE TERMINAL COVARIANCE
    # ========================================================

    sigma_z = (
        N_T
        @ sigma_w
        @ N_T.T

        + A_T
        @ sigma_0
        @ A_T.T
    )


    tr_sigma_z = np.trace(
        sigma_z
    )


    # ========================================================
    # TRAINING / CALIBRATION INITIAL STATES
    # ========================================================

    x_train = (
        sigma
        * z_x_train
    )


    x_cal_train = (
        sigma
        * z_x_cal
    )


    # ========================================================
    # STRUCTURE MLE
    # ========================================================

    (
        CT_hat,
        struct_sigma
    ) = estimate_structure(
        x_train,
        w_train
    )


    # ========================================================
    # CONFORMAL QUANTILE
    # ========================================================

    scores = get_scores(
        U_cal,
        N_cal,
        CT_hat,
        A,
        N_T,
        CT,
        T,
        w_cal_train,
        x_cal_train
    )


    q = conformal_quantile(
        scores,
        delta
    )


    q_sigma.append(
        q
    )


    cp_infeasible = (
        q >= epsilon
    )


    conf_infeasible.append(
        cp_infeasible
    )


    # ========================================================
    # SOLVE FOUR MEC METHODS
    # ========================================================

    (
        u_struct,
        u_struct_abs,
        u_known,
        u_known_abs
    ) = solve_main_methods(
        CT_hat,
        struct_sigma,
        sigma_z
    )


    # ========================================================
    # CONFORMAL
    # ========================================================

    u_conf = solve_conformal(
        CT_hat,
        q
    )


    # ========================================================
    # COSTS
    # ========================================================

    cost_struct.append(
        control_cost(
            u_struct
        )
    )


    cost_struct_abs.append(
        control_cost(
            u_struct_abs
        )
    )


    cost_known.append(
        control_cost(
            u_known
        )
    )


    cost_known_abs.append(
        control_cost(
            u_known_abs
        )
    )


    if u_conf is None:

        cost_conf.append(
            np.nan
        )

    else:

        cost_conf.append(
            control_cost(
                u_conf
            )
        )


    # ========================================================
    # MONTE-CARLO EVALUATION
    # ========================================================

    errors_struct = []
    errors_struct_abs = []
    errors_known = []
    errors_known_abs = []
    errors_conf = []


    for i in range(N_test):

        # ----------------------------------------------------
        # initial-state experiment:
        #
        # x0 uses varying sigma_0
        # w uses fixed sigma_w
        # ----------------------------------------------------

        x0 = (
            sigma
            * z_x_test[i]
        )


        w = (
            sigma_w_fixed
            * z_w_test[i]
        )


        # ----------------------------------------------------
        # Structure normalized
        # ----------------------------------------------------

        xT_struct = (
            A_T @ x0
            + CT @ u_struct
            + N_T @ w
        )


        # ----------------------------------------------------
        # Structure absolute
        # ----------------------------------------------------

        xT_struct_abs = (
            A_T @ x0
            + CT @ u_struct_abs
            + N_T @ w
        )


        # ----------------------------------------------------
        # Known normalized
        # ----------------------------------------------------

        xT_known = (
            A_T @ x0
            + CT @ u_known
            + N_T @ w
        )


        # ----------------------------------------------------
        # Known absolute
        # ----------------------------------------------------

        xT_known_abs = (
            A_T @ x0
            + CT @ u_known_abs
            + N_T @ w
        )


        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        errors_struct.append(
            xT_struct
            - xf
        )


        errors_struct_abs.append(
            xT_struct_abs
            - xf
        )


        errors_known.append(
            xT_known
            - xf
        )


        errors_known_abs.append(
            xT_known_abs
            - xf
        )


        # ----------------------------------------------------
        # Conformal
        # ----------------------------------------------------

        if u_conf is not None:

            xT_conf = (
                A_T @ x0
                + CT @ u_conf
                + N_T @ w
            )


            errors_conf.append(
                xT_conf
                - xf
            )


    # ========================================================
    # MEAN ERRORS
    # ========================================================

    mu_struct = np.mean(
        errors_struct,
        axis=0
    )


    mu_struct_abs = np.mean(
        errors_struct_abs,
        axis=0
    )


    mu_known = np.mean(
        errors_known,
        axis=0
    )


    mu_known_abs = np.mean(
        errors_known_abs,
        axis=0
    )


    # ========================================================
    # STRUCTURE NORMALIZED DEVIATION
    # ========================================================

    dev_struct.append(
        float(
            mu_struct.T
            @ np.linalg.pinv(
                sigma_z
            )
            @ mu_struct
        )
    )


    # ========================================================
    # STRUCTURE ABSOLUTE DEVIATION
    #
    # + trace(sigma_z)
    # ========================================================

    dev_struct_abs.append(
        float(
            mu_struct_abs.T
            @ mu_struct_abs
            + tr_struct
        )
    )


    # ========================================================
    # KNOWN NORMALIZED DEVIATION
    # ========================================================

    dev_known.append(
        float(
            mu_known.T
            @ np.linalg.pinv(
                sigma_z
            )
            @ mu_known
        )
    )


    # ========================================================
    # KNOWN ABSOLUTE DEVIATION
    #
    # + trace(sigma_z)
    # ========================================================

    dev_known_abs.append(
        float(
            mu_known_abs.T
            @ mu_known_abs
            + tr_sigma_z
        )
    )


    # ========================================================
    # CONFORMAL DEVIATION
    # ========================================================

    if u_conf is None:

        dev_conf.append(
            np.nan
        )

    else:

        mu_conf = np.mean(
            errors_conf,
            axis=0
        )


        dev_conf.append(
            float(
                mu_conf.T
                @ mu_conf
            )
        )


# ============================================================
# INITIAL-STATE UNCERTAINTY FIGURE
# ============================================================

make_ieee_figure(
    sigma_vals=sigma_vals,

    cost_struct=cost_struct,
    cost_struct_abs=cost_struct_abs,
    cost_conf=cost_conf,
    cost_known=cost_known,
    cost_known_abs=cost_known_abs,

    dev_struct=dev_struct,
    dev_struct_abs=dev_struct_abs,
    dev_conf=dev_conf,
    dev_known=dev_known,
    dev_known_abs=dev_known_abs,

    conf_infeasible=conf_infeasible,

    xlabel=r"$\sigma_0$",

    filename="all_methods_cost_dev_vs_sigma_0"
)


# ============================================================
# OPTIONAL:
# PRINT CONFORMAL THRESHOLD INFORMATION
# ============================================================

print("\nInitial-state uncertainty sweep:")
print("--------------------------------")

for sigma, q, infeasible in zip(
    sigma_vals,
    q_sigma,
    conf_infeasible
):

    if infeasible:

        print(
            f"sigma_0 = {sigma:.3f}, "
            f"q = {q:.3f}  --> CP infeasible"
        )


