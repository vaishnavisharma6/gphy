import numpy as np
import cvxpy as cp

from scipy.linalg import block_diag
from scipy.stats import ncx2
from scipy.optimize import brentq

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

    # Vector-compatible text in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # High-resolution PNG
    "savefig.dpi": 600,
})


# ============================================================
# FIGURE HELPERS
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
    Save both vector PDF and high-resolution PNG.
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


# ============================================================
# PARAMETERS
# ============================================================

np.random.seed(0)

n = 4
m = 4

T = 8

name = "stable"

xf = np.array([
    4.0,
    5.0,
    6.0,
    7.0
])

# Initial-state standard deviation
sigma_1 = 0.2

# Process-noise standard deviation
sigma_2 = 0.3

sigma_0 = (
    sigma_1**2
    * np.eye(n)
)

Q = np.diag([
    2.0,
    3.0,
    4.0,
    5.0
])

epsilon = 2.4

delta = 0.5


# ============================================================
# DATA SIZES
# ============================================================

N_list = [
    20,
    40,
    80,
    120,
    200
]

M_list = [
    5,
    10,
    20,
    50,
    100
]

K_runs = 30


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
# Control-cost matrix
# ------------------------------------------------------------

Q_tilde = block_diag(
    *([Q] * T)
)


# ------------------------------------------------------------
# Process-noise covariance
# ------------------------------------------------------------

sigma = (
    sigma_2**2
    * np.eye(n)
)

sigmaw = np.kron(
    np.eye(T),
    sigma
)


# ------------------------------------------------------------
# Noise propagation matrix
# ------------------------------------------------------------

NT = np.eye(n)

for i in range(1, T):

    NT = np.hstack(
        (
            NT,
            np.linalg.matrix_power(
                A,
                i
            )
        )
    )


# ------------------------------------------------------------
# True terminal covariance
# ------------------------------------------------------------

sigmaz = (
    NT
    @ sigmaw
    @ NT.T

    + A_T
    @ sigma_0
    @ A_T.T
)


# ============================================================
# NONCENTRAL CHI-SQUARE THRESHOLD
# ============================================================

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

    lam_lower = 0.0
    lam_upper = 1e11

    if (
        equation(lam_lower)
        * equation(lam_upper)
        > 0
    ):
        raise ValueError(
            "Root not bracketed. "
            "Increase lam_upper."
        )

    return brentq(
        equation,
        lam_lower,
        lam_upper
    )


lambda_star = find_lambda_star(
    n,
    epsilon,
    1 - delta
)


print(
    "lambda_star =",
    lambda_star
)


# ============================================================
# KNOWN-DYNAMICS OPTIMAL COST
# ============================================================

u_known = cp.Variable(
    m*T
)


known_objective = cp.Minimize(
    cp.quad_form(
        u_known,
        Q_tilde
    )
)


known_constraint = (
    cp.quad_form(
        CT @ u_known - xf,
        np.linalg.inv(sigmaz)
    )
    <= lambda_star
)


known_problem = cp.Problem(
    known_objective,
    [known_constraint]
)


known_problem.solve(
    solver=cp.SCS,
    verbose=False
)


if u_known.value is None:

    raise RuntimeError(
        "Known-dynamics optimization failed."
    )


J_known = float(
    u_known.value.T
    @ Q_tilde
    @ u_known.value
)


print(
    "Known optimal cost:",
    J_known
)


# ============================================================
# DATA-DRIVEN STRUCTURE-MLE COST
# ============================================================

def data_driven_cost(
    N,
    M
):
    """
    Compute the average structure-MLE optimal cost
    over K_runs independent datasets.

    For every run:

    1. Generate N control inputs.
    2. Generate M repeated terminal-state samples
       for every control input.
    3. Estimate C_T using structure.
    4. Estimate Sigma_z using structure residuals.
    5. Solve the stochastic MEC problem.
    """

    costs = []


    for run in range(K_runs):

        # ----------------------------------------------------
        # Independent RNG for each Monte-Carlo run
        # ----------------------------------------------------

        rng = np.random.default_rng(
            run
        )


        # ----------------------------------------------------
        # Input experiments
        # ----------------------------------------------------

        U = rng.standard_normal(
            (m*T, N)
        )


        # ----------------------------------------------------
        # Storage
        # ----------------------------------------------------

        X_bar = np.zeros(
            (n, N)
        )

        all_xT = np.zeros(
            (n, N, M)
        )


        # ====================================================
        # GENERATE DATA
        # ====================================================

        for i in range(N):

            for j in range(M):

                # --------------------------------------------
                # Initial condition
                # --------------------------------------------

                x0 = rng.multivariate_normal(
                    np.zeros(n),
                    sigma_0
                )


                # --------------------------------------------
                # Process noise
                # --------------------------------------------

                w = rng.multivariate_normal(
                    np.zeros(n*T),
                    sigmaw
                )


                # --------------------------------------------
                # Terminal state
                # --------------------------------------------

                xT = (
                    A_T @ x0
                    + CT @ U[:, i]
                    + NT @ w
                )


                all_xT[:, i, j] = xT


            # -----------------------------------------------
            # Average terminal state for input i
            # -----------------------------------------------

            X_bar[:, i] = np.mean(
                all_xT[:, i, :],
                axis=1
            )


        # ====================================================
        # STRUCTURE MLE OF C_T
        # ====================================================

        CT_hat = (
            X_bar
            @ U.T
            @ np.linalg.pinv(
                U @ U.T
            )
        )


        # ====================================================
        # STRUCTURE MLE OF TERMINAL COVARIANCE
        # ====================================================

        Sigma_hat = np.zeros(
            (n, n)
        )


        for i in range(N):

            for j in range(M):

                residual = (
                    all_xT[:, i, j]
                    - CT_hat @ U[:, i]
                )


                Sigma_hat += np.outer(
                    residual,
                    residual
                )


        Sigma_hat /= (
            N * M
        )


        # ----------------------------------------------------
        # Numerical symmetrization
        # ----------------------------------------------------

        Sigma_hat = (
            0.5
            * (
                Sigma_hat
                + Sigma_hat.T
            )
        )


        # ====================================================
        # DATA-DRIVEN OPTIMIZATION
        # ====================================================

        u = cp.Variable(
            m*T
        )


        objective = cp.Minimize(
            cp.quad_form(
                u,
                Q_tilde
            )
        )


        # pinv is safer for small N/M
        Sigma_hat_inv = np.linalg.pinv(
            Sigma_hat
        )


        # Symmetrize inverse for CVXPY
        Sigma_hat_inv = (
            0.5
            * (
                Sigma_hat_inv
                + Sigma_hat_inv.T
            )
        )


        constraint = (
            cp.quad_form(
                CT_hat @ u - xf,
                cp.psd_wrap(
                    Sigma_hat_inv
                )
            )
            <= lambda_star
        )


        problem = cp.Problem(
            objective,
            [constraint]
        )


        problem.solve(
            solver=cp.SCS,
            verbose=False
        )


        # ====================================================
        # STORE COST
        # ====================================================

        if u.value is not None:

            J = float(
                u.value.T
                @ Q_tilde
                @ u.value
            )


            costs.append(
                J
            )


    # ========================================================
    # CHECK SUCCESS
    # ========================================================

    if len(costs) == 0:

        return np.nan


    # ========================================================
    # AVERAGE OVER K RUNS
    # ========================================================

    return float(
        np.mean(costs)
    )


# ============================================================
# COST ERROR VS N
# ============================================================

M_fixed = 100

error_N = []

J_data_N = []


for N in N_list:

    print(
        f"Running N = {N}, "
        f"M = {M_fixed}"
    )


    J = data_driven_cost(
        N=N,
        M=M_fixed
    )


    J_data_N.append(
        J
    )


    relative_error = (
        (J - J_known)
        / J_known
    )


    error_N.append(
        relative_error
    )


    print(
        f"    J_data = {J:.6f}, "
        f"relative error = {relative_error:.6f}"
    )


# ============================================================
# COST ERROR VS M
# ============================================================

N_fixed = 100

error_M = []

J_data_M = []


for M in M_list:

    print(
        f"Running N = {N_fixed}, "
        f"M = {M}"
    )


    J = data_driven_cost(
        N=N_fixed,
        M=M
    )


    J_data_M.append(
        J
    )


    relative_error = (
        (J - J_known)
        / J_known
    )


    error_M.append(
        relative_error
    )


    print(
        f"    J_data = {J:.6f}, "
        f"relative error = {relative_error:.6f}"
    )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

error_N = np.asarray(
    error_N,
    dtype=float
)

error_M = np.asarray(
    error_M,
    dtype=float
)


# ============================================================
# IEEE FIGURE:
# RELATIVE OPTIMAL-COST ERROR VS N AND M
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
    top=0.93,
    wspace=0.42
)


# ============================================================
# LEFT:
# COST ERROR VS N
# ============================================================

ax[0].plot(
    N_list,
    error_N,
    linestyle="-",
    linewidth=0.9
)


# Zero-reference line
ax[0].axhline(
    0.0,
    linestyle=":",
    linewidth=0.65,
    alpha=0.70
)


ax[0].set_xlabel(
    r"$N$"
)


ax[0].set_ylabel(
    r"$(J_{\mathrm{data}}^*-J_{\mathrm{known}}^*)"
    r"/J_{\mathrm{known}}^*$",
    labelpad=1
)


# ============================================================
# RIGHT:
# COST ERROR VS M
# ============================================================

ax[1].plot(
    M_list,
    error_M,
    linestyle="-",
    linewidth=0.9
)


# Zero-reference line
ax[1].axhline(
    0.0,
    linestyle=":",
    linewidth=0.65,
    alpha=0.70
)


ax[1].set_xlabel(
    r"$M$"
)


# ============================================================
# IEEE AXIS STYLE
# ============================================================

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


# ============================================================
# NICE INTEGER X TICKS
# ============================================================

ax[0].set_xticks(
    N_list
)


ax[1].set_xticks(
    M_list
)


# ============================================================
# REMOVE UNNECESSARY TRAILING ZEROS ON Y AXES
# ============================================================

def clean_number(x, pos):

    if abs(x) < 1e-12:
        return "0"

    return f"{x:g}"


ax[0].yaxis.set_major_formatter(
    FuncFormatter(
        clean_number
    )
)


ax[1].yaxis.set_major_formatter(
    FuncFormatter(
        clean_number
    )
)


# ============================================================
# SAVE
# ============================================================

save_ieee_figure(
    fig,
    "relative_cost_error_vs_N_M"
)


plt.show()


# ============================================================
# PRINT FINAL NUMERICAL RESULTS
# ============================================================

print("\n==========================================")
print("Known-dynamics optimal cost")
print("==========================================")
print(
    f"J_known = {J_known:.8f}"
)


print("\n==========================================")
print(f"Cost convergence with M = {M_fixed}")
print("==========================================")

for N, J, err in zip(
    N_list,
    J_data_N,
    error_N
):

    print(
        f"N = {N:4d} : "
        f"J_data = {J:.8f}, "
        f"relative error = {err:.8f}"
    )


print("\n==========================================")
print(f"Cost convergence with N = {N_fixed}")
print("==========================================")

for M, J, err in zip(
    M_list,
    J_data_M,
    error_M
):

    print(
        f"M = {M:4d} : "
        f"J_data = {J:.8f}, "
        f"relative error = {err:.8f}"
    )