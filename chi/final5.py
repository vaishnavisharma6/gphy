import numpy as np
import cvxpy as cp
from scipy.stats import ncx2
from scipy.linalg import block_diag
from scipy.optimize import brentq

import matplotlib as mpl
import matplotlib.pyplot as plt


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

    "mathtext.fontset": "cm",

    "font.weight": "normal",
    "axes.labelweight": "normal",

    "font.size": 8.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,

    "lines.linewidth": 0.8,
    "axes.linewidth": 0.65,

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# SAME TICK SIZE EVERYWHERE
# ============================================================

TICK_SIZE = 7.0


def format_ticks(ax):

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=TICK_SIZE,
        pad=1.5
    )

    ax.tick_params(
        axis='both',
        which='minor',
        labelsize=TICK_SIZE
    )

    ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)


# ============================================================
# SYSTEM DIMENSIONS
# ============================================================

n = 4
m = 4


# ============================================================
# SYSTEM MATRICES
# ============================================================

A = np.array([
    [0.4,   0.1,  0.2,  0.6],
    [0.5,  -0.2,  0.2, -0.04],
    [-0.02, 0.1,  0.04, 0.4],
    [0.2,   0.03, 0.1,  0.09]
], dtype=float)


B = np.array([
    [1.4, -0.2,  0.3,  -0.8],
    [-2.5, 0.6,  0.8,  -0.7],
    [2.2, -1.4,  0.04, -0.1],
    [1.5,  1.4,  0.1,   0.3]
], dtype=float)


# ============================================================
# COVARIANCES
#
# Sigma_0 = sigma_0^2 I
# Sigma   = sigma^2 I
# ============================================================

sigma_0_std = 0.2
sigma_w_std = 0.3

Sigma_0 = sigma_0_std**2 * np.eye(n)
Sigma = sigma_w_std**2 * np.eye(n)


# ============================================================
# CHECK MONOTONICITY CONDITION
#
# A Sigma_0 A^T + Sigma >= Sigma_0
# ============================================================

condition_matrix = (
    A @ Sigma_0 @ A.T
    + Sigma
    - Sigma_0
)

condition_eigs = np.linalg.eigvalsh(condition_matrix)

print("\n==========================================")
print("MONOTONICITY CONDITION")
print("==========================================")

print(
    "Eigenvalues of "
    "A Sigma_0 A^T + Sigma - Sigma_0:"
)

print(condition_eigs)

print(
    "Condition satisfied:",
    np.all(condition_eigs >= -1e-10)
)


# ============================================================
# TARGET AND CONTROL COST
# ============================================================

x_f = np.array([
    4.0,
    5.0,
    6.0,
    7.0
])


Q = np.diag([
    2.0,
    3.0,
    4.0,
    5.0
])


# ============================================================
# CHANCE-CONSTRAINT PARAMETERS
# ============================================================

delta = 0.4
epsilon = 2.4


# ============================================================
# HORIZONS
# ============================================================

# Cost vs T
T_horizon = list(range(1, 30, 2))

# Cost vs gamma
T_gamma = [4, 6, 8]


# ============================================================
# MONTE CARLO SAMPLES
# ============================================================

N_MC = 5000


# ============================================================
# BUILD HORIZON MATRICES
#
# x_T = A^T x_0 + C_T u + N_T w
#
# C_T = [A^(T-1)B, ..., AB, B]
#
# N_T = [A^(T-1), ..., A, I]
# ============================================================

def build_horizon_matrices(T):

    # --------------------------------------------------------
    # A^T
    # --------------------------------------------------------

    A_T = np.linalg.matrix_power(A, T)


    # --------------------------------------------------------
    # Controllability matrix C_T
    # --------------------------------------------------------

    C_blocks = []

    for k in range(T):

        power = T - 1 - k

        C_blocks.append(
            np.linalg.matrix_power(A, power) @ B
        )

    C_T = np.hstack(C_blocks)


    # --------------------------------------------------------
    # Noise propagation matrix N_T
    # --------------------------------------------------------

    N_blocks = []

    for k in range(T):

        power = T - 1 - k

        N_blocks.append(
            np.linalg.matrix_power(A, power)
        )

    N_T = np.hstack(N_blocks)


    # --------------------------------------------------------
    # Stacked process-noise covariance
    # --------------------------------------------------------

    Sigma_wT = np.kron(
        np.eye(T),
        Sigma
    )


    # --------------------------------------------------------
    # Terminal covariance
    # --------------------------------------------------------

    Sigma_eT = (
        A_T @ Sigma_0 @ A_T.T
        +
        N_T @ Sigma_wT @ N_T.T
    )


    # --------------------------------------------------------
    # Stacked control cost
    # --------------------------------------------------------

    Q_T = block_diag(
        *([Q] * T)
    )


    return (
        A_T,
        C_T,
        N_T,
        Sigma_wT,
        Sigma_eT,
        Q_T
    )


# ============================================================
# CHECK Sigma_e(T+1) >= Sigma_e(T)
# ============================================================

print("\n==========================================")
print("TERMINAL COVARIANCE MONOTONICITY")
print("==========================================")

Sigma_previous = None

for T in range(1, 10):

    _, _, _, _, Sigma_eT, _ = (
        build_horizon_matrices(T)
    )

    if Sigma_previous is not None:

        difference = (
            Sigma_eT
            -
            Sigma_previous
        )

        min_eig = np.min(
            np.linalg.eigvalsh(difference)
        )

        print(
            f"T={T-1} -> T={T}: "
            f"min eigenvalue = {min_eig:.8f}"
        )

    Sigma_previous = Sigma_eT


# ============================================================
# FIND lambda*
#
# P{chi_n^2(lambda) <= epsilon^2} = 1-delta
# ============================================================

def find_lambda_star(
    n,
    epsilon,
    probability
):

    x = epsilon**2

    def equation(lam):

        return (
            ncx2.cdf(
                x,
                df=n,
                nc=lam
            )
            -
            probability
        )

    lam_lower = 0.0
    lam_upper = 1e11

    if (
        equation(lam_lower)
        *
        equation(lam_upper)
        > 0
    ):

        raise ValueError(
            "Root not bracketed."
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

print("\nlambda_star =", lambda_star)


# ============================================================
# SECOND-MOMENT THRESHOLD
#
# E[e^T Sigma_e^{-1} e]
# =
# mu_e^T Sigma_e^{-1} mu_e + n
#
# Hence
#
# mu_e^T Sigma_e^{-1} mu_e
# <= epsilon^2 - n
# ============================================================

gamma_mahal = epsilon**2

gamma_second = (
    epsilon**2
)

gamma_exact = lambda_star


if gamma_second <= 0:

    raise ValueError(
        "epsilon^2 - n must be positive "
        "for the second-moment constraint."
    )


print("\nConstraint thresholds:")

print(
    "Mahalanobis gamma =",
    gamma_mahal
)

print(
    "Second moment gamma =",
    gamma_second
)

print(
    "Exact chi-square gamma =",
    gamma_exact
)


# ============================================================
# CVXPY SOLVER
# ============================================================

def solve_problem(problem):

    installed = cp.installed_solvers()

    if "CLARABEL" in installed:

        try:

            problem.solve(
                solver=cp.CLARABEL,
                verbose=False
            )

            return

        except Exception:
            pass


    problem.solve(
        solver=cp.SCS,
        eps=1e-7,
        max_iters=100000,
        verbose=False
    )


# ============================================================
# SOLVE ONE ENERGY MINIMIZATION PROBLEM
#
# min u^T Q_T u
#
# s.t.
#
# (C_T u - x_f)^T
# Sigma_eT^{-1}
# (C_T u - x_f)
# <= gamma
# ============================================================

def solve_energy_problem(
    T,
    gamma
):

    (
        A_T,
        C_T,
        N_T,
        Sigma_wT,
        Sigma_eT,
        Q_T
    ) = build_horizon_matrices(T)


    Sigma_eT_inv = np.linalg.inv(
        Sigma_eT
    )


    b = -x_f


    u = cp.Variable(
        m * T
    )


    objective = cp.Minimize(
        cp.quad_form(
            u,
            Q_T
        )
    )


    constraint = (
        cp.quad_form(
            C_T @ u + b,
            Sigma_eT_inv
        )
        <= gamma
    )


    problem = cp.Problem(
        objective,
        [constraint]
    )


    solve_problem(problem)


    if u.value is None:

        return (
            np.nan,
            None,
            Sigma_eT,
            Sigma_eT_inv,
            C_T,
            N_T,
            A_T,
            Sigma_wT,
            Q_T
        )


    u_val = np.asarray(
        u.value
    ).reshape(-1)


    J_star = float(
        u_val.T
        @ Q_T
        @ u_val
    )


    return (
        J_star,
        u_val,
        Sigma_eT,
        Sigma_eT_inv,
        C_T,
        N_T,
        A_T,
        Sigma_wT,
        Q_T
    )


# ============================================================
# STORAGE: COST VS T
# ============================================================

epu = []
secu = []
exact = []


# ============================================================
# STORAGE: RATIO PLOTS
# ============================================================

ratios1 = []
ratios3 = []
ratios5 = []


# ============================================================
# PART I:
# COST VERSUS HORIZON T
# ============================================================

for T in T_horizon:

    print(
        f"\nSolving T = {T}"
    )


    # ========================================================
    # MAHALANOBIS
    # ========================================================

    (
        J_mahal,
        u_mahal,
        Sigma_eT,
        Sigma_eT_inv,
        C_T,
        N_T,
        A_T,
        Sigma_wT,
        Q_T
    ) = solve_energy_problem(
        T,
        gamma_mahal
    )


    # ========================================================
    # SECOND MOMENT
    # ========================================================

    (
        J_second,
        u_second,
        _,
        _,
        _,
        _,
        _,
        _,
        _
    ) = solve_energy_problem(
        T,
        gamma_second
    )


    # ========================================================
    # EXACT NONCENTRAL CHI-SQUARE
    # ========================================================

    (
        J_exact,
        u_exact,
        _,
        _,
        _,
        _,
        _,
        _,
        _
    ) = solve_energy_problem(
        T,
        gamma_exact
    )


    epu.append(J_mahal)
    secu.append(J_second)
    exact.append(J_exact)


    # ========================================================
    # MAHALANOBIS DETERMINISTIC MEAN QUANTITY
    # ========================================================

    if u_mahal is None:

        ratios3.append(np.nan)

    else:

        mu_mahal = (
            C_T @ u_mahal
            -
            x_f
        )

        mahal_mean_dev = float(
            mu_mahal.T
            @ Sigma_eT_inv
            @ mu_mahal
        )

        ratios3.append(
            mahal_mean_dev
        )


    # ========================================================
    # MONTE CARLO
    # ========================================================

    if (
        u_second is None
        or u_exact is None
    ):

        ratios1.append(np.nan)
        ratios5.append(np.nan)

        continue


    rng = np.random.default_rng(0)

    second_mc = []
    exact_success = 0


    for _ in range(N_MC):

        # ----------------------------------------------------
        # Initial condition
        # ----------------------------------------------------

        x0_sample = (
            rng.multivariate_normal(
                np.zeros(n),
                Sigma_0
            )
        )


        # ----------------------------------------------------
        # Stacked process noise
        # ----------------------------------------------------

        w_sample = (
            rng.multivariate_normal(
                np.zeros(n * T),
                Sigma_wT
            )
        )


        # ----------------------------------------------------
        # SECOND-MOMENT CONTROLLER
        # ----------------------------------------------------

        xT_second = (
            A_T @ x0_sample
            +
            C_T @ u_second
            +
            N_T @ w_sample
        )


        e_second = (
            xT_second
            -
            x_f
        )


        second_stat = float(
            e_second.T
            @ Sigma_eT_inv
            @ e_second
        )


        second_mc.append(
            second_stat
        )


        # ----------------------------------------------------
        # EXACT CONTROLLER
        # ----------------------------------------------------

        xT_exact = (
            A_T @ x0_sample
            +
            C_T @ u_exact
            +
            N_T @ w_sample
        )


        e_exact = (
            xT_exact
            -
            x_f
        )


        exact_stat = float(
            e_exact.T
            @ Sigma_eT_inv
            @ e_exact
        )


        if exact_stat <= epsilon**2:

            exact_success += 1


    # --------------------------------------------------------
    # Empirical second moment
    # --------------------------------------------------------

    ratios1.append(
        np.mean(second_mc)
    )


    # --------------------------------------------------------
    # Empirical probability
    # --------------------------------------------------------

    ratios5.append(
        exact_success
        /
        N_MC
    )


# ============================================================
# PRINT COST ARRAYS
# ============================================================

print("\n==========================================")
print("COST ARRAYS")
print("==========================================")

print(
    "Mahalanobis:"
)

print(
    np.asarray(epu)
)

print(
    "\nSecond moment:"
)

print(
    np.asarray(secu)
)

print(
    "\nNoncentral chi-square:"
)

print(
    np.asarray(exact)
)


# ============================================================
# CHECK MONOTONICITY OF COST VERSUS T
# ============================================================

def check_nonincreasing(values, name):

    values = np.asarray(
        values,
        dtype=float
    )

    differences = np.diff(
        values
    )

    tol = 1e-5

    condition = np.all(
        differences <= tol
    )

    print(
        f"{name} nonincreasing:",
        condition
    )

    if not condition:

        bad = np.where(
            differences > tol
        )[0]

        for i in bad:

            print(
                f"  violation: "
                f"T={T_horizon[i]} -> "
                f"T={T_horizon[i+1]}, "
                f"{values[i]:.8f} -> "
                f"{values[i+1]:.8f}"
            )


print("\n==========================================")
print("MONOTONICITY OF J* VS T")
print("==========================================")

check_nonincreasing(
    epu,
    "Mahalanobis"
)

check_nonincreasing(
    secu,
    "Second moment"
)

check_nonincreasing(
    exact,
    "Noncentral chi-square"
)


# ============================================================
# FIGURE 1:
# RATIO / CONSTRAINT SATISFACTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(3.4, 2.15)
)


ax.plot(
    T_horizon,
    ratios3,
    linestyle='--',
    linewidth=0.90,
    label=(
        r'$\hat{\mu}_{e_T}^{\top}'
        r'\Sigma_{e_T}^{-1}'
        r'\hat{\mu}_{e_T}$'
    )
)


ax.plot(
    T_horizon,
    ratios1,
    linestyle='--',
    linewidth=0.90,
    label=(
        r'$\hat{\mathbb{E}}'
        r'[e_T^\top\Sigma_{e_T}^{-1}e_T]$'
    )
)


ax.plot(
    T_horizon,
    ratios5,
    linestyle='--',
    linewidth=0.90,
    label=(
        r'$\hat{\mathbb{P}}'
        r'\{e_T^\top\Sigma_{e_T}^{-1}e_T'
        r'\leq\epsilon^2\}$'
    )
)


ax.axhline(
    y=epsilon**2,
    linestyle=':',
    linewidth=0.80,
    label=r'$\epsilon^2$'
)


# ax.axhline(
#     y=1-delta,
#     linestyle='-.',
#     linewidth=0.80,
#     label=r'$1-\delta$'
# )


ax.set_xlabel(
    r'$T$'
)


ax.set_xticks(
    [1, 7, 13, 19, 25, 29]
)


ax.grid(
    True,
    linestyle=':',
    linewidth=0.35,
    alpha=0.30
)


ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    fontsize=5.5,
    handlelength=1.4,
    handletextpad=0.35,
    columnspacing=0.8,
    labelspacing=0.25,
    borderpad=0.25,
    frameon=False
)


fig.subplots_adjust(
    left=0.16,
    right=0.99,
    bottom=0.19,
    top=0.72
)


format_ticks(ax)


fig.savefig(
    'ratio.pdf',
    format='pdf'
)


fig.savefig(
    'ratio.png',
    dpi=600,
    bbox_inches='tight'
)


plt.close(fig)


# ============================================================
# PART II:
# COST VERSUS gamma
#
# IMPORTANT:
# USE THE SAME gamma VALUES FOR EVERY T
# ============================================================


# ============================================================
# COMPUTE gamma_min AND gamma_max FOR EACH T
# ============================================================

gamma_min_dict = {}
gamma_max_dict = {}


for T in T_gamma:

    (
        A_T,
        C_T,
        N_T,
        Sigma_wT,
        Sigma_eT,
        Q_T
    ) = build_horizon_matrices(T)


    Sigma_eT_inv = np.linalg.inv(
        Sigma_eT
    )


    b = -x_f


    # --------------------------------------------------------
    # gamma_max:
    #
    # value corresponding to u = 0
    # --------------------------------------------------------

    gamma_max = float(
        b.T
        @ Sigma_eT_inv
        @ b
    )


    gamma_max_dict[T] = gamma_max


    # --------------------------------------------------------
    # gamma_min:
    #
    # min_u (C_T u + b)^T
    #       Sigma_eT^{-1}
    #       (C_T u + b)
    #
    # Weighted least squares
    # --------------------------------------------------------

    L = np.linalg.cholesky(
        Sigma_eT
    )


    C_white = np.linalg.solve(
        L,
        C_T
    )


    b_white = np.linalg.solve(
        L,
        b
    )


    u_ls = np.linalg.lstsq(
        C_white,
        -b_white,
        rcond=None
    )[0]


    residual = (
        C_white @ u_ls
        +
        b_white
    )


    gamma_min = float(
        residual.T
        @ residual
    )


    gamma_min_dict[T] = max(
        gamma_min,
        0.0
    )


print("\n==========================================")
print("GAMMA RANGES")
print("==========================================")

for T in T_gamma:

    print(
        f"T={T}: "
        f"gamma_min={gamma_min_dict[T]:.8f}, "
        f"gamma_max={gamma_max_dict[T]:.8f}"
    )


# ============================================================
# COMMON FEASIBLE gamma INTERVAL
# ============================================================

gamma_common_min = max(
    gamma_min_dict.values()
)

gamma_common_max = min(
    gamma_max_dict.values()
)


if gamma_common_min >= gamma_common_max:

    raise ValueError(
        "No common feasible gamma interval "
        "for the selected horizons."
    )


# Avoid exact feasibility boundary for numerical stability
gamma_start = max(
    gamma_common_min + 1e-6,
    1e-6
)


# Stay slightly below gamma_max so J* does not become exactly 0
gamma_end = (
    0.999
    *
    gamma_common_max
)


# SAME gamma values for T=4,6,8
gamma_vals = np.linspace(
    gamma_start,
    gamma_end,
    60
)


# ============================================================
# SOLVE COST VS gamma
# ============================================================

all_gamma = {}
all_cost = {}


for T in T_gamma:

    print(
        f"\nSolving common gamma sweep for T={T}"
    )


    cost_vals = []


    for gamma in gamma_vals:

        (
            J_star,
            u_val,
            Sigma_eT,
            Sigma_eT_inv,
            C_T,
            N_T,
            A_T,
            Sigma_wT,
            Q_T
        ) = solve_energy_problem(
            T,
            gamma
        )


        cost_vals.append(
            J_star
        )


    # Exactly the same gamma array
    all_gamma[T] = gamma_vals.copy()

    all_cost[T] = np.asarray(
        cost_vals
    )


# ============================================================
# DIRECTLY CHECK
#
# J_8*(gamma) <= J_6*(gamma) <= J_4*(gamma)
# ============================================================

print("\n==========================================")
print("COMMON-GAMMA MONOTONICITY CHECK")
print("==========================================")


J4 = np.asarray(
    all_cost[4]
)

J6 = np.asarray(
    all_cost[6]
)

J8 = np.asarray(
    all_cost[8]
)


tol = 1e-4


condition_46 = np.all(
    J6 <= J4 + tol
)

condition_68 = np.all(
    J8 <= J6 + tol
)


print(
    "J_6 <= J_4:",
    condition_46
)

print(
    "J_8 <= J_6:",
    condition_68
)

print(
    "Overall monotonic decrease:",
    condition_46 and condition_68
)


# Print any violations
for i, gamma in enumerate(gamma_vals):

    if (
        J6[i] > J4[i] + tol
        or
        J8[i] > J6[i] + tol
    ):

        print(
            "\nViolation at "
            f"gamma={gamma:.8f}:"
        )

        print(
            "J4 =",
            J4[i]
        )

        print(
            "J6 =",
            J6[i]
        )

        print(
            "J8 =",
            J8[i]
        )


# ============================================================
# FIGURE 2:
#
# LEFT  : J* vs gamma
# RIGHT : J* vs T
# ============================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(3.48, 1.50),
    constrained_layout=False
)


# ============================================================
# LEFT:
# COST VS gamma
# ============================================================

ax1 = axs[0]


for T in T_gamma:

    ax1.semilogy(
        all_gamma[T],
        all_cost[T],
        linewidth=0.90,
        linestyle='-',
        label=rf'$T={T}$'
    )


ax1.set_xlabel(
    r'$\gamma$'
)


ax1.set_ylabel(
    r'$J_T^\star$',
    labelpad=1.0
)


ax1.grid(
    True,
    which='both',
    linestyle=':',
    linewidth=0.30,
    alpha=0.28
)


ax1.legend(
    loc='best',
    ncol=1,
    fontsize=5.9,
    handlelength=1.4,
    handletextpad=0.35,
    labelspacing=0.20,
    borderpad=0.25,
    frameon=True,
    framealpha=0.85
)


format_ticks(ax1)


# ============================================================
# RIGHT:
# COST VS T
# ============================================================

ax2 = axs[1]


ax2.semilogy(
    T_horizon,
    epu,
    linestyle='-',
    linewidth=0.90,
    label='Mahalanobis',
    zorder=4
)


ax2.semilogy(
    T_horizon,
    secu,
    linestyle='--',
    linewidth=0.90,
    label='2nd moment',
    zorder=3
)


ax2.semilogy(
    T_horizon,
    exact,
    linestyle=':',
    linewidth=1.00,
    label=r'Noncentral $\chi^2$',
    zorder=5
)


ax2.set_xlabel(
    r'$T$'
)


ax2.set_ylabel(
    r'$J_T^\star$',
    labelpad=1.0
)


ax2.set_xticks(
    [1, 7, 13, 19, 25]
)


ax2.grid(
    True,
    which='both',
    linestyle=':',
    linewidth=0.30,
    alpha=0.28,
    zorder=0
)


ax2.legend(
    loc='best',
    ncol=1,
    fontsize=5.4,
    handlelength=1.55,
    handletextpad=0.30,
    labelspacing=0.18,
    borderpad=0.25,
    frameon=True,
    framealpha=0.85
)


format_ticks(ax2)


# ============================================================
# FIGURE SPACING
# ============================================================

fig.subplots_adjust(
    top=0.97,
    bottom=0.23,
    left=0.16,
    right=0.99,
    wspace=0.58
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    'combined_cost_gamma_T.pdf',
    format='pdf'
)


fig.savefig(
    'combined_cost_gamma_T.png',
    dpi=600,
    bbox_inches='tight'
)


plt.show()