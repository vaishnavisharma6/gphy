
import numpy as np
import cvxpy as cp
from scipy.stats import norm, ncx2
from scipy.linalg import block_diag
from scipy.optimize import brentq

import matplotlib as mpl
import matplotlib.pyplot as plt

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
# SYSTEM PARAMETERS
# ============================================================

n = 4
m = 4

sigma_0 = np.eye(n) * 2.0

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

delta = 0.4
epsilon = 2.4

name = 'stable'

# Horizons for J* vs T
T_horizon = list(range(1, 30, 2))

# Horizons for J* vs gamma
T_gamma = [4, 6, 8]

# Number of Monte-Carlo samples for ratio plot
N_MC = 5000


# ============================================================
# FIND lambda*
# ============================================================

def find_lambda_star(n, epsilon, delta):

    x = epsilon**2

    def equation(lam):
        return ncx2.cdf(
            x,
            df=n,
            nc=lam
        ) - delta

    lam_lower = 0.0
    lam_upper = 1e11

    if equation(lam_lower) * equation(lam_upper) > 0:
        raise ValueError(
            "Root not bracketed. Increase lam_upper."
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

print("lambda_star =", lambda_star)


# ============================================================
# STORAGE: COST VS T
# ============================================================

# Mahalanobis
epu = []

# Second moment
secu = []

# Exact noncentral chi-square
exact = []


# ============================================================
# STORAGE: RATIO PLOT
# ============================================================

# empirical E[e^T Sigma^-1 e]
# under second-moment design
ratios1 = []

# deterministic mean Mahalanobis deviation
# under Mahalanobis design
ratios3 = []

# empirical probability under exact design
ratios5 = []


# ============================================================
# PART I:
# SOLVE FOR DIFFERENT HORIZONS T
# ============================================================

for t in T_horizon:

    print(f"Solving T = {t}")

    # --------------------------------------------------------
    # Generate system
    # --------------------------------------------------------

    A, B, NT = get_system(
        name,
        n,
        m,
        t,
        seed=0
    )


    # --------------------------------------------------------
    # Construct N_T and C_T
    # --------------------------------------------------------

    N_T = np.eye(n)
    C_T = NT

    for i in range(1, t):

        N_T = np.hstack((
            N_T,
            np.linalg.matrix_power(A, i)
        ))


    # --------------------------------------------------------
    # Terminal covariance
    # --------------------------------------------------------

    sigma_w = np.eye(n * t) * 0.1

    A_T = np.linalg.matrix_power(A, t)

    sigma_z = (
        N_T @ sigma_w @ N_T.T
        +
        A_T @ sigma_0 @ A_T.T
    )

    sigma_z_inv = np.linalg.inv(sigma_z)


    # --------------------------------------------------------
    # Terminal mean offset
    # --------------------------------------------------------

    b = -x_f


    # --------------------------------------------------------
    # Cost matrix
    # --------------------------------------------------------

    Q_tilde = block_diag(
        *([Q] * t)
    )


    # ========================================================
    # THREE CONSTRAINT THRESHOLDS
    # ========================================================

    # --------------------------------------------------------
    # Mahalanobis:
    #
    # mu_e^T Sigma_e^{-1} mu_e <= epsilon^2
    # --------------------------------------------------------

    gamma_mahal = epsilon**2


    # --------------------------------------------------------
    # Second moment:
    #
    # E[e^T Sigma_e^{-1}e]
    # =
    # mu_e^T Sigma_e^{-1}mu_e + n
    #
    # Therefore:
    #
    # mu_e^T Sigma_e^{-1}mu_e <= epsilon^2 - n
    # --------------------------------------------------------

    gamma_second = epsilon**2


    # --------------------------------------------------------
    # Exact noncentral chi-square:
    #
    # mu_e^T Sigma_e^{-1}mu_e <= lambda_star
    # --------------------------------------------------------

    gamma_exact = lambda_star


    # ========================================================
    # OPTIMIZATION VARIABLES
    # ========================================================

    u_mahal = cp.Variable(m * t)

    u_second = cp.Variable(m * t)

    u_exact = cp.Variable(m * t)


    # ========================================================
    # OBJECTIVES
    # ========================================================

    obj_mahal = cp.Minimize(
        cp.quad_form(
            u_mahal,
            Q_tilde
        )
    )

    obj_second = cp.Minimize(
        cp.quad_form(
            u_second,
            Q_tilde
        )
    )

    obj_exact = cp.Minimize(
        cp.quad_form(
            u_exact,
            Q_tilde
        )
    )


    # ========================================================
    # NORMALIZED TERMINAL MEAN DEVIATIONS
    # ========================================================

    f_mahal = cp.quad_form(
        C_T @ u_mahal + b,
        sigma_z_inv
    )

    f_second = cp.quad_form(
        C_T @ u_second + b,
        sigma_z_inv
    )

    f_exact = cp.quad_form(
        C_T @ u_exact + b,
        sigma_z_inv
    )


    # ========================================================
    # OPTIMIZATION PROBLEMS
    # ========================================================

    prob_mahal = cp.Problem(
        obj_mahal,
        [
            f_mahal <= gamma_mahal
        ]
    )

    prob_second = cp.Problem(
        obj_second,
        [
            f_second <= gamma_second
        ]
    )

    prob_exact = cp.Problem(
        obj_exact,
        [
            f_exact <= gamma_exact
        ]
    )


    # ========================================================
    # SOLVE
    # ========================================================

    prob_mahal.solve(
        solver=cp.SCS,
        verbose=False
    )

    prob_second.solve(
        solver=cp.SCS,
        verbose=False
    )

    prob_exact.solve(
        solver=cp.SCS,
        verbose=False
    )


    # ========================================================
    # CHECK SOLUTIONS
    # ========================================================

    if (
        u_mahal.value is None
        or u_second.value is None
        or u_exact.value is None
    ):

        print(
            f"Solver failed at T = {t}"
        )

        epu.append(np.nan)
        secu.append(np.nan)
        exact.append(np.nan)

        ratios1.append(np.nan)
        ratios3.append(np.nan)
        ratios5.append(np.nan)

        continue


    # ========================================================
    # OPTIMAL CONTROLS
    # ========================================================

    u_mahal_val = np.asarray(
        u_mahal.value
    ).reshape(-1)

    u_second_val = np.asarray(
        u_second.value
    ).reshape(-1)

    u_exact_val = np.asarray(
        u_exact.value
    ).reshape(-1)


    # ========================================================
    # OPTIMAL COSTS
    # ========================================================

    J_mahal = float(
        u_mahal_val.T
        @ Q_tilde
        @ u_mahal_val
    )

    J_second = float(
        u_second_val.T
        @ Q_tilde
        @ u_second_val
    )

    J_exact = float(
        u_exact_val.T
        @ Q_tilde
        @ u_exact_val
    )


    epu.append(J_mahal)
    secu.append(J_second)
    exact.append(J_exact)


    # ========================================================
    # RATIO / CONSTRAINT-SATISFACTION QUANTITIES
    # ========================================================

    # --------------------------------------------------------
    # Mahalanobis deterministic mean quantity
    # --------------------------------------------------------

    mu_mahal = (
        C_T @ u_mahal_val
        + b
    )

    mahal_mean_dev = float(
        mu_mahal.T
        @ sigma_z_inv
        @ mu_mahal
    )

    ratios3.append(
        mahal_mean_dev
    )


    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    rng = np.random.default_rng(0)

    second_mc = []

    exact_success = 0

    for _ in range(N_MC):

        x0_sample = rng.multivariate_normal(
            np.zeros(n),
            sigma_0
        )

        w_sample = rng.multivariate_normal(
            np.zeros(n * t),
            sigma_w
        )


        # ----------------------------------------------------
        # Second-moment controller
        # ----------------------------------------------------

        xT_second = (
            A_T @ x0_sample
            +
            C_T @ u_second_val
            +
            N_T @ w_sample
        )

        e_second = (
            xT_second
            - x_f
        )

        second_mc.append(
            float(
                e_second.T
                @ sigma_z_inv
                @ e_second 
            )
        )


        # ----------------------------------------------------
        # Exact noncentral chi-square controller
        # ----------------------------------------------------

        xT_exact = (
            A_T @ x0_sample
            +
            C_T @ u_exact_val
            +
            N_T @ w_sample
        )

        e_exact = (
            xT_exact
            - x_f
        )

        exact_stat = float(
            e_exact.T
            @ sigma_z_inv
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
        exact_success / N_MC
    )


# ============================================================
# DIAGNOSTIC:
# CHECK THAT THREE COST CURVES ARE PRESENT
# ============================================================

print("\n====================================")
print("COST ARRAYS")
print("====================================")

print(
    "Mahalanobis:",
    np.asarray(epu)
)

print(
    "2nd moment:",
    np.asarray(secu)
)

print(
    "Noncentral chi-square:",
    np.asarray(exact)
)

print(
    "\nMahalanobis == 2nd moment:",
    np.allclose(
        epu,
        secu,
        equal_nan=True
    )
)

print(
    "Mahalanobis == chi-square:",
    np.allclose(
        epu,
        exact,
        equal_nan=True
    )
)

print(
    "2nd moment == chi-square:",
    np.allclose(
        secu,
        exact,
        equal_nan=True
    )
)


# ============================================================
# FIGURE 1:
# RATIO / CONSTRAINT-SATISFACTION PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(3.45, 2.15)
)


ax.plot(
    T_horizon,
    ratios3,
    linestyle='--',
    linewidth=0.90,
    label=(
        r'$\hat{\mu}_e^\top'
        r'\Sigma_e^{-1}'
        r'\hat{\mu}_e$'
    )
)


ax.plot(
    T_horizon,
    ratios1,
    linestyle='--',
    markersize=2.4,
    linewidth=0.90,
    label=(
         r'$\hat{\mu}_e^\top'
        r'\Sigma_e^{-1}'
        r'\hat{\mu}_e + n$'
    )
)


ax.plot(
    T_horizon,
    ratios5,
    linestyle='--',
    linewidth=0.90,
    label=(
        r'$\hat{\mathbb{P}}'
        r'\{e^\top\Sigma_e^{-1}e'
        r'\leq\epsilon_1^2\}$'
    )
)


ax.axhline(
    y=epsilon**2,
    linestyle=':',
    linewidth=0.80,
    label=r'$\epsilon^2$'
)


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


# Add some empty space at the top so the legend
# does not overlap the plotted curves
ymin, ymax = ax.get_ylim()

ax.set_ylim(
    ymin,
    ymax + 0.25 * (ymax - ymin)
)

# Legend INSIDE, in the added empty region
ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    ncol=1,
    fontsize=5.7,
    handlelength=1.4,
    handletextpad=0.35,
    labelspacing=0.20,
    borderpad=0.25,
    frameon=True,
    framealpha=0.90
)


fig.subplots_adjust(
    left=0.16,
    right=0.99,
    bottom=0.19,
    top=0.98
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
# COST VS gamma FOR FIXED HORIZONS
# ============================================================

all_gamma = {}
all_cost = {}


for T in T_gamma:

    print(
        f"Solving gamma sweep for T = {T}"
    )


    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

    A, B, NT = get_system(
        name,
        n,
        m,
        T,
        seed=0
    )


    # --------------------------------------------------------
    # Construct N_T and C_T
    # --------------------------------------------------------

    N_T = np.eye(n)
    C_T = NT

    for i in range(1, T):

        N_T = np.hstack((
            N_T,
            np.linalg.matrix_power(A, i)
        ))


    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    sigma_w = np.eye(n * T) * 0.1

    A_T = np.linalg.matrix_power(
        A,
        T
    )

    sigma_z = (
        N_T
        @ sigma_w
        @ N_T.T
        +
        A_T
        @ sigma_0
        @ A_T.T
    )

    sigma_z_inv = np.linalg.inv(
        sigma_z
    )


    # --------------------------------------------------------
    # Mean offset
    # --------------------------------------------------------

    b = -x_f


    # --------------------------------------------------------
    # Cost matrix
    # --------------------------------------------------------

    Q_tilde = block_diag(
        *([Q] * T)
    )


    # --------------------------------------------------------
    # Maximum gamma corresponding to u = 0
    # --------------------------------------------------------

    gamma_max = float(
        b.T
        @ sigma_z_inv
        @ b
    )


    # --------------------------------------------------------
    # Gamma values
    # --------------------------------------------------------

    gamma_vals = np.linspace(
        0.0,
        gamma_max,
        10
    )


    cost_vals = []


    # ========================================================
    # GAMMA SWEEP
    # ========================================================

    for gamma in gamma_vals:

        u = cp.Variable(
            m * T
        )


        objective = cp.Minimize(
            cp.quad_form(
                u,
                Q_tilde
            )
        )


        constraint = (
            cp.quad_form(
                C_T @ u + b,
                sigma_z_inv
            )
            <= gamma
        )


        prob = cp.Problem(
            objective,
            [constraint]
        )


        prob.solve(
            solver=cp.SCS,
            verbose=False
        )


        if u.value is None:

            cost_vals.append(
                np.nan
            )

            continue


        u_val = np.asarray(
            u.value
        ).reshape(-1)


        Jstar = float(
            u_val.T
            @ Q_tilde
            @ u_val
        )


        cost_vals.append(
            Jstar
        )


    all_gamma[T] = gamma_vals

    all_cost[T] = cost_vals


# ============================================================
# FIGURE 2:
#
# LEFT  : J* vs gamma
# RIGHT : J* vs T
#
# NO deviation plots
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
    r'$J_{T}^\star$',
    labelpad=1.0
)


ax1.grid(
    True,
    which='both',
    linestyle=':',
    linewidth=0.30,
    alpha=0.28
)


# T legend INSIDE
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
#
# Distinct styles + markers ensure all three appear.
# ============================================================

ax2 = axs[1]


# ------------------------------------------------------------
# Mahalanobis
# ------------------------------------------------------------

ax2.semilogy(
    T_horizon,
    epu,
    linestyle='-',
    markevery=1,
    linewidth=0.90,
    label='Mahalanobis',
    zorder=4
)


# ------------------------------------------------------------
# Second moment
# ------------------------------------------------------------

ax2.semilogy(
    T_horizon,
    secu,
    linestyle='--',
    markevery=1,
    linewidth=0.90,
    label='2nd moment',
    zorder=3
)


# ------------------------------------------------------------
# Exact noncentral chi-square
# ------------------------------------------------------------

ax2.semilogy(
    T_horizon,
    exact,
    linestyle=':',
    markevery=1,
    linewidth=1.00,
    label=r'Noncentral $\chi^2$',
    zorder=5
)


ax2.set_xlabel(
    r'$T$'
)

ax2.set_ylabel(
    r'$J_{T}^\star$',
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


# Only method labels.
# NO empirical / theoretical labels.
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