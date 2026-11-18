#results for stable case
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from datagen import get_system
from functools import reduce
# from plot import opt, diff
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib.patches import Circle
from scipy.stats import ncx2
from scipy.optimize import brentq
import GPy
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# ============================================================
# Publication-quality plotting settings for Overleaf / IEEE
# Designed for TWO PANELS inside ONE IEEE column
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

    # thinner-looking math
    "mathtext.fontset": "cm",

    # explicitly avoid bold axis text
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
# Consistent tick-label appearance across ALL exported figures
# ============================================================
TICK_SIZE = 7.0

def format_ticks(ax):
    """Force identical x/y tick-label font sizes on every axis."""
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, pad=1.5)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
#----------------------------------------------------------------------------------------
# fixed system parameters

n, m = 4, 4
sigma_0 = np.eye(n) * 2.0
x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)
x_f = np.array([4.0, 5.0, 6.0, 7.0])
T_list = range(1, 30, 2)  
sigma = np.eye(n) * 0.1
delta = 0.4
Q = np.diag([2.0, 3.0, 4.0, 5.0])
q = norm.ppf(1 - delta)
name = 'stable'
delta_folder = 'traj_{}'.format(delta)
epsilon = 2.4
os.makedirs(delta_folder, exist_ok = True)

#normalized norm
secu = []
epu = []
exact = []

#absolute norm
abs_secu = []
abs_exactu = []
abs_chebyu = []

#normalized dev
x1_dev = []
x3_dev = []
x5_dev = []

#absolute dev
abs_secdev = []
abs_exactdev = []
abs_chebydev = []

tdev1 = []
tdev3 = []
tdev5 = []

abs_tsecdev = []
abs_texactdev = []
abs_tchebydev = []

ratios1 = []
ratios3 = []
ratios5 = []

abs_ratiosec = []
abs_ratioexact = []
abs_ratiocheby = []

bT = []
abs_bT = []

#numerically find lambda*
def find_lambda_star(n, epsilon, delta):
    """
    Solves F_{chi^2_n(lambda)}(epsilon^2) = delta
    for lambda >= 0.
    Parameters:
    n       : degrees of freedom
    epsilon : radius (so we evaluate CDF at epsilon^2)
    delta   : probability level
    """
    x = epsilon**2

    # Define equation to solve
    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - delta

    # We search in a reasonable interval for lambda
    # Lower bound = 0
    # Upper bound chosen large enough
    lam_lower = 0.0
    lam_upper = 1e11

    # Ensure bracket contains root
    if equation(lam_lower) * equation(lam_upper) > 0:
        raise ValueError("Root not bracketed. Increase lam_upper.")

    lambda_star = brentq(equation, lam_lower, lam_upper)
    
    return lambda_star

#optimal lagrange multiplier
def solve_lambda(C_T, Qp, V, b, gamma, tol=1e-10, max_iter=1000):
    """
    Computes optimal lambda*.

    Parameters
    ----------
    C_T : matrix
    Qp : matrix (Q')
    V : matrix
    b : vector
    gamma : float
    """

    # V^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(V)
    V_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Q'^{-1}
    Qp_inv = np.linalg.inv(Qp)

    # build symmetric matrix
    M = V_inv_half @ C_T @ Qp_inv @ C_T.T @ V_inv_half

    # eigen decomposition
    delta, W = np.linalg.eigh(M)

    # compute b_tilde
    y = V_inv_half @ b
    b_tilde = W.T @ y

    # function to solve
    def f(lam):
        return np.sum((b_tilde**2) / (1 + lam * delta)**2) - gamma

    # bracket root
    lo = 0.0
    hi = 1.0
    while f(hi) > 0:
        hi *= 2

    # bisection
    for _ in range(max_iter):
        print(hi)
        mid = 0.5 * (lo + hi)
        val = f(mid)

        if abs(val) < tol:
            return mid

        if val > 0:
            lo = mid
        else:
            hi = mid

    return mid


for t in T_list:
    A,B, NT = get_system(name, n, m, t, seed=0)
    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = NT
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    sigma_w = np.eye(n*t) * 0.1
    sigma_z =  N_T @ sigma_w @ N_T.T + np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T
    traceSigma = np.trace(sigma_z)

    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = -1*  x_f
    vals = np.linalg.eigvals(sigma_z)
    eig = np.max(vals)
    gamma = epsilon**2
    dev = epsilon**2  #epsilon sd
    lambda_star = find_lambda_star(n, epsilon, 1-delta)
    marc = (epsilon - np.sqrt(-2 * math.log(1-delta)))**2

    cheby = (epsilon - np.sqrt(np.trace(sigma_z)/delta))**2
    sec = epsilon**2 - np.trace(sigma_z)
    Q_tilde = block_diag(*([Q] * t))


    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u3 = cp.Variable(m*t)
    u5 = cp.Variable(m*t)

    u1 = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    u4 = cp.Variable(m*t)

    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective5 = cp.Minimize(cp.quad_form(u5, Q_tilde))

    objective1 = cp.Minimize(cp.quad_form(u1, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))

    f = cp.quad_form(C_T @ u + b, np.linalg.inv(sigma_z))
    f3 = cp.quad_form(C_T @ u3 + b, np.linalg.inv(sigma_z))
    f5 = cp.quad_form(C_T@ u5 + b, np.linalg.inv(sigma_z))

    f1 = cp.quad_form(C_T @ u1 + b, np.identity(n))
    f2 = cp.quad_form(C_T @ u2 + b, np.identity(n))
    f4 = cp.quad_form(C_T @ u4 + b, np.identity(n))


    prob = cp.Problem(objective, [f <= gamma]) #second order
    prob3 = cp.Problem(objective3, [f3 <= dev]) #mahalanobis
    prob5 = cp.Problem(objective5, [f5 <= lambda_star]) #non-central chi

    prob1 = cp.Problem(objective1, [f1 <= lambda_star])# absolute exact
    prob2 = cp.Problem(objective2, [f2 <= sec])# absolute second order
    prob4 = cp.Problem(objective4, [f4 <= cheby])# absolute chebyshev

    prob.solve(solver=cp.SCS, verbose=False)
    prob3.solve(solver = cp.SCS, verbose = False)
    prob5.solve(solver = cp.SCS, verbose = False)

    prob1.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver=cp.SCS, verbose=False)
    prob4.solve(solver=cp.SCS, verbose=False)

    u_convex = u.value  #second order
    u_c3 = u3.value #epsilon sd
    u_c5 = u5.value # non-central

    u_c1 = u1.value #exact absolute
    u_c2 = u2.value # second order absolute
    u_c4 = u4.value # cheyshev absolute

    if u_convex is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c3 is None:
        print(f'Convex solver failed at T={t}')
    if u_c5 is None:
        print(f'Convex solver failed at T={t}') 

    if u_c1 is None:
        print(f"Convex solver failed at T={t}")
    if u_c2 is None:
        print(f"Convex solver failed at T={t}")
    if u_c4 is None:
        print(f"Convex solver failed at T={t}")        

    bT.append(b.T @ np.linalg.inv(sigma_z) @ b)  
    abs_bT.append(b.T @ b)  
    # --- Simulate both systems ---
    np.random.seed(0)
    x1 = []
    x3 = []
    x5 = []
    xm = []

    x2 = [] #absolute exact
    x4 = [] # absolute second
    x6 = [] #absolute cheby

    s1 = []
    s3 = []
    s5 = []

    s2 = []
    s4 = []
    s6 = []

    for j in range(0, 5000):
        x_0 = np.random.multivariate_normal(np.zeros(n), sigma_0)
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_convex + N_T @ w  # second order
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c3 + N_T @ w  # epsilon 
        x5T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c5 + N_T @ w #non-central
        
        x2T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c1 + N_T @ w # exact absolute
        x4T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c2 + N_T @ w # second absolute
        x6T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c4 + N_T @ w # chebyshev absolute

        
        x1.append((x1T-x_f).T @ np.linalg.inv(sigma_z) @ (x1T-x_f))
        x3.append((x3T-x_f))
        xm.append((x3T-x_f) @ np.linalg.inv(sigma_z) @ (x3T-x_f))
        
        x5.append((x5T - x_f).T @ np.linalg.inv(sigma_z) @ (x5T - x_f))
        count5 = sum(d <= epsilon**2 for d in x5)

        x2.append((x2T - x_f))# exact absolute
        x4.append((x4T - x_f)) # second absolute
        x6.append((x6T - x_f)) # cheby absolute

    tdev1.append((C_T @ u_convex + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_convex + b)+n)
    tdev3.append((C_T @ u_c3 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c3 + b)+n )
    tdev5.append((C_T @ u_c5 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c5 + b)+n)

    abs_texactdev.append((C_T @ u_c1 + b).T @ (C_T @ u_c1 + b))
    abs_tsecdev.append((C_T @ u_c2 + b).T @ (C_T @ u_c2 + b))
    abs_tchebydev.append((C_T @ u_c4 + b).T @ (C_T @ u_c4 + b))
    
    count1 = np.mean(x1, axis = 0)  
    ratios1.append(count1)
    count3 = np.mean(x3, axis = 0) @ np.linalg.inv(sigma_z) @ np.mean(x3, axis = 0)
    ratios3.append(count3)
    ratios5.append(count5/5000)
    count6 = (x3T-x_f).T @np.linalg.inv(sigma_z) @ (x3T - x_f)

    x1_dev.append((np.mean(x1, axis = 0)))
    x3_dev.append(np.mean(xm, axis = 0))
    x5_dev.append((np.mean(x5, axis = 0)))
    abs_exactdev.append(np.mean(x2, axis = 0) @ np.mean(x2, axis = 0))# deviation for abs exact
    abs_secdev.append(np.mean(x4, axis = 0) @ np.mean(x4, axis = 0)) # deviation for abs second
    abs_chebydev.append(np.mean(x6, axis = 0) @ np.mean(x6, axis = 0)) # deviation for abs chebyshev

    # --- Record metrics ---
    u_convex = u.value.reshape(-1, 1)
    u_c3  = u3.value.reshape(-1, 1)
    u_c5 = u5.value.reshape(-1, 1)

    u_c1 = u1.value.reshape(-1, 1)
    u_c2 = u2.value.reshape(-1, 1)
    u_c4 = u4.value.reshape(-1, 1)

    ep_norm = float(u_c3.T @ Q_tilde @ u_c3)
    sec_norm = float(u_convex.T @ Q_tilde @ u_convex)
    exact_norm = float(u_c5.T @ Q_tilde @ u_c5)

    abs_exactnorm = float(u_c1.T @ Q_tilde @ u_c1)
    abs_secnorm = float(u_c2.T @ Q_tilde @ u_c2)
    abs_chebynorm = float(u_c4.T @ Q_tilde @ u_c4)

    epu.append(ep_norm)
    secu.append(sec_norm)
    exact.append(exact_norm)

    abs_exactu.append(abs_exactnorm)
    abs_secu.append(abs_secnorm)
    abs_chebyu.append(abs_chebynorm)

N = np.arange(0, len(ratios1))


# ============================================================
# Figure 1: quantities entering the normalized-norm constraint
# SINGLE-COLUMN FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(3.45, 2.15))

ax.plot(
    T_list,
    ratios3,
    '--',
    linewidth=0.90,
    label=r'$\hat{\mu}_e^\top\Sigma_e^{-1}\hat{\mu}_e$'
)

ax.plot(
    T_list,
    ratios1,
    '--',
    linewidth=0.90,
    label=r'$\hat{\mathbb{E}}[e^\top\Sigma_e^{-1}e]$'
)

ax.plot(
    T_list,
    ratios5,
    '--',
    linewidth=0.90,
    label=r'$\mathbb{P}\{e^\top\Sigma_e^{-1}e\leq\epsilon^2\}$'
)

ax.axhline(
    y=epsilon**2,
    linestyle=':',
    linewidth=0.80,
    label=r'$\epsilon^2$'
)

ax.set_xlabel(r'$T$')
ax.set_xticks([1, 7, 13, 19, 25, 29])

ax.grid(
    True,
    linestyle=':',
    linewidth=0.35,
    alpha=0.30
)

# Put the legend INSIDE the plot
ax.legend(
    loc='best',
    ncol=1,
    fontsize=6.0,
    handlelength=1.5,
    handletextpad=0.4,
    labelspacing=0.25,
    borderpad=0.3,
    frameon=True,
    framealpha=0.85
)

fig.subplots_adjust(
    left=0.16,
    right=0.99,
    bottom=0.19,
    top=0.97
)

format_ticks(ax)

fig.savefig(
    'ratio.pdf'
)
plt.close(fig)


# ============================================================
# Figure 2: deviation and optimal cost versus horizon
# TWO PANELS SIDE-BY-SIDE IN ONE IEEE COLUMN
# ============================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(3.45, 1.90),
    constrained_layout=False
)

# ------------------------------------------------------------
# Left panel: empirical and theoretical deviation
# Same color = same method
# Solid = empirical, dotted = theoretical
# NO MARKERS
# ------------------------------------------------------------
ax1 = axes[0]

# Mahalanobis
p1, = ax1.plot(
    T_list,
    x3_dev,
    linestyle='--',
    linewidth=0.80,
    label='Mahalanobis'
)
ax1.plot(
    T_list,
    tdev3,
    linestyle='-',
    linewidth=0.65,
    color=p1.get_color()
)

# Second-order moment
p2, = ax1.plot(
    T_list,
    x1_dev,
    linestyle='--',
    linewidth=0.80,
    label='2nd moment'
)
ax1.plot(
    T_list,
    tdev1,
    linestyle='-',
    linewidth=0.65,
    color=p2.get_color()
)

# Noncentral chi-square
p3, = ax1.plot(
    T_list,
    x5_dev,
    linestyle='--',
    linewidth=0.80,
    label=r'Noncentral $\chi^2$'
)
ax1.plot(
    T_list,
    tdev5,
    linestyle='-',
    linewidth=0.65,
    color=p3.get_color()
)

ax1.set_xlabel(r'$T$')
ax1.set_ylabel(
    r'$\mathbb{E}\!\left[e^\top\Sigma_e^{-1}e\right]$',
    labelpad=1.5
)
ax1.set_xticks([1, 7, 13, 19, 25])

ax1.grid(
    True,
    linestyle=':',
    linewidth=0.30,
    alpha=0.28
)
format_ticks(ax1)


# ------------------------------------------------------------
# Right panel: optimal cost
# J* stays on the LEFT side of the right subplot
# ------------------------------------------------------------
ax2 = axes[1]
ax2.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f'{y:.2f}')
)


ax2.semilogy(
    T_list,
    epu,
    linestyle='-',
    linewidth=0.80,
    color=p1.get_color()
)

ax2.semilogy(
    T_list,
    secu,
    linestyle='-',
    linewidth=0.80,
    color=p2.get_color()
)

ax2.semilogy(
    T_list,
    exact,
    linestyle='-',
    linewidth=0.80,
    color=p3.get_color()
)

ax2.set_xlabel(r'$T$')

# IMPORTANT:
# Keep J* on the LEFT of the second subplot.
# Small labelpad + larger wspace prevents it from spilling into the first plot.
ax2.set_ylabel(
    r'$J^\star$',
    rotation=90,
    labelpad=1.0
)

ax2.yaxis.set_label_position('left')
ax2.yaxis.tick_left()

ax2.set_xticks([1, 7, 13, 19, 25])

ax2.grid(
    True,
    which='both',
    linestyle=':',
    linewidth=0.30,
    alpha=0.28
)
format_ticks(ax2)


# ------------------------------------------------------------
# One compact shared legend
# No markers
# ------------------------------------------------------------
method_handles = [
    Line2D(
        [0], [0],
        color=p1.get_color(),
        linestyle='--',
        linewidth=0.80,
        label='Mahal.'
    ),
    Line2D(
        [0], [0],
        color=p2.get_color(),
        linestyle='--',
        linewidth=0.80,
        label='2nd moment'
    ),
    Line2D(
        [0], [0],
        color=p3.get_color(),
        linestyle='--',
        linewidth=0.80,
        label=r'Noncentral $\chi^2$'
    ),
    Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        linewidth=0.80,
        label='Empirical'
    ),
    Line2D(
        [0], [0],
        color='black',
        linestyle='-',
        linewidth=0.65,
        label='Theoretical'
    )
]

fig.legend(
    handles=method_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.99),
    ncol=5,
    frameon=False,
    fontsize=6.4,
    columnspacing=0.70,
    handlelength=1.45,
    handletextpad=0.30,
    borderaxespad=0.0
)

# IMPORTANT:
# Larger wspace gives room for the second subplot's LEFT y-axis ticks + J*
# without overlapping the first subplot.
fig.subplots_adjust(
    top=0.79,
    bottom=0.23,
    left=0.17,
    right=0.99,
    wspace=0.68
)

fig.savefig(
    'combined_T.pdf',
    format='pdf'
)

# Optional raster preview only; use the PDF in Overleaf.
fig.savefig(
    'combined_T.png',
    dpi=600
)

plt.show()


# ============================================================
# Figure 3: feasibility-related quantities versus horizon
# TWO PANELS SIDE-BY-SIDE IN ONE IEEE COLUMN
# ============================================================
fig, axs = plt.subplots(
    1, 2,
    figsize=(3.48, 1.85),
    constrained_layout=False
)

# ---------------- Left ----------------
axs[0].plot(
    T_list,
    bT,
    '-',
    linewidth=0.90
)

axs[0].axhline(
    y=lambda_star,
    linestyle='--',
    linewidth=0.80
)

axs[0].set_xlabel(r'$T$')
axs[0].set_ylabel(r'$x_f^\top\Sigma_e^{-1}x_f$')
axs[0].set_xticks([1, 7, 13, 19, 25])

axs[0].grid(
    True,
    linestyle=':',
    linewidth=0.35,
    alpha=0.30
)

# ---------------- Right ----------------
axs[1].plot(
    T_list,
    abs_bT,
    '-',
    linewidth=0.90
)

axs[1].axhline(
    y=lambda_star,
    linestyle='--',
    linewidth=0.80
)

axs[1].set_xlabel(r'$T$')
axs[1].set_ylabel(r'$x_f^\top x_f$')
axs[1].set_xticks([1, 7, 13, 19, 25])

axs[1].grid(
    True,
    linestyle=':',
    linewidth=0.35,
    alpha=0.30
)

# Shared legend
shared_handles = [
    Line2D(
        [0], [0],
        linestyle='-',
        linewidth=0.90,
        label='Deviation'
    ),
    Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        linewidth=0.90,
        label=r'$\gamma$'
    )
]

fig.legend(
    handles=shared_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    fontsize=6.3,
    handlelength=1.5,
    columnspacing=1.1,
    handletextpad=0.35,
    borderaxespad=0.0
)

fig.subplots_adjust(
    top=0.78,
    bottom=0.22,
    left=0.15,
    right=0.99,
    wspace=0.48
)

for ax in axs:
    format_ticks(ax)

fig.savefig(
    f'combined_bT_{epsilon}_{delta}_{name}.pdf'
)

plt.show()


# --------------------------------------------------
# Cost vs gamma for fixed horizons
# --------------------------------------------------
T_list = [4, 6, 8]
lambda_star = find_lambda_star(n, epsilon, 1-delta)

all_gamma = {}
all_cost = {}
all_lambda = {}
all_dev = {}
all_cost1 = {}
all_dev1 = {}

for T in T_list:

    A, B, NT = get_system(name, n, m, T, seed=0)

    N_T = np.eye(n)
    C_T = NT
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))

    b = -1 * x_f

    sigma_w = np.eye(n*T) * 0.1
    sigma_z = (
        N_T @ sigma_w @ N_T.T
        + np.linalg.matrix_power(A, T)
        @ sigma_0
        @ np.linalg.matrix_power(A, T).T
    )

    Q_tilde = block_diag(*([Q] * T))

    gamma_min = 0.0
    gamma_max = float(
        b.T @ np.linalg.inv(sigma_z) @ b
    )

    gamma_vals = np.linspace(
        gamma_min,
        gamma_max,
        10
    )

    cost_vals = []
    lambda_vals = []
    dev_vals = []
    cost1_vals = []
    dev1_vals = []

    for gamma in gamma_vals:

        u = cp.Variable(m*T)
        u1 = cp.Variable(m*T)

        objective = cp.Minimize(
            cp.quad_form(u, Q_tilde)
        )

        objective1 = cp.Minimize(
            cp.quad_form(u1, Q_tilde)
        )

        constraint = (
            cp.quad_form(
                C_T @ u + b,
                np.linalg.inv(sigma_z)
            ) <= gamma
        )

        constraint1 = (
            cp.quad_form(
                C_T @ u1 + b,
                np.identity(n)
            ) <= gamma
        )

        prob = cp.Problem(
            objective,
            [constraint]
        )

        prob1 = cp.Problem(
            objective1,
            [constraint1]
        )

        prob.solve(
            solver=cp.SCS,
            verbose=False
        )

        prob1.solve(
            solver=cp.SCS,
            verbose=False
        )

        if u.value is None:
            cost_vals.append(np.nan)
            lambda_vals.append(np.nan)
            dev_vals.append(np.nan)
            continue

        if u1.value is None:
            cost1_vals.append(np.nan)
            dev1_vals.append(np.nan)
            continue

        Jstar = float(
            u.value.T
            @ Q_tilde
            @ u.value
        )

        J1star = float(
            u1.value.T
            @ Q_tilde
            @ u1.value
        )

        cost_vals.append(Jstar)
        cost1_vals.append(J1star)

        idev = []
        idev1 = []

        for _ in range(5000):

            x_0 = np.random.multivariate_normal(
                np.zeros(n),
                np.eye(n) * 0.2
            )

            w = np.random.multivariate_normal(
                np.zeros(n*T),
                np.eye(n*T) * 0.1
            )

            xT = (
                np.linalg.matrix_power(A, T) @ x_0
                + C_T @ u.value
                + N_T @ w
            )

            xT1 = (
                np.linalg.matrix_power(A, T) @ x_0
                + C_T @ u1.value
                + N_T @ w
            )

            e = xT - x_f
            e1 = xT1 - x_f

            idev.append(e.T @ e)
            idev1.append(e1.T @ e1)

        dev_vals.append(
            (C_T @ u.value + b).T
            @ np.linalg.inv(sigma_z)
            @ (C_T @ u.value + b)
        )

        dev1_vals.append(
            np.mean(idev1)
        )

    all_gamma[T] = gamma_vals
    all_cost[T] = cost_vals
    all_cost1[T] = cost1_vals
    all_lambda[T] = lambda_vals
    all_dev[T] = dev_vals
    all_dev1[T] = dev1_vals


# ============================================================
# Figure 4: optimal cost and deterministic deviation vs gamma
# TWO PANELS SIDE-BY-SIDE IN ONE IEEE COLUMN
# ============================================================
fig, axs = plt.subplots(
    1, 2,
    figsize=(3.48, 1.85),
    constrained_layout=False
)

gamma_lines = []

# ---------------- Left: cost vs gamma ----------------
for T in T_list:

    line, = axs[0].semilogy(
        all_gamma[T],
        all_cost[T],
        linewidth=0.90,
        label=rf'$T={T}$'
    )

    gamma_lines.append(line)

axs[0].set_xlabel(r'$\gamma$')
axs[0].set_ylabel(r'$J^\star(\gamma)$')

axs[0].grid(
    True,
    which='both',
    linestyle=':',
    linewidth=0.35,
    alpha=0.30
)

# ---------------- Right: deterministic deviation vs gamma ----------------
for line, T in zip(gamma_lines, T_list):

    axs[1].semilogy(
        all_gamma[T],
        all_dev[T],
        linewidth=0.90,
        color=line.get_color()
    )

axs[1].set_xlabel(r'$\gamma$')
axs[1].set_ylabel(
    r'$\mu_e^\top\Sigma_e^{-1}\mu_e$'
)

axs[1].grid(
    True,
    which='both',
    linestyle=':',
    linewidth=0.35,
    alpha=0.30
)

# One legend for both panels
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.99),
    ncol=3,
    fontsize=6.4,
    handlelength=1.5,
    columnspacing=1.0,
    handletextpad=0.35,
    borderaxespad=0.0
)

fig.subplots_adjust(
    top=0.78,
    bottom=0.22,
    left=0.15,
    right=0.99,
    wspace=0.48
)

for ax in axs:
    format_ticks(ax)

fig.savefig(
    f'combined_gamma_plots_{name}.pdf'
)

plt.show()

# ============================================================
# STABLE CASE:
#   1. Ratio / constraint-satisfaction plot vs T
#   2. Cost vs gamma and cost vs T side-by-side
#
# NO deviation plots
# ============================================================

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
        r'$\mathbb{P}'
        r'\{e^\top\Sigma_e^{-1}e'
        r'\leq\epsilon^2\}$'
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


# Legend INSIDE
ax.legend(
    loc='best',
    ncol=1,
    fontsize=6.0,
    handlelength=1.5,
    handletextpad=0.4,
    labelspacing=0.25,
    borderpad=0.3,
    frameon=True,
    framealpha=0.85
)


fig.subplots_adjust(
    left=0.16,
    right=0.99,
    bottom=0.19,
    top=0.97
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