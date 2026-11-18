
import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.stats import norm, ncx2, chi2
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import math
import os

from datagen import get_system


# ============================================================
# SYSTEM / EXPERIMENT PARAMETERS
# ============================================================

np.random.seed(0)

n, m = 4, 4

sigma_0 = 2.0 * np.eye(n)
x0_mean = np.zeros(n)

x_f = np.array([4.0, 5.0, 6.0, 7.0])

# Horizons
T_list = list(range(1, 30, 2))

# Process noise covariance
W = 0.1 * np.eye(n)

# Control cost
Q = np.diag([2.0, 3.0, 4.0, 5.0])

# Chance constraint
epsilon = 2.4
delta = 0.4

# Required probability
p_req = 1.0 - delta

name = "stable"

N_SIM = 5000


# ============================================================
# EXACT NON-CENTRAL CHI-SQUARE PARAMETER
# ============================================================

def find_lambda_star(n, epsilon, delta_cdf):
    """
    Find lambda satisfying

        F_{ncx2(n, lambda)}(epsilon^2) = delta_cdf.

    For the chance constraint

        P(||e||^2 <= epsilon^2) >= 1-delta,

    the deterministic constraint is

        ||Sigma_z^{-1/2} mu_e||^2 <= lambda_star,

    where delta_cdf = delta.
    """
    x = epsilon ** 2

    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - delta_cdf

    lo = 0.0
    hi = 1.0

    while equation(hi) < 0:
        hi *= 2.0

        if hi > 1e12:
            raise ValueError("Could not bracket lambda_star.")

    return brentq(equation, lo, hi)


lambda_star = find_lambda_star(
    n=n,
    epsilon=epsilon,
    delta_cdf=delta
)

print("lambda_star =", lambda_star)


# ============================================================
# COVARIANCE TARGET FOR CS
# ============================================================
#
# We choose an isotropic terminal covariance
#
#       Sigma_f = sigma_f^2 I
#
# such that, for a zero-mean Gaussian terminal error,
#
#       P(||e|| <= epsilon) = 1-delta.
#
# Since ||e||^2 / sigma_f^2 ~ chi2_n,
#
#       sigma_f^2 = epsilon^2 / chi2.ppf(1-delta,n).
#
# This is a probability-matched CS baseline, not an exact
# equivalence to the non-central-chi-square MEC constraint.
# ============================================================

sigma_f_sq = epsilon ** 2 / chi2.ppf(
    p_req,
    df=n
)

Sigma_f = sigma_f_sq * np.eye(n)

print("Sigma_f =")
print(Sigma_f)


# ============================================================
# BUILD HORIZON MATRICES
# ============================================================

def build_horizon_matrices(A, B, T):
    """
    Construct

        C_T = [A^(T-1)B ... B]

    for the convention

        x_T = A^T x_0 + C_T u + N_T w,

    where u = [u_0; ...; u_{T-1}].

    Also constructs

        N_T = [A^(T-1) ... I].
    """

    n = A.shape[0]
    m = B.shape[1]

    C_T = np.zeros((n, m * T))
    N_T = np.zeros((n, n * T))

    for k in range(T):
        # contribution of u_k to x_T
        C_T[:, k*m:(k+1)*m] = np.linalg.matrix_power(
            A, T - 1 - k
        ) @ B

        # contribution of w_k to x_T
        N_T[:, k*n:(k+1)*n] = np.linalg.matrix_power(
            A, T - 1 - k
        )

    return C_T, N_T


# ============================================================
# MEC: EXACT NON-CENTRAL CHI-SQUARE SOLVER
# ============================================================

def solve_mec_exact(A, B, sigma0, W, x_f,
                    Q, T, epsilon, delta):
    """
    Solve the exact Gaussian terminal chance-constrained
    minimum-energy problem used in the user's original code.

    The terminal uncontrolled covariance is

        Sigma_z =
            A^T Sigma0 A^(T)'
            + N_T Sigma_w N_T'.

    The exact chance constraint becomes

        (C_T u - x_f)' Sigma_z^{-1}
        (C_T u - x_f)
        <= lambda_star.
    """

    n = A.shape[0]
    m = B.shape[1]

    C_T, N_T = build_horizon_matrices(A, B, T)

    Sigma_w = np.kron(
        np.eye(T),
        W
    )

    A_T = np.linalg.matrix_power(A, T)

    Sigma_z = (
        A_T @ sigma0 @ A_T.T
        + N_T @ Sigma_w @ N_T.T
    )

    Q_T = block_diag(*([Q] * T))

    lambda_star = find_lambda_star(
        n=n,
        epsilon=epsilon,
        delta_cdf=delta
    )

    u = cp.Variable(m * T)

    mean_error = C_T @ u - x_f

    constraint = cp.quad_form(
        mean_error,
        np.linalg.inv(Sigma_z)
    ) <= lambda_star

    problem = cp.Problem(
        cp.Minimize(cp.quad_form(u, Q_T)),
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
            eps=1e-5,
            max_iters=100000
        )

    if problem.status not in [
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE
    ]:
        return None

    u_star = np.asarray(u.value).reshape(-1)

    return {
        "cost": float(
            u_star.T @ Q_T @ u_star
        ),
        "u": u_star,
        "C_T": C_T,
        "N_T": N_T,
        "Sigma_z": Sigma_z,
        "lambda_star": lambda_star
    }


# ============================================================
# COVARIANCE STEERING SOLVER
# ============================================================

def solve_covariance_steering(
        A, B, Sigma0, Sigma_f,
        x0_mean, x_f, Q, W, T):

    """
    Finite-horizon affine covariance steering.

    Policy:

        u_k = v_k + K_k (x_k - mu_k)

    Mean dynamics:

        mu_{k+1} = A mu_k + B v_k

    Covariance dynamics:

        Sigma_{k+1}
          = (A+B K_k) Sigma_k (A+B K_k)' + W

    The SDP uses

        Y_k = K_k Sigma_k

    and Schur-complement LMIs.

    Terminal constraints:

        mu_T = x_f
        Sigma_T <= Sigma_f

    Objective:

        sum E[u_k' Q u_k].
    """

    n = A.shape[0]
    m = B.shape[1]

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    mu = cp.Variable((n, T + 1))
    v = cp.Variable((m, T))

    Sigma = [
        cp.Variable((n, n), symmetric=True)
        for _ in range(T + 1)
    ]

    Y = [
        cp.Variable((m, n))
        for _ in range(T)
    ]

    U = [
        cp.Variable((m, m), symmetric=True)
        for _ in range(T)
    ]

    constraints = []

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    constraints += [
        mu[:, 0] == x0_mean,
        Sigma[0] == Sigma0
    ]

    # --------------------------------------------------------
    # Mean and covariance dynamics
    # --------------------------------------------------------

    for k in range(T):

        # Mean
        constraints += [
            mu[:, k+1]
            == A @ mu[:, k] + B @ v[:, k]
        ]

        # Covariance:
        #
        # Sigma_{k+1} >=
        # (A Sigma_k + B Y_k)
        # Sigma_k^{-1}
        # (A Sigma_k + B Y_k)' + W
        #
        # via Schur complement.

        AS_BY = (
            A @ Sigma[k]
            + B @ Y[k]
        )

        cov_lmi = cp.bmat([
            [
                Sigma[k+1] - W,
                AS_BY
            ],
            [
                AS_BY.T,
                Sigma[k]
            ]
        ])

        constraints += [
            cov_lmi >> 0
        ]

        # ----------------------------------------------------
        # U_k >= K Sigma K'
        # ----------------------------------------------------

        control_lmi = cp.bmat([
            [
                U[k],
                Y[k]
            ],
            [
                Y[k].T,
                Sigma[k]
            ]
        ])

        constraints += [
            control_lmi >> 0
        ]

        constraints += [
            Sigma[k] >> 1e-7 * np.eye(n),
            U[k] >> 1e-7 * np.eye(m)
        ]

    constraints += [
        Sigma[T] >> 1e-7 * np.eye(n)
    ]

    # --------------------------------------------------------
    # Terminal mean/covariance constraints
    # --------------------------------------------------------

    constraints += [
        mu[:, T] == x_f,
        Sigma[T] << Sigma_f
    ]

    # --------------------------------------------------------
    # Expected control cost
    #
    # E[u'Qu]
    # =
    # v'Qv + tr(Q K Sigma K')
    #
    # <= v'Qv + tr(Q U)
    # --------------------------------------------------------

    objective = 0

    for k in range(T):

        objective += cp.quad_form(
            v[:, k], Q
        )

        objective += cp.trace(
            Q @ U[k]
        )

    problem = cp.Problem(
        cp.Minimize(objective),
        constraints
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
            eps=1e-5,
            max_iters=100000
        )

    if problem.status not in [
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE
    ]:
        return None

    # --------------------------------------------------------
    # Recover feedback gains
    # --------------------------------------------------------

    K = []

    for k in range(T):

        Sigma_k = Sigma[k].value
        Y_k = Y[k].value

        K_k = Y_k @ np.linalg.inv(
            Sigma_k + 1e-8 * np.eye(n)
        )

        K.append(K_k)

    return {
        "cost": float(problem.value),
        "mu": mu.value,
        "v": v.value,
        "Sigma": [
            S.value for S in Sigma
        ],
        "Y": [
            Y_k.value for Y_k in Y
        ],
        "U": [
            U_k.value for U_k in U
        ],
        "K": K
    }


# ============================================================
# SIMULATE OPEN-LOOP MEC
# ============================================================

def simulate_mec(
        A, B, Sigma0, W,
        x_f, mec_result, T,
        epsilon, N_sim=5000):

    u = mec_result["u"]

    m = B.shape[1]

    u = u.reshape(T, m)

    terminal_errors = []

    for _ in range(N_sim):

        x = np.random.multivariate_normal(
            np.zeros(A.shape[0]),
            Sigma0
        )

        for k in range(T):

            w = np.random.multivariate_normal(
                np.zeros(A.shape[0]),
                W
            )

            x = (
                A @ x
                + B @ u[k]
                + w
            )

        terminal_errors.append(
            x - x_f
        )

    terminal_errors = np.asarray(
        terminal_errors
    )

    dev = np.mean(
        np.sum(
            terminal_errors ** 2,
            axis=1
        )
    )

    probability = np.mean(
        np.sum(
            terminal_errors ** 2,
            axis=1
        ) <= epsilon ** 2
    )

    return dev, probability, terminal_errors


# ============================================================
# SIMULATE COVARIANCE STEERING
# ============================================================

def simulate_cs(
        A, B, Sigma0, W,
        x_f, cs_result, T,
        epsilon, N_sim=5000):

    mu = cs_result["mu"]
    v = cs_result["v"]
    K = cs_result["K"]

    n = A.shape[0]

    terminal_errors = []

    for _ in range(N_sim):

        x = np.random.multivariate_normal(
            mu[:, 0],
            Sigma0
        )

        for k in range(T):

            # Affine feedback policy
            u = (
                v[:, k]
                + K[k] @ (
                    x - mu[:, k]
                )
            )

            w = np.random.multivariate_normal(
                np.zeros(n),
                W
            )

            x = (
                A @ x
                + B @ u
                + w
            )

        terminal_errors.append(
            x - x_f
        )

    terminal_errors = np.asarray(
        terminal_errors
    )

    dev = np.mean(
        np.sum(
            terminal_errors ** 2,
            axis=1
        )
    )

    probability = np.mean(
        np.sum(
            terminal_errors ** 2,
            axis=1
        ) <= epsilon ** 2
    )

    return dev, probability, terminal_errors


# ============================================================
# MAIN EXPERIMENT
# ============================================================

mec_cost = []
mec_dev = []
mec_prob = []

cs_cost = []
cs_dev = []
cs_prob = []

# Also store terminal covariance for CS
cs_terminal_trace = []

valid_T = []

for T in T_list:

    print("\n----------------------------------------")
    print(f"T = {T}")
    print("----------------------------------------")

    # --------------------------------------------------------
    # Get system
    # --------------------------------------------------------

    A, B, _ = get_system(
        name,
        n,
        m,
        T,
        seed=0
    )

    # --------------------------------------------------------
    # Solve MEC
    # --------------------------------------------------------

    mec_result = solve_mec_exact(
        A=A,
        B=B,
        sigma0=sigma_0,
        W=W,
        x_f=x_f,
        Q=Q,
        T=T,
        epsilon=epsilon,
        delta=delta
    )

    if mec_result is None:

        print("MEC failed.")

        mec_cost.append(np.nan)
        mec_dev.append(np.nan)
        mec_prob.append(np.nan)

        cs_cost.append(np.nan)
        cs_dev.append(np.nan)
        cs_prob.append(np.nan)

        cs_terminal_trace.append(np.nan)

        continue

    # --------------------------------------------------------
    # Solve CS
    # --------------------------------------------------------

    cs_result = solve_covariance_steering(
        A=A,
        B=B,
        Sigma0=sigma_0,
        Sigma_f=Sigma_f,
        x0_mean=x0_mean,
        x_f=x_f,
        Q=Q,
        W=W,
        T=T
    )

    if cs_result is None:

        print("CS failed.")

        mec_cost.append(
            mec_result["cost"]
        )

        mec_dev.append(np.nan)
        mec_prob.append(np.nan)

        cs_cost.append(np.nan)
        cs_dev.append(np.nan)
        cs_prob.append(np.nan)

        cs_terminal_trace.append(np.nan)

        continue

    # --------------------------------------------------------
    # Simulate MEC
    # --------------------------------------------------------

    dev_mec, prob_mec, _ = simulate_mec(
        A=A,
        B=B,
        Sigma0=sigma_0,
        W=W,
        x_f=x_f,
        mec_result=mec_result,
        T=T,
        epsilon=epsilon,
        N_sim=N_SIM
    )

    # --------------------------------------------------------
    # Simulate CS
    # --------------------------------------------------------

    dev_cs, prob_cs, _ = simulate_cs(
        A=A,
        B=B,
        Sigma0=sigma_0,
        W=W,
        x_f=x_f,
        cs_result=cs_result,
        T=T,
        epsilon=epsilon,
        N_sim=N_SIM
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    mec_cost.append(
        mec_result["cost"]
    )

    mec_dev.append(
        dev_mec
    )

    mec_prob.append(
        prob_mec
    )

    cs_cost.append(
        cs_result["cost"]
    )

    cs_dev.append(
        dev_cs
    )

    cs_prob.append(
        prob_cs
    )

    cs_terminal_trace.append(
        np.trace(
            cs_result["Sigma"][T]
        )
    )

    valid_T.append(T)

    print(
        f"MEC cost = {mec_result['cost']:.4f}"
    )

    print(
        f"CS cost  = {cs_result['cost']:.4f}"
    )

    print(
        f"MEC deviation = {dev_mec:.4f}"
    )

    print(
        f"CS deviation  = {dev_cs:.4f}"
    )

    print(
        f"MEC probability = {prob_mec:.4f}"
    )

    print(
        f"CS probability  = {prob_cs:.4f}"
    )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

mec_cost = np.asarray(mec_cost)
mec_dev = np.asarray(mec_dev)
mec_prob = np.asarray(mec_prob)

cs_cost = np.asarray(cs_cost)
cs_dev = np.asarray(cs_dev)
cs_prob = np.asarray(cs_prob)

cs_terminal_trace = np.asarray(
    cs_terminal_trace
)


# ============================================================
# PLOT 1: COST VS HORIZON
# ============================================================

plt.figure(figsize=(8, 5))

plt.semilogy(
    T_list,
    mec_cost,
    "o-",
    linewidth=2,
    label="MEC (exact)"
)

plt.semilogy(
    T_list,
    cs_cost,
    "s-",
    linewidth=2,
    label="Covariance steering"
)

plt.xlabel("Time horizon $T$")
plt.ylabel(r"Optimal cost $J^*$")
plt.legend()
plt.grid(True, which="both")

plt.tight_layout()

plt.savefig(
    "MEC_vs_CS_cost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2: TERMINAL STATE DEVIATION VS HORIZON
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    T_list,
    mec_dev,
    "o-",
    linewidth=2,
    label="MEC (exact)"
)

plt.plot(
    T_list,
    cs_dev,
    "s-",
    linewidth=2,
    label="Covariance steering"
)

plt.xlabel("Time horizon $T$")
plt.ylabel(
    r"$\mathbb{E}[\|x_T-x_f\|_2^2]$"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "MEC_vs_CS_deviation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 3: CHANCE-CONSTRAINT SATISFACTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    T_list,
    mec_prob,
    "o-",
    linewidth=2,
    label="MEC (exact)"
)

plt.plot(
    T_list,
    cs_prob,
    "s-",
    linewidth=2,
    label="Covariance steering"
)

plt.axhline(
    p_req,
    linestyle="--",
    linewidth=2,
    label=r"Required probability $1-\delta$"
)

plt.xlabel("Time horizon $T$")
plt.ylabel(
    r"$P(\|x_T-x_f\|_2\leq\epsilon)$"
)

plt.ylim(
    max(0, p_req - 0.25),
    1.02
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "MEC_vs_CS_probability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# OPTIONAL PLOT 4: CS TERMINAL COVARIANCE TRACE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    T_list,
    cs_terminal_trace,
    "s-",
    linewidth=2
)

plt.axhline(
    np.trace(Sigma_f),
    linestyle="--",
    linewidth=2,
    label=r"$\operatorname{tr}(\Sigma_f)$"
)

plt.xlabel("Time horizon $T$")
plt.ylabel(
    r"$\operatorname{tr}(\Sigma_T^{CS})$"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "CS_terminal_covariance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n====================================================")
print("SUMMARY")
print("====================================================")

print(
    f"Required probability = {p_req:.3f}"
)

print(
    f"epsilon = {epsilon:.3f}"
)

print(
    f"delta = {delta:.3f}"
)

print(
    f"Sigma_f scalar = {sigma_f_sq:.6f}"
)

for i, T in enumerate(T_list):

    print(
        f"T={T:2d} | "
        f"MEC cost={mec_cost[i]:10.4f} | "
        f"CS cost={cs_cost[i]:10.4f} | "
        f"MEC dev={mec_dev[i]:10.4f} | "
        f"CS dev={cs_dev[i]:10.4f} | "
        f"MEC P={mec_prob[i]:.4f} | "
        f"CS P={cs_prob[i]:.4f}"
    )
