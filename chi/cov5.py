import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.stats import ncx2, chi2, norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from datagen import get_system


# ============================================================
# USER / EXPERIMENT PARAMETERS
# ============================================================

np.random.seed(0)

n, m = 5, 5

# Initial state distribution x_0 ~ N(mu_0, Sigma_0)
mu_0 = np.zeros(n)
Sigma_0 = 2.0 * np.eye(n)

# Desired terminal state in your SMEC.
# For covariance steering we choose the prescribed terminal mean mu_f = x_f
# so that both problems refer to the same terminal target.
x_f = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
mu_f = x_f.copy()

T_list = list(range(1, 30, 2))

# Process noise covariance
W = 0.1 * np.eye(n)

# Stage control-energy weight
Q = np.diag([2.0, 3.0, 4.0, 5.0, 6.0])

# Your SMEC terminal chance constraint
#
# P{ (x_T-x_f)^T Sigma_e^{-1} (x_T-x_f) <= epsilon^2 } >= 1-delta
#
epsilon = 2.4
delta = 0.4
p_success = 1.0 - delta

name = "stable"

# Monte Carlo samples used only for evaluating the covariance-steering
# terminal chance probability (SMEC probability is available exactly).
N_MC_PROB = 50000
MC_SEED = 12345


# ============================================================
# TERMINAL COVARIANCE SPECIFICATION FOR COVARIANCE STEERING
# ============================================================
#
# Covariance steering needs a prescribed terminal covariance bound Sigma_f.
# The 2018 chance-constrained covariance-steering formulation uses a specified
# terminal mean and a maximum terminal covariance (Sigma_T <= Sigma_f).
#
# Runnable default below: isotropic Sigma_f chosen so that
# if x_T ~ N(x_f, sigma_f^2 I), then
#
#       P(||x_T-x_f||_2 <= epsilon) = 1-delta.
#
# This is ONLY a convenient way to choose Sigma_f from your existing epsilon,
# delta. Replace Sigma_f below by the covariance target you actually want.
#
sigma_f_sq = epsilon**2 / chi2.ppf(p_success, df=n)
Sigma_f = sigma_f_sq * np.eye(n)

print("Prescribed covariance upper bound Sigma_f =")
print(Sigma_f)
print("trace(Sigma_f) =", np.trace(Sigma_f))


# ============================================================
# OPTIONAL PAPER-STYLE PATH CHANCE CONSTRAINTS
# ============================================================
#
# The paper additionally allows linear state chance constraints
#
#       P{ a_i^T x_k <= b_i } >= 1-p_i.
#
# If your safe set is H_SAFE x <= h_SAFE, set USE_PATH_CHANCE_CONSTRAINTS=True
# and supply H_SAFE, h_SAFE below.  I leave it OFF by default because your
# supplied SMEC experiment does not define a safe polytope and I do not want
# to invent one.
#
# Example for a box lower <= x <= upper:
#
# lower = -10.0 * np.ones(n)
# upper =  10.0 * np.ones(n)
# H_SAFE = np.vstack([np.eye(n), -np.eye(n)])
# h_SAFE = np.concatenate([upper, -lower])
#
USE_PATH_CHANCE_CONSTRAINTS = False
H_SAFE = None
h_SAFE = None

# Joint path-failure budget. It is distributed uniformly over all
# scalar half-space constraints and constrained time steps via Boole's inequality.
delta_path = 0.05


# ============================================================
# BASIC MATRIX HELPERS
# ============================================================

def psd_sqrt(M, tol=1e-12):
    """Symmetric PSD square root."""
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.T


def build_terminal_matrices(A, B, T):
    """
    Chronological stacking convention:

        U = [u_0; ...; u_{T-1}],
        W_T = [w_0; ...; w_{T-1}],

        x_T = A^T x_0 + C_T U + N_T W_T.
    """
    n = A.shape[0]
    m = B.shape[1]

    C_T = np.zeros((n, m * T))
    N_T = np.zeros((n, n * T))

    for k in range(T):
        C_T[:, k*m:(k+1)*m] = (
            np.linalg.matrix_power(A, T - 1 - k) @ B
        )
        N_T[:, k*n:(k+1)*n] = np.linalg.matrix_power(
            A, T - 1 - k
        )

    return C_T, N_T


def build_stacked_state_matrices(A, B, T):
    """
    Build matrices for

        X = [x_1; ...; x_T]
          = Abar x_0 + Bbar U + Dbar W,

    with U=[u_0;...;u_{T-1}], W=[w_0;...;w_{T-1}].
    """
    n = A.shape[0]
    m = B.shape[1]

    Abar = np.zeros((n*T, n))
    Bbar = np.zeros((n*T, m*T))
    Dbar = np.zeros((n*T, n*T))

    for k in range(T):
        # Row block k corresponds to x_{k+1}
        Abar[k*n:(k+1)*n, :] = np.linalg.matrix_power(A, k + 1)

        for j in range(k + 1):
            A_power = np.linalg.matrix_power(A, k - j)

            Bbar[
                k*n:(k+1)*n,
                j*m:(j+1)*m
            ] = A_power @ B

            Dbar[
                k*n:(k+1)*n,
                j*n:(j+1)*n
            ] = A_power

    return Abar, Bbar, Dbar


def open_loop_terminal_covariance(A, T, Sigma0, W, N_T):
    A_T = np.linalg.matrix_power(A, T)
    Sigma_w = np.kron(np.eye(T), W)

    Sigma_T = (
        A_T @ Sigma0 @ A_T.T
        + N_T @ Sigma_w @ N_T.T
    )
    return 0.5 * (Sigma_T + Sigma_T.T)


# ============================================================
# EXACT NONCENTRAL-CHI^2 PARAMETER FOR YOUR SMEC
# ============================================================

def find_lambda_star(n, epsilon, p_success):
    """
    Find lambda_star such that

        F_{chi2_n(lambda_star)}(epsilon^2) = p_success.
    """
    x = epsilon**2

    def equation(lam):
        return ncx2.cdf(x, df=n, nc=lam) - p_success

    if equation(0.0) < 0:
        raise ValueError(
            "SMEC chance constraint is infeasible even at zero mean error: "
            "F_chi2_n(epsilon^2) < 1-delta."
        )

    lo = 0.0
    hi = 1.0

    while equation(hi) > 0:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("Could not bracket lambda_star.")

    return brentq(equation, lo, hi)


# ============================================================
# 1) YOUR OPEN-LOOP STOCHASTIC MINIMUM-ENERGY CONTROL
# ============================================================

def solve_smec(A, B, Sigma0, mu0, W, x_f, Q, T, epsilon, delta):
    """
    Your problem:

        min_U U^T Q_T U

        s.t.
        P{ e_T^T Sigma_e^{-1} e_T <= epsilon^2 } >= 1-delta,

    e_T = x_T-x_f.

    Since U is deterministic/open-loop, Sigma_e is independent of U, and
    the chance constraint is exactly equivalent to

        mu_e^T Sigma_e^{-1} mu_e <= lambda_star.

    No terminal mean equality is imposed.
    """
    n = A.shape[0]
    m = B.shape[1]

    C_T, N_T = build_terminal_matrices(A, B, T)
    A_T = np.linalg.matrix_power(A, T)

    Sigma_e = open_loop_terminal_covariance(
        A, T, Sigma0, W, N_T
    )

    Q_T = block_diag(*([Q] * T))

    lambda_star = find_lambda_star(
        n=n,
        epsilon=epsilon,
        p_success=1.0 - delta
    )

    U = cp.Variable(m * T)

    mu_T = A_T @ mu0 + C_T @ U
    mu_e = mu_T - x_f

    constraints = [
        cp.quad_form(mu_e, np.linalg.inv(Sigma_e)) <= lambda_star
    ]

    problem = cp.Problem(
        cp.Minimize(cp.quad_form(U, Q_T)),
        constraints
    )

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        problem.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-5,
            max_iters=150000
        )

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None

    U_star = np.asarray(U.value).reshape(-1)
    mu_T_star = A_T @ mu0 + C_T @ U_star
    mu_e_star = mu_T_star - x_f

    expected_sq_dev = (
        mu_e_star.T @ mu_e_star
        + np.trace(Sigma_e)
    )

    # Expected weighted squared terminal deviation, using the same
    # weighting matrix that appears in the SMEC chance constraint:
    #
    # E[e_T^T Sigma_e^{-1} e_T]
    #   = mu_e^T Sigma_e^{-1} mu_e
    #     + tr(Sigma_e^{-1} Sigma_e)
    #   = mu_e^T Sigma_e^{-1} mu_e + n.
    Sigma_e_inv = np.linalg.inv(Sigma_e)
    expected_weighted_dev = float(
        mu_e_star.T @ Sigma_e_inv @ mu_e_star
        + np.trace(Sigma_e_inv @ Sigma_e)
    )

    # Exact probability of the SAME weighted terminal error event used
    # in the SMEC chance constraint.  Since the open-loop terminal
    # covariance is exactly Sigma_e, the quadratic form follows a
    # noncentral chi-square distribution with noncentrality lambda.
    lambda_actual = float(
        mu_e_star.T @ np.linalg.inv(Sigma_e) @ mu_e_star
    )
    p_small_exact = float(
        ncx2.cdf(epsilon**2, df=n, nc=lambda_actual)
    )

    return {
        "cost": float(U_star.T @ Q_T @ U_star),
        "U": U_star,
        "mu_T": mu_T_star,
        "Sigma_T": Sigma_e,
        "expected_squared_deviation": float(expected_sq_dev),
        "expected_weighted_deviation": expected_weighted_dev,
        "lambda_star": float(lambda_star),
        "lambda_actual": lambda_actual,
        "p_small_exact": p_small_exact,
    }


# ============================================================
# 2) TERMINAL COVARIANCE STEERING
#    prescribed terminal mean + maximum terminal covariance
# ============================================================

def solve_covariance_steering(
        A, B, Sigma0, mu0, W,
        mu_f, Sigma_f, Q, T,
        H_safe=None, h_safe=None,
        delta_path=None):
    """
    Solve a convex affine disturbance-feedback covariance-steering problem.

    Policy:

        U = V + K_eta eta,

    where

        eta = [x_0-mu_0; w_0; ...; w_{T-1}].

    K_eta is block-causal:
        u_k may depend on x_0-mu_0 and w_0,...,w_{k-1},
        but not on w_k or future disturbances.

    Objective:

        min E[ U^T Q_T U ]

    Terminal distribution constraints:

        E[x_T] = mu_f,
        Cov(x_T) <= Sigma_f.

    The covariance inequality is imposed exactly through the Schur-complement LMI

        [ Sigma_f   M_T ] >= 0,
        [ M_T^T      I  ]

    where x_T-mu_T = M_T xi, xi~N(0,I).

    Optional paper-style linear state chance constraints:

        P{ H_safe[i] x_k <= h_safe[i] } >= 1-p_i,

    using Gaussian deterministic equivalents and uniform Boole risk allocation.
    """
    n = A.shape[0]
    m = B.shape[1]

    C_T, N_T = build_terminal_matrices(A, B, T)
    A_T = np.linalg.matrix_power(A, T)

    # Full stacked state matrices are needed for optional path chance constraints.
    Abar, Bbar, Dbar = build_stacked_state_matrices(A, B, T)

    # eta = [x0-mu0; w0; ...; w_{T-1}]
    Sigma_eta = block_diag(Sigma0, *([W] * T))
    Sigma_eta_sqrt = psd_sqrt(Sigma_eta)

    # Random part of terminal state before feedback:
    # x_T - E[x_T] = [A^T, N_T] eta + C_T K_eta eta.
    G_T = np.hstack([A_T, N_T])

    # Random part of stacked state X=[x_1;...;x_T].
    G_bar = np.hstack([Abar, Dbar])

    Q_T = block_diag(*([Q] * T))
    Q_T_sqrt = psd_sqrt(Q_T)

    # Decision variables.
    V = cp.Variable(m * T)
    K_eta = cp.Variable((m * T, n * (T + 1)))

    constraints = []

    # --------------------------------------------------------
    # Causality of disturbance feedback
    # --------------------------------------------------------
    # eta blocks:
    #   0: x0-mu0
    #   1: w0
    #   2: w1
    #   ...
    # For u_k, allowed blocks are 0,...,k.
    # Thus forbidden blocks begin at k+1.
    for k in range(T):
        rows = slice(k*m, (k+1)*m)
        first_forbidden_col = (k + 1) * n

        if first_forbidden_col < n * (T + 1):
            constraints.append(
                K_eta[rows, first_forbidden_col:] == 0
            )

    # --------------------------------------------------------
    # Terminal mean equality
    # --------------------------------------------------------
    mu_T = A_T @ mu0 + C_T @ V
    constraints.append(mu_T == mu_f)

    # --------------------------------------------------------
    # Terminal covariance upper bound
    # --------------------------------------------------------
    M_T = (G_T + C_T @ K_eta) @ Sigma_eta_sqrt

    r_eta = n * (T + 1)

    terminal_cov_lmi = cp.bmat([
        [Sigma_f, M_T],
        [M_T.T, np.eye(r_eta)]
    ])

    constraints.append(terminal_cov_lmi >> 0)

    # --------------------------------------------------------
    # OPTIONAL state chance constraints along the path
    # --------------------------------------------------------
    if H_safe is not None and h_safe is not None:
        H_safe = np.asarray(H_safe, dtype=float)
        h_safe = np.asarray(h_safe, dtype=float).reshape(-1)

        if H_safe.shape[1] != n:
            raise ValueError("H_safe must have n columns.")
        if H_safe.shape[0] != h_safe.size:
            raise ValueError("H_safe and h_safe dimensions do not match.")
        if delta_path is None:
            raise ValueError("Provide delta_path when path chance constraints are used.")

        n_faces = H_safe.shape[0]
        n_constrained_steps = max(T - 1, 0)  # x_1,...,x_{T-1}

        if n_constrained_steps > 0:
            p_face = delta_path / (n_faces * n_constrained_steps)

            if not (0.0 < p_face < 0.5):
                raise ValueError(
                    "Risk allocation produced invalid p_face. "
                    "Choose delta_path appropriately."
                )

            q_face = norm.ppf(1.0 - p_face)

            mu_X = Abar @ mu0 + Bbar @ V
            M_X = (G_bar + Bbar @ K_eta) @ Sigma_eta_sqrt

            # x_1,...,x_{T-1}; terminal distribution is handled separately.
            for k in range(T - 1):
                rows = slice(k*n, (k+1)*n)
                mu_k = mu_X[rows]
                M_k = M_X[rows, :]

                for i in range(n_faces):
                    a = H_safe[i, :]
                    b_i = h_safe[i]

                    constraints.append(
                        a @ mu_k
                        + q_face * cp.norm(a @ M_k, 2)
                        <= b_i
                    )

    # --------------------------------------------------------
    # Expected control energy
    # --------------------------------------------------------
    # E[U^T Q_T U]
    # = V^T Q_T V + tr(Q_T K_eta Sigma_eta K_eta^T)
    # = ||Q_T^{1/2}V||^2 + ||Q_T^{1/2}K_eta Sigma_eta^{1/2}||_F^2.
    objective = (
        cp.sum_squares(Q_T_sqrt @ V)
        + cp.sum_squares(Q_T_sqrt @ K_eta @ Sigma_eta_sqrt)
    )

    problem = cp.Problem(cp.Minimize(objective), constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        problem.solve(
            solver=cp.SCS,
            verbose=False,
            eps=2e-5,
            max_iters=200000
        )

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None

    V_star = np.asarray(V.value).reshape(-1)
    K_star = np.asarray(K_eta.value)

    mu_T_star = A_T @ mu0 + C_T @ V_star

    M_T_star = (
        G_T + C_T @ K_star
    ) @ Sigma_eta_sqrt

    Sigma_T_star = M_T_star @ M_T_star.T
    Sigma_T_star = 0.5 * (Sigma_T_star + Sigma_T_star.T)

    # Expected control-energy cost evaluated directly.
    cost = (
        V_star.T @ Q_T @ V_star
        + np.trace(
            Q_T @ K_star @ Sigma_eta @ K_star.T
        )
    )

    # General formula. Since mu_T = mu_f = x_f in this experiment,
    # the mean-error term should be numerical zero and the deviation is tr(Sigma_T).
    mean_error = mu_T_star - x_f
    expected_sq_dev = (
        mean_error.T @ mean_error
        + np.trace(Sigma_T_star)
    )

    # Numerical diagnostics for the terminal constraints.
    max_cov_eig = np.linalg.eigvalsh(
        Sigma_T_star - Sigma_f
    ).max()

    return {
        "cost": float(cost),
        "V": V_star,
        "K_eta": K_star,
        "mu_T": mu_T_star,
        "Sigma_T": Sigma_T_star,
        "expected_squared_deviation": float(expected_sq_dev),
        "terminal_mean_error_norm": float(np.linalg.norm(mean_error)),
        "max_eig_SigmaT_minus_Sigmaf": float(max_cov_eig),
        "status": problem.status,
    }


# ============================================================
# TERMINAL SMALL-ERROR PROBABILITY
# ============================================================

def terminal_small_error_probability_mc(
        mu_T, Sigma_T, x_f, Sigma_ref, epsilon,
        n_samples=50000, seed=12345):
    """
    Estimate

        P{ (x_T-x_f)^T Sigma_ref^{-1} (x_T-x_f) <= epsilon^2 }

    for x_T ~ N(mu_T, Sigma_T).

    For covariance steering, Sigma_T generally differs from Sigma_ref,
    so the quadratic form is not in general an ordinary noncentral
    chi-square random variable.  Monte Carlo is therefore used.
    """
    rng = np.random.default_rng(seed)

    Sigma_T = 0.5 * (Sigma_T + Sigma_T.T)
    samples = rng.multivariate_normal(
        mean=mu_T,
        cov=Sigma_T,
        size=n_samples
    )

    errors = samples - x_f
    Sigma_ref_inv = np.linalg.inv(Sigma_ref)

    qvals = np.einsum(
        "bi,ij,bj->b",
        errors,
        Sigma_ref_inv,
        errors
    )

    return float(np.mean(qvals <= epsilon**2))


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():
    smec_cost = []
    cs_cost = []

    smec_dev = []
    cs_dev = []

    # Expected weighted squared terminal deviation using the SAME
    # reference covariance for both controllers:
    #
    #   E[e_T^T Sigma_ref^{-1} e_T],
    #
    # where Sigma_ref = Sigma_e^{SMEC}(T), the open-loop terminal covariance.
    smec_weighted_dev = []
    cs_weighted_dev = []

    # Probability that the terminal weighted error is small:
    # P{e_T^T Sigma_ref^{-1} e_T <= epsilon^2}.
    smec_prob = []
    cs_prob = []

    cs_trace = []
    cs_mean_error = []

    for T in T_list:
        print("\n" + "=" * 60)
        print(f"T = {T}")
        print("=" * 60)

        A, B, _ = get_system(
            name,
            n,
            m,
            T,
            seed=0
        )

        # Use the open-loop terminal covariance as the common weighting
        # matrix in the small-error probability for BOTH controllers.
        # This is exactly the Sigma_e appearing in your SMEC constraint.
        _, N_T_ref = build_terminal_matrices(A, B, T)
        Sigma_ref = open_loop_terminal_covariance(
            A, T, Sigma_0, W, N_T_ref
        )

        # ----------------------------------------------------
        # Your SMEC
        # ----------------------------------------------------
        smec = solve_smec(
            A=A,
            B=B,
            Sigma0=Sigma_0,
            mu0=mu_0,
            W=W,
            x_f=x_f,
            Q=Q,
            T=T,
            epsilon=epsilon,
            delta=delta
        )

        # ----------------------------------------------------
        # Covariance steering
        # ----------------------------------------------------
        cs = solve_covariance_steering(
            A=A,
            B=B,
            Sigma0=Sigma_0,
            mu0=mu_0,
            W=W,
            mu_f=mu_f,
            Sigma_f=Sigma_f,
            Q=Q,
            T=T,
            H_safe=(H_SAFE if USE_PATH_CHANCE_CONSTRAINTS else None),
            h_safe=(h_SAFE if USE_PATH_CHANCE_CONSTRAINTS else None),
            delta_path=(delta_path if USE_PATH_CHANCE_CONSTRAINTS else None)
        )

        if smec is None:
            print("SMEC infeasible / solver failed.")
            smec_cost.append(np.nan)
            smec_dev.append(np.nan)
            smec_weighted_dev.append(np.nan)
            smec_prob.append(np.nan)
        else:
            smec_cost.append(smec["cost"])
            smec_dev.append(smec["expected_squared_deviation"])
            smec_weighted_dev.append(smec["expected_weighted_deviation"])
            smec_prob.append(smec["p_small_exact"])

            print(f"SMEC cost                  = {smec['cost']:.6g}")
            print(
                "SMEC E[||x_T-x_f||^2]      = "
                f"{smec['expected_squared_deviation']:.6g}"
            )
            print(
                "SMEC E[e^T Sigma_ref^-1 e] = "
                f"{smec['expected_weighted_deviation']:.6g}"
            )
            print(
                "SMEC P{e^T Sigma_ref^-1 e <= eps^2} = "
                f"{smec['p_small_exact']:.6f}"
            )

        if cs is None:
            print("Covariance steering infeasible / solver failed.")
            cs_cost.append(np.nan)
            cs_dev.append(np.nan)
            cs_weighted_dev.append(np.nan)
            cs_prob.append(np.nan)
            cs_trace.append(np.nan)
            cs_mean_error.append(np.nan)
        else:
            # Evaluate the SAME small-error event used for SMEC.
            # For CS, Sigma_T is controller-dependent, so use Monte Carlo.
            p_cs = terminal_small_error_probability_mc(
                mu_T=cs["mu_T"],
                Sigma_T=cs["Sigma_T"],
                x_f=x_f,
                Sigma_ref=Sigma_ref,
                epsilon=epsilon,
                n_samples=N_MC_PROB,
                seed=MC_SEED + T
            )

            # Expected weighted deviation using the SAME reference
            # covariance as in SMEC:
            #
            # E[e^T Sigma_ref^{-1} e]
            # = (mu_T-x_f)^T Sigma_ref^{-1} (mu_T-x_f)
            #   + tr(Sigma_ref^{-1} Sigma_T).
            #
            # Since covariance steering imposes mu_T=x_f, the first
            # term should be numerical zero.
            Sigma_ref_inv = np.linalg.inv(Sigma_ref)
            cs_mean_error_vec = cs["mu_T"] - x_f
            weighted_dev_cs = float(
                cs_mean_error_vec.T @ Sigma_ref_inv @ cs_mean_error_vec
                + np.trace(Sigma_ref_inv @ cs["Sigma_T"])
            )

            cs_cost.append(cs["cost"])
            cs_dev.append(cs["expected_squared_deviation"])
            cs_weighted_dev.append(weighted_dev_cs)
            cs_prob.append(p_cs)
            cs_trace.append(np.trace(cs["Sigma_T"]))
            cs_mean_error.append(cs["terminal_mean_error_norm"])

            print(f"CS cost                    = {cs['cost']:.6g}")
            print(
                "CS E[||x_T-x_f||^2]        = "
                f"{cs['expected_squared_deviation']:.6g}"
            )
            print(
                "CS E[e^T Sigma_ref^-1 e]   = "
                f"{weighted_dev_cs:.6g}"
            )
            print(
                "CS ||E[x_T]-x_f||          = "
                f"{cs['terminal_mean_error_norm']:.3e}"
            )
            print(
                "CS tr(Sigma_T)             = "
                f"{np.trace(cs['Sigma_T']):.6g}"
            )
            print(
                "CS max eig(Sigma_T-Sigma_f)= "
                f"{cs['max_eig_SigmaT_minus_Sigmaf']:.3e}"
            )
            print(
                "CS P{e^T Sigma_ref^-1 e <= eps^2}   = "
                f"{p_cs:.6f}"
            )

    smec_cost = np.asarray(smec_cost, dtype=float)
    cs_cost = np.asarray(cs_cost, dtype=float)
    smec_dev = np.asarray(smec_dev, dtype=float)
    cs_dev = np.asarray(cs_dev, dtype=float)
    smec_weighted_dev = np.asarray(smec_weighted_dev, dtype=float)
    cs_weighted_dev = np.asarray(cs_weighted_dev, dtype=float)
    smec_prob = np.asarray(smec_prob, dtype=float)
    cs_prob = np.asarray(cs_prob, dtype=float)
    cs_trace = np.asarray(cs_trace, dtype=float)
    cs_mean_error = np.asarray(cs_mean_error, dtype=float)

    # ========================================================
    # PLOT 1: OPTIMAL COST VS HORIZON
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.semilogy(
        T_list,
        smec_cost,
        "o-",
        linewidth=2,
        label="Open-loop SMEC"
    )

    plt.semilogy(
        T_list,
        cs_cost,
        "s-",
        linewidth=2,
        label="Covariance steering"
    )

    plt.xlabel(r"Time horizon $T$")
    plt.ylabel(r"Optimal control-energy cost $J^\star$")
    plt.xticks(T_list)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(
        "SMEC_vs_covariance_steering_cost.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ========================================================
    # PLOT 2: EXPECTED SQUARED TERMINAL DEVIATION VS HORIZON
    # ========================================================
    #
    # E[(x_T-x_f)^T(x_T-x_f)]
    # = ||E[x_T]-x_f||^2 + tr(Sigma_T).
    #
    # For covariance steering here, E[x_T]=x_f is imposed exactly,
    # so its curve is simply tr(Sigma_T^*).
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        T_list,
        smec_dev,
        "o-",
        linewidth=2,
        label="Open-loop SMEC"
    )

    plt.plot(
        T_list,
        cs_dev,
        "s-",
        linewidth=2,
        label="Covariance steering"
    )

    plt.xlabel(r"Time horizon $T$")
    plt.ylabel(
        r"$\mathbb{E}[(x_T-x_f)^\top(x_T-x_f)]$"
    )
    plt.xticks(T_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "SMEC_vs_covariance_steering_expected_squared_deviation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ========================================================
    # PLOT 3: PROBABILITY OF SMALL TERMINAL ERROR VS HORIZON
    # ========================================================
    # Same event for both controllers:
    #
    #   P{ (x_T-x_f)^T Sigma_ref^{-1} (x_T-x_f) <= epsilon^2 }.
    #
    # For SMEC this is evaluated exactly from the noncentral chi-square CDF.
    # For covariance steering it is evaluated by Monte Carlo because its
    # actual terminal covariance differs from Sigma_ref.
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        T_list,
        smec_prob,
        "o-",
        linewidth=2,
        label="Open-loop SMEC (exact)"
    )

    plt.plot(
        T_list,
        cs_prob,
        "s-",
        linewidth=2,
        label="Covariance steering (Monte Carlo)"
    )

    plt.axhline(
        p_success,
        linestyle="--",
        linewidth=2,
        label=r"Required probability $1-\delta$"
    )

    plt.xlabel(r"Time horizon $T$")
    plt.ylabel(
        r"$\mathbb{P}\{(x_T-x_f)^\top\Sigma_{e,\mathrm{OL}}^{-1}"
        r"(x_T-x_f)\leq\epsilon^2\}$"
    )
    plt.ylim(0.0, 1.02)
    plt.xticks(T_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "SMEC_vs_covariance_steering_small_error_probability.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ========================================================
    # PLOT 4: EXPECTED WEIGHTED TERMINAL DEVIATION VS HORIZON
    # ========================================================
    # Common weighting matrix for BOTH methods:
    #
    #   Sigma_ref(T) = Sigma_e^{SMEC}(T).
    #
    # We plot
    #
    #   E[e_T^T Sigma_ref^{-1} e_T].
    #
    # For SMEC:
    #
    #   E[e_T^T Sigma_e^{-1} e_T]
    #   = mu_e^T Sigma_e^{-1} mu_e + n.
    #
    # For covariance steering:
    #
    #   E[e_T^T Sigma_ref^{-1} e_T]
    #   = (mu_T-x_f)^T Sigma_ref^{-1}(mu_T-x_f)
    #     + tr(Sigma_ref^{-1} Sigma_T).
    #
    # Since CS enforces mu_T=x_f, this is approximately
    # tr(Sigma_ref^{-1} Sigma_T).
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        T_list,
        smec_weighted_dev,
        "o-",
        linewidth=2,
        label="Open-loop SMEC"
    )

    plt.plot(
        T_list,
        cs_weighted_dev,
        "s-",
        linewidth=2,
        label="Covariance steering"
    )

    plt.xlabel(r"Time horizon $T$")
    plt.ylabel(
        r"$\mathbb{E}[(x_T-x_f)^\top "
        r"\Sigma_{e,\mathrm{OL}}^{-1}(x_T-x_f)]$"
    )
    plt.xticks(T_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "SMEC_vs_covariance_steering_expected_weighted_deviation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ========================================================
    # OPTIONAL DIAGNOSTIC: CS TERMINAL COVARIANCE TRACE
    # ========================================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        T_list,
        cs_trace,
        "s-",
        linewidth=2,
        label=r"$\operatorname{tr}(\Sigma_T^\star)$"
    )

    plt.axhline(
        np.trace(Sigma_f),
        linestyle="--",
        linewidth=2,
        label=r"$\operatorname{tr}(\Sigma_f)$"
    )

    plt.xlabel(r"Time horizon $T$")
    plt.ylabel(r"Terminal covariance trace")
    plt.xticks(T_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "covariance_steering_terminal_covariance_trace.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("SMEC: terminal mean is free inside the chance-constraint region.")
    print("CS  : terminal mean is fixed at x_f and Sigma_T <= Sigma_f.")
    print("Sigma_f =")
    print(Sigma_f)

    if USE_PATH_CHANCE_CONSTRAINTS:
        print("Paper-style path chance constraints are ON.")
    else:
        print(
            "Paper-style path chance constraints are OFF because no safe "
            "polytope was supplied. Set H_SAFE, h_SAFE and the switch above "
            "to turn them on."
        )

    for i, T in enumerate(T_list):
        print(
            f"T={T:2d} | "
            f"SMEC J={smec_cost[i]:11.4g} | "
            f"CS J={cs_cost[i]:11.4g} | "
            f"SMEC E[e^T e]={smec_dev[i]:10.4g} | "
            f"CS E[e^T e]={cs_dev[i]:10.4g} | "
            f"SMEC E[e^T Sref^-1 e]={smec_weighted_dev[i]:9.4g} | "
            f"CS E[e^T Sref^-1 e]={cs_weighted_dev[i]:9.4g} | "
            f"SMEC Psmall={smec_prob[i]:7.4f} | "
            f"CS Psmall={cs_prob[i]:7.4f}"
        )


if __name__ == "__main__":
    main()
