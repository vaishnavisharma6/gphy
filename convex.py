# triangle_soc_cvxpy.py
import numpy as np
import cvxpy as cp
from scipy.stats import chi2
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ---------------------------
# USER: Replace these with your problem data
# ---------------------------
# Dimensions
n = 3           # dimension of z (state dimension after stacking)
mT = 6          # total dimension of control vector u (m * T)
# Matrices (shapes must match): C_T: n x mT, A: ??? (used to compute Sigma_z), N_T: n x qw
C_T = np.random.randn(n, mT)          # example
A = np.random.randn(n, n)             # example (modify for your model)
N_T = 0.1 * np.random.randn(n, n)     # example
# Covariances
Sigma0 = 0.5 * np.eye(n)              # covariance of x0
Sigma_w = 0.2 * np.eye(n)             # covariance of w
# Other vectors
mu = np.zeros(n)                      # mean of x0
x_f = np.ones(n)                     # target final state
# Objective weight (Q should be mT x mT, PSD)
Q = np.eye(mT)                        # can be block-diagonal of per-step Q_k
# Chance-constraint parameters
epsilon = 1.0
delta = 0.05

Sigma_z = A.T @ Sigma0 @ A + N_T @ Sigma_w @ N_T.T   # shape n x n

# Check positive definiteness (small regularization if needed)
eps_reg = 1e-9
# regularize if needed
eigvals = np.linalg.eigvalsh(Sigma_z)
if np.min(eigvals) <= 0:
    Sigma_z = Sigma_z + (abs(np.min(eigvals)) + 1e-8) * np.eye(n)
    eigvals = np.linalg.eigvalsh(Sigma_z)

# Chi-square radius
r = np.sqrt(chi2.ppf(1 - delta, df=n))

# triangle-inequality conservative RHS = epsilon - r * sqrt(lambda_max(Sigma_z))
lambda_max = np.max(eigvals)
RHS = epsilon - r * np.sqrt(lambda_max)

print("chi2 radius r:", r)
print("lambda_max(Sigma_z):", lambda_max)
print("RHS (epsilon - r*sqrt(lambda_max)):", RHS)

if RHS < 0:
    print("WARNING: RHS < 0. The triangle-inequality bound is infeasible for given epsilon/delta.")
    # You may still want to proceed (infeasible) or consider other approximations.

# ---------------------------
# CVXPY optimization
# ---------------------------
u = cp.Variable(mT)

# objective: u^T Q u  (Q must be PSD)
objective = cp.Minimize(cp.quad_form(u, Q))

# SOC constraint using triangle-inequality approximation:
# || C_T u + A^T mu - x_f ||_2 <= RHS
affine_term = C_T @ u + A.T @ mu - x_f

constraints = []
if RHS >= 0:
    constraints.append(cp.norm(affine_term, 2) <= RHS)
else:
    # If RHS < 0, no feasible u unless affine_term == 0. Enforce equality if you prefer.
    constraints.append(affine_term == 0.0)



prob = cp.Problem(objective, constraints)

# Solve (choose solver you have: ECOS, OSQP, SCS, MOSEK, etc.)
prob.solve(solver=cp.SCS, verbose=True)

print("status:", prob.status)
if prob.status in ["optimal", "optimal_inaccurate"]:
    u_opt = u.value
    print("u* =\n", u_opt)
    print("objective value:", prob.value)
else:
    print("Problem not solved to optimality. Try different solver or check data.")


if prob.status in ["optimal", "optimal_inaccurate"]:
    u_opt = u.value

    # Pick two coordinates to visualize
    idx1, idx2 = 0, 1
    u1_vals = np.linspace(-2, 2, 200)
    u2_vals = np.linspace(-2, 2, 200)
    U1, U2 = np.meshgrid(u1_vals, u2_vals)

    feasible_mask = np.zeros_like(U1, dtype=bool)
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            u_test = np.zeros(mT)
            u_test[idx1] = U1[i, j]
            u_test[idx2] = U2[i, j]
            lhs = np.linalg.norm(C_T @ u_test + A.T @ mu - x_f, 2)
            feasible_mask[i, j] = lhs <= RHS

    plt.figure(figsize=(6, 6))
    plt.contourf(U1, U2, feasible_mask, levels=[-0.5, 0.5, 1.5], colors=["white", "lightgreen"], alpha=0.6)
    plt.plot(u_opt[idx1], u_opt[idx2], "r*", markersize=12, label="Optimal $u^*$")
    plt.xlabel(f"$u_{idx1}$")
    plt.ylabel(f"$u_{idx2}$")
    plt.title("Feasible set projection (triangle SOC)")
    plt.legend()
    plt.axis("equal")
    plt.show()

    if prob.status in ["optimal", "optimal_inaccurate"]:
        u_opt = u.value

    # Grid
    u1_vals = np.linspace(-2, 2, 200)
    u2_vals = np.linspace(-2, 2, 200)
    U1, U2 = np.meshgrid(u1_vals, u2_vals)
    cost_vals = np.zeros_like(U1)

    idx1, idx2 = 0, 1
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            u_test = np.zeros(mT)
            u_test[idx1] = U1[i, j]
            u_test[idx2] = U2[i, j]
            cost_vals[i, j] = u_test @ Q @ u_test

    plt.figure(figsize=(6, 6))
    CS = plt.contour(U1, U2, cost_vals, levels=20, cmap="coolwarm", alpha=0.7)
    plt.clabel(CS, inline=True, fontsize=8)

    # feasible region shading
    feasible_mask = np.zeros_like(U1, dtype=bool)
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            u_test = np.zeros(mT)
            u_test[idx1] = U1[i, j]
            u_test[idx2] = U2[i, j]
            lhs = np.linalg.norm(C_T @ u_test + A.T @ mu - x_f, 2)
            feasible_mask[i, j] = lhs <= RHS
    plt.contourf(U1, U2, feasible_mask, levels=[-0.5, 0.5, 1.5], colors=["none", "lightgreen"], alpha=0.3)

    # mark optimum
    plt.plot(u_opt[idx1], u_opt[idx2], "r*", markersize=12, label="Optimal $u^*$")
    plt.xlabel(f"$u_{idx1}$")
    plt.ylabel(f"$u_{idx2}$")
    plt.title("Quadratic cost contours + feasible region")
    plt.legend()
    plt.axis("equal")
    plt.show()


    if prob.status in ["optimal", "optimal_inaccurate"]:
        u_opt = u.value

    # Project onto first 2 coordinates of u for visualization
    u1 = u_opt[0]
    u2 = u_opt[1]

    # Build grid for feasible region approximation
    grid_size = 100
    u1_vals = np.linspace(-2, 2, grid_size)
    u2_vals = np.linspace(-2, 2, grid_size)
    U1, U2 = np.meshgrid(u1_vals, u2_vals)

    feasible_mask = np.zeros_like(U1, dtype=bool)

    for i in range(grid_size):
        for j in range(grid_size):
            # construct candidate u with only first 2 free, others zero
            u_candidate = np.zeros(mT)
            u_candidate[0] = U1[i, j]
            u_candidate[1] = U2[i, j]

            lhs = np.linalg.norm(C_T @ u_candidate + A.T @ mu - x_f, 2)
            feasible_mask[i, j] = (lhs <= RHS + 1e-6)  # tolerance

    plt.figure(figsize=(6, 6))
    plt.contourf(U1, U2, feasible_mask, levels=[-0.5, 0.5, 1.5], alpha=0.3, colors=["white", "lightblue"])
    plt.plot(u1, u2, "ro", markersize=10, label="Optimal $u^*$ (proj)")

    plt.xlabel("$u_1$")
    plt.ylabel("$u_2$")
    plt.title("Feasible region (proj onto $u_1,u_2$) and optimal solution")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()