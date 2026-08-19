import numpy as np
import matplotlib.pyplot as plt
from datagen import get_system
import os
import math
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.stats import ncx2
from scipy.optimize import brentq
import matplotlib.patches as patches


# ============================================================
# 1. Define C_T, Q, Sigma_e, and x_f
# ============================================================

# Example dimensions
n = 5
m = 2
T = 2
sigma_1 = 0.2
sigma_2 = 0.3
sigma_0 = np.eye(n) * sigma_1**2
x_0 =  np.random.multivariate_normal(np.zeros(n), sigma_0)

A, B, C_T = get_system('stable', n, m, T, seed=0)

# Positive definite Q
Q = np.diag([2.0, 3.0])
Q = block_diag(*([Q] * T))
# Positive definite Sigma_e
sigma = 0.3**2 * np.eye(n*T)
It = np.eye(T)
sigmaw = np.kron(It, sigma)
NT = np.eye(n)
for i in range(1, T):
    NT = np.hstack((NT, np.linalg.matrix_power(A, i)))
sigmaw = np.eye(n*T) * sigma_2**2
Sigma_e = NT @ sigmaw @ NT.T + np.linalg.matrix_power(A, T) @ sigma_0 @ np.linalg.matrix_power(A, T).T

# Target state
x_f = np.random.randn(n, 1)


# ============================================================
# 2. Compute Sigma_e^{-1}
# ============================================================

Sigma_e_inv = np.linalg.inv(Sigma_e)


# ============================================================
# 3. Compute gamma_0
#
# gamma_0 =
# x_f^T [
#     Sigma_e^{-1}
#     - Sigma_e^{-1} C_T
#       (C_T^T Sigma_e^{-1} C_T)^dagger
#       C_T^T Sigma_e^{-1}
# ] x_f
# ============================================================

M = C_T.T @ Sigma_e_inv @ C_T

gamma_0 = (
    x_f.T
    @ (
        Sigma_e_inv
        - Sigma_e_inv
        @ C_T
        @ np.linalg.pinv(M)
        @ C_T.T
        @ Sigma_e_inv
    )
    @ x_f
).item()


# ============================================================
# 4. Compute upper limit
#
# f(0+) = x_f^T Sigma_e^{-1} x_f
# ============================================================

gamma_max = (
    x_f.T @ Sigma_e_inv @ x_f
).item()


print("============================================")
print("Problem parameters")
print("============================================")
print(f"gamma_0   = {gamma_0:.6f}")
print(f"f(0+)     = {gamma_max:.6f}")


# ============================================================
# 5. Choose gamma
#
# gamma_0 < gamma < x_f^T Sigma_e^{-1} x_f
#
# Choose gamma halfway between the two limits.
# ============================================================

gamma = gamma_0 + 0.5 * (gamma_max - gamma_0)

print(f"gamma     = {gamma:.6f}")

assert gamma_0 < gamma < gamma_max


# ============================================================
# 6. Define f(nu)
#
# S(nu) = Sigma_e / nu
#         + C_T Q^{-1} C_T^T
#
# f(nu) = (1/nu^2)
#         x_f^T S^{-1} Sigma_e S^{-1} x_f
# ============================================================

Q_inv = np.linalg.inv(Q)


def f(nu):

    S = (
        Sigma_e / nu
        + C_T @ Q_inv @ C_T.T
    )

    # Solve S y = x_f rather than explicitly computing S^{-1}
    y = np.linalg.solve(S, x_f)

    return (
        (1.0 / nu**2)
        * (y.T @ Sigma_e @ y)
    ).item()


# ============================================================
# 7. Evaluate f(nu)
# ============================================================

# Logarithmic grid because nu can span several orders
# of magnitude.

nu_values = np.logspace(-4, 4, 2000)

f_values = np.array([
    f(nu) for nu in nu_values
])


# ============================================================
# 8. Find nu* satisfying
#
# f(nu*) = gamma
#
# i.e.
#
# f(nu*) - gamma = 0
# ============================================================

g_values = f_values - gamma

# Find intervals in which the function changes sign
idx = np.where(
    g_values[:-1] * g_values[1:] < 0
)[0]


if len(idx) == 0:

    print("\nNo finite nu* found in the chosen range.")
    print("Try increasing the range of nu_values.")

    nu_star = None

else:

    # For the theoretically unique solution,
    # we use the first sign change.

    i = idx[0]

    nu_star = brentq(
        lambda nu: f(nu) - gamma,
        nu_values[i],
        nu_values[i + 1]
    )

    print("\n============================================")
    print("Solution")
    print("============================================")
    print(f"nu*       = {nu_star:.6f}")
    print(f"f(nu*)    = {f(nu_star):.6f}")
    print(f"gamma     = {gamma:.6f}")
    print(f"error     = {abs(f(nu_star) - gamma):.3e}")


# ============================================================
# 9. Plot f(nu)
# ============================================================

plt.figure(figsize=(8, 5))

# f(nu)
plt.plot(
    nu_values,
    f_values,
    linewidth=2,
    label=r"$f(\nu)$"
)

# gamma
plt.axhline(
    gamma,
    linestyle="--",
    linewidth=1.5,
    label=r"$\gamma$"
)

# gamma_0 = limiting value as nu -> infinity
plt.axhline(
    gamma_0,
    linestyle="-.",
    linewidth=1.5,
    label=r"$\gamma_0=f(\infty)$"
)

# f(0+) = x_f^T Sigma_e^{-1} x_f
plt.axhline(
    gamma_max,
    linestyle=":",
    linewidth=1.5,
    label=r"$f(0^+)=x_f^\top\Sigma_e^{-1}x_f$"
)


# ============================================================
# 10. Mark nu*
# ============================================================

if nu_star is not None:

    plt.axvline(
        nu_star,
        linestyle=":",
        linewidth=1.5,
        label=rf"$\nu^*={nu_star:.3f}$"
    )

    plt.scatter(
        [nu_star],
        [gamma],
        s=80,
        zorder=5
    )


# ============================================================
# 11. Formatting
# ============================================================

plt.xscale("log")

plt.xlabel(
    r"$\nu$",
    fontsize=15
)

plt.ylabel(
    r"$f(\nu)$",
    fontsize=15
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend(
    fontsize=11
)

plt.tight_layout()

plt.savefig('f.jpeg')