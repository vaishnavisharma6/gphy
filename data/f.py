import numpy as np
import matplotlib.pyplot as plt
from datagen import get_system
from scipy.linalg import block_diag
from scipy.optimize import brentq


# ============================================================
# 1. Define system
# ============================================================

n = 5
m = 2
T = 2

sigma_1 = 0.2
sigma_2 = 0.3

sigma_0 = np.eye(n) * sigma_1**2

x_0 = np.random.multivariate_normal(
    np.zeros(n),
    sigma_0
)

A, B, C_T = get_system(
    'stable',
    n,
    m,
    T,
    seed=0
)


# ============================================================
# 2. Control cost matrix Q
# ============================================================

Q_single = np.diag([2.0, 3.0])
Q = block_diag(*([Q_single] * T))

Q_inv = np.linalg.inv(Q)


# ============================================================
# 3. Construct Sigma_e
# ============================================================

sigmaw = np.eye(n * T) * sigma_2**2

NT = np.eye(n)

for i in range(1, T):
    NT = np.hstack(
        (
            NT,
            np.linalg.matrix_power(A, i)
        )
    )

Sigma_e = (
    NT @ sigmaw @ NT.T
    + np.linalg.matrix_power(A, T)
    @ sigma_0
    @ np.linalg.matrix_power(A, T).T
)


# ============================================================
# 4. Target state
# ============================================================

x_f = np.random.randn(n, 1)


# ============================================================
# 5. Compute Sigma_e^{-1}
# ============================================================

Sigma_e_inv = np.linalg.inv(Sigma_e)


# ============================================================
# 6. Compute gamma_0
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
# 7. Compute f(0+)
# ============================================================

gamma_max = (
    x_f.T
    @ Sigma_e_inv
    @ x_f
).item()


print("============================================")
print("Problem parameters")
print("============================================")
print(f"gamma_0   = {gamma_0:.6f}")
print(f"f(0+)     = {gamma_max:.6f}")


# ============================================================
# 8. Choose gamma
# ============================================================

gamma = gamma_0 + 0.5 * (gamma_max - gamma_0)

print(f"gamma     = {gamma:.6f}")

assert gamma_0 < gamma < gamma_max


# ============================================================
# 9. Define f(nu)
# ============================================================

def f(nu):

    S = (
        Sigma_e / nu
        + C_T @ Q_inv @ C_T.T
    )

    y = np.linalg.solve(
        S,
        x_f
    )

    return (
        (1.0 / nu**2)
        * (y.T @ Sigma_e @ y)
    ).item()


# ============================================================
# 10. Evaluate f(nu)
# ============================================================

nu_values = np.logspace(
    -4.5,
    4,
    2000
)

f_values = np.array([
    f(nu)
    for nu in nu_values
])


# ============================================================
# 11. Find nu*
# ============================================================

g_values = f_values - gamma

idx = np.where(
    g_values[:-1] * g_values[1:] < 0
)[0]


if len(idx) == 0:

    print("\nNo finite nu* found in the chosen range.")
    print("Try increasing the range of nu_values.")

    nu_star = None

else:

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
    print(
        f"error     = "
        f"{abs(f(nu_star) - gamma):.3e}"
    )


# ============================================================
# 12. Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))


# ------------------------------------------------------------
# f(nu)
# ------------------------------------------------------------

ax.plot(
    nu_values,
    f_values,
    linewidth=2
)


# ------------------------------------------------------------
# Log x-axis
# ------------------------------------------------------------

ax.set_xscale("log")

ax.set_xlim(
    nu_values[0],
    nu_values[-1]
)

ax.set_ylim(
    0,
    gamma_max * 1.05
)


# ------------------------------------------------------------
# Show a few x-values
# ------------------------------------------------------------

ax.set_xticks([
    1e-4,
    1e-2,
    1,
    1e2,
    1e4
])

ax.set_xticklabels([
    r"$10^{-4}$",
    r"$10^{-2}$",
    r"$10^{0}$",
    r"$10^{2}$",
    r"$10^{4}$"
], fontsize=11)


# ============================================================
# 13. gamma_0 horizontal line
# ============================================================

ax.axhline(
    gamma_0,
    linestyle="-.",
    linewidth=1.5
)


# ============================================================
# 14. gamma horizontal line
#
# Starts exactly at the left boundary of the axes and ends
# at nu*
# ============================================================

if nu_star is not None:

    ax.plot(
        [nu_values[0], nu_star],
        [gamma, gamma],
        linestyle="--",
        linewidth=1.5
    )


# ============================================================
# 15. nu* vertical line
#
# Starts exactly from the x-axis (y = 0)
# ============================================================

if nu_star is not None:

    ax.plot(
        [nu_star, nu_star],
        [0, gamma],
        linestyle=":",
        linewidth=1.5
    )

    ax.scatter(
        [nu_star],
        [gamma],
        s=60,
        zorder=5
    )


# ============================================================
# 16. Y-axis markings
# ============================================================

# Candidate intermediate values
candidates = np.linspace(
    0.15 * gamma_max,
    0.85 * gamma_max,
    8
)

# Remove values too close to gamma
candidates = [
    y for y in candidates
    if not np.isclose(
        y,
        gamma,
        rtol=0.05,
        atol=0.01
    )
]

# Choose two intermediate values
y1 = candidates[0]
y2 = candidates[-1]

ax.set_yticks([
    y1,
    y2,
    gamma
])

ax.set_yticklabels([
    rf"${y1:.2f}$",
    rf"${y2:.2f}$",
    r"$\gamma$"
], fontsize=12)


# ============================================================
# 17. X-axis marking for nu*
# ============================================================

if nu_star is not None:

    ax.annotate(
        r"$\nu_T^*$",
        xy=(nu_star, 0),
        xycoords="data",
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=14
    )


# ============================================================
# 18. Mark zero at left side of x-axis
# ============================================================

# ax.annotate(
#     r"$0$",
#     xy=(nu_values[0], 0),
#     xycoords="data",
#     xytext=(0, -12),
#     textcoords="offset points",
#     ha="center",
#     va="top",
#     fontsize=14
# )


# ============================================================
# 19. Labels
# ============================================================

ax.set_ylabel(
    r"$f(\nu_T)$",
    fontsize=16
)

ax.set_xlabel(
    r"$\nu_T$",
    fontsize=16
)


# ============================================================
# 20. Final formatting
# ============================================================

plt.tight_layout()

plt.savefig(
    "f.jpeg",
    dpi=300
)

plt.show()