import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datagen import get_system
from scipy.linalg import block_diag
from scipy.optimize import brentq


# ============================================================
# IEEE / OVERLEAF PUBLICATION SETTINGS
# ============================================================

mpl.rcParams.update({
    "text.usetex": True,

    # Times-like text
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif"
    ],

    # Lighter LaTeX-like mathematics
    "mathtext.fontset": "cm",

    # Avoid heavy-looking text
    "font.weight": "normal",
    "axes.labelweight": "normal",

    # IEEE-friendly sizes
    "font.size": 8.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,

    # Thin publication-quality axes / curves
    "lines.linewidth": 0.90,
    "axes.linewidth": 0.65,

    "xtick.major.width": 0.60,
    "ytick.major.width": 0.60,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "axes.labelpad": 2.0,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,

    "legend.frameon": False,

    # Vector font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


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

Q = block_diag(
    *([Q_single] * T)
)

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
    +
    np.linalg.matrix_power(A, T)
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

# Compact IEEE single-column figure
fig, ax = plt.subplots(
    figsize=(3.25, 1.70)
)


# ============================================================
# f(nu)
# ============================================================

ax.plot(
    nu_values,
    f_values,
    linewidth=0.90
)


# ============================================================
# Logarithmic x-axis
# ============================================================

ax.set_xscale("log")

ax.set_xlim(
    nu_values[0],
    nu_values[-1]
)

ax.set_ylim(
    0,
    gamma_max * 1.10
)


# ============================================================
# X-axis ticks
# ============================================================

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
])


# ============================================================
# gamma_0 = lower limit of f(nu)
# ============================================================

ax.axhline(
    gamma_0,
    linestyle="-.",
    linewidth=0.75
)


# ============================================================
# gamma horizontal line
# ============================================================

if nu_star is not None:

    ax.plot(
        [nu_values[0], nu_star],
        [gamma, gamma],
        linestyle="--",
        linewidth=0.75
    )


# ============================================================
# nu* vertical line
# ============================================================

if nu_star is not None:

    ax.plot(
        [nu_star, nu_star],
        [0, gamma],
        linestyle=":",
        linewidth=0.75
    )

    ax.scatter(
        [nu_star],
        [gamma],
        s=15,
        zorder=5
    )


# ============================================================
# Y-axis symbolic markings only
# ============================================================

ax.set_yticks([
    gamma,
    gamma_max
])

ax.set_yticklabels([
    r"$\gamma$",
    r"$x_f^\top \Sigma_{e_T}^{-1}x_f$"
])


# ============================================================
# X-axis marking for nu*
# ============================================================

if nu_star is not None:

    ax.annotate(
        r"$\nu_T^*$",
        xy=(nu_star, 0),
        xycoords="data",
        xytext=(0, -6),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7.5
    )


# ============================================================
# Axis labels
# ============================================================

ax.set_ylabel(
    r"$f(\nu_T)$"
)

ax.set_xlabel(
    r"$\nu_T$"
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    True,
    linestyle=':',
    linewidth=0.30,
    alpha=0.28
)


# ============================================================
# Final formatting
# ============================================================

fig.tight_layout(
    pad=0.25
)


# ============================================================
# Save
# ============================================================

# PDF — use this in Overleaf
fig.savefig(
    "f.pdf",
    bbox_inches="tight",
    pad_inches=0.01
)

# Optional PNG preview
fig.savefig(
    "f.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01
)

# Optional JPEG preview
fig.savefig(
    "f.jpg",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01
)

plt.show()
plt.close(fig)