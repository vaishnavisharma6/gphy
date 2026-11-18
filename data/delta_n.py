from scipy.stats import ncx2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# IEEE / OVERLEAF PUBLICATION SETTINGS
# ============================================================

mpl.rcParams.update({
    "text.usetex": False,

    # Times-like serif font
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif"
    ],

    # LaTeX-like mathematical symbols
    "mathtext.fontset": "cm",

    # Normal/light font weight
    "font.weight": "normal",
    "axes.labelweight": "normal",

    # Compact IEEE-friendly font sizes
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.0,
    "legend.fontsize": 6.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,

    # Lines / axes
    "lines.linewidth": 0.85,
    "axes.linewidth": 0.60,

    # Tick styling
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

    # Compact spacing
    "axes.labelpad": 1.0,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,

    # Legend
    "legend.frameon": False,

    # Vector font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Tight PDF saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


# ============================================================
# PARAMETERS
# ============================================================

epsilon_1 = 0.7
epsilon_2 = 1.4
epsilon_3 = 2.1

n_list = np.arange(2, 41)
lambda_list = np.arange(2, 41)

lambda_val = 5


# ============================================================
# FIGURE 1:
# Minimum feasible delta versus n
# ============================================================

delta_min1 = []
delta_min2 = []
delta_min3 = []

for n in n_list:

    delta1 = 1 - ncx2.cdf(
        epsilon_1**2,
        df=n,
        nc=lambda_val
    )

    delta2 = 1 - ncx2.cdf(
        epsilon_2**2,
        df=n,
        nc=lambda_val
    )

    delta3 = 1 - ncx2.cdf(
        epsilon_3**2,
        df=n,
        nc=lambda_val
    )

    delta_min1.append(delta1)
    delta_min2.append(delta2)
    delta_min3.append(delta3)


# Compact IEEE single-column figure
fig, ax = plt.subplots(
    figsize=(3.25, 1.65)
)

ax.plot(
    n_list,
    delta_min1,
    '-',
    linewidth=0.85,
    label=rf'$\epsilon={epsilon_1}$'
)

ax.plot(
    n_list,
    delta_min2,
    '-',
    linewidth=0.85,
    label=rf'$\epsilon={epsilon_2}$'
)

ax.plot(
    n_list,
    delta_min3,
    '-',
    linewidth=0.85,
    label=rf'$\epsilon={epsilon_3}$'
)

ax.set_xlabel(
    r'$n$',
    labelpad=1
)

ax.set_ylabel(
    r'Min. feasible $\delta$',
    labelpad=1
)

ax.set_xticks(
    [2, 10, 20, 30, 40]
)

ax.tick_params(
    axis='both',
    which='major',
    pad=1.5
)

ax.grid(
    True,
    linestyle=':',
    linewidth=0.30,
    alpha=0.30
)

ax.legend(
    loc='lower right',
    fontsize=6.0,
    handlelength=1.5,
    handletextpad=0.4,
    labelspacing=0.25,
    borderaxespad=0.3
)

fig.tight_layout(
    pad=0.25
)

fig.savefig(
    'delta_n_chi.pdf',
    bbox_inches='tight',
    pad_inches=0.01
)

plt.show()
plt.close(fig)


# ============================================================
# FIGURE 2:
# Minimum feasible delta versus lambda
# ============================================================

delta_min1 = []
delta_min2 = []
delta_min3 = []

n = 4

for l in lambda_list:

    delta1 = 1 - ncx2.cdf(
        epsilon_1**2,
        df=n,
        nc=l
    )

    delta2 = 1 - ncx2.cdf(
        epsilon_2**2,
        df=n,
        nc=l
    )

    delta3 = 1 - ncx2.cdf(
        epsilon_3**2,
        df=n,
        nc=l
    )

    delta_min1.append(delta1)
    delta_min2.append(delta2)
    delta_min3.append(delta3)


# Compact IEEE single-column figure
fig, ax = plt.subplots(
    figsize=(3.25, 1.65)
)

ax.plot(
    lambda_list,
    delta_min1,
    '-',
    linewidth=0.85,
    label=rf'$\epsilon={epsilon_1}$'
)

ax.plot(
    lambda_list,
    delta_min2,
    '--',
    linewidth=0.85,
    label=rf'$\epsilon={epsilon_2}$'
)

ax.plot(
    lambda_list,
    delta_min3,
    '-.',
    linewidth=0.85,
    label=rf'$\epsilon={epsilon_3}$'
)

ax.set_xlabel(
    r'$\lambda$',
    labelpad=1
)

ax.set_ylabel(
    r'Min. feasible $\delta$',
    labelpad=1
)

ax.set_xticks(
    [2, 10, 20, 30, 40]
)

ax.tick_params(
    axis='both',
    which='major',
    pad=1.5
)

ax.grid(
    True,
    linestyle=':',
    linewidth=0.30,
    alpha=0.30
)

ax.legend(
    loc='lower right',
    fontsize=6.0,
    handlelength=1.5,
    handletextpad=0.4,
    labelspacing=0.25,
    borderaxespad=0.3
)

fig.tight_layout(
    pad=0.25
)

fig.savefig(
    'delta_lambda_chi.pdf',
    bbox_inches='tight',
    pad_inches=0.01
)

plt.show()
plt.close(fig)
    