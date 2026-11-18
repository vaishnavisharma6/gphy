import os

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt



from datagen import get_system





# ============================================================

# IEEE / OVERLEAF PUBLICATION SETTINGS

# ============================================================



mpl.rcParams.update({

    "text.usetex": False,



    # IEEE-compatible serif text + Times-like math

    "font.family": "serif",

    "font.serif": [

        "Times New Roman",

        "Times",

        "Nimbus Roman No9 L",

        "DejaVu Serif",

    ],



    "mathtext.fontset": "stix",



    # IEEE-friendly sizes

    "font.size": 8.0,

    "axes.labelsize": 8.0,

    "axes.titlesize": 8.0,

    "legend.fontsize": 6.4,

    "xtick.labelsize": 7.0,

    "ytick.labelsize": 7.0,



    # Thin publication-style lines

    "font.weight": "normal",

    "axes.labelweight": "normal",

    "lines.linewidth": 0.65,

    "axes.linewidth": 0.65,



    # Compact ticks

    "xtick.major.width": 0.6,

    "ytick.major.width": 0.6,

    "xtick.major.size": 3.0,

    "ytick.major.size": 3.0,



    # Keep vector text editable/searchable in PDF

    "pdf.fonttype": 42,

    "ps.fonttype": 42,



    # High-resolution PNG fallback

    "savefig.dpi": 600,

})





# ============================================================

# IEEE FIGURE HELPERS

# ============================================================



def style_ieee_axis(ax):

    """Apply compact IEEE publication style."""



    ax.grid(

        True,

        which="major",

        linestyle=":",

        linewidth=0.45,

        alpha=0.45,

    )



    ax.tick_params(

        direction="in",

        top=True,

        right=True,

        pad=2,

    )



    ax.margins(x=0.02)





def set_compact_font_sizes(ax):

    """Same compact axis sizes as the reference plots."""



    ax.xaxis.label.set_size(6.5)

    ax.yaxis.label.set_size(6.5)



    ax.tick_params(

        labelsize=5.8

    )





def save_ieee_figure(fig, stem):

    """Save PDF and 600-dpi PNG."""



    fig.savefig(

        f"{stem}.pdf",

        bbox_inches="tight",

        pad_inches=0.02,

    )



    fig.savefig(

        f"{stem}.png",

        dpi=600,

        bbox_inches="tight",

        pad_inches=0.02,

    )





# ============================================================

# GLOBAL PARAMETERS

# ============================================================



name = "stable"



# Default dimensions

n_default = 4

m = 4



# Default horizon

T_default = 8



# Horizon used for sigma sensitivity study.

T_sensitivity = 4





# ============================================================

# UNCERTAINTY PARAMETERS

#

# Sigma_0 = sigma_0^2 I

# Sigma_w = sigma^2 I

# ============================================================



sigma0_default = 0.2

sigma_default = 0.4





# ============================================================

# DATA PARAMETERS

# ============================================================



# Repeated experiments for estimating C_T

M = 100



# Calibration samples

N_cal = 100



# Independent test samples for coverage

N_test = 2000



# Independent Monte-Carlo datasets

num_runs = 20





# ============================================================

# CONFORMAL PARAMETERS

# ============================================================



# Fixed delta for all q plots

delta_q = 0.30



# Miscoverage values for coverage plot

delta_list = [

    0.30,

    0.20,

    0.10,

    0.05,

    0.01,

]





# ============================================================

# PARAMETER SWEEPS

# ============================================================



# Only three T values for nominal-vs-empirical coverage

T_list_cov = [

    2,

    6,

    10,

]



# q vs T

T_list = list(

    range(1, 20, 2)

)



# q vs sigma

sigma_list = np.linspace(

    0.0,

    0.5,

    11,

)





# ============================================================

# RANDOM SEED

# ============================================================



BASE_SEED = 1000





# ============================================================

# OUTPUT DIRECTORY

# ============================================================



OUTDIR = "ieee_conformal_plots"



os.makedirs(

    OUTDIR,

    exist_ok=True,

)





# ============================================================

# CONFORMAL QUANTILE

# ============================================================



def conformal_quantile(

    scores,

    delta,

):

    """

    Finite-sample split-conformal quantile



        q_{1-delta}



    with



        k = ceil((N_cal + 1)(1-delta)).

    """



    N_cal_local = len(scores)



    k = int(

        np.ceil(

            (N_cal_local + 1)

            *

            (1 - delta)

        )

    )



    k = min(

        k,

        N_cal_local,

    )



    return np.sort(scores)[

        k - 1

    ]





# ============================================================

# BUILD SYSTEM MATRICES

# ============================================================



def build_system(

    n,

    m,

    T,

):

    """

    Build A, B, C_T and N_T such that



        x_T = A^T x_0 + C_T u + N_T w.

    """



    A, B, CT = get_system(

        name,

        n,

        m,

        T,

        seed=0,

    )



    NT = np.eye(n)



    for k in range(1, T):



        NT = np.hstack((

            NT,

            np.linalg.matrix_power(

                A,

                k,

            ),

        ))



    return (

        A,

        B,

        CT,

        NT,

    )





# ============================================================

# STRUCTURE-BASED ESTIMATE OF C_T

# ============================================================



def estimate_CT(

    n,

    m,

    T,

    sigma0_std,

    sigma_std,

    M,

    rng,

):

    """

    Structure-based estimator



        C_T_hat

        =

        X_bar U^T (U U^T)^dagger.



    where



        x_0 ~ N(0, sigma_0^2 I),



        w ~ N(0, sigma^2 I).

    """



    (

        A,

        _,

        CT,

        NT,

    ) = build_system(

        n,

        m,

        T,

    )



    # --------------------------------------------------------

    # Number of training inputs

    # --------------------------------------------------------



    N = m * T + 10





    # --------------------------------------------------------

    # Input matrix

    # --------------------------------------------------------



    U = rng.standard_normal(

        (m * T, N)

    )





    # --------------------------------------------------------

    # Initial-state samples

    #

    # Shape:

    #     N x M x n

    # --------------------------------------------------------



    x0 = (

        sigma0_std

        *

        rng.standard_normal(

            (N, M, n)

        )

    )





    # --------------------------------------------------------

    # Process-noise samples

    #

    # Shape:

    #     N x M x nT

    # --------------------------------------------------------



    w = (

        sigma_std

        *

        rng.standard_normal(

            (N, M, n * T)

        )

    )





    AT = np.linalg.matrix_power(

        A,

        T,

    )





    # --------------------------------------------------------

    # A^T x_0

    # --------------------------------------------------------



    x0_term = np.einsum(

        "ab,ijb->ija",

        AT,

        x0,

    )





    # --------------------------------------------------------

    # N_T w

    # --------------------------------------------------------



    w_term = np.einsum(

        "ab,ijb->ija",

        NT,

        w,

    )





    # --------------------------------------------------------

    # C_T u

    #

    # Shape:

    #     N x 1 x n

    # --------------------------------------------------------



    control_term = (

        CT @ U

    ).T[:, None, :]





    # --------------------------------------------------------

    # Terminal-state samples

    # --------------------------------------------------------



    xT = (

        x0_term

        +

        control_term

        +

        w_term

    )





    # --------------------------------------------------------

    # Mean terminal state for each training input

    #

    # X_bar in R^{n x N}

    # --------------------------------------------------------



    X_bar = np.mean(

        xT,

        axis=1,

    ).T





    # --------------------------------------------------------

    # Structure-based estimator

    # --------------------------------------------------------



    CT_hat = (

        X_bar

        @ U.T

        @ np.linalg.pinv(

            U @ U.T

        )

    )





    return (

        A,

        CT,

        NT,

        CT_hat,

    )





# ============================================================

# CALIBRATION SCORES

# ============================================================



def calibration_scores(

    A,

    CT,

    NT,

    CT_hat,

    n,

    m,

    T,

    sigma0_std,

    sigma_std,

    N_cal,

    rng,

):

    """

    Calibration scores



        S_i

        =

        ||x_T^(i) - C_T_hat u_i||_2.

    """



    U_cal = rng.standard_normal(

        (m * T, N_cal)

    )



    x0 = (

        sigma0_std

        *

        rng.standard_normal(

            (N_cal, n)

        )

    )



    w = (

        sigma_std

        *

        rng.standard_normal(

            (N_cal, n * T)

        )

    )



    AT = np.linalg.matrix_power(

        A,

        T,

    )





    # Model-estimation error

    model_error = (

        (CT - CT_hat)

        @ U_cal

    ).T





    # Initial-state contribution

    x0_error = (

        x0

        @ AT.T

    )





    # Process-noise contribution

    w_error = (

        w

        @ NT.T

    )





    error = (

        model_error

        +

        x0_error

        +

        w_error

    )





    return np.linalg.norm(

        error,

        axis=1,

    )





# ============================================================

# TEST SCORES

# ============================================================



def test_scores(

    A,

    CT,

    NT,

    CT_hat,

    n,

    m,

    T,

    sigma0_std,

    sigma_std,

    N_test,

    rng,

):

    """

    Independent scores for empirical coverage.

    """



    U_test = rng.standard_normal(

        (m * T, N_test)

    )



    x0 = (

        sigma0_std

        *

        rng.standard_normal(

            (N_test, n)

        )

    )



    w = (

        sigma_std

        *

        rng.standard_normal(

            (N_test, n * T)

        )

    )



    AT = np.linalg.matrix_power(

        A,

        T,

    )





    model_error = (

        (CT - CT_hat)

        @ U_test

    ).T





    x0_error = (

        x0

        @ AT.T

    )





    w_error = (

        w

        @ NT.T

    )





    error = (

        model_error

        +

        x0_error

        +

        w_error

    )





    return np.linalg.norm(

        error,

        axis=1,

    )





# ============================================================

# COMPUTE q FOR ONE INDEPENDENT DATASET

# ============================================================



def compute_q_once(

    n,

    m,

    T,

    sigma0_std,

    sigma_std,

    delta,

    seed,

):

    """

    Estimate C_T and calculate q_{1-delta}.

    """



    rng = np.random.default_rng(

        seed

    )



    (

        A,

        CT,

        NT,

        CT_hat,

    ) = estimate_CT(

        n=n,

        m=m,

        T=T,

        sigma0_std=sigma0_std,

        sigma_std=sigma_std,

        M=M,

        rng=rng,

    )





    scores = calibration_scores(

        A=A,

        CT=CT,

        NT=NT,

        CT_hat=CT_hat,

        n=n,

        m=m,

        T=T,

        sigma0_std=sigma0_std,

        sigma_std=sigma_std,

        N_cal=N_cal,

        rng=rng,

    )





    return conformal_quantile(

        scores,

        delta,

    )





# ============================================================

# AVERAGE q OVER INDEPENDENT DATASETS

#

# Used for q vs T.

# ============================================================



def average_q(

    n,

    m,

    T,

    sigma0_std,

    sigma_std,

    delta,

    num_runs,

):

    q_runs = []





    for run in range(num_runs):



        q = compute_q_once(

            n=n,

            m=m,

            T=T,

            sigma0_std=sigma0_std,

            sigma_std=sigma_std,

            delta=delta,

            seed=BASE_SEED + run,

        )



        q_runs.append(q)





    return (

        np.mean(q_runs),

        np.std(q_runs),

    )





# ============================================================

# A. NOMINAL VS EMPIRICAL COVERAGE

# ============================================================



coverage_results = {



    T: {

        delta: []

        for delta in delta_list

    }



    for T in T_list_cov

}





for T in T_list_cov:



    print(

        f"Coverage: T={T}"

    )





    for run in range(num_runs):



        rng = np.random.default_rng(

            BASE_SEED

            + 10000 * T

            + run

        )





        (

            A,

            CT,

            NT,

            CT_hat,

        ) = estimate_CT(

            n=n_default,

            m=m,

            T=T,

            sigma0_std=sigma0_default,

            sigma_std=sigma_default,

            M=M,

            rng=rng,

        )





        # ----------------------------------------------------

        # Calibration set

        # ----------------------------------------------------



        scores_cal = calibration_scores(

            A=A,

            CT=CT,

            NT=NT,

            CT_hat=CT_hat,

            n=n_default,

            m=m,

            T=T,

            sigma0_std=sigma0_default,

            sigma_std=sigma_default,

            N_cal=N_cal,

            rng=rng,

        )





        # ----------------------------------------------------

        # Independent test set

        # ----------------------------------------------------



        scores_test = test_scores(

            A=A,

            CT=CT,

            NT=NT,

            CT_hat=CT_hat,

            n=n_default,

            m=m,

            T=T,

            sigma0_std=sigma0_default,

            sigma_std=sigma_default,

            N_test=N_test,

            rng=rng,

        )





        for delta in delta_list:



            q = conformal_quantile(

                scores_cal,

                delta,

            )





            coverage = np.mean(

                scores_test <= q

            )





            coverage_results[

                T

            ][

                delta

            ].append(

                coverage

            )





# Nominal coverage

nominal = np.asarray([

    1.0 - delta

    for delta in delta_list

])





# ============================================================

# COVERAGE FIGURE

# ============================================================



fig, ax = plt.subplots(

    figsize=(3.48, 2.05),

)





fig.subplots_adjust(

    left=0.16,

    right=0.98,

    bottom=0.22,

    top=0.78,

)





coverage_colors = {

    2: "C0",

    6: "C1",

    10: "C2",

}





for T in T_list_cov:



    empirical = np.asarray([



        np.mean(

            coverage_results[T][delta]

        )



        for delta in delta_list

    ])





    ax.plot(

        nominal,

        empirical,

        color=coverage_colors[T],

        linestyle="-",

        linewidth=0.70,

        label=rf"$T={T}$",

    )





# Ideal line

ideal_x = np.linspace(

    np.min(nominal),

    1.0,

    100,

)





ax.plot(

    ideal_x,

    ideal_x,

    color="black",

    linestyle=":",

    linewidth=0.70,

    label="Ideal",

)





style_ieee_axis(ax)





ax.set_xlabel(

    "Nominal coverage"

)



ax.set_ylabel(

    "Empirical coverage"

)





ax.set_xlim(

    np.min(nominal) - 0.01,

    1.005,

)



ax.set_ylim(

    np.min(nominal) - 0.03,

    1.01,

)





set_compact_font_sizes(ax)





# Legend above plot

ax.legend(

    loc="lower center",

    bbox_to_anchor=(

        0.5,

        1.015,

    ),

    ncol=4,

    frameon=False,

    fontsize=4.7,

    handlelength=1.7,

    handletextpad=0.35,

    columnspacing=0.8,

    labelspacing=0.25,

    borderaxespad=0.0,

)





coverage_stem = os.path.join(

    OUTDIR,

    "coverage_vs_nominal",

)





save_ieee_figure(

    fig,

    coverage_stem,

)



plt.show()





# ============================================================

# B. q_{1-delta} VS T

# ============================================================



q_T = []

q_T_std = []





for T in T_list:



    print(

        f"q vs T: T={T}"

    )





    q_mean, q_std = average_q(

        n=n_default,

        m=m,

        T=T,

        sigma0_std=sigma0_default,

        sigma_std=sigma_default,

        delta=delta_q,

        num_runs=num_runs,

    )





    q_T.append(

        q_mean

    )



    q_T_std.append(

        q_std

    )





q_T = np.asarray(

    q_T

)



q_T_std = np.asarray(

    q_T_std

)





# ============================================================

# C. q_{1-delta} VS sigma

# Process-noise standard-deviation sensitivity study.

# For every sigma value, C_T_hat is re-estimated from a new

# training dataset and q_{1-delta} is computed from an

# independent calibration dataset.

# ============================================================



num_runs_sensitivity = 200



q_sigma = []

q_sigma_std = []



for sigma_value in sigma_list:



    print(

        "q vs sigma: "

        f"sigma={sigma_value:.3f}"

    )



    q_runs = []



    for run in range(num_runs_sensitivity):



        # Same run seed across all sigma values = common random numbers.

        # C_T_hat is re-estimated for every sigma because sigma_std

        # changes the generated training data X inside estimate_CT().

        seed = (

            BASE_SEED

            + 30000

            + run

        )



        q = compute_q_once(

            n=n_default,

            m=m,

            T=T_sensitivity,

            sigma0_std=sigma0_default,

            sigma_std=sigma_value,

            delta=delta_q,

            seed=seed,

        )



        q_runs.append(q)



    q_sigma.append(

        np.mean(q_runs)

    )



    q_sigma_std.append(

        np.std(q_runs)

    )



q_sigma = np.asarray(

    q_sigma

)



q_sigma_std = np.asarray(

    q_sigma_std

)





# ============================================================

# D. COMBINED q_{1-delta} FIGURE

#

#               q vs T        q vs sigma

#

# Compact IEEE single-column width.

# No markers.

# ============================================================



fig, ax = plt.subplots(

    1,

    2,

    figsize=(3.48, 1.75),

)



fig.subplots_adjust(

    left=0.15,

    right=0.98,

    bottom=0.24,

    top=0.96,

    wspace=0.44,

)





# ============================================================

# LEFT

# q_{1-delta} VS T

# ============================================================



ax[0].plot(

    T_list,

    q_T,

    color="C0",

    linestyle="-",

    linewidth=0.70,

)



ax[0].set_xlabel(

    r"$T$"

)



ax[0].set_ylabel(

    r"$q_{1-\delta}$"

)



ax[0].set_xticks(

    [1, 5, 9, 13, 17]

)





# ============================================================

# RIGHT

# q_{1-delta} VS sigma

# ============================================================



ax[1].plot(

    sigma_list,

    q_sigma,

    color="C0",

    linestyle="-",

    linewidth=0.70,

)



ax[1].set_xlabel(

    r"$\sigma$"

)



ax[1].set_ylabel(

    r"$q_{1-\delta}$"

)



ax[1].set_xticks(

    [0.0, 0.2, 0.4]

)





# ============================================================

# COMMON AXIS FORMATTING

# ============================================================



for a in ax:



    style_ieee_axis(a)



    set_compact_font_sizes(a)





# Slightly adjust right-panel y label

ax[1].yaxis.set_label_coords(

    -0.14,

    0.5,

)





# ============================================================

# SAVE COMBINED q FIGURE

# ============================================================



quantile_stem = os.path.join(

    OUTDIR,

    "conformal_quantiles",

)



save_ieee_figure(

    fig,

    quantile_stem,

)



plt.show()





# ============================================================

# PRINT NUMERICAL VALUES

# ============================================================



print(

    "\n========================================"

)



print(

    "q vs T"

)



print(

    "========================================"

)



for T, q in zip(

    T_list,

    q_T,

):



    print(

        f"T={T:2d}: q={q:.6f}"

    )





print(

    "\n========================================"

)



print(

    "q vs sigma"

)



print(

    "========================================"

)



for sigma_value, q in zip(

    sigma_list,

    q_sigma,

):



    print(

        f"sigma={sigma_value:.2f}: "

        f"q={q:.6f}"

    )





# ============================================================

# SUMMARY

# ============================================================



print(

    "\n========================================"

)



print(

    "Finished."

)



print(

    "========================================"

)



print(

    f"Saved: {coverage_stem}.pdf"

)



print(

    f"Saved: {coverage_stem}.png"

)



print(

    f"Saved: {quantile_stem}.pdf"

)



print(

    f"Saved: {quantile_stem}.png"

)



print(

    f"All q plots use delta={delta_q}"

)



print(

    f"sigma sensitivity uses T={T_sensitivity}"

)


