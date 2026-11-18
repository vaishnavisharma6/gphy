import numpy as np

import cvxpy as cp



from scipy.linalg import block_diag

from scipy.stats import ncx2

from scipy.optimize import brentq



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from matplotlib.ticker import FuncFormatter



from datagen import get_system





# ============================================================

# IEEE / OVERLEAF PLOTTING SETTINGS

# ============================================================



mpl.rcParams.update({

    "text.usetex": False,

    "font.family": "serif",

    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],

    "mathtext.fontset": "stix",

    "font.size": 10.0,

    "axes.labelsize": 10.0,

    "axes.titlesize": 10.0,

    "legend.fontsize": 8.0,

    "xtick.labelsize": 8.5,

    "ytick.labelsize": 8.5,

    "font.weight": "normal",

    "axes.labelweight": "normal",

    "lines.linewidth": 0.90,

    "axes.linewidth": 0.65,

    "xtick.major.width": 0.6,

    "ytick.major.width": 0.6,

    "xtick.major.size": 3.0,

    "ytick.major.size": 3.0,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "savefig.dpi": 600,

})





# ============================================================

# FIGURE HELPERS

# ============================================================



def style_ieee_axis(ax):

    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.45)

    ax.tick_params(direction="in", top=True, right=True, pad=2)

    ax.margins(x=0.02)





def save_ieee_figure(fig, stem):

    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)

    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)





def shade_infeasible_regions(ax, x_vals, infeasible):

    """Shade contiguous regions where conformal prediction is infeasible (q >= epsilon)."""

    x_vals = np.asarray(x_vals)

    infeasible = np.asarray(infeasible, dtype=bool)



    if not np.any(infeasible):

        return



    edges = np.empty(len(x_vals) + 1)

    edges[1:-1] = 0.5 * (x_vals[:-1] + x_vals[1:])

    edges[0] = x_vals[0] - 0.5 * (x_vals[1] - x_vals[0])

    edges[-1] = x_vals[-1] + 0.5 * (x_vals[-1] - x_vals[-2])

    edges[0] = max(edges[0], 1e-8)  # log x-axis on the right panel



    i = 0

    while i < len(infeasible):

        if not infeasible[i]:

            i += 1

            continue

        start = i

        while i + 1 < len(infeasible) and infeasible[i + 1]:

            i += 1

        end = i

        ax.axvspan(edges[start], edges[end + 1],

                   facecolor="0.82", alpha=0.45, edgecolor="none", zorder=0)

        i += 1





# ============================================================

# CONFORMAL FUNCTIONS

# ============================================================



def conformal_quantile(scores, delta):

    N_cal = len(scores)

    k = int(np.ceil((N_cal + 1) * (1 - delta)))

    k = min(k, N_cal)

    return np.sort(scores)[k - 1]





def get_scores(U_cal, N_cal, CT_hat, A, NT, CT, T, w_cal_train, x_cal_train):

    scores = []

    A_T = np.linalg.matrix_power(A, T)



    for i in range(N_cal):

        w = w_cal_train[i]

        x0 = x_cal_train[i]

        x_true = A_T @ x0 + CT @ U_cal[:, i] + NT @ w

        x_pred = CT_hat @ U_cal[:, i]

        scores.append(np.linalg.norm(x_true - x_pred))



    return np.asarray(scores)





def find_lambda_star(n, epsilon, probability):

    """Solve F_{chi_n^2(lambda)}(epsilon^2) = probability for lambda >= 0."""

    x = epsilon ** 2



    def equation(lam):

        return ncx2.cdf(x, df=n, nc=lam) - probability



    return brentq(equation, 0.0, 1e11)





# ============================================================

# SYSTEM PARAMETERS

# ============================================================



np.random.seed(0)



n = 4

m = 4

T = 3



xf = np.array([4.0, 5.0, 6.0, 7.0])



delta = 0.5

epsilon = 3.0



Q = np.diag([2.0, 3.0, 4.0, 5.0])



name = "stable"



M = 100

N = m * T + 10

N_cal = 60

N_test = 5000



sigma_vals = np.linspace(0.3, 5.0, 40)





# ============================================================

# SYSTEM MATRICES

# ============================================================



A, B, CT = get_system(name, n, m, T, seed=0)



A_T = np.linalg.matrix_power(A, T)



# Noise controllability-type matrix N_T

N_T = np.eye(n)

for i in range(1, T):

    N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))



# Block control-cost matrix

Q_tilde = block_diag(*([Q] * T))



I_T = np.eye(T)



# Noncentral chi-square threshold

lambda_star = find_lambda_star(n, epsilon, 1 - delta)





# ============================================================

# COMMON TRAINING INPUTS

# ============================================================



U = np.random.randn(m * T, N)

U_cal = np.random.randn(m * T, N_cal)





# ============================================================

# OPTIMIZATION PROBLEMS

# ============================================================



def solve_main_methods(CT_hat, struct_sigma, sigma_z):

    """

    Solve four MEC formulations:

      1. Structure MLE   -- normalized

      2. Structure MLE   -- absolute

      3. Known dynamics  -- normalized

      4. Known dynamics  -- absolute

    """



    u_struct = cp.Variable(m * T)

    u_struct_abs = cp.Variable(m * T)

    u_known = cp.Variable(m * T)

    u_known_abs = cp.Variable(m * T)



    # 1. Structure MLE -- normalized

    problem_struct = cp.Problem(

        cp.Minimize(cp.quad_form(u_struct, Q_tilde)),

        [cp.quad_form(CT_hat @ u_struct - xf, np.linalg.pinv(struct_sigma)) <= lambda_star]

    )

    problem_struct.solve(solver=cp.SCS, verbose=False)



    # 2. Structure MLE -- absolute

    problem_struct_abs = cp.Problem(

        cp.Minimize(cp.quad_form(u_struct_abs, Q_tilde)),

        [cp.sum_squares(CT_hat @ u_struct_abs - xf) <= epsilon ** 2]

    )

    problem_struct_abs.solve(solver=cp.SCS, verbose=False)



    # 3. Known dynamics -- normalized

    problem_known = cp.Problem(

        cp.Minimize(cp.quad_form(u_known, Q_tilde)),

        [cp.quad_form(CT @ u_known - xf, np.linalg.pinv(sigma_z)) <= lambda_star]

    )

    problem_known.solve(solver=cp.SCS, verbose=False)



    # 4. Known dynamics -- absolute

    problem_known_abs = cp.Problem(

        cp.Minimize(cp.quad_form(u_known_abs, Q_tilde)),

        [cp.sum_squares(CT @ u_known_abs - xf) <= epsilon ** 2]

    )

    problem_known_abs.solve(solver=cp.SCS, verbose=False)



    if u_struct.value is None:

        raise RuntimeError("Structure normalized problem failed.")

    if u_struct_abs.value is None:

        raise RuntimeError("Structure absolute problem failed.")

    if u_known.value is None:

        raise RuntimeError("Known normalized problem failed.")

    if u_known_abs.value is None:

        raise RuntimeError("Known absolute problem failed.")



    return (

        np.asarray(u_struct.value).reshape(-1),

        np.asarray(u_struct_abs.value).reshape(-1),

        np.asarray(u_known.value).reshape(-1),

        np.asarray(u_known_abs.value).reshape(-1),

    )





def solve_conformal(CT_hat, q):

    """

    min u' Q u   s.t.  ||C_hat u - xf||_2 <= epsilon - q.

    If q >= epsilon the problem is marked infeasible (returns None).

    """

    if q >= epsilon:

        return None



    radius = epsilon - q

    u_conf = cp.Variable(m * T)



    problem = cp.Problem(

        cp.Minimize(cp.quad_form(u_conf, Q_tilde)),

        [cp.sum_squares(CT_hat @ u_conf - xf) <= radius ** 2]

    )

    problem.solve(solver=cp.SCS, verbose=False)



    if u_conf.value is None:

        return None



    return np.asarray(u_conf.value).reshape(-1)





# ============================================================

# STRUCTURE-MLE ESTIMATION

# ============================================================



def estimate_structure(x_train, w_train):

    """Estimate C_T (structured least squares / MLE) and the residual covariance."""



    X_bar = np.zeros((n, N))

    all_xT = np.zeros((n, N, M))



    for i in range(N):

        for j in range(M):

            all_xT[:, i, j] = (

                A_T @ x_train[i, j]

                + CT @ U[:, i]

                + N_T @ w_train[i, j]

            )

        X_bar[:, i] = np.mean(all_xT[:, i, :], axis=1)



    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)



    struct_sigma = np.zeros((n, n))

    for i in range(N):

        for j in range(M):

            residual = all_xT[:, i, j] - CT_hat @ U[:, i]

            struct_sigma += np.outer(residual, residual)

    struct_sigma /= (N * M)



    return CT_hat, struct_sigma





def control_cost(u):

    return float(u.T @ Q_tilde @ u)





# ============================================================

# DEVIATION METRICS

#

#   normalized :  mu_e' Sigma_e^{-1} mu_e

#   absolute   :  mu_e' mu_e + tr(Sigma_e)

#

#   known dynamics : mu_e  = empirical (Monte-Carlo) mean of x_T - xf

#                    Sigma_e = true Sigma_z

#   data-driven    : hat quantities

#                    mu_e_hat    = CT_hat u - xf

#                    Sigma_e_hat = struct_sigma

#   conformal      : no covariance of its own -> absolute metric with

#                    hat mean and hat covariance

# ============================================================



def mc_mean_error(u, x0_scale, w_scale, z_x_test, z_w_test):

    """Empirical mean of x_T - xf under the TRUE system (vectorised)."""

    x0s = x0_scale * z_x_test          # (N_test, n)

    ws = w_scale * z_w_test            # (N_test, n*T)

    errs = x0s @ A_T.T + ws @ N_T.T + (CT @ u - xf)

    return errs.mean(axis=0)





def maha(mu, S):

    return float(mu @ np.linalg.pinv(S) @ mu)





def eval_deviations(u_struct, u_struct_abs, u_known, u_known_abs, u_conf,

                    CT_hat, struct_sigma, sigma_z,

                    x0_scale, w_scale, z_x_test, z_w_test):



    tr_true = np.trace(sigma_z)

    tr_hat = np.trace(struct_sigma)



    # known dynamics: empirical mean, true covariance

    mu_k = mc_mean_error(u_known, x0_scale, w_scale, z_x_test, z_w_test)

    mu_ka = mc_mean_error(u_known_abs, x0_scale, w_scale, z_x_test, z_w_test)



    # data-driven: hat mean, hat covariance

    mu_s = CT_hat @ u_struct - xf

    mu_sa = CT_hat @ u_struct_abs - xf



    out = dict(

        known=maha(mu_k, sigma_z),

        known_abs=float(mu_ka @ mu_ka + tr_true),

        struct=maha(mu_s, struct_sigma),

        struct_abs=float(mu_sa @ mu_sa + tr_hat),

    )



    if u_conf is None:

        out["conf"] = np.nan

    else:

        mu_c = CT_hat @ u_conf - xf

        out["conf"] = float(mu_c @ mu_c + tr_hat)



    return out





# ============================================================

# PLOT FUNCTION

# ============================================================



def make_ieee_figure(sigma_vals, cost, dev, conf_infeasible, xlabel, filename):

    """

    cost / dev: dicts with keys struct, struct_abs, conf, known, known_abs.

    """



    fig, ax = plt.subplots(1, 2, figsize=(3.48, 2.10))

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.22, top=0.68, wspace=0.42)



    series = [

        ("struct",     "Data-driven (MLE\nbased), normalised",   "-"),

        ("struct_abs", "Data-driven (MLE\nbased), absolute",     "-"),

        ("conf",       "Conformal",              "-"),

        ("known",      "Known model,\nnormalised", ":"),

        ("known_abs",  "Known model,\nabsolute",   ":"),

    ]



    # Left: optimal cost

    for key, label, ls in series:

        ax[0].plot(sigma_vals, cost[key], linestyle=ls, linewidth=0.9,

                   label=label, zorder=3)

    shade_infeasible_regions(ax[0], sigma_vals, conf_infeasible)



    # Right: deviation

    for key, label, ls in series:

        ax[1].semilogy(sigma_vals, dev[key], linestyle=ls, linewidth=0.9, zorder=3)

    shade_infeasible_regions(ax[1], sigma_vals, conf_infeasible)



    for a in ax:

        style_ieee_axis(a)

        a.xaxis.label.set_size(10.0)

        a.yaxis.label.set_size(10.0)

        a.tick_params(labelsize=8.5)



    ax[0].set_xlabel(xlabel)

    ax[0].set_ylabel(r"$J_T^*$")

    ax[1].set_xlabel(xlabel)

    ax[1].set_ylabel("Mean deviation", labelpad=0)

    ax[1].yaxis.set_label_coords(-0.13, 0.5)



    # Log y-axis: plain numbers

    ax[1].yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0,)))

    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))

    ax[1].yaxis.set_minor_locator(

        mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    ax[1].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())



    # Keep the sigma sweep (0.3 to 5.0), with identical ticks
    # and x-limits on both panels; only the mean-deviation y-axis is log.
    shared_xticks = [sigma_vals[0], 2.0, 3.0, 4.0, sigma_vals[-1]]
    for a in ax:
        a.set_xlim(sigma_vals[0], sigma_vals[-1])
        a.set_xticks(shared_xticks)
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))



    # Shared legend

    handles, labels = ax[0].get_legend_handles_labels()

    handles.append(Patch(facecolor="0.82", alpha=0.45, edgecolor="none",

                         label="CP infeasible"))

    labels.append("CP infeasible")



    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),

               ncol=3, frameon=False, fontsize=8.0, handlelength=1.25,

               handletextpad=0.25, columnspacing=0.4, borderaxespad=0.0)



    save_ieee_figure(fig, filename)

    plt.show()





# ============================================================

# GENERIC SWEEP

#

# mode = "w":  vary process-noise std sigma, x0 std fixed

# mode = "x0": vary initial-state std sigma, process-noise std fixed

# ============================================================



KEYS = ["struct", "struct_abs", "conf", "known", "known_abs"]





def run_sweep(mode, sigma_0_fixed=0.2, sigma_w_fixed=0.3):



    # Common random numbers (drawn in the same order as the original script)

    z_x_train = np.random.randn(N, M, n)

    z_x_cal = np.random.randn(N_cal, n)

    z_w_train = np.random.randn(N, M, n * T)

    z_w_cal = np.random.randn(N_cal, n * T)

    z_x_test = np.random.randn(N_test, n)

    z_w_test = np.random.randn(N_test, n * T)



    cost = {k: [] for k in KEYS}

    dev = {k: [] for k in KEYS}

    conf_infeasible = []

    q_list = []



    for sigma in sigma_vals:



        if mode == "w":

            x0_scale, w_scale = sigma_0_fixed, sigma

        elif mode == "x0":

            x0_scale, w_scale = sigma, sigma_w_fixed

        else:

            raise ValueError(mode)



        # True covariances

        sigma_0 = x0_scale ** 2 * np.eye(n)

        sigma_w = np.kron(I_T, w_scale ** 2 * np.eye(n))

        sigma_z = N_T @ sigma_w @ N_T.T + A_T @ sigma_0 @ A_T.T



        # Training / calibration data

        x_train = x0_scale * z_x_train

        x_cal_train = x0_scale * z_x_cal

        w_train = w_scale * z_w_train

        w_cal_train = w_scale * z_w_cal



        # Structure MLE

        CT_hat, struct_sigma = estimate_structure(x_train, w_train)



        # Conformal quantile

        scores = get_scores(U_cal, N_cal, CT_hat, A, N_T, CT, T,

                            w_cal_train, x_cal_train)

        q = conformal_quantile(scores, delta)

        q_list.append(q)

        conf_infeasible.append(q >= epsilon)



        # Solve

        u_struct, u_struct_abs, u_known, u_known_abs = solve_main_methods(

            CT_hat, struct_sigma, sigma_z)

        u_conf = solve_conformal(CT_hat, q)



        # Costs

        cost["struct"].append(control_cost(u_struct))

        cost["struct_abs"].append(control_cost(u_struct_abs))

        cost["known"].append(control_cost(u_known))

        cost["known_abs"].append(control_cost(u_known_abs))

        cost["conf"].append(np.nan if u_conf is None else control_cost(u_conf))



        # Deviations

        d = eval_deviations(u_struct, u_struct_abs, u_known, u_known_abs, u_conf,

                            CT_hat, struct_sigma, sigma_z,

                            x0_scale, w_scale, z_x_test, z_w_test)

        for k in KEYS:

            dev[k].append(d[k])



    return cost, dev, conf_infeasible, q_list





def print_infeasible(title, label, q_list, conf_infeasible):

    print(f"\n{title}:")

    print("-" * (len(title) + 1))

    for s, q, inf in zip(sigma_vals, q_list, conf_infeasible):

        if inf:

            print(f"{label} = {s:.3f}, q = {q:.3f}  --> CP infeasible")





# ============================================================

# EXPERIMENT 1: vary process-noise std sigma

# ============================================================



cost1, dev1, inf1, q1 = run_sweep("w", sigma_0_fixed=0.2)



make_ieee_figure(sigma_vals, cost1, dev1, inf1,

                 xlabel=r"$\sigma$",

                 filename="all_methods_cost_dev_vs_sigma_w")



print_infeasible("Process-noise sweep", "sigma", q1, inf1)





# ============================================================

# EXPERIMENT 2: vary initial-state std sigma_0

# ============================================================



cost2, dev2, inf2, q2 = run_sweep("x0", sigma_w_fixed=0.3)



make_ieee_figure(sigma_vals, cost2, dev2, inf2,

                 xlabel=r"$\sigma_0$",

                 filename="all_methods_cost_dev_vs_sigma_0")



print_infeasible("Initial-state uncertainty sweep", "sigma_0", q2, inf2)