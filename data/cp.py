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

#first define system and all parameters
#design data matrices
# find estimates of mu and sigma for sample based method
# find estimate of C_T for structure aware method
# use that C_T to find conformal scores using calibration set and find 1-delta quantile.
#define QCQP for all cases and solve.
# simulate trajectories using solution of all
# plot optimal cost, avg state deviation and fraction of trajectories with time horizon
# plot estimation error wrt M and N in mu, sigma, and C_T

#system and parameters

n,m = 2,2
x_0 = np.array([0.0, 0.0])
T_list = range(1, 50, 5)
sigma_0 = np.eye(n) * 0.1
delta = 0.1
Q = np.diag([2.0, 3.0])
name = 'stable'
epsilon = 3.0
M = 60

sampleu = []
structu = []
cpu = []
originalu = []

ratios1 = []
ratios2 = []
ratiosk = []
ratios3 = []

x1_dev = []
x2_dev = []
xk_dev = []
x3_dev = []
delta_folder = 'traj_{}'.format(delta)
os.makedirs(delta_folder, exist_ok = True)


def conformal_quantile(scores, delta):
    N_cal = len(scores)

    k = int(np.ceil((N_cal + 1) * (1 - delta)))
    k = min(k, N_cal)

    q = np.sort(scores)[k - 1]

    return q


def get_scores(U_cal, N_cal, CT_hat, NT, CT, t):
    scores = []
    for i in range(N_cal):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x_true = CT @ U_cal[:, i] + NT@w
        x_cal = CT_hat @ U_cal[:, i]
        scores.append(np.linalg.norm(x_true - x_cal))

    return(np.array(scores))

def find_lambda_star(n, epsilon, delta):
    """
    Solves F_{chi^2_n(lambda)}(epsilon^2) = delta
    for lambda >= 0.
    
    Parameters:
    n       : degrees of freedom
    epsilon : radius (so we evaluate CDF at epsilon^2)
    delta   : probability level
    
    Returns:
    lambda_star
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

for t in T_list:
    A, B, CT = get_system(name, n, m, t, seed=0)
    uf = np.random.randn(m*t, 1)
    xf = CT@ uf
    xf = xf.reshape(-1)
    Q_tilde = block_diag(*([Q] * t))
    NT = np.eye(n)
    for i in range(1, t):
        NT = np.hstack((NT, np.linalg.matrix_power(A, i)))

    sigmaw = np.eye(n*t) * 0.1
    sigmaz = NT @ sigmaw @ NT.T
    
    #estimation
    N = m*t +3
    N_cal = 9

    U = np.random.randn(m*t, N) # input matrix
    U_cal = np.random.randn(m*t, N_cal)
    #create bar{X}
    X_bar = np.zeros((n, N))
    all_xT = np.zeros((n, N, M))
    for i in range(N):
        sumx = np.zeros(n)
        for j in range(M):
            w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
            xT = CT @ U[:,i] + NT@ w
            all_xT[:, i, j] = xT
            sumx = sumx + xT
        x_bar = sumx/M     
        X_bar[:, i] = x_bar
    
    CT_hat = X_bar @ U.T @ np.linalg.pinv(U @ U.T)
    sample_sigma = np.zeros((n, n))
    struct_sigma = np.zeros((n, n))

    for i in range(N):
       for j in range(M):
          diff = all_xT[:, i, j] - X_bar[:, i]
          diff1 = all_xT[:, i, j] - CT_hat @ U[:, i]
          sample_sigma += np.outer(diff, diff)
          struct_sigma += np.outer(diff1, diff1)

    sample_sigma = sample_sigma / (N*M)
    struct_sigma = struct_sigma/ (N*M)
    
    #find 1-delta quantile of scores
    scores = get_scores(U_cal, N_cal, CT_hat, NT, CT, t)
    q = conformal_quantile(scores, delta)

    #rhs of all inequalities
    lambda_star = find_lambda_star(n, epsilon, delta)
    lambda_cp = (epsilon - q)**2    

    alpha1 = cp.Variable(N)
    alpha2 = cp.Variable(N)
    alphak = cp.Variable(N)
    u = cp.Variable(m*t)
    


    objective1 = cp.Minimize(cp.quad_form((U @ alpha1), Q_tilde))
    objective2 = cp.Minimize(cp.quad_form((U @ alpha2), Q_tilde))
    objectivek = cp.Minimize(cp.quad_form((U @ alphak), Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u, Q_tilde))

    f1 = cp.quad_form(X_bar @ alpha1 - xf, np.linalg.inv(sample_sigma))
    f2 = cp.quad_form(CT_hat @ U @ alpha2 - xf, np.linalg.inv(struct_sigma))
    fk = cp.quad_form(CT @ U @ alphak - xf, np.linalg.inv(sigmaz))
    f3 = cp.quad_form(CT_hat @ u - xf, np.eye(n))

    prob1 = cp.Problem(objective1, [f1 <= lambda_star])
    prob2 = cp.Problem(objective2, [f2 <= lambda_star])
    probk = cp.Problem(objectivek, [fk <= lambda_star])
    prob3 = cp.Problem(objective3, [f3 <= lambda_cp])

    prob1.solve(solver = cp.SCS, verbose = False)
    prob2.solve(solver = cp.SCS, verbose = False)
    probk.solve(solver = cp.SCS, verbose = False)
    prob3.solve(solver = cp.SCS, verbose = False)

    a1 = alpha1.value
    a2 = alpha2.value
    ak = alphak.value
    u3 = u.value

    u1 = U @ a1
    u2 = U @ a2
    uk = U @ ak

    w = np.random.randn(n*t)
    if a1 is None:
        print(f"1st Convex solver failed at T={t}")
        continue

    if a2 is None:
        print(f"2nd Convex solver failed at T={t}")
        continue

    if ak is None:
        print(f"original Convex solver failed at T={t}")
        continue

    if u3 is None:
        print(f"cp Convex solver failed at T={t}")
        continue

    np.random.seed(0)

    x1 = []
    x2 = []
    xk = []
    x3 = []

    s1 = []
    s2 = []
    sk = []
    s3 = []

    for j in range(0, 2000):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = CT @ u1 + NT @ w  
        x2T =  CT @ u2 + NT @ w  
        xkT = CT @ uk + NT @ w  
        x3T = CT @ u3 + NT @ w
        s1.append(x1T)
        s2.append(x2T)
        sk.append(xkT)
        s3.append(x3T)
      
        x1.append(np.linalg.norm(x1T-xf))

        x2.append(np.linalg.norm(x2T-xf))
        xk.append(np.linalg.norm(xkT-xf))
        x3.append(np.linalg.norm(x3T-xf))

    count1 = sum(d <= epsilon for d in x1)   
    count2 = sum(d <= epsilon for d in x2)   
    countk = sum(d <= epsilon for d in xk)   
    count3 = sum(d <= epsilon for d in x3)   

    

    ratios1.append(count1/2000)
    ratios2.append(count2/2000)
    ratiosk.append(countk/2000)
    ratios3.append(count3/2000)

    x1_dev.append(np.average(x1))
    x2_dev.append(np.average(x2))
    xk_dev.append(np.average(xk))
    x3_dev.append(np.average(x3))

    s1 = np.array(s1)
    s2 = np.array(s2)
    sk = np.array(sk)
    s3 = np.array(s3)

    u1 = u1.reshape(-1, 1)
    u2 = u2.reshape(-1, 1)
    uk = uk.reshape(-1, 1)
    u3 = u3.reshape(-1, 1)

    sampleu.append(float(u1.T @ Q_tilde @ u1))
    structu.append(float(u2.T @ Q_tilde @ u2))
    originalu.append(float(uk.T @ Q_tilde @ uk))
    cpu.append(float(u3.T @ Q_tilde @ u3))

    center = (xf[0], xf[1])

    theta = np.linspace(0, 2*np.pi, 400)
    x = center[0] + epsilon * np.cos(theta)
    y = center[1] + epsilon * np.sin(theta)




    filename1 = os.path.join(delta_folder, 'sa-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('sample based MLE, horizon = {}, ratio = {}'.format(t, count1/2000))
    plt.savefig(filename1)

    filename2 = os.path.join(delta_folder, 'st-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s2[:, 0], s2[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('structure based MLE, horizon = {}, ratio = {}'.format(t, count2/2000))

    plt.savefig(filename2)

    filename3 = os.path.join(delta_folder, 'known-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(sk[:, 0], sk[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('known dynamics, horizon = {}, ratio = {}'.format(t, countk/2000))

    plt.savefig(filename3)

    filename4 = os.path.join(delta_folder, 'cp-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s3[:, 0], s3[:, 1], marker = 'x')
    plt.scatter(xf[0], xf[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((xf[0], xf[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('conformal, horizon = {}, ratio = {}'.format(t, count3/2000))

    plt.savefig(filename4)

plt.close()    
plt.figure(figsize = (10, 6))
plt.plot(T_list, x1_dev, 'o-', label = 'sample based MLE')
plt.plot(T_list, x2_dev, 'o-', label = 'structure MLE')
plt.plot(T_list, xk_dev, 'o--', color = 'black', label = 'known dynamics')
plt.plot(T_list, x3_dev, 'o-', label = 'conformal')
plt.xlabel('time horizon')
plt.ylabel('average deviation in terminal state')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('dev_{}_{}_{}.jpeg'.format(epsilon, delta, M))

plt.figure(figsize = (10, 6))
plt.plot(T_list, sampleu, 'o-', label = 'sample based MLE')
plt.plot(T_list, structu, 'o-', label = 'structure MLE')
plt.plot(T_list, originalu, 'o--', color = 'black', label = 'known dynamics')
plt.plot(T_list, cpu, 'o-', label = 'conformal')
plt.xlabel('time horizon')
plt.ylabel('optimal cost')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('norm_{}_{}_{}.jpeg'.format(epsilon, delta, M))


plt.figure(figsize = (10, 6))
plt.plot(T_list, ratios1, 'o-', label = 'sample based MLE')
plt.plot(T_list, ratios2, 'o-', label = 'structure MLE')
plt.plot(T_list, ratiosk, 'o--', color = 'black', label = 'known dynamics')
plt.plot(T_list, ratios3, 'o-', label = 'conformal')
plt.xlabel('time horizon')
plt.ylabel('ratio')
plt.title('ratio of terminal states inside epsilon radius ball')
plt.legend()
plt.grid(True)
plt.xticks(T_list)
plt.savefig('ratios_{}.jpeg'.format(M))




    