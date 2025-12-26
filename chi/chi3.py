import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from datagen import get_system
from plot import opt, diff
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib.patches import Circle
import GPy

#----------------------------------------------------------------------------------------
# fixed system parameters
n, m = 2, 2
x_0 = np.array([0.0, 0.0])

T_list = range(1, 40)  
sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
delta = 0.99
Q = np.diag([2.0, 3.0])
q = norm.ppf(1 - delta)
u_norm_diff_list = []
state_dev_diff_list = []
name = 'stable'
norm_folder = 'norms_{}'.format(name)
state_folder = 'states_{}'.format(name)
diff_folder = 'diffs_{}'.format(name)

delta_folder = 'traj_{}'.format(delta)
epsilon = 0.85
os.makedirs(norm_folder, exist_ok=True)
os.makedirs(state_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)
os.makedirs(delta_folder, exist_ok = True)
chebd = []
secd = []
epd = []
firstd = []
modeld = []

chebu = []
secu = []
epu = []
firstu = []
modelu = []
alleps = []

x1_dev = []
x2_dev = []
x3_dev = []
x4_dev = []



for t in T_list:
    A,B, NT = get_system(name, n, m, t, seed=0)
    U = np.random.randn(m * t, 1)

        # Compute reachable final state
    x_f = np.linalg.matrix_power(A, t) @ x_0.reshape(-1, 1) + NT @ U
    x_f =  x_f.reshape(-1)
   
    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = NT
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
        # C_T = np.hstack((C_T, np.linalg.matrix_power(A, i) @ B))

    # covariance and bias
    # D = np.random.randn(n*t, n*t)
    # sigma_w = D @ D.T
    sigma_w = np.eye(n*t) * 0.1
    sigma_z = np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T + N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)

    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = (np.linalg.matrix_power(A, t) @ x_0) - x_f
    b_cheb =  x_f - (np.linalg.matrix_power(A, t) @ x_0) 
    # convex constraint gamma
    gamma = epsilon**2 - traceSigma  #second order constraint
    r_cheb = epsilon**2 - 2* epsilon * math.sqrt(traceSigma/delta) + (traceSigma/delta)  #chebyshev
    dev = epsilon**2  #epsilon sd

    #choose proper epsilon
   
    
    # --- Convex QCQP ---
    u = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    u3 = cp.Variable(m*t)
    u4 = cp.Variable(m*t)
    Q_tilde = block_diag(*([Q] * t))
    objective = cp.Minimize(cp.quad_form(u, Q_tilde))
    objective2 = cp.Minimize(cp.quad_form(u2, Q_tilde))
    objective3 = cp.Minimize(cp.quad_form(u3, Q_tilde))
    objective4 = cp.Minimize(cp.quad_form(u4, Q_tilde))
    f = cp.quad_form(C_T @ u + b, np.eye(C_T.shape[0]))
    f2 = cp.quad_form(C_T @ u2 + b, np.eye(C_T.shape[0]))
    f3 = cp.quad_form(C_T @ u3 + b, np.linalg.inv(sigma_z))
    f4 = C_T @ u4 + b
    prob = cp.Problem(objective, [f <= gamma])
    prob2 = cp.Problem(objective2, [f2 <= r_cheb])
    prob3 = cp.Problem(objective3, [f3 <= dev])
    prob4 = cp.Problem(objective4, [f4 == 0])
    prob.solve(solver=cp.SCS, verbose=False)
    prob2.solve(solver = cp.SCS, verbose = False)
    prob3.solve(solver = cp.SCS, verbose = False)
    prob4.solve(solver = cp.SCS, verbose = False)
    u_convex = u.value  #second order
    u_c2 = u2.value #chebyshev
    u_c3 = u3.value #epsilon sd
    u_c4 = u4.value # first order
    print('RHS, 2nd order:', gamma)
    print('residue 2nd order:', (C_T @ u_convex + b).T @ (C_T @ u_convex + b))
    print('RHS, chebyshev:', r_cheb)
    print('residue, chebyshev:', (C_T @ u_c2 + b).T @ (C_T @ u_c2 + b))
    print('RHS, epsilon sd:', dev)
    print('residue, epsilon sd:', (C_T @ u_c3 + b).T @ np.linalg.inv(sigma_z) @ (C_T @ u_c3 + b))
    w = np.random.randn(n*t)
    x = x_f - N_T @ w
    AT = np.linalg.matrix_power(A, t)
    u_model =  np.linalg.pinv(C_T) @ (x - AT@ x_0)
    if u_convex is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c2 is None:
        print(f"Convex solver failed at T={t}")
        continue
    if u_c3 is None:
        print(f'Convex solver failed at T={t}')
    if u_c4 is None:
        print(f'Convex solver failed at T={t}')

    # --- Simulate both systems ---
    np.random.seed(0)
   
    x1 = []
    x2 = []
    x3 = []
    x4 = []

    s1 = []
    s2 = []
    s3 = []
    s4 = []
    for j in range(0, 5000):
        w = np.random.multivariate_normal(np.zeros(n*t), np.eye(n*t)*0.1)
        x1T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_convex + N_T @ w  # second order
        x2T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c2 + N_T @ w  # chebyshev
        x3T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c3 + N_T @ w  # epsilon 
        x4T = np.linalg.matrix_power(A, t) @ x_0 + C_T @ u_c4 + N_T @ w
        x1.append(np.linalg.norm(x1T-x_f))
        x2.append(np.linalg.norm(x2T-x_f))
        x3.append(np.linalg.norm(x3T-x_f))
        x4.append(np.linalg.norm(x4T - x_f))
        s1.append(x1T)
        s2.append(x2T)
        s3.append(x3T)
        s4.append(x4T)
    
    x1_dev.append(np.average(x1))
    x2_dev.append(np.average(x2))
    x3_dev.append(np.average(x3))
    x4_dev.append(np.average(x4))
   
    s1 = np.array(s1)
    s2 = np.array(s2)
    s3 = np.array(s3)
    s4 = np.array(s4)

    # --- Record metrics ---
    cheb_norm = u_c2.T @ Q_tilde @ u_c2
    ep_norm = u_c3.T @ Q_tilde @ u_c3
    sec_norm = u_convex.T @ Q_tilde @ u_convex
    det_norm = u_c4.T @ Q_tilde @ u_c4
    model_norm = u_model.T @ Q_tilde @ u_model
    # u_norm_diff = np.linalg.norm(u_c3) - np.linalg.norm(u_convex)
    
    
    # state_dev_diff = np.linalg.norm(x2[:, -1] - x1[:, -1])
    # u_norm_diff_list.append(u_norm_diff)
    # state_dev_diff_list.append(state_dev_diff)
    u_opt = u.value.reshape(t, m)
    u_n = np.linalg.norm(u_opt, axis = 1) #second order
    U = u_c3.reshape(t, m)
    u_norms = np.linalg.norm(U, axis = 1) #epsilon sd
    u_cheb = u2.value.reshape(t, m) 
    uc_norms = np.linalg.norm(u_cheb, axis = 1) #chebyshev
    u_det = u_c4.reshape(t, m)
    udet_norms = np.linalg.norm(u_det, axis = 1)



    chebu.append(cheb_norm)
    epu.append(ep_norm)
    secu.append(sec_norm)
    firstu.append(det_norm)

    center = (x_f[0], x_f[1])

    theta = np.linspace(0, 2*np.pi, 400)
    x = center[0] + epsilon * np.cos(theta)
    y = center[1] + epsilon * np.sin(theta)



    # modelu.append(model_norm)
    filename1 = os.path.join(delta_folder, '2nd-{}-{}.jpeg'.format(delta, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s1[:, 0], s1[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    plt.plot(x, y)
    plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('2nd order, horizon = {}'.format(t))
    plt.legend()
    plt.savefig(filename1)


    filename2 = os.path.join(delta_folder, 'cheb-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s2[:, 0], s2[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    # plt.plot(x, y)
    # plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('chebyshev, horizon = {}'.format(t))
    plt.legend()
    plt.savefig(filename2)


    filename3 = os.path.join(delta_folder, 'eps-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s3[:, 0], s3[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    # plt.plot(x, y)
    # plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('epsilon, horizon = {}'.format(t))
    plt.legend()
    plt.savefig(filename3)


    filename4 = os.path.join(delta_folder, 'det-{}-{}-{}.jpeg'.format(delta, epsilon, t))
    plt.figure(figsize = (10,6))
    plt.scatter(s4[:, 0], s4[:, 1], marker = 'x')
    plt.scatter(x_f[0], x_f[1], marker = 'x', color = 'red')
    # plt.plot(x, y)
    # plt.scatter(*center)
    plt.gca().set_aspect('equal', adjustable='box')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('det, horizon = {}'.format(t))
    plt.legend()
    plt.savefig(filename4)
   
    # opt(t, x1, x2, x3, x4, x_f, epsilon, u_n, u_norms, uc_norms, udet_norms, state_folder, norm_folder) #plot state dev, norms
    

# diff(T_list, u_norm_diff_list, state_dev_diff_list, diff_folder)   #plot diffs

plt.figure(figsize = (10,6))
N = np.arange(0, len(x2_dev))
plt.plot(N, x2_dev, label = 'chebyshev')
plt.plot(N, x3_dev, label = 'epsilon sd')
plt.plot(N, x1_dev, label = '2nd order state dev')
plt.plot(N, x4_dev, label = 'first order')

# plt.semilogy(N, first_avg, label = 'det state dev')
# plt.semilogy(N, modeld, label = 'model state dev')
plt.xlabel('time horizon')
plt.ylabel('State deviation')
plt.title('avg state deviation vs time horizon, delta = {}'.format(delta))
plt.legend()
plt.savefig('dev-T-{}-{}.jpeg'.format(epsilon, delta))

plt.figure(figsize = (10,6))
N = np.arange(0, len(chebu))
plt.plot(N, chebu, label = 'cheby norm')
plt.plot(N, epu, label = 'epsilon norm')
plt.plot(N, secu, label = 'second order norm')
plt.plot(N, firstu, label = 'first order norm')
# plt.plot(N, modelu, label = 'model norm')
plt.xlabel('time horizon')
plt.ylabel('u* norms')
plt.title('u* norm vs time horizon, delta = {}'.format(delta))
plt.legend()
plt.savefig('norm-T-{}-{}.jpeg'.format(epsilon, delta))


plt.figure(figsize = (10, 6))
N = np.arange(1, len(alleps)+1)
plt.plot(N, alleps, label = 'feasible epsilon vs Time horizon')
plt.xlabel('time horizon')
plt.ylabel('epsilon')
plt.title('delta = {}'.format(delta))
plt.legend()
plt.savefig('eps-T.jpeg')


#cdf of GCS using imhof's method and satterthwaite and their comparison


#simulate all systems



 
    

  

  
