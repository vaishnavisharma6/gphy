import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from datagen import get_system
import math
import cvxpy as cp
import GPy
#gp code for different scenerios

#1. unknown cost, known model, set of u with dim mT
#Simple GP regression
#constraint parameters
n,m = 2,2
epsilon = 0.8
delta = 0.9
x_0 = np.array([0.0, 0.0])

sigma_0 = np.eye(n) * 0.1
sigma_w_base = np.eye(n) * 0.1
q = norm.ppf(1 - delta)
name = 'stable'
Q = np.diag([2.0, 1.0])

gp_norm_folder = 'norms_{}'.format(name)
gp_state_folder = 'states_{}'.format(name)
gp_diff_folder = 'diffs_{}'.format(name)


for t in range(1,3):
    u = cp.Variable(m*t)
    u2 = cp.Variable(m*t)
    A,B, NT = get_system(name, n=2, m=2, T = t, seed=2)
    U = np.random.randn(m * t, 1)

        # Compute reachable final state
    x_f = np.linalg.matrix_power(A, t) @ x_0.reshape(-1, 1) + NT @ U
    x_f =  x_f.reshape(-1)

    # Construct N_T and C_T
    N_T = np.eye(n)
    C_T = B
    for i in range(1, t):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i)))
        C_T = np.hstack((C_T, np.linalg.matrix_power(A, i) @ B))

    sigma_w = np.eye(n*t) * 0.1
    sigma_z = np.linalg.matrix_power(A, t) @ sigma_0 @ np.linalg.matrix_power(A, t).T + N_T @ sigma_w @ N_T.T
    traceSigma = np.trace(sigma_z)
    tau = 2 * np.trace(sigma_z @ sigma_z)
    b = (np.linalg.matrix_power(A, t) @ x_0) - x_f
    b_cheb =  x_f - (np.linalg.matrix_power(A, t) @ x_0) 
    # convex constraint gamma
    gamma = epsilon**2 - traceSigma - q * math.sqrt(tau) 
    r_cheb = epsilon - math.sqrt(traceSigma/delta)    
    f = cp.quad_form(C_T @ u + b, np.eye(C_T.shape[0]))
    f2 = cp.norm2((C_T @ u2) - b_cheb)
    # constraint1 = [f <= gamma]
    # constraint2 = [f2 <= r_cheb]
 #----------------------------------------------------------------------------------------------------------------------   
    #GP-UCB sampling to learn cost using constraint

    def sample_from_ellipsoid(C_T, b, r, num_samples):
  
        n, mT = C_T.shape

        # Pseudoinverse and center of ellipsoid
        C_pinv = np.linalg.pinv(C_T)
        u_center = C_pinv @ b

        # Orthogonal basis for nullspace (directions with no effect on C_T u)
        U, S, Vt = np.linalg.svd(C_T)
        nullspace_basis = Vt.T[:, np.where(S < 1e-10)[0]]  # empty if full rank

        # Approximate covariance (shape matrix)
        A = C_pinv @ np.eye(n) @ C_pinv.T  # any scaling gives same shape

        # Cholesky factor (or SVD-based)
        L = np.linalg.cholesky(A + 1e-8 * np.eye(A.shape[0]))

        # Samples on unit ball
        z = np.random.randn(mT, num_samples)
        z /= np.linalg.norm(z, axis=0, keepdims=True)

        # Random radii for uniform sampling in ball
        r_scales = np.random.rand(num_samples) ** (1 / mT)
        z *= r_scales * r

        # Map to ellipsoid
        u_samples = u_center.reshape(-1, 1) + L @ z

        return u_samples.T 
    
    num_samples = 10
    
    def cost(t, u):
        w = np.random.rand(1)
        c = []
        for j in range(0, u.shape[0]):
            cu = 0
            for i in range(t+1):
                if i%m == 0:
                    cu = cu + (u[j, i: i+m]  @ Q  @ np.transpose(u[j, i:i+m]))
            cu = cu + w
            c.append(cu)
        return(c)
    u_train = sample_from_ellipsoid(C_T, b, r_cheb, num_samples)
    print(u_train)
    y_train = cost(t, u_train)
    
    #GP prior
    mu = np.zeros((num_samples, 1))
    k = GPy.kern.RBF(m*t, variance = 1., lengthscale = 0.1, name = 'RBF')
    C = k.K(u_train, u_train)
    Sigma = C.copy()
    u_obs = np.empty((0, m*t))
    y_obs = np.empty((0, 1))

    n = t//2  #training horizon

    beta_t = lambda t: 2 * np.log(num_samples * np.pi**2 * t**2 / (6*0.1)) # idk what is this

    means = {}
    
    #training
    for l in range(0, n+1):
        std = np.sqrt(np.diag(Sigma)).reshape(-1,1)
        ucb = mu + np.sqrt(beta_t(l)) * std

        idx = np.argmax(ucb)
        u_t = u_train[idx, :]
        u_t = u_t.reshape(-1,1)
        y_t = cost(t, u_t)
        
        #update data
        u_obs = np.vstack((u_obs, u_t))
        y_obs = np.vstack((y_obs, y_t))

        #Necessary parameters
        k_uu = k.K(u_train, u_train) 
        k_unu = k.K(u_train, u_obs)
        k_nu = k.K(u_obs, u_obs)
        k_uu_inv = np.linalg.inv(k_uu)

        mu = k_unu @ k_uu_inv @ y_obs
        Sigma = C - k_nu @ k_uu_inv @ k_unu.T

        print(np.shape(mu))
        print(np.shape(Sigma))

        #plot final posterior
        indices = np.arange(len(u_train[0]))
        plt.clf()
        plt.plot(indices, mu, label="Posterior mean", color='blue')
        plt.fill_between(indices.ravel(), (mu - 1*std).ravel(), (mu + 1*std).ravel(), alpha=0.2, color='blue', label='±2 std')
        plt.plot(indices, cost(t, u_train), 'r--', label="True function")
        plt.scatter(u_obs, y_obs, color='black', zorder=5, label="Observations")
        plt.title("GP-UCB")
        plt.legend()
        # plt.grid(True)
        plt.pause(0.4)
plt.savefig('updates.jpeg')

        
    
    





