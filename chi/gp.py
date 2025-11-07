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


t = 5
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
    
    
num_samples = 100

def cost(t, u):
    w = np.random.rand(1)
    c = []
    for j in range(u.shape[0]):                   
        cu = 0
        for i in range(0, m * t, m):                
            cu = cu + (u[j, i:i+m] @ Q @ u[j, i:i+m].T)
        cu = cu + w
        c.append(cu)
    return np.array(c).reshape(-1, 1)             

u_train = sample_from_ellipsoid(C_T, b, r_cheb, num_samples)
print(u_train)
y_train = cost(t, u_train)
    
# GP prior
mu = np.zeros((num_samples, 1))
k = GPy.kern.RBF(input_dim=m*t, variance=1., lengthscale=0.1, name='RBF')  
C = k.K(u_train, u_train)
Sigma = C.copy()
u_obs = np.empty((0, m*t))
y_obs = np.empty((0, 1))

n = 20                             
beta_t = lambda t: 2 * np.log(num_samples * np.pi**2 * (t+1)**2 / (6*0.1))  

means = {}
indices1 = []
# training
for l in range(0, n+1):
    std = np.sqrt(np.diag(Sigma)).reshape(-1,1)
    ucb = -1*mu + np.sqrt(beta_t(l)) * std

    idx = np.argmax(ucb)
    u_t = u_train[idx, :].reshape(1, -1)            
    y_t = y_train[idx].reshape(1, 1)                
    
    # update data
    u_obs = np.vstack((u_obs, u_t))
    y_obs = np.vstack((y_obs, y_t))

    k_uu = k.K(u_train, u_train)
    print(np.shape(k_uu))
    k_unu = k.K(u_train, u_obs)    # (N, n_obs)
    print(np.shape(k_unu))
    k_nu = k.K(u_obs, u_obs)       # (n_obs, n_obs)
    print(np.shape(k_nu))

    # invert the observed-observed kernel (k_nu), not k_uu
    k_nu_inv = np.linalg.inv(k_nu + 1e-6 * np.eye(k_nu.shape[0]))
    print(np.shape(k_nu_inv))

# posterior at the test points u_train given observations (u_obs, y_obs)
    mu = k_unu @ k_nu_inv @ y_obs          # shape (N, 1)
    Sigma = k_uu - k_unu @ k_nu_inv @ k_unu.T

    # plot final posterior

    indices = np.arange(len(u_train))  
    indices1.append(idx)

    # print('u norm:', u_obs[indices1[-1]])      
    plt.clf()
    plt.plot(indices, mu, label="Posterior mean", color='blue')
    plt.fill_between(indices.ravel(),
                     (mu - std).ravel(),
                     (mu + std).ravel(),
                     alpha=0.2, color='blue', label='±1 std')
    plt.plot(indices, y_train, 'r--', label="True function")
    plt.scatter(indices1, y_obs, color='red', marker = 'x', label="Observations") 
    plt.title("GP-UCB, chosen u:{}".format(idx))
    plt.legend()
    plt.pause(0.4)
plt.savefig('updates.jpeg')

plt.figure(figsize = (10, 6))
x = np.arange(len(indices1))
plt.plot(x, indices1, marker= 'x', color = 'blue', label ='index of chosen input')
plt.xlabel('iteration #')
plt.ylabel('index number')
plt.grid(True)
plt.legend()
plt.savefig('chosen.jpeg')





