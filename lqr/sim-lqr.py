import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.datasets
from sklearn.datasets import make_spd_matrix
import GPy
from scipy.linalg import solve_discrete_are


# LQR parameters
A = np.array([[1.0]])
B = np.array([[1.0]])
Q = np.array([[1.0]])
R = np.array([[0.1]])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# Dynamics
def dynamics(x, u):
    return A @ x + B @ u

# True cost function
def true_cost(x, u):
    return float(x.T @ Q @ x + u.T @ R @ u)

# Noisy cost observation
def observe_cost(x, u):
    noise = np.random.normal(0, 1.0)
    return true_cost(x, u) + noise

# Domain of control inputs
U = np.linspace(-5, 5, 500)[:, None]

# GP setup
kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=0.3)
beta = 0.5  # UCB parameter

# Storage
xu = np.empty((0,1))  # actions tried
yu = np.empty((0,1))  # cost observed

# Initial state
x0 = np.array([[1.0]])
x_opt = [1.0]
cost_opt = []
cum_cost = [0]
total = 0
plt.figure(figsize=(10,6))
N = 100
for t in range(N):
    if t == 0:
        # No data: sample random action
        ut = np.array([[np.random.uniform(-2, 2)]])
    else:
        # Fit GP
        model = GPy.models.GPRegression(xu, yu, kernel)
        model.Gaussian_noise.variance = 0.01
        model.Gaussian_noise.variance.fix()

        # Predict over domain
        mu, var = model.predict(U)
        std = np.sqrt(var)
        ucb = mu - np.sqrt(beta) * std  # minimize cost

        # Choose control
        idx = np.argmin(ucb)
        ut = U[idx]

  
    
    # Observe cost
    ct = observe_cost(x0, ut)
    total = ct + total
    cum_cost.append(total)
    # xnext = dynamics(x0, ut)
    # x0 = xnext   
    
    # Update data
    xu = np.vstack((xu, ut.reshape(1,1)))
    yu = np.vstack((yu, np.array([[ct]]))) 
    
    u_opt = -K @ np.array([[x_opt[-1]]])
    cost = x_opt[-1]**2 * Q[0,0] + u_opt[0,0]**2 * R[0,0]
    cost_opt.append(cost)
    x_next = A @ np.array([[x_opt[-1]]]) + B @ u_opt
    x_opt.append(float(x_next[0]))


    # Plot GP
    if t > 0:
        plt.clf()
        mu, var = model.predict(U)
        std = np.sqrt(var)
       
        plt.plot(U, mu, label='Mean')
        plt.fill_between(U.ravel(),
                         (mu - 1.96 * std).ravel(),
                         (mu + 1.96 * std).ravel(),
                         color='blue', alpha=0.2, label='95% CI')
        plt.scatter(xu, yu, color='red', label='Data')
        plt.title(f"GP-UCB: Iteration {t+1}")
        plt.xlabel('Control input u')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.pause(0.4)

    plt.savefig('gp_ucb_lqr.png')

plt.figure(figsize = (10,6))
T = np.arange(0, len(yu))
T1 = np.arange(0, len(cum_cost))
plt.plot(T, yu, color = 'red')
# plt.plot(T, cost_opt, color = 'blue')
plt.plot(T1, cum_cost, color = 'green')
plt.pause(0.4)
plt.savefig('cost.jpeg')


    

    
        






