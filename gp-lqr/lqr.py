#code for gp-ucb to model synthetic data

import numpy as np
import matplotlib.pyplot as plt
import GPy


#Code for GP regression

#RBF kernel-------
k = GPy.kern.RBF(1)
print(k)

X = np.linspace(-4, 4, 100)[:, None]
print(np.shape(X))

plt.figure(figsize = (10,6))
K = k.K(X, np.array([[0.]]))
plt.plot(X, K)
plt.xlabel('x')
plt.ylabel('covariance value between x and 0')
plt.legend()
plt.savefig('zero-x.png')
plt.close()

K = k.K(X, X)
plt.pcolor(X.T, X, K)

plt.gca().invert_yaxis()
plt.gca().axis("image")
plt.xlabel("x")
plt.ylabel("x")
plt.colorbar()
plt.title('rbf')
plt.savefig('x-x.png')
plt.close()


#different lengthscales
ls =  [0.25, 0.5, 1.0, 2.0, 4.0]
for l in ls:

    k.lengthscale = l
  
    C = k.K(X, np.array([[0.]]))
    plt.plot(X, C, label = 'ls = {}'.format(l))
    plt.xlabel('x')
    plt.ylabel('covariance of x with zero')
    plt.title('different lengthscales for RBF kernel')
    plt.legend()
    plt.savefig('ls.png')
   


for l in ls:
    plt.figure(figsize = (10,6))
    k.lengthscale = l 

    C2 = k.K(X, X)
    plt.pcolor(X.T, X, C2)
    plt.gca().invert_yaxis()
    plt.gca().axis("image")
    plt.xlabel("x")
    plt.ylabel("x")
    plt.colorbar()
    plt.title('ls = {}'.format(l))
    plt.savefig('xls{}.png'.format(l))
    plt.close()


# Code to show variation in variance parameter

vs =  [0.25, 0.5, 1.0, 2.0, 4.0]
plt.figure(figsize = (10,6))
for v in vs:

    k.variance = v
  
    C3 = k.K(X, np.array([[0.]]))
    plt.plot(X, C3, label = 'vs = {}'.format(v))
    plt.xlabel('x')
    plt.ylabel('covariance of x with zero')
    plt.title('different lengthscales for RBF kernel')
    plt.legend()
    plt.savefig('vs.png')
   


for v in vs:
    plt.figure(figsize = (10,6))
    k.variance = v

    C4 = k.K(X, X)
    plt.pcolor(X.T, X, C4)
    plt.gca().invert_yaxis()
    plt.gca().axis("image")
    plt.xlabel("x")
    plt.ylabel("x")
    plt.colorbar()
    plt.title('vs = {}'.format(v))
    plt.savefig('vxs{}.png'.format(v))
    plt.close()



#other kernels
X = np.linspace(-4.,4.,250)[:, None]

# RBF kernel
k_R = GPy.kern.RBF(1)
C_R = k_R.K(X, np.array([[0.]]))

# Matern 5/2
k_M = GPy.kern.Matern52(1)
C_M = k_M.K(X, np.array([[0.]]))

# Cosine
k_C = GPy.kern.Cosine(1)
C_C = k_C.K(X, np.array([[0.]]))

plt.figure(figsize=(18,7))
plt.plot(X, C_R, X, C_M, X, C_C);
plt.xlabel("x"), plt.ylabel("$\kappa$")
plt.legend(labels=["Gaussian RBF", "Matérn 5/2", "Cosine"])
plt.savefig('kernels.png')

# sampling from Gaussian process (similar to sampling from a multi-variate normal distribution)
# various kernels

ks = [GPy.kern.RBF(1, lengthscale = 1),
      GPy.kern.RBF(1, lengthscale = 0.5),
      GPy.kern.RBF(1, lengthscale = 0.25, variance = 2.),
      GPy.kern.RBF(1, lengthscale = 2.5)]

kname= ["RBF ls=1", "RBF ls = 0.5", "RBF ls = 0.25", "RBF ls = 2.5"]      

X = np.linspace(-5, 5, 250)[:, None]

for i, k in enumerate(ks):
    mu = np.zeros((250))
    C = k.K(X, X)

    Z = np.random.multivariate_normal(mu, C, 3)
    plt.figure(figsize = (10, 6))

    plt.subplot(121)
    for j in range(3):
        plt.plot(X[:], Z[j,:])
        plt.title(kname[i])

    plt.subplot(122)
    plt.pcolor(X.T, X, C)
    plt.gca().invert_yaxis(), plt.gca().axis("image")
    plt.colorbar()  
    plt.savefig('GP_sample_kernel_matrix{}.jpg'.format(kname[i]))  


#GP regression 

f = lambda x: -np.cos(2*np.pi*x) + 0.5*np.sin(6*np.pi*x)
X = np.linspace(0.05, 2.0, 10)[:, None]

Y = f(X) + np.random.normal(0., 0.1, (10, 1))
plt.figure(figsize = (10, 6))
plt.plot(X, Y, "kx")
plt.xlabel("x")
plt.ylabel("f")

plt.legend(labels = ["sample points"])
plt.savefig('data-points.jpg')
 
k = GPy.kern.RBF(1, variance = 1., lengthscale = 0.1, name = "RBF") 

#testing points

Xnew = np.linspace(0.3, 1.5, 100)[:, None]

kxx = k.K(X, X) + 1*np.eye(10)
ksx = k.K(Xnew, X)
kss = k.K(Xnew, Xnew)

mean = ksx @ np.linalg.inv(kxx) @Y
Cov = kss - ksx @ np.linalg.inv(kxx) @ksx.T


plt.figure(figsize = (10, 6))
plt.fill_between(Xnew[:, 0], mean[:, 0]-1.96*np.sqrt(np.diag(Cov)), mean[:, 0]+1.96*np.sqrt(np.diag(Cov)), alpha = 0.5) 
plt.plot(Xnew, mean)
plt.legend(labels = ["GP fit"])

plt.xlabel("x")
plt.ylabel("y")

X_, Y_ = (X, Y)
plt.plot(X_, Y_, "kx", mew = 2)


plt.savefig('GP fit.jpg')


#outline: backend: system equation, have a prior on input equation, sample input, apply, observe state, record cost, get posterior
# gp-ucb: we have some prior currently. 

# D = np.linspace(-100, 100, 10000)[:, None]
# N = 50

# k = GPy.kern.RBF(1, variance = 1., lengthscale = 0.1, name = 'RBF')

# def imean(X):
#     return (np.zeros(len(X)))

# def updated_mean(ksx, kxx, y):
#     return (ksx @ np.linalg.inv(kxx) @ y)

# def updated_cov(kss, ksx, kxx):
#     return(kss - ksx @ np.linalg.inv(kxx) @ksx.T)

# def evaluate(x, D, k):
#     kxx = k.K(x, x) + np.eye(len(x))
#     ksx = k.K(D, x)
#     kss = k.K(D, D)

#     return(kxx, ksx, kss)

# def func(x):
#     y = -np.cos(2*np.pi*x) + 0.5*np.sin(6*np.pi*x) 
#     ep = np.random.normal(0, 0.1, 1)
#     return (y + ep)


# beta = 0.5
# print(k.K(D, D))
# print(imean(D))
# emp = np.empty((0,1))
# empy = np.empty((0,1))
# for t in range(3):
#     if t == 0:
#         var = np.diag(k.K(D, D)) 
#         score = imean(D) + (beta**0.5 * var)
#         xt = np.argmax(score)
#         yt = func(D[xt])
#         xs = np.array([[D[xt]]])
#         ys = np.array(yt)
#         x = np.vstack((emp, xs))
#         y = np.vstack((empy, ys))
#         kxx, ksx, kss = evaluate(x, D, k)
     
#         mean = updated_mean(ksx, kxx, y)
        
#         Cov = updated_cov(kss, ksx, kxx)
        
        
#     else:
#         var = np.diag(Cov)    
#         score =  + (beta**0.5 * var)

    # xt = np.argmax(score)
    # yt = func(D[xt])
    # xs = np.array(xt).reshape(-1, 1)
    # ys = np.array(yt).reshape(-1, 1)
    # x = np.vstack((emp, xs))
    # y = np.vstack((empy, ys))
    # print(np.shape(x))
    # print(np.shape(y))
    # kxx, ksx, kss = evaluate(x, x, k)
    # mean = updated_mean(ksx, kxx, y)
    # Cov = updated_cov(kss, ksx, kxx)
    

    

    
        
    #  print(xt)

# Define kernel
k =  GPy.kern.RBF(1, variance = 1., lengthscale = 0.1, name = 'RBF')

# Domain
D = np.linspace(-5, 5, 500).reshape(-1, 1)
N = D.shape[0]

# GP prior
mu = np.zeros((N, 1))
C = k.K(D, D)
sigma_noise = 1e-2
Sigma = C.copy()

# Storage
x_obs = np.empty((0, 1))
y_obs = np.empty((0, 1))

# GP-UCB parameters
T = 12
beta_t = lambda t: 2 * np.log(N * np.pi**2 * t**2 / (6 * 0.1))  # δ=0.1 confidence level
means = {}
plt.figure(figsize=(10, 5))
for t in range(0, T + 1):
    std = np.sqrt(np.diag(Sigma)).reshape(-1, 1)
    ucb = mu + np.sqrt(beta_t(t)) * std

    # Select next point
    idx = np.argmax(ucb)
    x_t = D[idx].reshape(1, 1)

    # Sample from function with noise (ground truth: sin)
    y_t = np.sin(x_t) + np.random.normal(0, sigma_noise, size=(1, 1))

    # Update data
    x_obs = np.vstack((x_obs, x_t))
    y_obs = np.vstack((y_obs, y_t))
    # GP Bayesian update
    K_xx = k.K(x_obs, x_obs) + sigma_noise**2 * np.eye(len(x_obs))
    print(np.shape(K_xx))
    K_Dx = k.K(D, x_obs)
    K_xx_inv = np.linalg.inv(K_xx)

    mu = K_Dx @ K_xx_inv @ y_obs
    Sigma = C - K_Dx @ K_xx_inv @ K_Dx.T

    print(np.shape(mu))
    print(np.shape(Sigma))
# Plot final posterior
    plt.clf()
    plt.plot(D, mu, label="Posterior mean", color='blue')
    plt.fill_between(D.ravel(), (mu - 1*std).ravel(), (mu + 1*std).ravel(), alpha=0.2, color='blue', label='±2 std')
    plt.plot(D, np.sin(D), 'r--', label="True function")
    plt.scatter(x_obs, y_obs, color='black', zorder=5, label="Observations")
    plt.title("GP-UCB")
    plt.legend()
    # plt.grid(True)
    plt.pause(0.4)
plt.savefig('updates.jpeg')


