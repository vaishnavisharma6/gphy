import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.datasets
from sklearn.datasets import make_spd_matrix
import GPy



#1d state and input
class linear_system():
    def state(xi, ui, A, B):
        xj = A*xi + B*ui
        e = np.random.normal(0., 0.1, 1)
        return(xj + e)
    
    def cost(ci, xi, ui, Q, R):
        c = np.transpose(xi) * Q * xi + np.transpose(ui)*R*ui
        e = np.random.normal(0, 1, 1)
        return(c + ci + e)
    
    def lqr_opt(K, xi):
        return (-K * xi)
    
    def ricati(A, B, Q, R):
        P = 0
        K = 0
        return (P, K)


    def gp_prior():
        k = GPy.kern.RBF(1, variance = 1., lengthscale = 0.1, name = 'RBF')
        return(k)

    def choose_input(self):
        #GP-UCB
        beta = 0.5
        mean, Cov = self.update_prior(c, u, self.gp_prior())
        u = np.argmax(mean(u) + beta**0.5 * Cov(u)**0.5)
        return u
    
    def update_prior(self):
        k = self.gp_prior()
        uo = u
        unew = u.append(ui)
        kuu = k.K(uo, uo) + 1*np.eye(1)
        knu = k.K(unew, uo)
        knn = k.K(unew, unew)
        Y = self.cost(ci, xi, ui, Q, R)
        mean = knu @ np.linalg.inv(kuu) @ Y
        Cov = knn @ knu @ np.linalg.inv(kuu) @ knu.T
        return (mean, Cov)



    

xi = 0
ui = random.random() 
A = np.random.rand(1,1)
B = np.random.rand(1,1)
Q = make_spd_matrix(n_dim = 1)
R = make_spd_matrix(n_dim = 1)
ci = 0
ls = linear_system


states = [xi]
costs = [0]
N = 300
plt.figure(figsize = (10, 6))
for i in range(N):
   ui = random.random()
   ns = ls.state(xi, ui, A, B)
   uj = 0
   c = ls.cost(ci, xi, ui, Q, R)
   ci = c
   states.append(float(ns))
   costs.append(float(c))
   xi = ns
   t = np.arange(0, len(states))

   plt.plot(t, states, color = 'red')
   plt.plot(t, costs, color = 'blue')
   plt.xlabel('step')
   plt.ylabel('state')
   plt.pause(0.4)

plt.savefig('linear.jpg')










