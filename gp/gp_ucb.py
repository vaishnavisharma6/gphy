import numpy as np
import matplotlib.pyplot as plt
import GPy


#GP-UCB Code

#1. Initialize with a prior over function space.
#2. For loop over time steps
#3. Sample initial
#4. Get y value
#5. Update prior to get posterior
#6. Sample using sampling rule. So 3 parts: initialize, sample for t =0, cost function, updation

class gp_ucb:
    def __init__(self, domain):
        self.X = domain
    
    def initialize(self, k):
        pr_mean = np.zeros((len(self.X), 1))
        pr_cov = k.K(self.X, self.X)
        return(pr_mean, pr_cov)
        
    
    def sample(self, mean, cov):
        beta = 0.5
        var = np.diag(cov).reshape(-1, 1)
        print(np.shape(var))
        score = mean + (beta**0.5 * var)
        # print(np.shape(score))
        i = np.argmax(score)
        print(i)
        x = self.X[i]
        return(x)

    def cost(self, x):
        c = 3*x**2 + 2 
        w = np.random.normal()
        y = c + w
        return(y)
    

    def updation(self, Y_obs, X_obs):
            mean =  k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ Y_obs
            cov = k.K(self.X, self.X) - k.K(self.X, X_obs) @ np.linalg.inv(k.K(X_obs, X_obs) + 1*np.eye(X_obs.shape[0])) @ k.K(X_obs, self.X)
            
            return(mean, cov)

X = np.linspace(0, 100, 1000)[:, None]
k = GPy.kern.RBF(1, variance = 1., lengthscale = 50)
algo = gp_ucb(X)
X_obs = []
Y_obs = []
reg = []
mean, cov = algo.initialize(k)
print(mean)
yb = algo.cost(100)
plt.figure(figsize = (10, 6))
sr = 0
for t in range(0,30):
    x = algo.sample(mean, cov)
    X_obs.append(x)
    y = algo.cost(x)
    Y_obs.append(y)
    regret = yb - y
    sr = sr + regret
    reg.append(sr)
    mean, cov = algo.updation(np.array(Y_obs), np.array(X_obs))
    plt.fill_between(X[:,0], mean[:, 0] - 1*np.sqrt(np.diag(cov)), mean[:, 0] + 1*np.sqrt(np.diag(cov)), alpha=0.2, color='blue')
    plt.plot(X, mean, color = 'blue')
    plt.scatter(X_obs, Y_obs, color ='black')
    plt.xlabel('X')
    plt.ylabel('Mean function')
    plt.legend()
    plt.pause(0.4)
plt.savefig('gp_ucb.jpeg')
    
plt.figure(figsize=(10, 6))
N = np.arange(0, len(reg))
plt.plot(N, reg, label = 'regret')
plt.legend()
plt.savefig('regret.jpeg')
