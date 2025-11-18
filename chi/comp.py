import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------------------------------------------
# Imhof’s Method for quadratic form sum λ_i * Y_i^2
# ------------------------------------------------------
def imhof_cdf(x, lambdas, deltas):
    def integrand(t):
        rho = 1
        theta2 = mp.mpf('0')
        theta3 = mp.mpf('0')

        for i in range(len(lambdas)):
            lam = mp.mpf(str(lambdas[i]))
            delta = mp.mpf(str(deltas[i]))

            rho *= mp.sqrt(1 + 4 * lam**2 * t**2)
            theta2 += mp.atan(2 * lam * t)
            theta3 += (delta**2) * lam * t / (1 + 4 * lam**2 * t**2)

        theta = t * x - 0.5 * theta2 + theta3
        if t == 0:
            return mp.mpf('0')
        return mp.sin(theta) / (t * rho)

    return float(0.5 + (1/mp.pi) * mp.quad(integrand, [0, mp.inf]))


# ------------------------------------------------------
# CDF of ||Z||^2 for Z ~ N(mu, Sigma)
# ------------------------------------------------------
def cdf_normsq_imhof(x, mu, Sigma):
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-14)
    mu_t = eigvecs.T @ mu

    lambdas = eigvals.copy()
    deltas = mu_t / np.sqrt(eigvals)

    return imhof_cdf(x, lambdas, deltas)


# ------------------------------------------------------
# Example: choose mu and Sigma
# ------------------------------------------------------
Sigma = np.array([[1.0, 0.3, 0.2],
                  [0.3, 0.8, 0.1],
                  [0.2, 0.1, 0.5]])
mu = np.array([0.5, -0.2, 0.3])

# Mean and variance from your formulas
mean_Q = np.trace(Sigma) + mu @ mu
var_Q = 2 * np.trace(Sigma @ Sigma) + 4 * mu @ (Sigma @ mu)
std_Q = np.sqrt(var_Q)

print("Mean:", mean_Q)
print("Variance:", var_Q)

# Domain for x
xs = np.linspace(0.01, mean_Q + 6*std_Q, 120)

# Exact CDF (Imhof)
cdf_exact = [cdf_normsq_imhof(x, mu, Sigma) for x in xs]

# Normal Approx
cdf_normal = norm.cdf(xs, loc=mean_Q, scale=std_Q)

# Monte–Carlo (for validation)
rng = np.random.default_rng(0)
Nmc = 200000
samples = rng.multivariate_normal(mu, Sigma, size=Nmc)
q_samples = np.sum(samples**2, axis=1)
cdf_emp = [np.mean(q_samples <= xx) for xx in xs]

# ------------------------------------------------------
# Plot
# ------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.plot(xs, cdf_exact, label="Imhof exact CDF")
plt.plot(xs, cdf_normal, '--', label="Normal approx")
plt.plot(xs, cdf_emp, 'o', markersize=3, label="Empirical (MC)")
plt.xlabel("x")
plt.ylabel("CDF")
plt.title(r"CDF of $\|Z\|^2$  where  $Z \sim \mathcal{N}(\mu,\Sigma)$")
plt.legend()
plt.grid()
plt.show()

