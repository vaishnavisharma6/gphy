import numpy as np
import matplotlib.pyplot as plt
import math
import os


# script to generate data for different kinds of systems



def type_1(n, m, T, seed=None):
    #stable system
    if seed is not None:
        np.random.seed(seed)
    
    # Start with a random matrix
    A = np.random.randn(n, n)
    
    # Scale it to make it stable (shrink eigenvalues)
    eigvals, eigvecs = np.linalg.eig(A)
    max_abs_eig = np.max(np.abs(eigvals))
    A = A / (1.2 * max_abs_eig)   # ensures spectral radius < 1
    
    # Random input matrix
    B = np.random.randn(n, m)
    N_T = B
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i) @ B))
    
    return A, B

def type_2(n, m, T, seed=None):
    #unstable system
    if seed is not None:
        np.random.seed(seed)
    
    # Start with a random matrix
    A = np.random.randn(n, n)
    
    # Scale to make eigenvalues > 1
    eigvals, eigvecs = np.linalg.eig(A)
    max_abs_eig = np.max(np.abs(eigvals))
    A = A / (0.5 * max_abs_eig)  # ensures at least one eigenvalue > 1
    
    # Random input matrix
    B = np.random.randn(n, m)
    N_T = B
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i) @ B))
    
    return A, B

def type_3(n,m, T, seed=None, stable=True):
    # diagonal system
    if seed is not None:
        np.random.seed(seed)
    
    if stable:
        # Diagonal entries between -0.9 and 0.9
        diag_entries = np.random.uniform(-0.9, 0.9, n)
    else:
        # Some entries greater than 1 in magnitude
        diag_entries = np.random.uniform(1.1, 2.0, n)
    
    A = np.diag(diag_entries)
    B = np.random.randn(n, m)
    N_T = B
    for i in range(1, T):
        N_T = np.hstack((N_T, np.linalg.matrix_power(A, i) @ B))
    
    return A, B


def get_system(name, n, m, T, seed=None):
    systems = {
        'stable': type_1,
        'unstable': type_2,
        'diagonal': type_3

    }
    return systems[name](n=n, m=m, T=T, seed=seed)


