import numpy as np
import math
import torch

def flat_prior(min_val: float, max_val: float):
    return np.random.uniform(min_val, max_val)

def inverse_chi1_squared(x, median):
    gamma = math.gamma(1/2)
    if isinstance(x, float):
        if x <= 0:
            return 0
        return 2**(-1/2)/gamma * x ** (-3/2) * np.exp(-1/(2*x))
    
    pos = x[x > 0]
    res_pos = np.zeros(len(pos))
    res_pos = 2**(-1/2)/gamma * pos ** (-3/2) * np.exp(-1/(2*pos))

    scaling_factor = median/np.median(res_pos)
    shifted = scaling_factor * res_pos

    res = np.zeros(len(x))
    res[x > 0] = shifted
    return res

def inverse_squared_prior(x, median):
    pr = inverse_chi1_squared(x, median)
    pr = pr/np.sum(pr)
    return np.random.choice(x, p = pr)

def prior_from_half_life(min_val, max_val, n):
    grid = np.linspace(min_val, max_val, n)
    h = np.random.choice(grid)
    return (1/2)**(1/h)

def symmetric_triangular_prior(a, b, n):
    half_points = (n - 1) // 2
    peak_index = half_points
    
    x = np.linspace(a, b, n)
    y = np.zeros_like(x, dtype=float)
    
    for i in range(peak_index + 1):
        y[i] = i / peak_index
        y[n - 1 - i] = i / peak_index
    
    return np.random.choice(x, p=y / np.sum(y))

def common_priors(theta_min, theta_max, n, alpha, m):
    thetas = np.linspace(theta_min, theta_max, n)
    params = np.ones(n)*alpha/n
    prior = np.random.dirichlet(params)
    return np.random.choice(thetas, size=m, p=prior)

def persistence_u():
    x = torch.linspace(0, 1-0.00001, 5)
    y = torch.linspace(0, 1-0.00001, 5)
    z = torch.linspace(0, 1-0.00001, 4)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1) 

    U1, U2, zeta = grid[np.random.choice(len(grid))]

    h1 = 25+775*U1**2
    h2 = 25+775*U2**2

    rho1 = 0.5**(1/max(h1, 0.00001))
    rho2 = 0.5**(1/max(h2, 0.00001))
    
    return rho1.item(), rho2.item(), zeta.item()

def group_factors(n, m):
    """
    n: no of entities to be grouped
    m: no of groups
    """
    groups = np.arange(m)
    return np.random.choice(groups, size=n)

def multivariate_normal_prior(mean, cov, n=1):
    return np.random.multivariate_normal(mean=mean, cov=cov, size=n)