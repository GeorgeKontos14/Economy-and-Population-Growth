import numpy as np
import scipy as sci

def flat_prior(theta, min_val: float, max_val: float):
    c: float = 1.0 / (max_val - min_val)
    if isinstance(theta, float):
        if theta >= min_val and theta <= max_val:
            return c
        return 0
    return np.where((theta >= min_val) & (theta <= max_val), c, 0.0)

def inverse_chi1_squared(x, median):
    gamma = sci.special.gamma(1/2)
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

def prior_from_half_life(min_val, max_val, n):
    grid = np.linspace(min_val, max_val, n)
    result = np.zeros(n)
    for i, h in enumerate(grid):
        result[i] = (1/2)**(1/h)
    return result

def symmetric_triangular_prior(min_val, max_val, n):
    c = (min_val+max_val)/2
    points = np.linspace(min_val, max_val, n)
    res_support = ((max_val-c)-np.abs(c - points))/(max_val-c)**2

    res = np.zeros(len(points))
    res[(points >= min_val) & (points <= max_val)] = res_support
    return res

def common_priors(theta_min, theta_max, n, alpha, m):
    thetas = np.linspace(theta_min, theta_max, n)
    params = np.ones(n)*alpha/n
    prior = np.random.dirichlet(params)
    return np.random.choice(thetas, size=m, p=prior)

def persistence_u(n):
    x = np.linspace(0,1,n)
    xx, yy, zz = np.meshgrid(x,x,x)
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    U1, U2, zeta = grid[np.random.choice(len(grid))]

    h1 = 25+775*U1**2
    h2 = 25+775*U2**2

    rho1 = 0.5**(1/max(h1, 0.0001)) # Avoid division by zero
    rho2 = 0.5**(1/max(h2, 0.0001))
    
    return rho1, rho2, zeta

def group_factors(n, m):
    """
    n: no of entities to be grouped
    m: no of groups
    """
    groups = np.arange(m)
    return np.random.choice(groups, size=n)