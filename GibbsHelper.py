import numpy as np
import Priors
from Trends import LowFrequencyTrends

# Parameter objects
class Lambda_Parameters:
    def __init__(self, l_c_i, l_g_j):
        self.lambda_c_i = l_c_i
        self.lambda_g_j = l_g_j

class Kappa_Parameters:
    def __init__(self, k_c_i, k_g_j, k_h_k):
        self.kappa_c_i = k_c_i
        self.kappa_g_j = k_g_j
        self.kappa_h_k = k_h_k

class P_Parameters:
    def __init__(self, c_lambda, g_lambda, c_theta, g_theta, h_theta, c_kappa, g_kappa, h_k):
        self.p_c_lambda = c_lambda
        self.p_g_lambda = g_lambda
        self.p_c_theta = c_theta
        self.p_g_theta = g_theta
        self.p_h_theta = h_theta
        self.p_c_kappa = c_kappa
        self.p_g_kappa = g_kappa
        self.p_h_k = h_k
        

# Methods for initialization of variables
def initialize_X(data, q):
    """
    Produces the low-frequency trends of all time series
    in the data
    data: The list consisting of all time series
    q: Cut-off
    """
    X_i = []
    for y in data:
        X_i.append(LowFrequencyTrends.find_baseline_trend(y, q))
    return np.array(X_i)

def initialize_F(X_i, w, q_hat):
    """
    Initialize F as w*X_i-Y0
    X_i: The low-frequency trends of all time series
    w: The population weights for each country.
    q_hat: Cut-off
    """
    Delta = np.identity(q_hat+1)
    Y0 = Priors.multivariate_normal_prior(np.zeros(Delta.shape[0]), Delta)
    return np.matmul(w, X_i) - Y0

def initialize_S_m(T, R_hat):
    """
    Draw S_m from a multivariate normal distribution
    T: Length of the time series
    R_hat: low-frequency regressors
    """
    sigma_m = Priors.symmetric_triangular_prior(0.001, 0.02, 25)
    rho_m = Priors.prior_from_half_life(50,150,25)
    Sigma_m_hat = cov_matrix(T, rho_m)
    Sigma_m = np.linalg.inv(R_hat.T@R_hat)@R_hat.T@Sigma_m_hat@R_hat@np.linalg.inv(R_hat.T@R_hat)
    return Priors.multivariate_normal_prior(np.zeros(Sigma_m.shape[0]), sigma_m**2*Sigma_m)

def initialize_lambdas(n, m):
    """
    Initialize group factor parameters
    n: number of countries
    m: number of groups
    """
    lambda_c_i = Priors.common_priors(0, 0.95, n, 25)
    lambda_g_j = Priors.common_priors(0, 0.95, m, 25)
    return Lambda_Parameters(lambda_c_i, lambda_g_j)

def initialize_kappas(n, m, l):
    """
    Initialize the kappa parameters
    n: number of countries
    m: number of groups
    l: number of groups-of-groups
    """
    kappa_c_i = Priors.common_priors(1/3, 1, n, 25)
    kappa_g_j = Priors.common_priors(1/3, 1, m, 25)
    kappa_h_k = Priors.common_priors(1/3, 1, l, 15)
    return Kappa_Parameters(kappa_c_i, kappa_g_j, kappa_h_k)

def initialize_p():
    """
    Initialize p parameters
    """
    p_c_lambda = np.random.dirichlet(np.ones(25)*20/25)
    p_g_lambda = np.random.dirichlet(np.ones(25)*20/25)
    p_c_theta = np.random.dirichlet(np.ones(100)*20/100)
    p_g_theta = np.random.dirichlet(np.ones(100)*20/100)
    p_h_theta = np.random.dirichlet(np.ones(100)*20/100)
    p_c_kappa = np.random.dirichlet(np.ones(25)*20/25)
    p_g_kappa = np.random.dirichlet(np.ones(25)*20/25)
    p_h_k = np.random.dirichlet(np.ones(25)*20/25)
    return P_Parameters(p_c_lambda, p_g_lambda, p_c_theta, p_g_theta, p_h_theta,
                        p_c_kappa, p_g_kappa, p_h_k)

def inverse_squared(median):
    return Priors.inverse_squared_prior(np.linspace(0, 1000, 100000), median)

def cov_matrix(T, rho_m):
    Sigma_m_hat = np.zeros((T,T))
    innov_variance = 1/(1-rho_m**2)
    for i in range(T):
        for j in range(T):
            Sigma_m_hat[i][j] = (innov_variance/(1-rho_m**2))*(rho_m**abs(i-j))
    return Sigma_m_hat    

def initialize_U_trend(T, omega_squared, kappa, R_hat, lam=None):
    rho_1, rho_2, zeta = Priors.persistence_u(100)
    mat1 = cov_matrix(T, rho_1)
    mat2 = cov_matrix(T, rho_2)
    weight_2 = (1-zeta**2)**(1/2)
    Sigma_U_hat = zeta*mat1+weight_2*mat2
    Sigma_U = np.linalg.inv(R_hat.T@R_hat)@R_hat.T@Sigma_U_hat@R_hat@np.linalg.inv(R_hat.T@R_hat)
    factor = omega_squared*kappa
    if lam is not None:
        factor = factor*(1-lam**2)

    return Priors.multivariate_normal_prior(np.zeros(Sigma_U.shape[0]), factor*Sigma_U)