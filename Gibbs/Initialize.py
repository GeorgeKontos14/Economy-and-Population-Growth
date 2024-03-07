import numpy as np
import Gibbs.Priors as Priors
from Trends import LowFrequencyTrends

# Methods for initialization of the Gibbs state
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
    Delta = np.identity(q_hat+1)*0.01**2
    Y0 = Priors.multivariate_normal_prior(np.zeros(Delta.shape[0]), Delta)
    mat = np.matmul(w, X_i) - Y0
    return mat[0]

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
    mat = Priors.multivariate_normal_prior(np.zeros(Sigma_m.shape[0]), sigma_m**2*Sigma_m)
    return mat[0], sigma_m, Sigma_m

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
    a = zeta/(zeta+weight_2)
    b = weight_2/(zeta+weight_2)
    Sigma_U_hat = a*mat1+b*mat2
    Sigma_U = np.linalg.inv(R_hat.T@R_hat)@R_hat.T@Sigma_U_hat@R_hat@np.linalg.inv(R_hat.T@R_hat)
    factor = omega_squared*kappa
    if lam is not None:
        factor = factor*(1-lam**2)

    mat = Priors.multivariate_normal_prior(np.zeros(Sigma_U.shape[0]), factor*Sigma_U)
    return mat[0], factor*Sigma_U

def init_Sigma_A(R_hat, T, q_hat, s_Da):
    V: np.ndarray = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            V[i][j] = min(i+1, j+1)

    help = np.linalg.inv(R_hat.T@R_hat)
    Sigma_A = help@R_hat.T@V@R_hat@help
    A = Priors.multivariate_normal_prior(np.zeros(Sigma_A.shape[0]), s_Da**2*Sigma_A)
    return A, Sigma_A