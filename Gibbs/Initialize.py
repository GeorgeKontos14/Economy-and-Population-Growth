import numpy as np
import Gibbs.Priors as Priors
from Trends import LowFrequencyTrends
import torch

# Methods for initialization of the Gibbs state
def initialize_X(data, q):
    """
    Produces the low-frequency trends of all time series
    in the data
    data: The list consisting of all time series
    q: Cut-off
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    X_i = []
    for y in data:
        X_i.append(LowFrequencyTrends.find_baseline_trend(y, q))
    return torch.tensor(np.array(X_i), device = device)

def initialize_F(X_i, w, q_hat):
    """
    Initialize F as w*X_i-Y0
    X_i: The low-frequency trends of all time series
    w: The population weights for each country.
    q_hat: Cut-off
    """
    print(X_i.device)
    print(w.device)
    # Delta = np.identity(q_hat+1)*0.01**2
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    Delta = torch.eye(q_hat+1, device=device)*0.01**2
    print(Delta.device)
    dist = torch.distributions.MultivariateNormal(torch.zeros(q_hat+1, device=device), Delta)
    Y0 = dist.sample()
    Y0 = Y0.cuda()
    print(Y0.device)
        

    mat = torch.matmul(w, X_i) - Y0
    return mat

def initialize_S_m(T, R_hat, device):
    """
    Draw S_m from a multivariate normal distribution
    T: Length of the time series
    R_hat: low-frequency regressors
    """
    sigma_m = float(Priors.symmetric_triangular_prior(0.001, 0.02, 25))
    rho_m = Priors.prior_from_half_life(50,150,25)
    Sigma_m_hat = cov_matrix(T, rho_m)
    Sigma_m = np.linalg.inv(R_hat.T@R_hat)@R_hat.T@Sigma_m_hat@R_hat@np.linalg.inv(R_hat.T@R_hat)
    Sigma_m = torch.tensor(Sigma_m, dtype=float, device = device)
    # mat = Priors.multivariate_normal_prior(np.zeros(Sigma_m.shape[0]), sigma_m**2*Sigma_m)
    dist = torch.distributions.MultivariateNormal(torch.zeros(Sigma_m.shape[0], dtype=float, device=device), sigma_m**2*Sigma_m)
    mat = dist.sample()
    return mat, sigma_m, Sigma_m

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
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
    return torch.tensor(mat[0], device=device), torch.tensor(factor*Sigma_U, device=device)

def init_Sigma_A(R_hat, T, q_hat, s_Da):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    

    V: np.ndarray = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            V[i][j] = min(i+1, j+1)

    help = np.linalg.inv(R_hat.T@R_hat)
    Sigma_A = help@R_hat.T@V@R_hat@help
    A = Priors.multivariate_normal_prior(np.zeros(Sigma_A.shape[0]), s_Da**2*Sigma_A)
    return torch.tensor(A[0], device=device), torch.tensor(Sigma_A, device=device)