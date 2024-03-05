import numpy as np
import Priors
from InitializeGibbs import initialize_U_trend

class Lambda_Parameters:
    def __init__(self, n, m):
        self.lambda_c_i = Priors.common_priors(0, 0.95, 25, 20, n)
        self.lambda_g_j = Priors.common_priors(0, 0.95, 25, 20, m)

class Kappa_Parameters:
    def __init__(self, n, m, l):
        self.kappa_c_i = Priors.common_priors(1/3, 1, 25, 20, n)
        self.kappa_g_j = Priors.common_priors(1/3, 1, 25, 20, m)
        self.kappa_h_k = Priors.common_priors(1/3, 1, 25, 20, l)

class P_Parameters:
    def __init__(self):
        self.p_c_lambda = np.random.dirichlet(np.ones(25)*20/25)
        self.p_g_lambda = np.random.dirichlet(np.ones(25)*20/25)
        self.p_c_theta = np.random.dirichlet(np.ones(100)*20/100)
        self.p_g_theta = np.random.dirichlet(np.ones(100)*20/100)
        self.p_h_theta = np.random.dirichlet(np.ones(100)*20/100)
        self.p_c_kappa = np.random.dirichlet(np.ones(25)*20/25)
        self.p_g_kappa = np.random.dirichlet(np.ones(25)*20/25)
        self.p_h_k = np.random.dirichlet(np.ones(25)*20/25)

class U_Parameters:
    def __init__(self, omega_squared: float, lambdas: Lambda_Parameters, kappas: Kappa_Parameters, R_hat, n, m, l, T):
        self.U_c = []
        self.S_U_c = []
        for i in range(n):
            u_c, s_c = initialize_U_trend(T, omega_squared, kappas.kappa_c_i[i], R_hat, lam=lambdas.lambda_c_i[i])
            self.S_U_c.append(s_c)
            self.U_c.append(u_c)
            
        self.U_g = []
        self.S_U_g = []
        for j in range(m):
            u_g, s_g = initialize_U_trend(T, omega_squared, kappas.kappa_g_j[j], R_hat, lam=lambdas.lambda_g_j[j])
            self.U_g.append(u_g)
            self.S_U_g.append(s_g)
        
        self.H = []
        self.S_h = []
        for k in range(l):
            h, s_h = initialize_U_trend(T, omega_squared, kappas.kappa_h_k[k], R_hat)
            self.H.append(h)
            self.S_h.append(s_h)