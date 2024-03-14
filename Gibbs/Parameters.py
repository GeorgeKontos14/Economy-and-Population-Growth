import numpy as np
import Gibbs.Priors as Priors
from Gibbs.Initialize import initialize_U_trend
import torch

class Lambda_Parameters:
    def __init__(self, n, m, device):
        self.lambda_c_i = torch.tensor(Priors.common_priors(0, 0.95, 25, 20, n), device=device)
        self.lambda_g_j = torch.tensor(Priors.common_priors(0, 0.95, 25, 20, m), device=device)

class Kappa_Parameters:
    def __init__(self, n, m, l, device):
        self.kappa_c_i = torch.tensor(Priors.common_priors(1/3, 1, 25, 20, n), device = device)
        self.kappa_g_j = torch.tensor(Priors.common_priors(1/3, 1, 25, 20, m), device = device)
        self.kappa_h_k = torch.tensor(Priors.common_priors(1/3, 1, 25, 20, l), device = device)

class P_Parameters:
    def __init__(self, device):
        self.p_c_lambda = torch.tensor(np.random.dirichlet(np.ones(25)*20/25), device = device)
        self.p_g_lambda = torch.tensor(np.random.dirichlet(np.ones(25)*20/25), device = device)
        self.p_c_theta = torch.tensor(np.random.dirichlet(np.ones(100)*20/100), device = device)
        self.p_g_theta = torch.tensor(np.random.dirichlet(np.ones(100)*20/100), device = device)
        self.p_h_theta = torch.tensor(np.random.dirichlet(np.ones(100)*20/100), device = device)
        self.p_c_kappa = torch.tensor(np.random.dirichlet(np.ones(25)*20/25), device = device)
        self.p_g_kappa = torch.tensor(np.random.dirichlet(np.ones(25)*20/25), device = device)
        self.p_h_k = torch.tensor(np.random.dirichlet(np.ones(25)*20/25), device = device)

class U_Parameters:
    def __init__(self, omega_squared: float, lambdas: Lambda_Parameters, kappas: Kappa_Parameters, R_hat, n, m, l, T, device):
        self.U_c = []
        self.S_U_c = []
        self.theta_c_i = []
        for i in range(n):
            u_c, s_c, theta = initialize_U_trend(T, omega_squared, kappas.kappa_c_i[i], R_hat, device, lam=lambdas.lambda_c_i[i])
            self.S_U_c.append(s_c)
            self.U_c.append(u_c)
            self.theta_c_i.append(theta)
            
        self.U_g = []
        self.S_U_g = []
        self.theta_g_j = []
        for j in range(m):
            u_g, s_g, theta = initialize_U_trend(T, omega_squared, kappas.kappa_g_j[j], R_hat, device, lam=lambdas.lambda_g_j[j])
            self.U_g.append(u_g)
            self.S_U_g.append(s_g)
            self.theta_g_j.append(theta)
        
        self.H = []
        self.S_h = []
        self.theta_h_k = []
        for k in range(l):
            h, s_h, theta = initialize_U_trend(T, omega_squared, kappas.kappa_h_k[k], R_hat, device)
            self.H.append(h)
            self.S_h.append(s_h)
            self.theta_h_k.append(theta)
