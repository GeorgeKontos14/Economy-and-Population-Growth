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
        for i in range(n):
            u_c, s_c = initialize_U_trend(T, omega_squared, kappas.kappa_c_i[i], R_hat, device, lam=lambdas.lambda_c_i[i])
            self.S_U_c.append(s_c)
            self.U_c.append(u_c)
            
        self.U_g = []
        self.S_U_g = []
        for j in range(m):
            u_g, s_g = initialize_U_trend(T, omega_squared, kappas.kappa_g_j[j], R_hat, device, lam=lambdas.lambda_g_j[j])
            self.U_g.append(u_g)
            self.S_U_g.append(s_g)
        
        self.H = []
        self.S_h = []
        for k in range(l):
            h, s_h = initialize_U_trend(T, omega_squared, kappas.kappa_h_k[k], R_hat, device)
            self.H.append(h)
            self.S_h.append(s_h)


class Step1_Parameters:
    def __init__(self, C, w, n, q_hat, S_U_c, X_i, device):
        self.mu_C = torch.zeros(n*(q_hat+1), device=device)
        
        dim = (q_hat+1)*n
        self.Sigma = torch.zeros((dim, dim), device=device)
        for i in range(n):
            self.Sigma[i*(q_hat+1):(i+1)*(q_hat+1), i*(q_hat+1):(i+1)*(q_hat+1)] = S_U_c[i]

        self.X = torch.flatten(X_i)
        self.B = np.zeros((dim, n*(q_hat-14)))
        I = np.identity(q_hat+1)
        self.ws = torch.tensor(np.kron(w.T, I).T, device=device)
        self.Delta = torch.eye(q_hat+1, device=device)*0.01**2
        self.Cs = torch.flatten(C)

        self.V = torch.zeros((dim, dim), device=device)


class Step2_Parameters:
    def __init__(self, n, q_hat, device):
        self.mu_F = torch.zeros(q_hat+1)
        self.Sigma_F = torch.zeros((q_hat+1, q_hat+1), device=device)

        I = np.identity(q_hat+1)
        self.e = torch.tensor(np.kron(np.ones(n).T, I).T, device=device)

        self.V_F = torch.zeros((q_hat+1, q_hat+1), device=device)
        self.m_F = torch.zeros(q_hat+1, device=device) 

        self.Sigma_S = torch.zeros((q_hat+1, q_hat+1), device=device)
