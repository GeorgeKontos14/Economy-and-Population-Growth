import numpy as np
import Gibbs.Priors as Priors
from Gibbs.Initialize import initialize_U_trend, calculate_Sigma_U, cov_matrix
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
        peak_index = 12
        self.p_s_m = torch.zeros(25, device=device)
        for i in range(13):
            self.p_s_m[i] = i / peak_index
            self.p_s_m[24-i] = i / peak_index
        
class U_Parameters:
    def __init__(self, omega_squared: float, lambdas: Lambda_Parameters, kappas: Kappa_Parameters, R_hat, n, m, l, T, device):
        x = torch.linspace(0, 1-0.00001, 5)
        y = torch.linspace(0, 1-0.00001, 5)
        z = torch.linspace(0, 1-0.00001, 4)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        self.thetas = []
        for point in points:
            h1 = 25+775*point[0]**2
            h2 = 25+775*point[1]**2

            rho_1 = 0.5**(1/max(h1, 0.00001))
            rho_2 = 0.5**(1/max(h2, 0.00001))
            self.thetas.append((rho_1.item(), rho_2.item(), point[2].item()))    
        self.Sigma_U = {}
        self.inv_Sigma = {}
        for theta in self.thetas:
            mat = calculate_Sigma_U(theta, T, torch.tensor(R_hat, device=device), device)
            self.Sigma_U[theta] = mat
            self.inv_Sigma[theta] = torch.inverse(mat)

        half_lifes = np.linspace(50, 150, 25)
        self.rhos = []
        self.Sigma_ms = {}
        self.inv_Sigma_ms = {}
        for h in half_lifes:
            rho = (1/2)**(1/h)
            self.rhos.append(rho)
            Sigma_m_hat = cov_matrix(T, rho)
            Sigma_m = np.linalg.inv(R_hat.T@R_hat)@R_hat.T@Sigma_m_hat@R_hat@np.linalg.inv(R_hat.T@R_hat)
            Sigma_m = torch.tensor(Sigma_m, dtype=float, device = device) 
            self.Sigma_ms[rho] = Sigma_m     
            self.inv_Sigma_ms[rho] = torch.inverse(Sigma_m)       

        self.U_c = []
        self.theta_c_i = []
        for i in range(n):
            u_c, theta = initialize_U_trend(T, omega_squared, kappas.kappa_c_i[i], self.Sigma_U, device, lam=lambdas.lambda_c_i[i])
            self.U_c.append(u_c)
            self.theta_c_i.append(theta)
            
        self.U_g = []

        self.theta_g_j = []
        for j in range(m):
            u_g, theta = initialize_U_trend(T, omega_squared, kappas.kappa_g_j[j], self.Sigma_U, device, lam=lambdas.lambda_g_j[j])
            self.U_g.append(u_g)
            self.theta_g_j.append(theta)
        
        self.H = []
        self.theta_h_k = []
        for k in range(l):
            h, theta = initialize_U_trend(T, omega_squared, kappas.kappa_h_k[k], self.Sigma_U, device)
            self.H.append(h)
            self.theta_h_k.append(theta)

    def S_U_c(self, i):
        return self.Sigma_U[self.theta_c_i[i]]
    
    def S_U_g(self, i):
        return self.Sigma_U[self.theta_g_j[i]]
    
    def S_h(self, i):
        return self.Sigma_U[self.theta_h_k[i]]
    
    def inv_U_c(self, i):
        return self.inv_Sigma[self.theta_c_i[i]]
    
    def inv_U_g(self, i):
        return self.inv_Sigma[self.theta_g_j[i]]
    
    def inv_h(self, i):
        return self.inv_Sigma[self.theta_h_k[i]]
    
    def lookup(self, i):
        return self.Sigma_U[self.thetas[i]]
    
    def lookup_inv(self, i):
        return self.inv_Sigma[self.thetas[i]]
    
    def Sigma(self, rho):
        return self.Sigma_ms[rho]
    
    def inv_Sigma_m(self, rho):
        return self.inv_Sigma_ms[rho]


def step1_params(q_hat, n, U, w, device):
    I = torch.eye(q_hat+1, device=device)
    ws = torch.kron(w.t(), I).t()
    Delta = torch.eye(q_hat+1, device=device)*0.01**2
    return I, ws, Delta

def step2_params(n, I, device):
   return torch.kron(torch.ones(n, device=device).t(), I).t()

def step3_params(m, n, lambdas, U, omega_squared, kappas):
    Sigma_g_j = []
    inv_g_j = []
    for i in range(m):
        g_j = omega_squared*kappas.kappa_g_j[i]**2*(1-lambdas.lambda_g_j[i]**2)*U.S_U_g(i)
        Sigma_g_j.append(g_j)
        inv_g_j.append(torch.inverse(g_j))
    Sigma_c_i = []
    inv_c_i = []
    for i in range(n):
        c_i = omega_squared*kappas.kappa_c_i[i]**2*(1-lambdas.lambda_c_i[i]**2)*U.S_U_c(i)
        Sigma_c_i.append(c_i)
        inv_c_i.append(torch.inverse(c_i))

    return Sigma_g_j, inv_g_j, Sigma_c_i, inv_c_i

def step4_params(omega_squared, U, kappas, l):
    Sigma_H_k = []
    inv_h_k = []
    for i in range(l):
        h_k = omega_squared*kappas.kappa_h_k[i]**2*U.S_h(i)
        Sigma_H_k.append(h_k)
        inv_h_k.append(torch.inverse(h_k))
    return Sigma_H_k, inv_h_k