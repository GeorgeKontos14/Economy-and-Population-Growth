from Gibbs.Parameters import *
import Gibbs.StepsHelper as StepsHelper
import torch
import numpy as np
import math

def iterate_step3(omega_squared, lambdas, mu_c, K, inv_c_i, inv_sigma, J, ind, q_hat, C, U, device):
  find_ind = torch.where(J == ind)[0]
  sum1 = torch.zeros((q_hat+1, q_hat+1), device=device)
  for i in find_ind:
    sum1 += lambdas.lambda_c_i[i]**2*inv_c_i[i]
  V_G_ind = torch.inverse(inv_sigma+sum1)
  i_1 = torch.zeros(q_hat+1, device=device)
  i_1[0] = 1
  sum2 = torch.zeros(q_hat+1, device=device)
  for i in find_ind:
    sum2 += lambdas.lambda_c_i[i]*torch.matmul(inv_c_i[i].float(), C[i]-mu_c*i_1)

  m_G_ind = torch.matmul(V_G_ind.float(), lambdas.lambda_g_j[ind].float()*torch.matmul(inv_sigma.float(), U.H[K[ind]].float()).float()+sum2.float())
  dist = torch.distributions.MultivariateNormal(m_G_ind.float(), StepsHelper.make_positive_definite(V_G_ind).float())
  return dist.sample()

def iterate_step4(lambdas, K, G, ind, inv_g_j, inv_sigma, q_hat, device):
  find_ind = torch.where(K == ind)[0]
  sum1 = torch.zeros((q_hat+1, q_hat+1), device=device)
  for i in find_ind:
    sum1 += lambdas.lambda_g_j[i]**2*inv_g_j[i]
  V_H_k = torch.inverse(inv_sigma+sum1)
  sum2 = torch.zeros(q_hat+1, device=device)
  for i in find_ind:
    sum2 += lambdas.lambda_g_j[i].float()*torch.matmul(inv_g_j[i].float(), G[i].float())
  m_H_k = torch.matmul(V_H_k.float(), sum2.float())
  dist = torch.distributions.MultivariateNormal(m_H_k.float(), StepsHelper.make_positive_definite(V_H_k).float())
  return dist.sample()

def iterate_step6(omega_squared, p_parameters, kappa_param, S_U, mu_c, C_vec, G_vec, q_hat, grid_points, device):
    i_1 = torch.zeros(q_hat+1, device=device)
    i_1[0] = 1
    probs = []
    for i, point in enumerate(grid_points):
        gamma = C_vec -mu_c*i_1-point*G_vec
        tau = omega_squared*kappa_param**2*(1-point**2)*S_U
        inv_tau = torch.inverse(tau)
        psi = -0.5* torch.linalg.multi_dot([gamma.t().float(), inv_tau.float(), gamma.float()]).item()
        p = p_parameters.p_c_lambda[i]*math.exp(psi)*(1-point**2)**(-(q_hat+1)/2)
        probs.append(p.item()) 
    if sum(probs) == 0:
        probs = torch.ones(25, device=device)
    else:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
    ind = torch.multinomial(probs, 1).item()
    return grid_points[ind].item()    

def iterate_step7(omega_squared, p_parameters, kappa_param, S_U, G_vec, H_vec, q_hat, grid_points, device):
    probs = []
    for i, point in enumerate(grid_points):
        gamma = G_vec-point*H_vec
        tau = omega_squared*kappa_param**2*(1-point**2)*S_U
        inv_tau = torch.inverse(tau)
        psi = -0.5*torch.linalg.multi_dot([gamma.t().float(), inv_tau.float(), gamma.float()]).item()
        p = p_parameters.p_g_lambda[i]*math.exp(psi)*(1-point**2)**(-(q_hat+1)/2)
        probs.append(p.item())
    if sum(probs) == 0:
        probs = torch.ones(25, device=device)
    else:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
    ind = torch.multinomial(probs, 1).item()
    return grid_points[ind].item()    

def iterate_step8(U, p_parameters, kappa_param, U_c_i, omega_squared, lambda_param, device):
    probs = []
    for j in range(100):
        mat = U.lookup(j)
        inv_mat = U.lookup_inv(j)
        deter = torch.abs(torch.det(mat))**(-0.5)
        denom = omega_squared*kappa_param*(1-lambda_param**2)
        num = torch.linalg.multi_dot([U_c_i.t().float(), inv_mat.float(), U_c_i.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon.item()*10**100
        p = p_parameters.p_c_theta[j]*deter*scaled
        probs.append(p.item())
    ind = -1
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(100, device=device)
        ind = torch.multinomial(probs, 1).item()
    return U.thetas[ind] 

def iterate_step9(U, p_parameters, kappa_param, U_g_j, omega_squared, lambda_param, device):
    probs = []
    for j in range(100):
        mat = U.lookup(j)
        inv_mat = U.lookup_inv(j)
        deter = torch.abs(torch.det(mat))**(-0.5)
        denom = omega_squared*kappa_param*(1-lambda_param**2)
        num = torch.linalg.multi_dot([U_g_j.t().float(), inv_mat.float(), U_g_j.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon.item()*10**30
        p = p_parameters.p_g_theta[j]*deter*scaled
        probs.append(p.item())
    ind = -1
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(100, device=device)
        ind = torch.multinomial(probs, 1).item()
    return U.thetas[ind]

def iterate_step10(U, p_parameters, kappa_param, H_k, omega_squared, device):
    probs = []
    for j in range(100):
        mat = U.lookup(j)
        inv_mat = U.lookup_inv(j)
        deter = torch.abs(torch.det(mat))**(-0.5)
        denom = omega_squared*kappa_param
        num = torch.linalg.multi_dot([H_k.t().float(), inv_mat.float(), H_k.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon.item()*10**30
        p = p_parameters.p_g_theta[j]*deter*scaled
        probs.append(p.item())
    ind = -1
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(100, device=device)
        ind = torch.multinomial(probs, 1).item()
    return U.thetas[ind]

def iterate_step12(p_parameters, lambda_param, U_c, inv_U, omega_squared, grid_points, q_hat, device):
    probs = []
    for j, point in enumerate(grid_points):
        denom = omega_squared*point**2*(1-lambda_param**2)
        num = torch.linalg.multi_dot([U_c.t().float(), inv_U.float(), U_c.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon
        p = p_parameters.p_c_kappa[j]*scaled*point**(-(q_hat+1))
        probs.append(p.item())
    ind = -1
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(25, device=device)
        ind = torch.multinomial(probs, 1).item()
    return grid_points[ind].item()

def iterate_step13(p_parameters, lambda_param, U_g, inv_U, omega_squared, grid_points, q_hat, device):
    probs = []
    for j, point in enumerate(grid_points):
        denom = omega_squared*point**2*(1-lambda_param**2)
        num = torch.linalg.multi_dot([U_g.t().float(), inv_U.float(), U_g.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon
        p = p_parameters.p_g_kappa[j]*scaled*point**(-(q_hat+1))
        probs.append(p.item())
    ind = -1
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(25, device=device)
        ind = torch.multinomial(probs, 1).item()
    return grid_points[ind].item()

def iterate_step14(p_parameters, H, inv_U, omega_squared, grid_points, q_hat, device):
    probs = []
    for j, point in enumerate(grid_points):
        denom = omega_squared*point**2
        num = torch.linalg.multi_dot([H.t().float(), inv_U.float(), H.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon
        p = p_parameters.p_h_k[j]*scaled*point**(-(q_hat+1))
        probs.append(p.item())
    ind = -1
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(25, device=device)
        ind = torch.multinomial(probs, 1).item()
    return grid_points[ind].item()

def iterate_step15(omega_squared, kappa_param, lambda_param, inv_U, C_i, mu_c, G, m, q_hat, device):
    probs = []
    i_1 = torch.zeros(q_hat+1, device=device)
    i_1[0] = 1
    for j in range(m):
        gamma = C_i-mu_c*i_1-lambda_param*G[j]
        factor = 1/(omega_squared*kappa_param**2*(1-lambda_param**2))
        mat = torch.linalg.multi_dot([gamma.float(), inv_U.float(), gamma.float()])
        expon = -0.5*factor*mat
        p = torch.exp(expon)
        probs.append(p.item())
    
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        return torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(m, device=device)
        return torch.multinomial(probs, 1).item()
    
def iterate_step16(omega_squared, kappa_param, lambda_param, inv_U, G_j, H, l, device):
    probs = []
    for k in range(l):
        gamma = G_j - lambda_param*H[k]
        factor = 1/(omega_squared*kappa_param**2*(1-lambda_param**2))
        mat = torch.linalg.multi_dot([gamma.float(), inv_U.float(), gamma.float()])
        expon = -0.5*factor*mat
        p = torch.exp(expon)
        probs.append(p.item())
    
    try:
        probs = np.array(probs)
        probs = torch.from_numpy(probs)
        probs.to(device=device)
        return torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(l, device=device)
        return torch.multinomial(probs, 1).item()
    
