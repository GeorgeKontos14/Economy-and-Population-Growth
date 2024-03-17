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

  m_G_ind = torch.matmul(V_G_ind, lambdas.lambda_g_j[ind]*torch.matmul(inv_sigma, U.H[K[ind]])+sum2)
  dist = torch.distributions.MultivariateNormal(m_G_ind, StepsHelper.make_positive_definite(V_G_ind))
  return dist.sample()

def iterate_step4(lambdas, K, G, ind, inv_g_j, inv_sigma, q_hat, device):
  find_ind = torch.where(K == ind)[0]
  sum1 = torch.zeros((q_hat+1, q_hat+1), device=device)
  for i in find_ind:
    sum1 += lambdas.lambda_g_j[i]**2*inv_g_j[i]
  V_H_k = torch.inverse(inv_sigma+sum1)
  sum2 = torch.zeros(q_hat+1, device=device)
  for i in find_ind:
    sum2 += lambdas.lambda_g_j[i]*torch.matmul(inv_g_j[i], G[i])
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
        psi = -0.5* torch.linalg.multi_dot([gamma.t(), inv_tau, gamma]).item()
        probs.append(p_parameters.p_c_lambda[i]*math.exp(psi)*(1-point**2)**(-(q_hat+1)/2)) 
    if sum(probs) == 0:
        probs = torch.ones(25, device=device)
    else:
        probs = torch.tensor(probs, device=device)
    ind = torch.multinomial(probs, 1).item()
    return grid_points[ind]    

def iterate_step7(omega_squared, p_parameters, kappa_param, S_U, G_vec, H_vec, q_hat, grid_points, device):
    probs = []
    for i, point in enumerate(grid_points):
        gamma = G_vec-point*H_vec
        tau = omega_squared*kappa_param**2*(1-point**2)*S_U
        inv_tau = torch.inverse(tau)
        psi = -0.5*torch.linalg.multi_dot([gamma.t(), inv_tau, gamma]).item()
        probs.append(p_parameters.p_g_lambda[i]*math.exp(psi)*(1-point**2)**(-(q_hat+1)/2))
    if sum(probs) == 0:
        probs = torch.ones(25, device=device)
    else:
        probs = torch.tensor(probs, device=device)
    ind = torch.multinomial(probs, 1).item()
    return grid_points[ind]    

def iterate_step8(U, p_parameters, kappa_param, U_c_i, omega_squared, lambda_param, device):
    probs = []
    for j in range(100):
        mat = U.lookup(j)
        inv_mat = U.lookup_inv(j)
        deter = torch.det(mat)**(-0.5)
        denom = omega_squared*kappa_param*(1-lambda_param**2)
        num = torch.linalg.multi_dot([U_c_i.t(), inv_mat, U_c_i])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon.item()*10**100
        p = p_parameters.p_c_theta[j]*deter*scaled
        probs.append(p.item())
    ind = -1
    try:
        probs = torch.tensor(probs, device=device)
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
        deter = torch.det(mat)**(-0.5)
        denom = omega_squared*kappa_param*(1-lambda_param**2)
        num = torch.linalg.multi_dot([U_g_j.t(), inv_mat, U_g_j])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon.item()*10**30
        p = p_parameters.p_g_theta[j]*deter*scaled
        probs.append(p.item())
    ind = -1
    try:
        probs = torch.tensor(probs, device=device)
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
        deter = torch.det(mat)**(-0.5)
        denom = omega_squared*kappa_param
        num = torch.linalg.multi_dot([H_k.t().float(), inv_mat.float(), H_k.float()])
        expon = torch.exp(-0.5*num/denom)
        scaled = expon.item()*10**30
        p = p_parameters.p_g_theta[j]*deter*scaled
        probs.append(p.item())
    ind = -1
    try:
        probs = torch.tensor(probs, device=device)
        ind = torch.multinomial(probs, 1).item()
    except Exception as e:
        probs = torch.ones(100, device=device)
        ind = torch.multinomial(probs, 1).item()
    return U.thetas[ind]

