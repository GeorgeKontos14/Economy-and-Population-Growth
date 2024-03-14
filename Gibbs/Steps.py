from Gibbs import Initialize
from Gibbs.Parameters import *
from Trends import Regressors
from Trends import LowFrequencyTrends
from Gibbs import Priors
import Gibbs.StepsHelper as StepsHelper
import torch
import random
import numpy as np
import math

def initialize(data, n, m, l, T, q, q_hat, device):
  R_hat = Regressors.find_regressors(T, q_hat).T
  S_m, sigma_m, Sigma_m = Initialize.initialize_S_m(T, R_hat, device)
  K = torch.tensor(Priors.group_factors(m, l), device=device)
  J = torch.tensor(Priors.group_factors(n, m), device=device)
  lambdas = Lambda_Parameters(n, m, device)
  kappas = Kappa_Parameters(n, m, l, device)
  p_parameters = P_Parameters(device)

  s_Da = Initialize.inverse_squared(0.03**2)
  A, Sigma_A = Initialize.init_Sigma_A(R_hat, T, q_hat, s_Da)

  f_0 = random.uniform(-1, 1)
  mu_m = random.uniform(-1, 1)
  mu_c = random.uniform(-1, 1)
  omega_squared = Initialize.inverse_squared(1)

  U = U_Parameters(omega_squared, lambdas, kappas, R_hat, n, m, l, T, device)

  G = []
  for j in range(m):
      G_j = lambdas.lambda_g_j[j]*U.H[K[j]]+U.U_g[j]
      G.append(G_j)
  G = torch.stack(G)

  C = []
  i_1 = torch.zeros(q_hat+1, device=device)
  i_1[0] = 1
  for i in range(n):
      C_i = mu_c*i_1+lambdas.lambda_c_i[i]*G[J[i]]+U.U_c[i]
      C.append(C_i)
  C = torch.stack(C)
  Ys = []
  for i in range(n):
      Y_i, _ = LowFrequencyTrends.find_trends(data[i], q)
      Ys.append(Y_i)

  Y = np.array(Ys).flatten()
  Y = torch.tensor(Y, device=device)

  i_2 = torch.zeros(q_hat+1, device=device)
  i_2[1] = 1
  F = i_1*f_0+i_2*mu_m+S_m+A
  X_i = F + C
  return X_i, F, S_m, sigma_m, Sigma_m, K, J, lambdas, kappas, p_parameters, s_Da, A, Sigma_A, f_0, mu_m, mu_c, omega_squared, U, G, C, Y

def step1_params(q_hat, n, U, w, device):
    dim = (q_hat+1)*n
    Sigma = torch.zeros((dim, dim), device=device)
    for i in range(n):
        Sigma[i*(q_hat+1):(i+1)*(q_hat+1), i*(q_hat+1):(i+1)*(q_hat+1)] = U.S_U_c[i]

    I = torch.eye(q_hat+1, device=device)
    ws = torch.kron(w.t(), I).t()
    Delta = torch.eye(q_hat+1, device=device)*0.01**2
    return Sigma, I, ws, Delta

def step2_params(n, I, device):
   return torch.kron(torch.ones(n, device=device).t(), I).t()

def step3_params(m, n, lambdas, U, omega_squared, kappas):
    Sigma_g_j = []
    inv_g_j = []
    for i in range(m):
        g_j = omega_squared*kappas.kappa_g_j[i]**2*(1-lambdas.lambda_g_j[i]**2)*U.S_U_g[i]
        Sigma_g_j.append(g_j)
        inv_g_j.append(torch.inverse(g_j))
    Sigma_c_i = []
    inv_c_i = []
    for i in range(n):
        c_i = omega_squared*kappas.kappa_c_i[i]**2*(1-lambdas.lambda_c_i[i]**2)*U.S_U_c[i]
        Sigma_c_i.append(c_i)
        inv_c_i.append(torch.inverse(c_i))

    return Sigma_g_j, inv_g_j, Sigma_c_i, inv_c_i

def step4_params(omega_squared, U, kappas, l):
    Sigma_H_k = []
    inv_h_k = []
    for i in range(l):
        h_k = omega_squared*kappas.kappa_h_k[i]**2*U.S_h[i]
        Sigma_H_k.append(h_k)
        inv_h_k.append(torch.inverse(h_k))
    return Sigma_H_k, inv_h_k

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

def step1(X_i, w, ws, Delta, Y, C, G, J, F, mu_c, lambdas, Sigma, n, q_hat, device):
  Y0 = torch.matmul(w, C)
  i_1 = torch.zeros(q_hat+1, device=device)
  i_1[0] = 1
  mu_C = torch.cat([mu_c*i_1.t()+lambdas.lambda_c_i[i]*G[J[i]].t() for i in range(n)]).t()
  X = torch.flatten(X_i)
  B = StepsHelper.find_B(Y, X, device)
  Cs = torch.flatten(C)
  helper = torch.linalg.multi_dot([Sigma, B, torch.inverse(torch.linalg.multi_dot([B.t(), Sigma, B])), B.t()])
  V = Sigma-torch.matmul(helper, Sigma)
  normal_V = V/torch.norm(V)
  Z0 = StepsHelper.draw_independent_samples(mu_C.float(), normal_V.float(), n)
  Z1 = Z0-torch.matmul(helper, Z0-Cs.float())
  e_dist = torch.distributions.MultivariateNormal(Y0.float(), Delta.float())
  epsilon = e_dist.sample()
  ep = torch.repeat_interleave(epsilon, repeats=n)
  inv_help = torch.inverse(torch.linalg.multi_dot([ws.t().float(), V, ws.float()])+Delta)
  Cs = Z1-torch.linalg.multi_dot([V, ws.float(), inv_help, ws.t().float(), Z1-ep])
  CC = torch.reshape(Cs, (n, q_hat+1))
  XX = CC-F
  return XX, CC, Sigma, mu_C, Y0, X

def step2(e, f_0, mu_m, sigma_m, Sigma_m, s_Da, Sigma_A, Sigma, Delta, mu_C, ws, q_hat, X, Y0, device):
  i_1 = torch.zeros(q_hat+1, device=device)
  i_1[0] = 1
  i_2 = torch.zeros(q_hat+1, device=device)
  i_2[1] = 1
  mu_F = i_1*f_0+i_2*mu_m
  Sigma_F = sigma_m**3*Sigma_m+s_Da**2*Sigma_A
  V_F = torch.inverse(torch.inverse(Sigma_F)+torch.linalg.multi_dot([e.t().float(), torch.inverse(Sigma).float(), e.float()])+torch.inverse(Delta))
  helper = X.double()-mu_C.double()-torch.matmul(e.double(), mu_F.double()).double()
  add1 = torch.linalg.multi_dot([e.t().float(), torch.inverse(Sigma).float(), helper.float()])
  add2 = torch.matmul(torch.inverse(Delta).float(), torch.matmul(ws.t().float(), X.float()).float()-Y0.float()-mu_F.float())
  m_F = torch.matmul(V_F.float(), add1.float()+add2.float())
  dist = torch.distributions.MultivariateNormal(m_F.float(), StepsHelper.make_positive_definite(V_F).float())
  draw = dist.sample()
  FF = draw+mu_F
  Sigma_S = sigma_m**2*Sigma_m
  mult = FF-mu_F
  mean_S_m = torch.linalg.multi_dot([Sigma_S.float(), torch.inverse(Sigma_F).float(), mult.float()])
  var_S_m = Sigma_S-torch.linalg.multi_dot([Sigma_S.float(), torch.inverse(Sigma_F).float(), Sigma_S.float()])
  S_m_dist = torch.distributions.MultivariateNormal(mean_S_m.float(), StepsHelper.make_positive_definite(var_S_m.float()))
  SS_m = S_m_dist.sample()
  return FF, SS_m

def step3(omega_squared, lambdas, mu_c, K, inv_c_i, inv_g_j, J, q_hat, m, C, U, device):
  new_G = []
  for j in range(m):
    new_G.append(iterate_step3(omega_squared, lambdas, mu_c, K, inv_c_i, inv_g_j[j], J, j, q_hat, C, U, device))

  return torch.stack(new_G)

def step4(lambdas, K, G, inv_g_j, inv_h_k, l, q_hat, device):
  new_H = []
  for k in range(l):
    new_H.append(iterate_step4(lambdas, K, G, k, inv_g_j, inv_h_k[k], q_hat, device))
  return new_H

def step5(q_hat, lambdas, C, G, J, inv_c_i, n, device):
  i_1 = torch.zeros(q_hat+1, device=device)
  i_1[0] = 1
  denominator = 0
  numerator = 0
  for i in range(n):
    add = C[i]-lambdas.lambda_c_i[i]*G[J[i]]
    numerator += torch.linalg.multi_dot([i_1.t(), inv_c_i[i].float(), add.float()]).item()
    denominator += torch.linalg.multi_dot([i_1.t().float(), inv_c_i[i].float(), i_1.float()]).item()

  mean_mu_c = numerator/denominator
  var_mu_c = 1/denominator
  return torch.normal(mean = mean_mu_c, std = torch.sqrt(torch.tensor(var_mu_c, device=device))).item()

def step6(omega_squared, kappas, U, p_parameters, mu_c, C, G, J, q_hat, n, device):
    grid_points = torch.linspace(0, 0.95, 25)
    new_lambda_c_i = []
    for i in range(n):
        new_lambda_c_i.append(iterate_step6(omega_squared, p_parameters=p_parameters, kappa_param=kappas.kappa_c_i[i], S_U=U.S_U_c[i], mu_c=mu_c, C_vec=C[i], G_vec = G[J[i]], q_hat=q_hat, grid_points=grid_points, device=device))
    return new_lambda_c_i

def step7(omega_squared, p_parameters, kappas, U, G, m, K, q_hat, device):
    grid_points = torch.linspace(0, 0.95, 25)
    new_lambda_g_j = []
    for i in range(m):
        new_lambda_g_j.append(iterate_step7(omega_squared, p_parameters, kappas.kappa_g_j[i], U.S_U_g[i], G[i], U.H[K[i]], q_hat, grid_points, device=device))
    return new_lambda_g_j