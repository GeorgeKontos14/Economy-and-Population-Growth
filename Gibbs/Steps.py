from Gibbs import Initialize
from Gibbs.Parameters import *
from Trends import Regressors
from Trends import LowFrequencyTrends
from Gibbs import Priors
import Gibbs.StepsHelper as StepsHelper
from Gibbs.Iterations import *
import torch
import random
import numpy as np

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
  return X_i, F, S_m, sigma_m, Sigma_m, K, J, lambdas, kappas, p_parameters, s_Da, A, Sigma_A, f_0, mu_m, mu_c, omega_squared, U, G, C, Y, torch.tensor(R_hat, device=device)

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
        new_lambda_c_i.append(iterate_step6(omega_squared, p_parameters=p_parameters, kappa_param=kappas.kappa_c_i[i], S_U=U.S_U_c(i), mu_c=mu_c, C_vec=C[i], G_vec = G[J[i]], q_hat=q_hat, grid_points=grid_points, device=device))
    return torch.tensor(new_lambda_c_i, device=device)

def step7(omega_squared, p_parameters, kappas, U, G, m, K, q_hat, device):
    grid_points = torch.linspace(0, 0.95, 25)
    new_lambda_g_j = []
    for i in range(m):
        new_lambda_g_j.append(iterate_step7(omega_squared, p_parameters, kappas.kappa_g_j[i], U.S_U_g(i), G[i], U.H[K[i]], q_hat, grid_points, device=device))
    return torch.tensor(new_lambda_g_j, device=device)

def step8(omega_squared, p_parameters, kappas, lambdas, U, X_i, F, mu_c, G, J, q_hat, n, device):
    new_U = StepsHelper.update_U_c_i(U.U_c, X_i, F, mu_c, lambdas, G, J, q_hat, device)
    new_thetas = []
    for i in range(n):
        new_thetas.append(iterate_step8(U, p_parameters, kappas.kappa_c_i[i], new_U[i], omega_squared, lambdas.lambda_c_i[i], device))
    return new_U, new_thetas

def step9(omega_squared, p_parameters, kappas, lambdas, U, G, K, m, device):
    new_U = StepsHelper.update_U_g_j(U.U_g, G, lambdas, K, U.H)
    new_thetas = []
    for i in range(m):
        new_thetas.append(iterate_step9(U, p_parameters, kappas.kappa_g_j[i], new_U[i], omega_squared, lambdas.lambda_g_j[i], device))
    return new_U, new_thetas

def step10(omega_squared, p_parameters, kappas, U, l, device):
    new_thetas = []
    for i in range(l):
        new_thetas.append(iterate_step10(U, p_parameters, kappas.kappa_h_k[i], U.H[i], omega_squared, device))
    return new_thetas

def step11(n, m, l, kappas, lambdas, U, q_hat):
    S_c_squared = 0
    for i in range(n):
        factor = 1/(kappas.kappa_c_i[i]**2*(1-lambdas.lambda_c_i[i]**2))
        prod = torch.linalg.multi_dot([U.U_c[i].t(), U.inv_U_c(i), U.U_c[i]])
        S_c_squared += prod.item()*factor.item()
    
    S_g_squared = 0
    for i in range(m):
        factor = 1/(kappas.kappa_g_j[i]**2*(1-lambdas.lambda_g_j[i]**2))
        prod = torch.linalg.multi_dot([U.U_g[i].t(), U.inv_U_g(i), U.U_g[i]])
        S_g_squared += prod.item()*factor.item()

    S_h_squared = 0
    for i in range(l):
        factor = 1/(kappas.kappa_h_k[i]**2)
        prod = torch.linalg.multi_dot([U.H[i].t().float(), U.inv_h(i).float(), U.H[i].float()])
        S_h_squared += prod.item()*factor.item()

    freedom = 1+(n+m+l)*(q_hat+1)
    dist = torch.distributions.Chi2(df=freedom)
    samp = dist.sample()
    added = 1/2.198
    return (added+S_c_squared+S_g_squared+S_h_squared)/samp.item()

def step12(omega_squared, p_parameters, lambdas, U, q_hat, n, device):
    grid_points = torch.linspace(1/3,1, 25)
    new_kappas = []
    for i in range(n):
        new_kappas.append(iterate_step12(p_parameters, lambdas.lambda_c_i[i], U.U_c[i], U.inv_U_c(i), omega_squared, grid_points, q_hat, device))
    return torch.tensor(new_kappas, device=device)

def step13(omega_squared, p_parameters, lambdas, U, q_hat, m, device):
    grid_points = torch.linspace(1/3,1, 25)
    new_kappas = []
    for i in range(m):
        new_kappas.append(iterate_step13(p_parameters, lambdas.lambda_g_j[i], U.U_g[i], U.inv_U_g(i), omega_squared, grid_points, q_hat, device))
    return torch.tensor(new_kappas, device=device)

def step14(omega_squared, p_parameters, U, q_hat, l, device):
    grid_points = torch.linspace(1/3,1, 25)
    new_kappas = []
    for i in range(l):
        new_kappas.append(iterate_step14(p_parameters, U.H[i], U.inv_h(i), omega_squared, grid_points, q_hat, device))
    return torch.tensor(new_kappas, device=device)

def step15(omega_squared, kappas, lambdas, C, G, U, n, m, mu_c, q_hat, device):
    new_J = []
    for i in range(n):
        new_J.append(iterate_step15(omega_squared, kappas.kappa_c_i[i], lambdas.lambda_c_i[i], U.inv_U_c(i), C[i], mu_c, G, m, q_hat, device))
    return torch.tensor(new_J, device=device)

def step16(omega_squared, kappas, lambdas, G, U, m, l, device):
    new_K = []
    for i in range(m):
        new_K.append(iterate_step16(omega_squared, kappas.kappa_g_j[i], lambdas.lambda_g_j[i], U.inv_U_g(i), G[i], U.H, l, device))
    return torch.tensor(new_K, device=device)

