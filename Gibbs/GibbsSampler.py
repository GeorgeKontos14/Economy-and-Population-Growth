import numpy as np
import Gibbs.Priors as Priors
from Trends import Regressors
from Trends import LowFrequencyTrends
import Gibbs.Initialize as Initialize
from Gibbs.Parameters import *
import Gibbs.StepsHelper as StepsHelper
import torch

class GibbsSampler:
    def __init__(self, data, q, w):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # Basic variables
        self.n = len(data)
        self.m = 25
        self.l = 10
        self.T = len(data[0])
        self.q_hat = q+15
        self.R_hat = Regressors.find_regressors(self.T, self.q_hat).T
        self.w = torch.tensor(w, device=device)
        # Initialize {X_i}, i = 1,...,n
        self.X_i = Initialize.initialize_X(data, q)

        # Initialize (F, S_m)
        self.F = Initialize.initialize_F(self.X_i, self.w, self.q_hat)
        self.S_m, self.sigma_m, self.Sigma_m = Initialize.initialize_S_m(self.T, self.R_hat)

        # Initialize K, J
        self.K = torch.tensor(Priors.group_factors(self.m, self.l))
        self.J = torch.tensor(Priors.group_factors(self.n, self.m))    

        # Initialize lambda parameters
        self.lambdas = Lambda_Parameters(self.n, self.m, device)
        
        # Initialize the kappa parameters
        self.kappas = Kappa_Parameters(self.n, self.m, self.l, device)

        # Initialize p parameters
        self.p_parameters = P_Parameters(device)

        # Initialize sigma_Da^2
        self.s_Da = Initialize.inverse_squared(0.03**2)
        self.A, self.Sigma_A = Initialize.init_Sigma_A(self.R_hat, self.T, self.q_hat, self.s_Da)
        # Initialize f_0, mu_m, mu_c
        self.f_0 = Priors.flat_prior(0, 10**6)
        self.mu_m = Priors.flat_prior(0, 10**6)
        self.mu_c = Priors.flat_prior(0, 10**6)

        # Initialize omega^2
        self.omega_squared = Initialize.inverse_squared(1)

        # Initialize U parameters 
        self.U = U_Parameters(self.omega_squared, self.lambdas, self.kappas, self.R_hat, self.n, self.m, self.l, self.T)

        self.G = []
        for j in range(self.m):
            G_j = self.lambdas.lambda_g_j[j]*self.U.H[self.K[j]]+self.U.U_g[j]
            self.G.append(G_j)
        self.G = torch.stack(self.G)

        self.C = []
        i_1 = torch.zeros(self.q_hat+1, device=device)
        i_1[0] = 1
        for i in range(self.n):
            C_i = self.mu_c*i_1+self.lambdas.lambda_c_i[i]*self.G[self.J[i]]+self.U.U_c[i]
            self.C.append(C_i)
        self.C = torch.stack(self.C)
        Ys = []
        for i in range(self.n):
            Y_i, _ = LowFrequencyTrends.find_trends(data[i], q)
            Ys.append(Y_i)

        self.Y = np.array(Ys).flatten()
        self.Y = torch.tensor(self.Y, device=device)
        self.step1 = Step1_Parameters(self.C, self.w, self.n, self.q_hat, self.U.S_U_c, self.X_i, device)
        self.step2 = Step2_Parameters(self.n, self.q_hat, device)


    def step_1(self):
        """
        Calculate X_i
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # self.Y0 = self.w@self.C
        self.Y0 = torch.matmul(self.w, self.C)
        i_1 = torch.zeros(self.q_hat+1, device=device)
        i_1[0] = 1
        self.step1.mu_C = torch.cat([self.mu_c*i_1.T+self.lambdas.lambda_c_i[i]*self.G[self.J[i]].T for i in range(self.n)]).t()

        self.step1.X = torch.flatten(self.X_i)
        B = StepsHelper.find_B(self.Y, self.step1.X, device)

        Cs = torch.flatten(self.C)

        # helper = self.step1.Sigma@B@np.linalg.inv(B.T@self.step1.Sigma@B)@B.T
        helper = torch.linalg.multi_dot([self.step1.Sigma, B, torch.inverse(torch.chain_matmul(B.t(), self.step1.Sigma, B)), B.t()])
        # self.step1.V = self.step1.Sigma-helper@self.step1.Sigma
        self.step1.V = self.step1.Sigma-torch.matmul(helper, self.step1.Sigma)
        # Z0 = Priors.multivariate_normal_prior(self.step1.mu_C, self.step1.V/np.linalg.norm(self.step1.V))[0]
        normal_V = self.step1.V/torch.norm(self.step1.V)
        Z0 = StepsHelper.draw_independent_samples(self.step1.mu_C.float(), normal_V.float(), self.n)
        # Z1 = Z0-helper@(Z0-Cs)
        Z1 = Z0-torch.matmul(helper, Z0-Cs.float())
        e_dist = torch.distributions.MultivariateNormal(self.Y0.float(), self.step1.Delta.float())
        epsilon = e_dist.sample()
        # epsilon = Priors.multivariate_normal_prior(self.Y0, self.step1.Delta)[0]
        # e = np.repeat(epsilon, self.n)
        e = torch.repeat_interleave(epsilon, repeats=self.n)

        # self.step1.Cs = Z1 - self.step1.V@self.step1.ws@(self.step1.ws.T@self.step1.V@self.step1.ws+self.step1.Delta)@self.step1.ws.T@(Z1-e)
        inv_help = torch.inverse(torch.linalg.multi_dot([self.step1.ws.t().float(), self.step1.V, self.step1.ws.float()])+self.step1.Delta)
        self.step1.Cs = Z1-torch.linalg.multi_dot([self.step1.V, self.step1.ws.float(), inv_help, self.step1.ws.t().float(), Z1-e])
        self.C = torch.reshape(Cs, (self.n, self.q_hat+1))
        self.X_i = self.C-self.F

    def step_2(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")        
        i_1 = torch.zeros(self.q_hat+1, device=device)
        i_1[0] = 1
        i_2 = torch.zeros(self.q_hat+1, device=device)
        i_2[1] = 1

        self.step2.mu_F = i_1*self.f_0+i_2*self.mu_m

        self.step2.Sigma_F = self.sigma_m**3*self.Sigma_m+self.s_Da**2*self.Sigma_A
        self.step2.V_F = torch.inverse(torch.inverse(self.step2.Sigma_F)+torch.linalg.multi_dot([self.step2.e.t().float(), torch.inverse(self.step1.Sigma).float(), self.step2.e.float()])+torch.inverse(self.step1.Delta))
        helper = self.step1.X.double()-self.step1.mu_C.double()-torch.matmul(self.step2.e.double(), self.step2.mu_F.double()).double()
        add1 = torch.linalg.multi_dot([self.step2.e.t().float(), torch.inverse(self.step1.Sigma).float(), helper.float()])
        add2 = torch.matmul(torch.inverse(self.step1.Delta).float(), torch.matmul(self.step1.ws.t().float(), self.step1.X.float()).float()-self.Y0.float()-self.step2.mu_F.float())
        self.step2.m_F = torch.matmul(self.step2.V_F.float(), add1.float()+add2.float())

        dist = torch.distributions.MultivariateNormal(self.step2.m_F.float(), StepsHelper.make_positive_definite(self.step2.V_F).float())
        draw = dist.sample()
        self.F = draw+self.step2.mu_F

        self.Sigma_S = self.sigma_m**2*self.Sigma_m

        # mean_S_m = self.step2.Sigma_S@np.linalg.inv(self.step2.Sigma_F)@(self.F-self.step2.mu_F)
        mult = self.F-self.step2.mu_F
        mean_S_m = torch.linalg.multi_dot([self.step2.Sigma_S.float(), torch.inverse(self.step2.Sigma_F).float(), mult.float()])
        # var_S_m = self.step2.Sigma_S-self.step2.Sigma_S@np.linalg.inv(self.step2.Sigma_F)@self.step2.Sigma_S
        var_S_m = self.step2.Sigma_S-torch.linalg.multi_dot([self.step2.Sigma_S.float(), torch.inverse(self.step2.Sigma_F).float(), self.step2.Sigma_S.float()])
        # self.S_m = Priors.multivariate_normal_prior(mean_S_m, var_S_m)[0]
        S_m_dist = torch.distributions.MultivariateNormal(mean_S_m.float(), StepsHelper.make_positive_definite(var_S_m.float()))
        self.S_m = S_m_dist.sample()