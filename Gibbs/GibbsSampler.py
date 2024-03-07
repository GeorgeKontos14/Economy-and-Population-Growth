import numpy as np
import Gibbs.Priors as Priors
from Trends import Regressors
from Trends import LowFrequencyTrends
import Gibbs.Initialize as Initialize
from Gibbs.Parameters import *
import Gibbs.StepsHelper as StepsHelper

class GibbsSampler:
    def __init__(self, data, q, w):
        # Basic variables
        self.n = len(data)
        self.m = 25
        self.l = 10
        self.T = len(data[0])
        self.q_hat = q+15
        self.R_hat = Regressors.find_regressors(self.T, self.q_hat).T
        self.w = w
        # Initialize {X_i}, i = 1,...,n
        self.X_i = Initialize.initialize_X(data, q)

        # Initialize (F, S_m)
        self.F = Initialize.initialize_F(self.X_i, w, self.q_hat)
        self.S_m, self.sigma_m, self.Sigma_m = Initialize.initialize_S_m(self.T, self.R_hat)

        # Initialize K, J
        self.K = Priors.group_factors(self.m, self.l)
        self.J = Priors.group_factors(self.n, self.m)    

        # Initialize lambda parameters
        self.lambdas = Lambda_Parameters(self.n, self.m)
        
        # Initialize the kappa parameters
        self.kappas = Kappa_Parameters(self.n, self.m, self.l)

        # Initialize p parameters
        self.p_parameters = P_Parameters()

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
        self.G = np.array(self.G)

        self.C = []
        i_1 = np.zeros(self.q_hat+1)
        i_1[0] = 1
        for i in range(self.n):
            C_i = self.mu_c*i_1+self.lambdas.lambda_c_i[i]*self.G[self.J[i]]+self.U.U_c[i]
            self.C.append(C_i)
        self.C = np.array(self.C)
        Ys = []
        for i in range(self.n):
            Y_i, _ = LowFrequencyTrends.find_trends(data[i], q)
            Ys.append(Y_i)

        self.Y = np.array(Ys).flatten()

        self.step1 = Step1_Parameters(self.C, self.w, self.n, self.q_hat, self.U.S_U_c, self.X_i)
        self.step2 = Step2_Parameters(self.n, self.q_hat)


    def step_1(self):
        """
        Calculate X_i
        """
        self.Y0 = self.w@self.C
        i_1 = np.zeros(self.q_hat+1)
        i_1[0] = 1
        self.step1.mu_C = np.concatenate([self.mu_c*i_1.T+self.lambdas.lambda_c_i[i]*self.G[self.J[i]].T for i in range(self.n)]).T

        self.step1.X = self.X_i.flatten()
        B = StepsHelper.find_B(self.Y, self.step1.X)

        Cs = self.C.flatten()

        helper = self.step1.Sigma@B@np.linalg.inv(B.T@self.step1.Sigma@B)@B.T
        self.step1.V = self.step1.Sigma-helper@self.step1.Sigma

        Z0 = Priors.multivariate_normal_prior(self.step1.mu_C, self.step1.V/np.linalg.norm(self.step1.V))[0]
        Z1 = Z0-helper@(Z0-Cs)
        epsilon = Priors.multivariate_normal_prior(self.Y0, self.step1.Delta)[0]
        e = np.repeat(epsilon, self.n)

        self.step1.Cs = Z1 - self.step1.V@self.step1.ws@(self.step1.ws.T@self.step1.V@self.step1.ws+self.step1.Delta)@self.step1.ws.T@(Z1-e)
        self.C = np.reshape(Cs, (self.n, self.q_hat+1))
        self.X_i = self.C-self.F

    def step_2(self):
        i_1 = np.zeros(self.q_hat+1)
        i_1[0] = 1
        i_2 = np.zeros(self.q_hat+1)
        i_2[1] = 1

        self.step2.mu_F = i_1*self.f_0+i_2*self.mu_m

        self.step2.Sigma_F = self.sigma_m**3*self.Sigma_m+self.s_Da**2*self.Sigma_A

        self.step2.V_F = np.linalg.inv(np.linalg.inv(self.step2.Sigma_F)+self.step2.e.T@np.linalg.inv(self.step1.Sigma)@self.step2.e+np.linalg.inv(self.step1.Delta))

        self.step2.m_F = self.step2.V_F@(self.step2.e.T@np.linalg.inv(self.step1.Sigma)@(self.step1.X-self.step1.mu_C-self.step2.e@self.step2.mu_F)+np.linalg.inv(self.step1.Delta)@((self.step1.ws.T@self.step1.X)-self.Y0-self.step2.mu_F))

        draw = Priors.multivariate_normal_prior(self.step2.m_F, self.step2.V_F)[0]
        self.F = draw+self.step2.mu_F

        self.Sigma_S = self.sigma_m**2*self.Sigma_m

        mean_S_m = self.step2.Sigma_S@np.linalg.inv(self.step2.Sigma_F)@(self.F-self.step2.mu_F)
        var_S_m = self.step2.Sigma_S-self.step2.Sigma_S@np.linalg.inv(self.step2.Sigma_F)@self.step2.Sigma_S

        self.S_m = Priors.multivariate_normal_prior(mean_S_m, var_S_m)[0]