import numpy as np
import Priors
from Trends import Regressors
from Trends import LowFrequencyTrends
import InitializeGibbs
from GibbsParameters import *
import GibbsStepsHelper

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
        self.X_i = InitializeGibbs.initialize_X(data, q)

        # Initialize (F, S_m)
        self.F = InitializeGibbs.initialize_F(self.X_i, w, self.q_hat)
        self.S_m = InitializeGibbs.initialize_S_m(self.T, self.R_hat)

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
        self.s_Da = InitializeGibbs.inverse_squared(0.03**2)

        # Initialize f_0, mu_m, mu_c
        self.f_0 = Priors.flat_prior(0, 10**6)
        self.mu_m = Priors.flat_prior(0, 10**6)
        self.mu_c = Priors.flat_prior(0, 10**6)

        # Initialize omega^2
        self.omega_squared = InitializeGibbs.inverse_squared(1)

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

        self.Y = np.concatenate([Ys[i] for i in range(len(Ys))])

    def step_1(self):
        """
        Calculate X_i
        """
        self.Y0 = self.w@self.C
        i_1 = np.zeros(self.q_hat+1)
        i_1[0] = 1
        mu_C = np.concatenate([self.mu_c*i_1.T+self.lambdas.lambda_c_i[i]*self.G[self.J[i]].T for i in range(self.n)]).T
        dim = (self.q_hat+1)*self.n
        Sigma = np.zeros((dim, dim))
        Sigma[0:(self.q_hat+1), 0:(self.q_hat+1)] = self.U.S_U_c[0]
        for i in range(self.n):
            Sigma[i*(self.q_hat+1):(i+1)*(self.q_hat+1), i*(self.q_hat+1):(i+1)*(self.q_hat+1)] = self.U.S_U_c[i]

        X = self.X_i.flatten()
        B = GibbsStepsHelper.find_B(self.Y, X)
        I = np.identity(self.q_hat+1)
        ws = np.kron(self.w.T, I).T

        Cs = self.C.flatten()

        helper = Sigma@B@np.linalg.inv(B.T@Sigma@B)@B.T
        mm = helper@(Cs-mu_C)
        V = Sigma-helper@Sigma

        Z0 = Priors.multivariate_normal_prior(mu_C, V/np.linalg.norm(V))[0]
        Z1 = Z0-helper@(Z0-Cs)
        Delta = np.identity(self.q_hat+1)*0.01**2
        epsilon = Priors.multivariate_normal_prior(self.Y0, Delta)[0]
        e = np.repeat(epsilon, self.n)

        Cs = Z1 - V@ws@(ws.T@V@ws+Delta)@ws.T@(Z1-e)
        self.C = np.reshape(Cs, (self.n, self.q_hat+1))
        self.X_i = self.C-self.F