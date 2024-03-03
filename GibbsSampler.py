import numpy as np
import Priors
from Trends import Regressors
import GibbsHelper

class GibbsSampler:
    def __init__(self, data, q, w):
        self.n = len(data)
        self.m = 25
        self.l = 10
        self.T = len(data[0])
        self.q_hat = q+15
        self.R_hat = Regressors.find_regressors(self.T, self.q_hat).T

        # Initialize {X_i}, i = 1,...,n
        self.X_i = GibbsHelper.initialize_X(data, q)

        # Initialize (F, S_m), sigma_m, rho_m
        self.F = GibbsHelper.initialize_F(self.X_i, w, self.q_hat)
        self.S_m = GibbsHelper.initialize_S_m(self.T, self.R_hat)

        # Initialize K, J
        self.K = Priors.group_factors(self.n, self.m)
        self.J = Priors.group_factors(self.m, self.l)    

        # Initialize lambda parameters
        self.lambdas = GibbsHelper.initialize_lambdas(self.n, self.l)
        
        # Initialize the kappa parameters
        self.kappas = GibbsHelper.initialize_kappas(self.n, self.m, self.l)

        # Initialize p parameters
        self.p_parameters = GibbsHelper.initialize_p()

        # Initialize sigma_Da^2
        self.s_Da = GibbsHelper.inverse_squared(0.03**2)

        # Initialize f_0, mu_m, mu_c
        self.f_0 = Priors.flat_prior(0, 10**6)
        self.mu_m = Priors.flat_prior(0, 10**6)
        self.mu_c = Priors.flat_prior(0, 10**6)

        # Initialize omega^2
        self.omega_squared = GibbsHelper.inverse_squared(1)

        # Initialize U persistence trends
        self.U_c = []
        for i in range(self.n):
            self.U_c.append(GibbsHelper.initialize_U_trend(self.T, 
                                                           self.omega_squared,
                                                           self.kappas.kappa_c_i[i],
                                                           self.R_hat,
                                                           lam=self.lambdas.lambda_c_i[i]))
        
        self.U_g = []
        for j in range(self.m):
            self.U_g.append(GibbsHelper.initialize_U_trend(self.T,
                                                           self.omega_squared,
                                                           self.kappas.kappa_g_j[j],
                                                           self.R_hat,
                                                           lam=self.lambdas.lambda_g_j[j]))
        
        self.H = []
        for k in range(self.l):
            self.H.append(GibbsHelper.initialize_U_trend(self.T,
                                                           self.omega_squared,
                                                           self.kappas.kappa_h_k[k],
                                                           self.R_hat))
            
        self.G = []
        for j in range(self.m):
            G_j = self.lambdas.lambda_g_j[j]*self.H[self.K[j]]+self.U_g[j]
            self.G.append(G_j)
        
        self.C = []
        i_1 = np.zeros(self.q_hat+1)
        i_1[0] = 1
        for i in range(self.n):
            C_i = self.mu_c*i_1+self.lambdas.lambda_c_i[i]*self.G[self.J[i]]+self.U_c[i]
            self.C.append(C_i)