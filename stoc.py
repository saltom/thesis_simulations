#!/usr/bin/env python3

import numpy as np

class StochasticProcess:
    def __init__(self, process_function, k, n, N, t_end, phi_star, lambda_star, seed=None):
        self.process_function = process_function
        self.k = k
        self.n = n
        self.N = N
        self.t_end = t_end
        self.phi_star = phi_star
        self.lambda_star = lambda_star
        self.epsilon_star = lambda_star/phi_star
        self.seed = seed

    def simulate(self, IC_list, *args):
        if self.seed is not None:
            np.random.seed(self.seed)   #for reprducibility

        realizations = np.zeros((self.k, self.N, self.n))

        for i in range(self.N):
            initial_conditions = [IC[i] for IC in IC_list]
            realizations[:, i, :] = self.process_function(self, *initial_conditions, *args)
                
        return [realizations[i,:,:] for i in range(self.k)]

    def data_like_corr(self, realizations1, realizations2):     #this is how data anlaysis was performed
        
        correlation_matrix = np.zeros((self.N, 2*self.n-1))

        for i in range(self.N):
            m1 = np.mean(realizations1[i,:])
            m2 = np.mean(realizations2[i,:])
            v1 = np.var(realizations1[i,:])
            v2 = np.var(realizations2[i,:])
            correlation_matrix[i,:] = np.correlate(realizations1[i,:] - m1, realizations2[i,:] - m2, mode='full')/ (np.sqrt(v1) * np.sqrt(v2) * self.n)

        return np.mean(correlation_matrix, axis=0)
        
    def theoretical_corr(self, realizations1, realizations2, cut_time):   #this is implemented as analytic calculations of stochastic process theory are made
    
        q = int(cut_time/(self.t_end/self.n))    
        correlation_matrix = np.zeros((self.N, 2*q-1))

        for i in range(self.N):
            correlation_matrix[i,:] = np.correlate(realizations1[i,-q:], realizations2[i,-q:], mode='full')/ q

        return np.mean(correlation_matrix, axis=0)/np.mean((np.sqrt(np.var(realizations1[:,-q:], axis=0)*np.var(realizations2[:,-q:], axis=0))))

    def mean_temp(self, realizations):        
    
        mean_temp=np.zeros((self.N,self.n))
    
        for j in range(self.N): 
            for i in range(self.n):
                mean_temp[j,i] = np.mean(realizations[j,:i+1])
    
        return mean_temp    

    def calculate_sigma_epsilon(self, z, rho, sigma_omega, tau_chi, var_lambda):     #for omega white epsilon colored
     
        b = 2 * np.sqrt(2*z) * self.phi_star * self.epsilon_star * self.lambda_star * rho * sigma_omega / ( (z * tau_chi + 1) * (self.lambda_star + z) )
        c = 0.5 * self.epsilon_star**2 * self.lambda_star * (sigma_omega**2) / (self.lambda_star * tau_chi + 1) - var_lambda    
        
        return ( -b + np.sqrt(b**2 - 4 * self.phi_star**2 * c) ) / (2 * self.phi_star**2)











