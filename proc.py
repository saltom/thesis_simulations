#!/usr/bin/env python3

from stoc import StochasticProcess
import numpy as np

class Process(StochasticProcess):

    def epsilon_white(self, phi0, sigma_epsilon):  
  
        delta_phi = np.zeros(self.n)
        delta_phi[0] = phi0
        dt = self.t_end/self.n
        dt_sqrt = np.sqrt(dt)
        eta_epsilon = np.random.normal(scale=sigma_epsilon, size=self.n)
  
        for i in range(self.n-1):
            delta_phi[i + 1] = delta_phi[i] - self.lambda_star * delta_phi[i] * dt - self.phi_star * delta_phi[i] * eta_epsilon[i] * dt_sqrt
    
        return delta_phi

    def epsilon_OU(self, phi0, epsilon0, sigma_epsilon, z):  

        delta_phi = np.zeros(self.n)
        delta_epsilon = np.zeros(self.n)
        delta_phi[0] = phi0        
        delta_epsilon[0] = epsilon0        
        dt = self.t_end/self.n
        sigma_eta = (np.sqrt(2*z)) * sigma_epsilon
        eta = np.random.normal(scale=sigma_eta, size=self.n)

        for i in range(self.n-1):
            delta_epsilon[i+1] = delta_epsilon[i] - z * delta_epsilon[i] * dt + eta[i] * np.sqrt(dt)

        for i in range(self.n-1):
            delta_phi[i + 1] = delta_phi[i] - self.lambda_star * delta_phi[i] * dt -  self.phi_star * delta_phi[i] * delta_epsilon[i] * dt 
      
        return delta_phi, delta_epsilon
        
    def omega_white_epsilon_white(self, phi0, chi0, sigma_omega, sigma_epsilon, tau_chi, rho):
        
        delta_phi = np.zeros(self.n)
        delta_phi[0] = phi0
        delta_chi = np.zeros(self.n)
        delta_phi[0] = phi0
   
        mean = np.zeros(2)
        cov = [[sigma_omega**2, rho * sigma_omega * sigma_epsilon], [rho * sigma_omega * sigma_epsilon, sigma_epsilon**2]]
        size = self.n
   
        appo = np.random.multivariate_normal(mean, cov, size)
        
        eta_omega = appo[:,0]
        eta_epsilon = appo[:,1]
             
        dt = self.t_end/self.n
        
        for i in range(self.n-1):      
            delta_chi[i+1] = delta_chi[i] - (1/tau_chi) * delta_chi[i] * dt + (1/tau_chi) * eta_omega[i] * np.sqrt(dt)
    
        for i in range(self.n-1):            
            delta_phi[i+1] = delta_phi[i] - self.lambda_star * (delta_phi[i] - delta_chi[i]) * dt
    
        delta_lambda = self.phi_star * eta_epsilon + self.epsilon_star * delta_phi
    
        return delta_phi, delta_lambda #return growth rate instead of chi because chi is not experimentally accessible
        
    def omega_white_epsilon_colored(self, phi0, chi0, epsilon0, sigma_omega, sigma_epsilon, tau_chi, z, rho):
        
        delta_phi = np.zeros(self.n)
        delta_chi = np.zeros(self.n)
        delta_epsilon = np.zeros(self.n)  
        delta_phi[0] = phi0
        delta_chi[0] = chi0
        delta_epsilon[0] = epsilon0  #for further applications I will need to set data-like initial conditions
   
        mean = np.zeros(2)
        cov = [[sigma_omega**2, rho * sigma_omega * sigma_epsilon * np.sqrt(2*z)], [rho*sigma_omega*sigma_epsilon*np.sqrt(2*z), 2*z*sigma_epsilon**2]]
       #so that sigma_epsilon is actually the variance of epsilon
        size = self.n
   
        appo = np.random.multivariate_normal(mean, cov, size)
        
        eta_omega = appo[:,0]
        eta_epsilon = appo[:,1]
               
        dt = self.t_end/self.n
        
        for i in range(self.n-1):
            delta_chi[i+1] = delta_chi[i] - (1/tau_chi) * delta_chi[i] * dt + (1/tau_chi) * eta_omega[i] * np.sqrt(dt)
        
        for i in range(self.n-1):
            delta_epsilon[i+1] = delta_epsilon[i] - z * delta_epsilon[i] * dt + eta_epsilon[i] * np.sqrt(dt) 
    
        for i in range(self.n-1):           
            delta_phi[i+1] = delta_phi[i] - self.lambda_star * delta_phi[i] * dt + self.lambda_star * delta_chi[i] * dt
    
        delta_lambda = self.epsilon_star * delta_phi + self.phi_star * delta_epsilon
    
        return delta_phi, delta_lambda, delta_epsilon
           
    def omega_white_epsilon_colored_nsa(self, phi0, chi0, epsilon0, sigma_omega, sigma_epsilon, tau_chi, z, rho, sigma_phi_star, sigma_epsilon_star, rho_star):
        
        delta_phi = np.zeros(self.n)
        delta_chi = np.zeros(self.n)
        delta_epsilon = np.zeros(self.n)  
        delta_phi[0] = phi0
        delta_chi[0] = chi0
        delta_epsilon[0] = epsilon0 
        
        mean_star = [self.phi_star, self.epsilon_star]
        cov_star = [[sigma_phi_star**2, rho_star*sigma_phi_star*sigma_epsilon_star], [rho_star*sigma_phi_star*sigma_epsilon_star, sigma_epsilon_star**2]]
        
        a = np.random.multivariate_normal(mean_star, cov_star, 1)
        true_phi_star = a[:,0]
        true_epsilon_star = a[:,1]
        true_lambda_star = true_phi_star * true_epsilon_star

        mean = np.zeros(2)
        cov = [[sigma_omega**2, rho*sigma_omega*sigma_epsilon*np.sqrt(2*z)], [rho*sigma_omega*sigma_epsilon*np.sqrt(2*z), 2*z*sigma_epsilon**2]]
        size = self.n
   
        appo = np.random.multivariate_normal(mean, cov, size)
        
        eta_omega = appo[:,0]
        eta_epsilon = appo[:,1]
        
        dt = self.t_end/self.n
        
        for i in range(self.n-1):
            delta_chi[i+1] = delta_chi[i] - (1/tau_chi) * delta_chi[i] * dt + (1/tau_chi) * eta_omega[i] * np.sqrt(dt) 
                
        for i in range(self.n-1):
            delta_epsilon[i+1] = delta_epsilon[i] - z * delta_epsilon[i] * dt + eta_epsilon[i] * np.sqrt(dt) 
    
        for i in range(self.n-1):            
            delta_phi[i+1] = delta_phi[i] - self.lambda_star * delta_phi[i] * dt + self.lambda_star * delta_chi[i] * dt
    
        delta_lambda = true_epsilon_star * delta_phi + true_phi_star * delta_epsilon
    
        return true_phi_star + delta_phi, true_lambda_star + delta_lambda, true_epsilon_star + delta_epsilon 

    def propagated_omega(self, phi0, chi0, epsilon0, sigma_omega, tau_chi, nu_star, phi_max, epsilon_max, k_a):
    
        dt = self.t_end/self.n      
          
        delta_chi = np.zeros(self.n)
        delta_phi = np.zeros(self.n)
        delta_epsilon = np.zeros(self.n)
        
        delta_phi[0] = phi0
        delta_chi[0] = chi0
        delta_epsilon[0] = epsilon0         
        
        psi_star = nu_star * (phi_max - self.phi_star) / self.lambda_star - 1
        k_1 = self.lambda_star + self.phi_star * (1 + psi_star) * ((epsilon_max - self.epsilon_star)**2) / (k_a * epsilon_max)
        k_2 = ( nu_star + self.epsilon_star * (1 + psi_star) ) * ((epsilon_max - self.epsilon_star)**2) / (k_a * epsilon_max)
        
        eta_omega = np.random.normal(scale=sigma_omega, size=self.n)
    
        for i in range(self.n-1):
            delta_chi[i+1] = delta_chi[i] - (1/tau_chi) * delta_chi[i] * dt + (1/tau_chi) * eta_omega[i] * np.sqrt(dt)
        
        for i in range(self.n-1):
            delta_phi[i+1] = delta_phi[i] - self.lambda_star * delta_phi[i] * dt + self.lambda_star * delta_chi[i] * dt 
        
        for i in range(self.n-1):
            delta_epsilon[i+1] = delta_epsilon[i] - k_1 * delta_epsilon[i] * dt - k_2 * delta_phi[i] * dt 
        
        delta_lambda = self.epsilon_star * delta_phi + self.phi_star * delta_epsilon
    
        return delta_phi, delta_lambda, delta_epsilon     
        
        
        
        
        
