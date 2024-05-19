#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from proc import Process
import read as r
import time

start = time.time()

min_length = 10
params = r.read_params(min_length)
phi_star = params[1]
lambda_star = params[2]
tau_chi = 0.16666
sigma_omega = np.sqrt(2*(lambda_star*tau_chi+1)*params[3]/lambda_star)
N = params[0]
z = 1/tau_chi
k = 3
n = 96 #data-like
t_end = 6
seed = 0
rho = -0.7
n_tries = 4
sigma_phi_star = np.linspace(0.01, 0.013, n_tries)
sigma_epsilon_star = np.linspace(1.1, 1.3, n_tries)
rho_star = np.linspace(-0.7, -0.5, n_tries)

process = Process(Process.omega_white_epsilon_colored_nsa, k, n, N, t_end, phi_star, lambda_star, seed)  
sigma_epsilon = process.calculate_sigma_epsilon(z, rho, sigma_omega, tau_chi, params[4])

np.random.seed(seed)  #seed for data-like initial conditions
IC_phi = np.random.normal(scale = np.sqrt(params[3]), size=N)
IC_chi = np.zeros(N)
IC_epsilon = np.random.normal(scale=sigma_epsilon, size=N)
IC = [IC_phi, IC_chi, IC_epsilon] 

a = r.read_scatter()

std_phi_data = np.std(a[0])
std_lam_data = np.std(a[1])
rho_data = np.corrcoef(a[0], a[1])[0, 1]

error_matrix = np.zeros((n_tries, n_tries, n_tries))

for i in range(n_tries):
    for j in range(n_tries):
        for k in range(n_tries):
            x = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, z, rho, sigma_phi_star[i], sigma_epsilon_star[j], rho_star[k]) 
            std_phi_sim = np.std(process.mean_temp(x[0])[:,n-1])
            error_phi = np.abs(std_phi_sim-std_phi_data)/std_phi_data
            std_lam_sim = np.std(process.mean_temp(x[1])[:,n-1])
            error_lam = np.abs(std_lam_sim-std_lam_data)/std_lam_data
            ro = np.corrcoef(process.mean_temp(x[0])[:,n-1], process.mean_temp(x[1])[:,n-1])[0, 1]
            error_rho = np.abs(ro-rho_data)/np.abs(rho_data)
            error_matrix[i,j,k] = (error_phi + error_lam + error_rho) / 3

min_error, best_index = min([(value, (i, j, k)) for i, matrix2d in enumerate(error_matrix) for j, row in enumerate(matrix2d) for k, value in enumerate(row)])

end = time.time()

print('min error')            
print(min_error)
print('for sigma phi')
print(sigma_phi_star[best_index[0]])          
print('sigma epsilon')
print(sigma_epsilon_star[best_index[1]])
print('and rho')
print(rho_star[best_index[2]])            
print('time in minutes')            
print((end-start)/60)       









