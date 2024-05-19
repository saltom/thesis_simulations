#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import read as r
from proc import Process

min_length = 10
params = r.read_params(min_length)
phi_star = params[1]
lambda_star = params[2]
tau_chi = 0.16666
sigma_omega = np.sqrt(2*(lambda_star*tau_chi+1)*params[3]/lambda_star)
N = params[0]
IC = [np.zeros(N), np.zeros(N), np.zeros(N)] 
z = 1/tau_chi
k = 3
n = 96
t_end = 8
seed = 0
n_tries = 20
rho = np.linspace(-1, 1, n_tries)

h = r.read_corr(3)

data_matrix = np.zeros((3,191))
data_matrix[0,:] = h[7]
data_matrix[1,:] = h[7]
data_matrix[2,:] = h[9]

data_cross = np.mean(data_matrix, axis=0)

fit_matrix = np.zeros((2, n_tries))

process = Process(Process.omega_white_epsilon_colored, k, n, N, t_end, phi_star, lambda_star, seed)  

for i in range(n_tries):
    sigma_epsilon = process.calculate_sigma_epsilon(z, rho[i], sigma_omega, tau_chi, params[4])   #growth rate variance depends on rho so this automatically calculates the noise amplitude to match experimental variance
    x = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, z, rho[i])   #I give as input experimental variance because for each rho the process computes the right sigma_epsilon
    cross = process.data_like_corr(x[1], x[0])
    fit_matrix[0,i] = rho[i]
    fit_matrix[1,i] = np.abs(np.max(data_cross)-np.max(cross)) + np.abs(np.min(data_cross)-np.min(cross))  #this time I fitted maximum and minumum

best_index = np.argmin(fit_matrix[1,:])
print(fit_matrix[:,best_index])
rho = fit_matrix[0,best_index]

sigma_epsilon = process.calculate_sigma_epsilon(z, rho, sigma_omega, tau_chi, params[4])

x = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, z, rho)
cross = process.data_like_corr(x[1], x[0])

font = 18
ticks = 15
width = 4
size = (6,4)

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.15)
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.set_xlabel('Lag time $\\tau$ [h]', fontsize=font)
ax.set_ylabel('Crosscorrelation $C_{\phi_R \lambda}$', fontsize=font)
ax.plot(h[0], data_cross, label='Data', lw=width, color='darkorange')
ax.plot(h[0], cross, label='Simulation', lw=width, color='purple', ls='--')
ax.legend(fontsize=ticks, loc='lower right')
fig.savefig('cross_omega_white_epsilon_colored.pdf')
    















