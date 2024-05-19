#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from proc import Process
import read as r

#In order to account for finite size effects, simulations are performed both in ideal conditions and in data-like conditions

min_length = 10
params = r.read_params(min_length)
phi_star = params[1]
lambda_star = params[2]
epsilon_star = lambda_star / phi_star
tau_chi = 0.16666
sigma_omega = np.sqrt(2*(lambda_star*tau_chi+1)*params[3]/lambda_star)  #analytically computed to match steady-state experimental variance
sigma_epsilon = np.sqrt(params[4] - (epsilon_star**2)*params[3]) / phi_star  #analytically computed to match steady-state experimental variance
rho = 0
N_theo = 2000
N_data_like = params[0]
IC_theo = [np.zeros(N_theo), np.zeros(N_theo)]
IC_data_like = [np.random.normal(scale=np.sqrt(params[3]), size=N_data_like), np.zeros(N_data_like)]
k = 2
n_theo = 1000
n_data_like = 96
t_end_theo = 10
t_end_data_like = 8
seed = 0
cut_time = 8
q = int(cut_time/(t_end_theo/n_theo))

y = r.read_corr(3)
data_matrix = np.zeros((3,96))
data_matrix[0,:] = y[1]
data_matrix[1,:] = y[2]
data_matrix[2,:] = y[3]
data_corr = np.mean(data_matrix, axis=0)

#all properties of phi do not depend on rho

process_theo = Process(Process.omega_white_epsilon_white, k, n_theo, N_theo, t_end_theo, phi_star, lambda_star, seed)

x_theo = process_theo.simulate(IC_theo, sigma_omega, sigma_epsilon, tau_chi, rho)
t_theo = np.linspace(0, t_end_theo, n_theo)
phi_var = np.var(x_theo[0], axis=0) #in order to consider steady-state correlation samples are taken once variance is relaxed 
tau_theo = np.linspace(0, cut_time, q)
phi_corr_theo = process_theo.theoretical_corr(x_theo[0], x_theo[0], cut_time)[q-1:]

process_data_like = Process(Process.omega_white_epsilon_white, k, n_data_like, N_data_like, t_end_data_like, phi_star, lambda_star, seed)
x_data_like = process_data_like.simulate(IC_data_like, sigma_omega, sigma_epsilon, tau_chi, rho)
phi_corr_data_like = process_data_like.data_like_corr(x_data_like[0], x_data_like[0])[n_data_like-1:]
tau_data_like = np.linspace(0, t_end_data_like, n_data_like)

font = 18
ticks = 15
width = 4
size = (6,4)

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.15)
ax.plot(t_theo, phi_var, lw=width, color='purple', ls = '--')
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax.yaxis.offsetText.set_size(ticks)
ax.set_xlabel('Time t [h]', fontsize=font)
ax.set_ylabel('R sector fluctuations\nvariance $\mathrm{Var}(\delta \phi_R)$', fontsize=font)
fig.savefig('var_phi_omega_white_epsilon_white.pdf')

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.15)
ax.plot(y[0][95:], data_corr, lw=width, color='darkorange', label='Data')
ax.plot(tau_theo, phi_corr_theo, lw=width, color='purple', ls = '--', label='Theoretical simulation')
ax.plot(tau_data_like, phi_corr_data_like, lw=width, color='magenta', ls = '--', label='Data-like simulation')
ax.legend(fontsize=ticks)
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.set_xlabel('Lag time t [h]', fontsize=font)
ax.set_ylabel('R sector fluctuations \n auto-correlation $\mathrm{C}_{\delta \phi_R \delta \phi_R}$', fontsize=font)
fig.savefig('theo_corr_phi_omega_white_epsilon_white.pdf')



















