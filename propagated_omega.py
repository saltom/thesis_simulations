#!/usr/bin/env python3

#this shows the need of an additiona noise source on epsilon

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
k = 3
n = 1000
t_end = 10
cut_time = 8
q = int(cut_time/(t_end/n))
seed = 0
nu_star = 5.884
phi_max = 0.33
epsilon_max = 10.48
k_a = 0.005

process = Process(Process.propagated_omega, k, n, N, t_end, phi_star, lambda_star, seed)
x = process.simulate(IC, sigma_omega, tau_chi, nu_star, phi_max, epsilon_max, k_a)
t = np.linspace(0, t_end, n)
var_lambda = np.var(x[1], axis=0)
tau = np.linspace(-cut_time, cut_time, 2*q-1)
cross = process.theoretical_corr(x[1], x[0], cut_time)

font = 18
ticks = 15
width = 4
size = (6,4)

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
ax.plot(t, var_lambda, lw=width, color='purple', ls = '--')
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax.yaxis.offsetText.set_size(ticks)
ax.set_xlabel('Time t [h]', fontsize=font)
ax.set_ylabel('Growth rate fluctuations\nvariance $\mathrm{Var}(\delta \lambda)$ ' '$[h^{-1}]$', fontsize=font)
fig.savefig('var_lambda_propagated_omega.pdf')

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.2)
ax.plot(cross, lw=width, color='purple', ls = '--')
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.set_xlabel('Lag time $\\tau$ [h]', fontsize=font)
ax.set_ylabel('Fluctuations \ncross-correlation $C_{\delta \phi_R \delta \lambda}$', fontsize=font)
fig.savefig('cross_propagated_omega.pdf')










