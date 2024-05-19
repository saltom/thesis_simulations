#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from proc import Process
import read as r

min_length = 10
params = r.read_params(min_length)
phi_star = params[1]
lambda_star = params[2]
sigma_epsilon = np.sqrt(params[4]/(phi_star**2)) 
N = 10000
tau_chi = 0.16666
z = 1/tau_chi
N = 10000
IC = [np.ones(N), np.zeros(N)]
k = 2
n = 1000
t_end = 20
seed = 0

process = Process(Process.epsilon_OU, k, n, N, t_end, phi_star, lambda_star, seed)
x=process.simulate(IC, sigma_epsilon, z)

t = np.linspace(0,t_end,n)

lim_1 = 3
lim_2 = 4
n_bins = 30

trasl = lambda_star*t
phi_axis_1 = np.linspace(-lim_1,lim_1,n)
phi_axis_2 = np.linspace(-lim_2,lim_2,n)
var_teo = (phi_star**2)*(sigma_epsilon**2)*tau_chi*2*t

phi_teo_1 = (1/(np.sqrt(2*np.pi*var_teo[500])))*np.exp(-(phi_axis_1**2)/(2*var_teo[500]))
phi_teo_2 = (1/(np.sqrt(2*np.pi*var_teo[900])))*np.exp(-(phi_axis_2**2)/(2*var_teo[900]))

font = 18
ticks = 15
width = 4
size = (6,4)

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.18)
ax.set_xlim(-lim_1,lim_1)
ax.set_ylim(0,0.7)
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.set_xlabel('Shifted logarithmic fluctuation $\ln(\delta \phi_R) + \lambda^* t$', fontsize=font)
ax.set_ylabel('Probability density $\mathrm{P}(\ln(\delta \phi_R))$', fontsize=font)
ax.hist(trasl[500]+np.log((x[0])[:,500]), density=True, bins=n_bins, label='Simulation')
ax.plot(phi_axis_1, phi_teo_1, color='crimson', lw=width, label='Theoretical distribution')
ax.legend(fontsize=ticks)
fig.savefig('OU_lognormal_10.pdf')

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.18)
ax.set_xlim(-lim_2,lim_2)
ax.set_ylim(0,0.7)
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.set_xlabel('Shifted logarithmic fluctuation $\ln(\delta \phi_R) + \lambda^* t$', fontsize=font)
ax.set_ylabel('Probability density $\mathrm{P}(\ln(\delta \phi_R))$', fontsize=font)
ax.hist(trasl[900]+np.log((x[0])[:,900]), density=True, bins=n_bins, label='Simulation')
ax.plot(phi_axis_2, phi_teo_2, color='crimson', lw=width, label='Theoretical distribution')
ax.legend(fontsize=ticks)
fig.savefig('OU_lognormal_18.pdf')


