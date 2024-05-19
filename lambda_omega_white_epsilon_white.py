#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from proc import Process
import read as r
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#No need to account for finite size effects here due to fast convergence 

min_length = 10
params = r.read_params(min_length)
phi_star = params[1]
lambda_star = params[2]
epsilon_star = lambda_star / phi_star
tau_chi = 0.16666
sigma_omega = np.sqrt(2*(lambda_star*tau_chi+1)*params[3]/lambda_star)    #analytically computed to match steady-state experimental variance
sigma_epsilon = np.sqrt(params[4] - (epsilon_star**2)*params[3]) / phi_star     #analytically computed to match steady-state experimental variance
N = 2000
IC = [np.zeros(N), np.zeros(N)]
k = 2
n = 1000
t_end = 8
seed = 0

process = Process(Process.omega_white_epsilon_white, k, n, N, t_end, phi_star, lambda_star, seed)

x = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, 0)
y = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, 1)
w = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, -1)
t = np.linspace(0, t_end, n) 

lam_var = np.var(x[1], axis=0)   #in this framework growth rate variance does not depend on rho
lam_corr_zero = process.theoretical_corr(x[1], x[1], t_end)[n-1:]    #no need to cut transients as variance is almost constant
lam_corr_uno = process.theoretical_corr(y[1], y[1], t_end)[n-1:]
lam_corr_menouno = process.theoretical_corr(w[1], w[1], t_end)[n-1:]

zoom_xmin = -0.2
zoom_xmax = 0.75
zoom_ymin = 0
zoom_ymax = 0.125

font = 18
ticks = 15
width = 4
size = (6,4)

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
ax.plot(t, lam_var, lw=width, color='purple')
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.set_ylim(0,0.3)
ax.set_xlabel('Time t [h]', fontsize=font)
ax.set_ylabel('Growth rate fluctuations\nvariance $\mathrm{Var}(\delta \lambda)$ $[h^{-2}]$', fontsize=font)
fig.savefig('var_lambda_omega_white_epsilonwhite.pdf')

fig, ax = plt.subplots(figsize=size)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
ax.plot(t, lam_corr_zero, lw=width, label='$\\rho=0$', color='green')
ax.plot(t, lam_corr_uno, lw=width, label='$\\rho=1$', color='royalblue', ls='--')
ax.plot(t, lam_corr_menouno, lw=width, label='$\\rho=-1$', color='red', ls=':')
axin = inset_axes(ax, width='50%', height='50%', bbox_to_anchor=(0.05, 0.05, 0.75, 0.75), bbox_transform=ax.transAxes, loc='center')
axin.plot(t, lam_corr_zero, lw=width, color='green')
axin.plot(t, lam_corr_uno, lw=width, color='royalblue', ls='--')
axin.plot(t, lam_corr_menouno, lw=width, color='red', ls=':')
axin.tick_params(axis='x', labelsize=ticks) 
axin.tick_params(axis='y', labelsize=ticks) 
axin.set_xlim(zoom_xmin, zoom_xmax)
axin.set_ylim(zoom_ymin, zoom_ymax)
ax.tick_params(axis='x', labelsize=ticks) 
ax.tick_params(axis='y', labelsize=ticks) 
ax.legend(fontsize=ticks)
ax.set_xlabel('Lag time $\\tau$ [h]', fontsize=font)
ax.set_ylabel('Growth rate fluctuations\nauto-correlation $\mathrm{C}_{\delta \lambda \delta \lambda}$', fontsize=font)
fig.savefig('corr_lambda_omega_white_epsilon_white.pdf')






