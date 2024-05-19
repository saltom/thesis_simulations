#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from proc import Process
import read as r
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

min_length = 10
params = r.read_params(min_length)
phi_star = params[1]
lambda_star = params[2]
tau_chi = 0.16666
sigma_omega = np.sqrt(2*(lambda_star*tau_chi+1)*params[3]/lambda_star)
N = params[0]
z = 1/tau_chi
k = 3
n = 96
t_end = 6
seed = 0
rho = -0.7
sigma_phi_star = 0.011666666666666667
sigma_epsilon_star = 1.2555555555555555
rho_star = -0.611111111111111

process = Process(Process.omega_white_epsilon_colored_nsa, k, n, N, t_end, phi_star, lambda_star, seed)  
sigma_epsilon = process.calculate_sigma_epsilon(z, rho, sigma_omega, tau_chi, params[4])

np.random.seed(seed)  #seed for data-like initial conditions
IC_phi = np.random.normal(scale = np.sqrt(params[3]), size=N)
IC_chi = np.zeros(N)
IC_epsilon = np.random.normal(scale=sigma_epsilon, size=N)
IC = [IC_phi, IC_chi, IC_epsilon] 

x = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, z, rho, sigma_phi_star, sigma_epsilon_star, rho_star) 

phi = process.mean_temp(x[0])[:,n-1]
lam = process.mean_temp(x[1])[:,n-1]

font = 18
ticks = 15

xy = np.vstack([phi,lam])
z = gaussian_kde(xy)(xy)

colors = ["red", "orange"]  
n_bins = 100  
cmap_name = "my_custom_map"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

jp = sns.jointplot(x=phi, y=lam, color="blue", marginal_kws=dict(bins=20, fill=True), space=0)
jp.fig.suptitle('Simulated non self average model', fontsize=24)
jp.ax_joint.set_xlim(0.09, 0.19)
jp.ax_joint.set_ylim(0.4, 1.6)
jp.ax_joint.scatter(phi, lam, c=z, cmap=custom_cmap, edgecolor=None)
jp.set_axis_labels('R sector time average $\mu_{\phi_R}$', 'Growth rate time average $\mu_\lambda [h^{-1}]$', fontsize=font)
jp.ax_joint.tick_params(axis='both', which='major', labelsize=ticks) 
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('scatter_nsa.pdf')

