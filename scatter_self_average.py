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
N = 763  #number of scattered data points
seed = 0
z = 1/tau_chi
k = 3
n = 96
t_end = 6
rho = -0.7 #reference value obtained from the fit in cross_omega_white_epsilon_colored

process = Process(Process.omega_white_epsilon_colored, k, n, N, t_end, phi_star, lambda_star, seed)  
sigma_epsilon = process.calculate_sigma_epsilon(z, rho, sigma_omega, tau_chi, params[4])

np.random.seed(seed)  #seed for data-like initial conditions
IC_phi = np.random.normal(scale = np.sqrt(params[3]), size=N)
IC_chi = np.zeros(N)
IC_epsilon = np.random.normal(scale=sigma_epsilon, size=N)
IC = [IC_phi, IC_chi, IC_epsilon] 

x = process.simulate(IC, sigma_omega, sigma_epsilon, tau_chi, z, rho)

a = r.read_scatter()

data_phi = a[0]
data_lam = a[1]
phi = process.mean_temp(phi_star+x[0])[:,n-1]
lam = process.mean_temp(lambda_star+x[1])[:,n-1]

font = 18
ticks = 15

xy = np.vstack([phi,lam])
z = gaussian_kde(xy)(xy)

xy_data = np.vstack([data_phi,data_lam])
z_data = gaussian_kde(xy_data)(xy_data)

colors = ["orange", "yellow"]
n_bins = 100  
cmap_name = "my_custom_map"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

jp = sns.jointplot(x=phi, y=lam, color="blue", marginal_kws=dict(bins=20, fill=True), space=0)
jp.ax_joint.set_xlim(0.09, 0.19)
jp.ax_joint.set_ylim(0.4, 1.6)
jp.fig.suptitle('Simulated self average model', fontsize=24)
jp.ax_joint.scatter(phi, lam, c=z, cmap=custom_cmap, edgecolor=None)
jp.set_axis_labels('R sector time average $\mu_{\phi_R}$', 'Growth rate time average $\mu_\lambda [h^{-1}]$', fontsize=font)
jp.ax_joint.tick_params(axis='both', which='major', labelsize=ticks) 
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('scatter_self_average.pdf')
  
colors = ["blue", "lightblue"]  
n_bins = 100  
cmap_name = "my_custom_map"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

jp = sns.jointplot(x=data_phi, y=data_lam, color="green", marginal_kws=dict(bins=20, fill=True), space=0)
jp.ax_joint.set_xlim(0.09, 0.19)
jp.ax_joint.set_ylim(0.4, 1.6)
jp.fig.suptitle('Data', fontsize=24)
jp.ax_joint.scatter(data_phi, data_lam, c=z_data, cmap=custom_cmap, edgecolor=None)
jp.set_axis_labels('R sector time average $\mu_{\phi_R}$', 'Growth rate time average $\mu_\lambda [h^{-1}]$', fontsize=font)
jp.ax_joint.tick_params(axis='both', which='major', labelsize=ticks) 
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('scatter_data.pdf')



