#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_lineage_matrices(min_length):  

    lineages_all_data = pd.read_csv("ALL_lineages_growth_data.csv")
    lineages_P1_20160229 = lineages_all_data[lineages_all_data["expday"] == 20160229]
    lineages_P1_20160308 = lineages_all_data[lineages_all_data["expday"] == 20160308]
    lineages_P1_20160323 = lineages_all_data[lineages_all_data["expday"] == 20160323]
    lineages_P1 = pd.concat([lineages_P1_20160229, lineages_P1_20160308, lineages_P1_20160323], axis=0)
 
    lineage_counts = lineages_P1['lineageID'].value_counts()
    long_lineages = lineage_counts[lineage_counts > min_length].index
    filtered_df = lineages_P1[lineages_P1['lineageID'].isin(long_lineages)] 
    filtered_df_sorted = filtered_df.sort_values(by=['lineageID', 'T'])
    filtered_df_sorted['growth_rate'] = filtered_df_sorted['growth_rate'] * 60

    lineages_matrices = []    
    lineages_ID = filtered_df_sorted.drop_duplicates(['lineageID'])['lineageID']    
    lineages_ID_list = lineages_ID.values    
    lineages_ID_list.sort()

    for ID in lineages_ID_list:
        lineage_matrix = filtered_df_sorted[filtered_df_sorted['lineageID'] == ID][['T', 'phi', 'growth_rate']].values 
        lineages_matrices.append(lineage_matrix)

    return lineages_matrices    
    
def read_params(min_length):
   
    x = read_lineage_matrices(min_length)
    num_lineages=len(x)

    phi_means=np.zeros(num_lineages)
    lam_means=np.zeros(num_lineages)
    
    for i in range(num_lineages):
        phi_means[i]=np.mean(x[i][:,1])
        lam_means[i]=np.mean(x[i][:,2])

    phi_mean=np.mean(phi_means)    
    lam_mean=np.mean(lam_means)
    
    phi_vars=np.zeros(num_lineages)
    lam_vars=np.zeros(num_lineages)
    
    for i in range(num_lineages):
        phi_vars[i]=np.var(x[i][:,1]-np.mean(x[i][:,1]))
        lam_vars[i]=np.var(x[i][:,2]-np.mean(x[i][:,2]))
    
    phi_var=np.mean(phi_vars)
    lam_var=np.mean(lam_vars)
    
    x = read_lineage_matrices(min_length)
    phi_list = []
    lam_list = []
    
    for i in range(len(x)):
        phi_list = np.append(phi_list,x[i][:,1])   
        lam_list = np.append(lam_list,x[i][:,2])  
    
    bulk_phi_var = phi_list.var()
    bulk_lam_var = lam_list.var()
    
    lengths = np.zeros(len(x))
    for i in range(len(x)):
        lengths[i] = len(x[i][:,0])    
   
    return num_lineages, phi_mean, lam_mean, phi_var, lam_var, bulk_phi_var, bulk_lam_var, lengths

def read_corr(n):      

    def moving_average(data, window_size):
        moving_average = []  
        moving_average.append(data[0])
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            mean = sum(window) / window_size
            moving_average.append(mean)
        moving_average.append(data[-1])
        return moving_average
    
    phi_phi_df = pd.read_csv('autocorr_phi_phi.txt', delimiter=',', header=None) 
    phi_phi_df = phi_phi_df.iloc[1:] #esclude la prima riga
    phi_phi_matrix = phi_phi_df.values
    
    times = phi_phi_matrix[:,0].astype(float)
    times = times/60
    phi_phi_20160229 = phi_phi_matrix[:,1].astype(float)
    phi_phi_20160308 = phi_phi_matrix[:,2].astype(float)
    phi_phi_20160323 = phi_phi_matrix[:,3].astype(float)
    
    lam_lam_df = pd.read_csv('autocorr_GR_GR.txt', delimiter=',', header=None) 
    lam_lam_df = lam_lam_df.iloc[1:] 
    lam_lam_matrix = lam_lam_df.values
    
    lam_lam_20160229 = lam_lam_matrix[:,1].astype(float)
    lam_lam_20160308 = lam_lam_matrix[:,2].astype(float)
    lam_lam_20160323 = lam_lam_matrix[:,3].astype(float)
        
    phi_lam_df = pd.read_csv('autocorr_phi_GR.txt', delimiter=',', header=None) 
    phi_lam_df = phi_lam_df.iloc[1:] 
    phi_lam_matrix = phi_lam_df.values
    
    phi_lam_20160229 = phi_lam_matrix[:,1].astype(float)
    phi_lam_20160308 = phi_lam_matrix[:,2].astype(float)
    phi_lam_20160323 = phi_lam_matrix[:,3].astype(float)    
    
    return times, moving_average(phi_phi_20160229[95:], n), moving_average(phi_phi_20160308[95:], n), moving_average(phi_phi_20160323[95:], n), moving_average(lam_lam_20160229[95:], n), moving_average(lam_lam_20160308[95:], n), moving_average(lam_lam_20160323[95:], n), moving_average(phi_lam_20160229, n), moving_average(phi_lam_20160308, n), moving_average(phi_lam_20160323, n) 

def read_scatter():
 
    scatter_all_data = pd.read_csv('lineage_points_to_scatter.txt', delimiter=',', header=None) 
    scatter_all_data = scatter_all_data.iloc[1:]

    matrix = scatter_all_data.values    
    phi = matrix[:,3].astype(float)
    lam = matrix[:,4].astype(float)*60

    return phi, lam









    
 
    
    

