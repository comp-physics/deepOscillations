#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:18:56 2022
@author: ethanpickering
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
import shutil

# Plotting Code
seed            = 1
samples         = 2500
exponent_truth  = 13
epochs          = 10000
b_layers        = 3
neurons         = 125
func_str='EvansWebster1'
save_dir='/Users/anshumansinha/Desktop/Project/results3/'

normalized_MSE = np.zeros(10,)
normalized_MSE_NN = np.zeros(10,)
normalized_MSE_NN_obs = np.zeros(10,)


n_array=[3, 5 ,10, 25]
b_array=[1, 2, 3, 5]
s_array=[2500]

for exponent_approx in range(1,11):
    for neurons in n_array:
        for b_layers in b_array:
            # String Values
            save_str = func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)
            d = sio.loadmat(save_dir+func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+
                    '_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat')
            # Error Metric
            normalized_MSE[exponent_approx-4] = d['normalized_MSE']
            normalized_MSE_NN[exponent_approx-4] = d['NN_MSEs_test']
            normalized_MSE_NN_obs[exponent_approx-4] = d['NN_MSEs_train']
    
#plt.semilogy(normalized_MSE)
plt.semilogy(normalized_MSE_NN)
plt.semilogy(normalized_MSE_NN_obs)
plt.show()