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
samples         = 1000
exponent_truth  = 13
epochs          = 10000
b_layers        = 3
neurons         = 500

    

n_array=[100, 125, 250]
b_array=[1, 2]
s_array=[5000]

n_array=[75, 100, 125]
b_array=[1, 2,3,5,7]
s_array=[5000]

n_array=[3, 5 ,10, 25]
b_array=[1, 2, 3, 5]
s_array=[5000]

n_array=[5] 
b_array=[3, 5, 7, 8]
s_array=[1000]

func_str='EvansWebster1'
save_dir= '/Users/anshumansinha/Desktop/Project/results3/'

normalized_MSE = np.zeros(12,)
normalized_MSE_NN = np.zeros(12,)
normalized_MSE_NN_obs = np.zeros(12,)
import itertools
ax = plt.gca()

y_axs = np.zeros(1,)
x_axs = np.zeros(1,)
y_axs_tr = np.zeros(1,)
x_tr = np.zeros(1,)

markers = ["o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s", "o" , "x" , "D" , "s", "o" , "x" , "D" , "s","o" , "x" , "D" , "s"]
colors = ['b', 'g','r','c','m','y','b', 'g','r','c','m','y']
Z = 1

counteri = 0 

for exponent_approx in range(1,5):

    xt  = 2**(exponent_approx)+1
    x_tr = np.append(x_tr, xt +1)
    #normalized_MSE[exponent_approx] = d['normalized_MSE']

    #color = next(ax._get_lines.prop_cycler)['color']
    color = colors[exponent_approx-1]
    

    for neurons in n_array:
        for b_layers in b_array:
        # String Values
            save_str = func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)
            d = sio.loadmat(save_dir+func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+
                    '_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat')
            # Error Metric
            
            normalized_MSE[exponent_approx] = d['normalized_MSE']
            normalized_MSE_NN[exponent_approx] = d['NN_MSEs_test'] 
            #normalized_MSE_NN_obs[exponent_approx] = d['NN_MSEs_train']
            zin = 2**(exponent_approx)+1
            #x = ((2*10*zin-1)*neurons + neurons) + ((2*neurons-1)+1) + (((2*neurons-1)*neurons + neurons)*(b_layers-1))
            x = (b_layers-1)*(2*neurons*neurons) + 2*neurons*(1+zin)
            y_axs = np.append(y_axs, d['NN_MSEs_test'])
            x_axs = np.append(x_axs, x)

            
            #if(exponent_approx>3):

            marker = markers[counteri]
            counteri = counteri+1

            #plt.loglog(normalized_MSE)
            plt.semilogy(normalized_MSE_NN,label='NN Test'+''+str(neurons)+'x'+str(b_layers), color = color, linestyle="",marker="x")
            #plt.loglog(x,normalized_MSE_NN,label='NN Test'+''+str(neurons)+'x'+str(b_layers), color = color, linestyle="",marker= marker)
            #plt.loglog(x,d['NN_MSEs_test'],label='NN Test'+''+str(neurons)+'x'+str(b_layers), color = color, linestyle="",marker= marker)
            #plt.semilogy(normalized_MSE_NN_obs,label='NN Train',color = color,linestyle="",marker="o")
    
    y_axs_tr = np.append(y_axs_tr, d['normalized_MSE'])


plt.loglog(x_tr,y_axs_tr, color='k', label='Trap',linestyle="",marker="o")
plt.legend() #loc='top right'
plt.grid(linestyle = '--')
plt.minorticks_on()
plt.show()


