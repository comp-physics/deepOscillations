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

First=True
Second=False
Third=False
Fourth=False

if First:
    # Collecting Data Code
    seed            = 1
    exponent_truth  = 13
    epochs          = 10000
    func_str= 'Levin1'
    #func_str= 'Levin2' 
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/results3/'

    func_str= 'EvansWebster3'
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/Res/Results3_EW3/'

    func_str= 'EvansWebster3'
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/Res/Results3_EW3/'

    func_str= 'RP'
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/Res/Results3_RP/'


    func_str= 'sinx'
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/Res/Res_sinx1_k_5/'

    e_array= [1,2,3,4,5,6,7,8,9,10,11] 
    n_array= [2,3,5,6,7]
    b_array= [2,3,4,5]
    s_array= [10000]

    n_array=[2 ,3, 4, 5] 
    b_array=[2, 3 ,4 ]
    s_array=[1000]

    e_array2= [1,2,3,4,5,6,7,8,9,10,11] 
    n_array2= 1
    b_array2= 1
    s_array2= 10
    itr1 = 'mid'
    itr2 = 'mid'
    #itr = 'trapz'

    # Levin1_Seed_1_Samples_5000_X_13_4_epochs_100_blayers_7_neurons_125.mat'

    normalized_MSE = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
    normalized_MSE2 = np.zeros((np.size(e_array2),np.size(n_array2),np.size(b_array2),np.size(s_array2)))
    normalized_MSE_NN = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
    normalized_MSE_NN2 = np.zeros((np.size(e_array2),np.size(n_array2),np.size(b_array2),np.size(s_array2)))
    normalized_MSE_NN_obs = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))

    for i in range(0,np.size(e_array)):
        exponent_approx = e_array[i]
        for j in range(0,np.size(n_array)):
            neurons = n_array[j]
            for k in range(0,np.size(b_array)):
                b_layers = b_array[k]
                for l in range(0,np.size(s_array)):
                    samples = s_array[l]
                    #d = sio.loadmat(save_dir+func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat', variable_names=['normalized_MSE', 'NN_MSEs_test', 'NN_MSEs_train'])
                    d = sio.loadmat(save_dir + func_str+ itr1 +'_'+ str(exponent_approx)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat', variable_names=['normalized_MSE', 'NN_MSEs_test'])
                    p = sio.loadmat(save_dir + func_str + itr2 +'_Seed_'+str(seed)+'_Samples_'+str(s_array2)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_array2)+'_neurons_'+str(n_array2)+'.mat', variable_names=['normalized_MSE', 'NN_MSEs_test', 'NN_MSEs_train'])
                    # Error Metric
                    normalized_MSE[i,j,k,l] = d['normalized_MSE']
                    normalized_MSE2[i,0,0,0] = p['normalized_MSE']
                    normalized_MSE_NN[i,j,k,l] = d['NN_MSEs_test']
                    normalized_MSE_NN2[i,0,0,0] = p['NN_MSEs_test']
                    #normalized_MSE_NN_obs[i,j,k,l] = d['NN_MSEs_train']


    #sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN_obs':normalized_MSE_NN_obs, 'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})

    sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})
    sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN2':normalized_MSE_NN2, 'normalized_MSE2':normalized_MSE2})

if Second:
    # Collecting Data Code
    seed            = 1
    exponent_truth  = 13
    epochs          = 10000
    # func_strs=['EvansWebster1', 'EvansWebster3', 'EvansWebster6']
    func_strs=['EvansWebster6']
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/Res/Results3_EW6'
    

    e_array= [1,2,3,4,5,6,7,8,9,10,11] 
    n_array= [2,3,5,6,7]
    b_array= [2,3,4,5]
    s_array= [10000]

    for func_str in func_strs:
        normalized_MSE = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        normalized_MSE_NN = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        normalized_MSE_NN_obs = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        for i in range(0,np.size(e_array)):
            exponent_approx = e_array[i]
            for j in range(0,np.size(n_array)):
                neurons = n_array[j]
                for k in range(0,np.size(b_array)):
                    b_layers = b_array[k]
                    for l in range(0,np.size(s_array)):
                        samples = s_array[l]
                        d = sio.loadmat(save_dir+func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat', variable_names=['normalized_MSE', 'NN_MSEs_test', 'NN_MSEs_train'])
                        # Error Metric
                        normalized_MSE[i,j,k,l] = d['normalized_MSE']
                        normalized_MSE_NN[i,j,k,l] = d['NN_MSEs_test']
                        #normalized_MSE_NN_obs[i,j,k,l] = d['NN_MSEs_train']


        #sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN_obs':normalized_MSE_NN_obs, 'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})
        sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})

if Third:
    # Collecting Data Code
    seed            = 1
    exponent_truth  = 13
    epochs          = 10000
    #func_strs=['EvansWebster1', 'EvansWebster3', 'EvansWebster6']
    save_dir='./results/'
    save_dir='/scratch/epickeri/Oscillations/results/'
    func_strs=['RP']

    n_array=[25, 50, 75, 100, 125]
    b_array=[3]
    s_array=[1000, 5000, 10000, 20000]

    e_array=[4,5,6,7,8,9,10,11]
#   n_array=[125]
    b_array=[3]
    #s_array=[1000, 5000, 10000, 20000]

    for func_str in func_strs:
        normalized_MSE = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        normalized_MSE_NN = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        normalized_MSE_NN_obs = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        for i in range(0,np.size(e_array)):
            exponent_approx = e_array[i]
            for j in range(0,np.size(n_array)):
                neurons = n_array[j]
                for k in range(0,np.size(b_array)):
                    b_layers = b_array[k]
                    for l in range(0,np.size(s_array)):
                        samples = s_array[l]
                        d = sio.loadmat(save_dir+func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat', variable_names=['normalized_MSE', 'NN_MSEs_test', 'NN_MSEs_train'])
                        # Error Metric
                        normalized_MSE[i,j,k,l] = d['normalized_MSE']
                        normalized_MSE_NN[i,j,k,l] = d['NN_MSEs_test']
                        normalized_MSE_NN_obs[i,j,k,l] = d['NN_MSEs_train']


        sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN_obs':normalized_MSE_NN_obs, 'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})

if Fourth:
    # Collecting Data Code
    seed            = 1
    exponent_truth  = 13
    epochs          = 10000

    seed            = 1
    exponent_truth  = 13
    epochs          = 10000
    func_strs=['RP']
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/results3/'

    e_array= [1,2,3,4,5,6,7,8,9,10,11] 
    n_array= [2,3,5,6,7]
    b_array= [2,3,4,5]
    s_array= [10000]


    for func_str in func_strs:
        normalized_MSE = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        normalized_MSE_NN = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        normalized_MSE_NN_obs = np.zeros((np.size(e_array),np.size(n_array),np.size(b_array),np.size(s_array)))
        for i in range(0,np.size(e_array)):
            exponent_approx = e_array[i]
            for j in range(0,np.size(n_array)):
                neurons = n_array[j]
                for k in range(0,np.size(b_array)):
                    b_layers = b_array[k]
                    for l in range(0,np.size(s_array)):
                        samples = s_array[l]
                        d = sio.loadmat(save_dir+func_str+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat', variable_names=['normalized_MSE', 'NN_MSEs_test', 'NN_MSEs_train'])
                        # Error Metric
                        normalized_MSE[i,j,k,l] = d['normalized_MSE']
                        normalized_MSE_NN[i,j,k,l] = d['NN_MSEs_test']
                        #normalized_MSE_NN_obs[i,j,k,l] = d['NN_MSEs_train']


        #sio.savemat(func_str+'_2_Errors.mat', {'normalized_MSE_NN_obs':normalized_MSE_NN_obs, 'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})
        sio.savemat(func_str+'_Errors.mat', {'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})

