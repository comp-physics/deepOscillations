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

First=False
Second=True
Third=False
Fourth=False

if First:
    # Collecting Data Code
    seed            = 1
    exponent_truth  = 13
    epochs          = 10000
    func_str='Levin1'
    #save_dir='./results/'
    save_dir= '/Users/anshumansinha/Desktop/Project/results3/'

    e_array=[4,5,6,7,8,9,10]
    n_array=[125, 250, 500]
    b_array=[3, 5, 7]
    s_array=[100, 500, 1000]
    e_array=[1,2,3,4]

    n_array=[3, 5, 10,25]
    b_array=[1, 2,3,5]
    s_array=[5000]

    # Levin1_Seed_1_Samples_5000_X_13_4_epochs_100_blayers_7_neurons_125.mat'

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


if Second:
    # Collecting Data Code
    seed            = 1
    exponent_truth  = 13
    epochs          = 10000
    # func_strs=['EvansWebster1', 'EvansWebster3', 'EvansWebster6']
    func_strs=['EvansWebster1']
    save_dir='./results/'
    save_dir='/Users/anshumansinha/Desktop/Project/results3/'

    e_array=[4,5,6,7,8,9,10]
    n_array=[250]
    b_array=[3]
    s_array=[1000, 5000, 10000, 20000]

    e_array=[1,2,3,4,5,6,7,8,9,10,11]
    n_array=[3, 5, 10,25]
    b_array=[1, 2,3,5]
    s_array=[2500]

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
    #func_strs=['EvansWebster1', 'EvansWebster3', 'EvansWebster6']
    #save_dir='./results/'
    #save_dir='/scratch/epickeri/Oscillations/results/'
    func_strs=['RP']

    #func_str='RP'
    #save_dir='./results/'
    save_dir= '/Users/anshumansinha/Desktop/Project/results2/'

    n_array=[8, 16, 25]
    b_array=[7]
    s_array=[10, 50, 100, 200, 500, 1000]
    e_array=[4,5,6,7,8,9,10,11]

    n_array=[125]
    b_array=[3]
    s_array=[10000]


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


        sio.savemat(func_str+'_2_Errors.mat', {'normalized_MSE_NN_obs':normalized_MSE_NN_obs, 'normalized_MSE_NN':normalized_MSE_NN, 'normalized_MSE':normalized_MSE})
