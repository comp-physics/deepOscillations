#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:54:40 2022

@author: ethanpickering
@sub_author: anshumansinha16

"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
import shutil
from scipy.integrate import odeint 
import numpy as np
import random
from scipy import integrate

# Bash Parameters
# Integer Values
seed            = int(sys.argv[1])
samples         = int(sys.argv[2])
exponent_truth  = int(sys.argv[3])
exponent_approx = int(sys.argv[4])
epochs          = int(sys.argv[5])
b_layers        = int(sys.argv[6])
neurons         = int(sys.argv[7]) #number of neurons

# String Values
func_str = sys.argv[8]
save_dir = sys.argv[9]
itr = 'mid'
#itr = 'simp'
#itr = 'trapz'

save_str = func_str+itr+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+'_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)

# Set seed and calculate number of points
np.random.random(seed)
points = 2**exponent_truth+1

# I believe the functions we want to structure are I(x) = int_a^b g(x) S(rx) dx

if func_str == 'Levin1':
    def oscil_func(x,nu,r): # Levin paper Bessel function
        y = 1/(x**2+1)* sp.jv(nu, r*x)
        return y 
    # Create Bounds for integegration for the Levin paper
    a = 1
    b = 2
    x = np.linspace(a,b,points)
    
    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        nu = np.random.random()*0
        r = np.random.random()*50+125
        y[:,i] =oscil_func(x,nu,r)
        
elif func_str == 'Levin2':
    def oscil_func(x,nu,r,k1): # Levin paper Bessel function
        y = 1/(x**2+1)* np.cos(k1*x)* sp.jv(nu, r*x)
        return y 
    
    # Create Bounds for integegration for the Levin paper
    a = 1
    b = 2
    x = np.linspace(a,b,points)
    
    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        nu = np.random.random()*0
        r = np.random.random()*50+125
        k1 = np.random.random()*50+75
        y[:,i] =oscil_func(x,nu,r,k1)

elif func_str == 'EvansWebster1':
    def oscil_func(x,k1,k2): # EvansWebster Functions
        y = np.cos(k1*x**2)*np.sin(k2*x)
        return y 
    
    a = 0
    b = 1
    x = np.linspace(a,b,points)
    
    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        k1 = np.random.random()*10+5
        k2 = np.random.random()*50+25
        y[:,i] =oscil_func(x,k1,k2)
        
elif func_str == 'EvansWebster2':
    def oscil_func(x,k1): # EvansWebster Functions
        y = np.cos(x)*np.cos(k1*np.cos(x))
        return y 
    
    a = 0
    b = 1
    x = np.linspace(a,b,points)
    
    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        k1 = np.random.random()*40+20
        y[:,i] =oscil_func(x,k1) 
        
elif func_str == 'EvansWebster3':
    def oscil_func(x,k1): # EvansWebster Functions
        y = np.sin(x) * np.cos(k1*(x**2+x))
        return y 
    
    a = 0
    b = 1
    x = np.linspace(a,b,points)
    
    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        k1 = np.random.random()*50+475
        y[:,i] =oscil_func(x,k1)      

elif func_str == 'EvansWebster6':
    def oscil_func(x,k1): # EvansWebster Functions
        y = np.exp(x) * np.sin(k1*np.cosh(x)) 
        return y 

    a = 0
    b = 2
    x = np.linspace(a,b,points)
    
    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        k1 = np.random.random()*50+25
        y[:,i] =oscil_func(x,k1) 
        
        
               # 4 &\int_0^\pi \cos(30x)\cos(30\cos(x)) \dd x \\
               # 5 &\int_0^{\pi/2} \sin(x)\cos(\cos(x)) \cos(100 \cos(x)) \dd x \\
               # 7 &\int_{-1}^1 \cos(47 \pi x^2/4)\cos(41\pi x /4)\dd x
elif func_str == 'RP':
    # Define RP
    def equation(y0, t): 
        R, u = y0 
        return u, (P_g-P_0-1317000*np.cos(2*np.pi*26500*t)-2*sigma/R-4*miu*u/R+(2*sigma/R_0+P_0-P_g)*(R_0/R)**(3*k))/(R*rho)-3*u**2/(2*R) 
    
    # parameters 
    a = 0
    b = 2
    time = np.linspace(0, b/1000000, points) 
    sigma = 0.0725 
    miu = 8.9*10**(-4)
    P_g = 2330 
    P_0 = 10000 
    k = 1.33 

    # initial conditions
    R_0 = 0.0000026 
    u_0 = 0 
    x = time*10**6 

    def oscil_func(time): # EvansWebster Functions
        R_1 = odeint(equation, [R_0, u_0], time) 
        y = R_1[:,0]*10**6 
        return y

    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        rho = np.random.random(1)*500+500
        y[:,i] =oscil_func(time).reshape(points,)

elif func_str == 'Levin_Breaker1':
    print('not in yet')
    # traingle functions
    # heavyside functions
    # w'(x) /= A(x) w(x)

elif func_str == 'sinx':
    def oscil_func(x,k): # Levin paper Bessel function
        y = np.sin(k*x)
        return y 
    
    # Create Bounds for integegration for the Levin paper
    a = 1
    b = 2
    x = np.linspace(a,b,points)

    # Calculate the function
    y = np.zeros((points,samples))

    for i in range(0,samples):
        k = np.random.random()*10+25
        y[:,i] =oscil_func(x,k)

else:
    print('Functions not defined for integration')
    
# Define the Trapezoidal Integrating Function
def Integrate_funcs(x,y,samples):
    I = np.zeros((samples,))
    for i in range(0,samples):
        I[i] = np.trapz(y[:,i], x)
    return I

#def Integrate_funcs(x,y,samples):
#    I = np.zeros((samples,))
#    for i in range(0,samples):
#        I[i] = integrate.simpson(y[:,i], x)
#    return I


# Integrate the functions
I = Integrate_funcs(x,y,samples)
# Center and Normalize the data Determine 
I_max = np.max(np.abs(I))

I_mean = np.mean(I)
I = (I - I_mean) / I_max

# Integrate with smaller bin sizes
Is = np.zeros((samples,))
#Idiff = np.zeros((samples,))

approx_points = 2**(exponent_approx)+1
xs      = np.linspace(a,b,approx_points)
#xs = a+ (b-a)*np.random.random(approx_points)
inds    = ((xs-1)*(points-1)).astype(int)

Is = Integrate_funcs(xs,y[inds,:],samples)
Is = (Is - I_mean) / I_max

Idiff = I-Is
# Error Metric
normalized_MSE = np.mean(Idiff**2, axis = 0)/np.mean(I**2)

def Xs(a,b,col_points,total_points):
    xs = np.linspace(a,b,col_points)
    inds    = ((xs-1)*(total_points-1)).astype(int)
    return xs, inds

xs , inds= Xs(a,b,approx_points,points)

split = int(samples*0.75)
NN_MSEs_test = 0

sio.savemat(save_dir+func_str+itr+'_Seed_'+str(seed)+'_Samples_'+str(samples)+'_X_'+str(exponent_truth)+'_'+str(exponent_approx)+
            '_epochs_'+str(epochs)+'_blayers_'+str(b_layers)+'_neurons_'+str(neurons)+'.mat', 
            {'NN_MSEs_test':NN_MSEs_test,
             'y':y, 'I':I, 'Is':Is, 'x':x, 'xs':xs, 'inds':inds, 'normalized_MSE':normalized_MSE})


