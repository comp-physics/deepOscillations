#!/usr/bin/env python3

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
seed = int(sys.argv[1])
samples = int(sys.argv[2])
exponent_truth = int(sys.argv[3])
exponent_approx = int(sys.argv[4])
epochs = int(sys.argv[5])
b_layers = int(sys.argv[6])
neurons = int(sys.argv[7])

# String Values
func_str = sys.argv[8]
save_dir = sys.argv[9]

itr = "mid"
# itr = 'trapz'

save_str = func_str + itr + "_Seed_" + str(seed) + "_Samples_" + str(samples) + "_X_" + str(exponent_truth) + "_" + str(exponent_approx) + "_epochs_" + str(epochs) + "_blayers_" + str(b_layers) + "_neurons_" + str(neurons)

# Set seed and calculate number of points
np.random.random(seed)
points = 2**exponent_truth + 1

# I believe the functions we want to structure are I(x) = int_a^b g(x) S(rx) dx

if func_str == "Levin1":

    def oscil_func(x, nu, r):  # Levin paper Bessel function
        y = 1 / (x**2 + 1) * sp.jv(nu, r * x)
        return y

    # Create Bounds for integegration for the Levin paper
    a = 1
    b = 2
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        nu = np.random.random() * 0
        r = np.random.random() * 50 + 125
        y[:, i] = oscil_func(x, nu, r)

elif func_str == "Levin2":

    def oscil_func(x, nu, r, k1):  # Levin paper Bessel function
        y = 1 / (x**2 + 1) * np.cos(k1 * x) * sp.jv(nu, r * x)
        return y

    # Create Bounds for integegration for the Levin paper
    a = 1
    b = 2
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        nu = np.random.random() * 0
        r = np.random.random() * 50 + 125
        k1 = np.random.random() * 50 + 75
        y[:, i] = oscil_func(x, nu, r, k1)

elif func_str == "EvansWebster1":

    def oscil_func(x, k1, k2):  # EvansWebster Functions
        y = np.cos(k1 * x**2) * np.sin(k2 * x)
        return y

    a = 0
    b = 1
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        k1 = np.random.random() * 10 + 5
        k2 = np.random.random() * 50 + 25
        y[:, i] = oscil_func(x, k1, k2)

elif func_str == "EvansWebster2":

    def oscil_func(x, k1):  # EvansWebster Functions
        y = np.cos(x) * np.cos(k1 * np.cos(x))
        return y

    a = 0
    b = 1
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        k1 = np.random.random() * 40 + 20
        y[:, i] = oscil_func(x, k1)

elif func_str == "EvansWebster3":

    def oscil_func(x, k1):  # EvansWebster Functions
        y = np.sin(x) * np.cos(k1 * (x**2 + x))
        return y

    a = 0
    b = 1
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        k1 = np.random.random() * 50 + 475
        y[:, i] = oscil_func(x, k1)

elif func_str == "EvansWebster6":

    def oscil_func(x, k1):  # EvansWebster Functions
        y = np.exp(x) * np.sin(k1 * np.cosh(x))
        return y

    a = 0
    b = 2
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        k1 = np.random.random() * 50 + 25
        y[:, i] = oscil_func(x, k1)

        # 4 &\int_0^\pi \cos(30x)\cos(30\cos(x)) \dd x \\
        # 5 &\int_0^{\pi/2} \sin(x)\cos(\cos(x)) \cos(100 \cos(x)) \dd x \\
        # 7 &\int_{-1}^1 \cos(47 \pi x^2/4)\cos(41\pi x /4)\dd x
elif func_str == "RP":
    # Define RP
    def equation(y0, t):
        R, u = y0
        return u, (P_g - P_0 - 1317000 * np.cos(2 * np.pi * 26500 * t) - 2 * sigma / R - 4 * miu * u / R + (2 * sigma / R_0 + P_0 - P_g) * (R_0 / R) ** (3 * k)) / (R * rho) - 3 * u**2 / (2 * R)

    # parameters
    a = 0
    b = 2
    time = np.linspace(0, b / 1000000, points)
    sigma = 0.0725
    miu = 8.9 * 10 ** (-4)
    P_g = 2330
    P_0 = 10000
    k = 1.33

    # initial conditions
    R_0 = 0.0000026
    u_0 = 0
    x = time * 10**6

    def oscil_func(time):  # EvansWebster Functions
        R_1 = odeint(equation, [R_0, u_0], time)
        y = R_1[:, 0] * 10**6
        return y

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        rho = np.random.random(1) * 500 + 500
        y[:, i] = oscil_func(time).reshape(
            points,
        )

elif func_str == "sinx":

    def oscil_func(x, k):  # Levin paper Bessel function
        y = np.sin(k * x)
        return y

    # Create Bounds for integegration for the Levin paper
    a = 1
    b = 2
    x = np.linspace(a, b, points)

    # Calculate the function
    y = np.zeros((points, samples))

    for i in range(0, samples):
        k = np.random.random() * 10 + 25
        y[:, i] = oscil_func(x, k)

else:
    print("Functions not defined for integration")

# Define the Trapezoidal Integrating Function

# def Integrate_funcs_trapz(x,y,samples):
#    I = np.zeros((samples,))
#    for i in range(0,samples):
#        I[i] = np.trapz(y[:,i], x)
#    return I

# Define the Trapezoidal Integrating Function

# Define the Simpson Integrating Function

# def Integrate_funcs_simp(x,y,samples):
#    I = np.zeros((samples,))
#    for i in range(0,samples):
#        I[i] = integrate.simpson(y[:,i], x)
#    return I

# Define the Midpoint Integrating Function


def Integrate_funcs_mid(x, y, samples):
    I = np.zeros((samples,))
    len(x)
    dx = (x[len(x) - 1] - x[0]) / (len(x) - 2)

    for i in range(0, samples):
        I_i = 0
        for m in range(1, len(x) - 1):
            I_i += y[m, i] * dx
        I[i] = I_i
    return I


# Integrate the functions
I = Integrate_funcs_mid(x, y, samples)
# Center and Normalize the data Determine
I_max = np.max(np.abs(I))

I_mean = np.mean(I)
I = (I - I_mean) / I_max

# Integrate with smaller bin sizes
Is = np.zeros((samples,))
# Idiff = np.zeros((samples,))

approx_points = 2 ** (exponent_approx) + 1
xs = np.linspace(a, b, approx_points)
# xs = a+ (b-a)*np.random.random(approx_points)
inds = ((xs - 1) * (points - 1)).astype(int)

Is = Integrate_funcs_mid(xs, y[inds, :], samples)
Is = (Is - I_mean) / I_max

Idiff = I - Is
# Error Metric
normalized_MSE = np.mean(Idiff**2, axis=0) / np.mean(I**2)


def Xs(a, b, col_points, total_points):
    xs = np.linspace(a, b, col_points)
    inds = ((xs - 1) * (total_points - 1)).astype(int)
    return xs, inds


xs, inds = Xs(a, b, approx_points, points)


# %% DeepONet implementation
def DeepONet(samples, split, points, approx_points, y, I, inds, neurons, epochs, b_layers):

    import deepxde as dde

    # define error metrics

    def mean_squared_error(y_true, y_pred):
        error = np.ravel((y_true - y_pred) ** 2)
        return np.mean(error)

    def mean_relative_error(y_true, y_pred):
        error = np.ravel((((y_true - y_pred) / y_true) ** 2) ** (1 / 2))
        return np.mean(error)

    X_train0 = np.transpose(y[inds, 0:split])
    y_train = I[0:split,].reshape(split, 1)
    X_train1 = np.ones(np.size(y_train)).reshape(split, 1)

    X_test0 = np.transpose(y[inds, split:samples])
    y_test = I[split:samples,].reshape(samples - split, 1)
    X_test1 = np.ones(np.size(y_test)).reshape(samples - split, 1)

    X_train = (X_train0, X_train1)
    X_test = (X_test0, X_test1)

    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    m = np.size(y[inds, 0])  # 604*2
    print(m)

    dim_x = 1
    lr = 0.0001
    t_layers = 1
    activation = "relu"
    branch = [neurons] * (b_layers + 1)
    branch[0] = m
    trunk = [neurons] * (t_layers + 1)
    trunk[0] = dim_x

    net = dde.maps.DeepONet(
        branch,
        trunk,
        "relu",
        "Glorot normal",
        use_bias=True,
        stacked=False,
    )  #   "relu","Glorot normal",  # batch_size =

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error])
    checker = dde.callbacks.ModelCheckpoint("/Users/anshumansinha/Desktop/Project/model/" + save_str + "model.ckpt", save_better_only=False, period=100)

    # Training for different input points from 2^4 to 2^11.
    # Training will be done for 10,000 epochs.
    # isplot = True will generate 10 plots for each simulation.
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])  # Training Model batch_size = 10000

    # For plotting the residuals and the training history: isplot=True will plot
    if exponent_approx == 10 or exponent_approx == 6:
        dde.saveplot(losshistory, train_state, issave=False, isplot=False)

    NN_obs = model.predict(X_train)
    NN_test = model.predict(X_test)
    NN_obs_Idiff = (
        NN_obs.reshape(
            split,
        )
        - I[0:split,]
    )
    NN_test_Idiff = (
        NN_test.reshape(
            samples - split,
        )
        - I[split:samples,]
    )

    normalized_MSE_NN = np.mean(NN_test_Idiff**2) / np.mean(I[split:samples] ** 2)
    normalized_MSE_NN_obs = np.mean(NN_obs_Idiff**2) / np.mean(I[0:split] ** 2)

    print("Neuron", neurons)
    print("Exponent_approx.", exponent_approx)
    print(normalized_MSE_NN)

    return normalized_MSE_NN

split = int(samples * 0.75)
NN_MSEs_test = DeepONet(samples, split, points, approx_points, y / np.max(np.abs(y)), I, inds, neurons, epochs, b_layers)

sio.savemat(save_dir + func_str + itr + "_" + str(exponent_approx) + "_blayers_" + str(b_layers) + "_neurons_" + str(neurons) + ".mat", {"NN_MSEs_test": NN_MSEs_test, "normalized_MSE": normalized_MSE})
