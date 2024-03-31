#!/usr/bin/env python3

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
import shutil

# Plotting Code
seed = 1
samples = 10000
samples2 = 10
exponent_truth = 13
epochs = 10000
b_layers = 3
neurons = 500


e_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
n_array = [2, 3, 5, 10, 15]
b_array = [2, 3, 5, 7]
s_array = [10000]

n_array2 = 1  # for baseline integral
b_array2 = 1  # for baseline integral

itr1 = "mid"
itr2 = "mid"  # #itr = 'trapz'


func_str = "sinx"
save_dir = "/Users/anshumansinha/Desktop/Project/Res/Res_sinx1_k_45/"

normalized_MSE = np.zeros(
    12,
)
normalized_MSE_NN = np.zeros(
    12,
)
normalized_MSE_NN_obs = np.zeros(
    12,
)

normalized_MSE2 = np.zeros(
    12,
)
normalized_MSE_NN2 = np.zeros(
    12,
)
normalized_MSE_NN_obs2 = np.zeros(
    12,
)

import itertools

ax = plt.gca()

y_axs = np.zeros(
    1,
)
x_axs = np.zeros(
    1,
)
y_axs_tr = np.zeros(
    1,
)
y_axs_tr2 = np.zeros(
    1,
)
x_tr = np.zeros(
    1,
)

markers = ["o", "x", "D", "s", "^", "*", "o", "x", "D", "s", "^", "*"]
colors = ["b", "g", "r", "c", "m", "y", "b", "g", "r", "c", "m", "y"]
Z = 1

for exponent_approx in range(1, 11):

    if exponent_approx >= 1:
        xt = 2 ** (exponent_approx) + 1
        x_tr = np.append(x_tr, (2**exponent_approx + 1) * Z * (xt + 1))  # (2**exponent_approx-1) more points
    else:
        xt = 2 ** (exponent_approx) + 1
        x_tr = np.append(x_tr, Z * (xt + 1))

    color = colors[exponent_approx - 1]
    counteri = 0

    for neurons in n_array:

        for b_layers in b_array:

            # color = colors[b_layers]

            # String Values
            save_str = func_str + "_Seed_" + str(seed) + "_Samples_" + str(samples) + "_X_" + str(exponent_truth) + "_" + str(exponent_approx) + "_epochs_" + str(epochs) + "_blayers_" + str(b_layers) + "_neurons_" + str(neurons)

            d = sio.loadmat(save_dir + func_str + itr1 + "_" + str(exponent_approx) + "_blayers_" + str(b_layers) + "_neurons_" + str(neurons) + ".mat", variable_names=["normalized_MSE", "NN_MSEs_test"])
            p = sio.loadmat(save_dir + func_str + itr2 + "_Seed_" + str(seed) + "_Samples_" + str(samples2) + "_X_" + str(exponent_truth) + "_" + str(exponent_approx) + "_epochs_" + str(epochs) + "_blayers_" + str(b_array2) + "_neurons_" + str(n_array2) + ".mat")

            # Error Metric
            normalized_MSE[exponent_approx] = d["normalized_MSE"]
            normalized_MSE2[exponent_approx] = p["normalized_MSE"]
            normalized_MSE_NN[exponent_approx] = d["NN_MSEs_test"]

            zin = 2 ** (exponent_approx) + 1 

            if exponent_approx >= 1:
                x = (b_layers - 1) * (2 * neurons * neurons) + 2 * neurons * (1 + Z * zin) * zin
                y_axs = np.append(y_axs, d["NN_MSEs_test"])
                x_axs = np.append(x_axs, x)
            else:
                x = (b_layers - 1) * (2 * neurons * neurons) + 2 * neurons * (1 + Z * zin)
                y_axs = np.append(y_axs, d["NN_MSEs_test"])
                x_axs = np.append(x_axs, x)

            marker = markers[counteri]

            plt.loglog(x, d["NN_MSEs_test"], label="NN Test" + "" + str(neurons) + "x" + str(b_layers), color=color, linestyle="", marker=marker)

        counteri = counteri + 1

    y_axs_tr = np.append(y_axs_tr, d["normalized_MSE"])
    y_axs_tr2 = np.append(y_axs_tr2, p["normalized_MSE"])


plt.loglog(x_tr, y_axs_tr, color="k", label="Trap", linestyle="", marker="o")
plt.loglog(x_tr, y_axs_tr2, color="r", label="mid", linestyle="", marker="o")
plt.xlim([0.1e1, 1e7])
plt.grid(linestyle="--")
plt.minorticks_on()
plt.show()
