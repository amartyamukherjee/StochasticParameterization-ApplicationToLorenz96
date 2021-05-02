#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:09:39 2020

@author: marty
"""

import numpy as np
from scipy.io import loadmat

from RungeKuttaUtils import run_lorenz96_regression_truth, APSE

cur_seed = 411
K = 40  # Number of variables

time_step = 0.001
num_steps = int(2000/time_step)
burn_in = 0
skip = 1

multiplier = 10
Dt = time_step*multiplier

Output = loadmat("Lorenz-96-data_"+str(cur_seed)+".mat")
x_shrunken = Output["x"]
u_shrunken = Output["u"]
shrunken_t = Output["t"][0,:]

start_interval = int(500/Dt)
end_interval = int(1000/Dt)

x_1D_vals = x_shrunken.copy()
u_1D_vals = u_shrunken.copy()

x_1D_vals = x_1D_vals[start_interval:end_interval, :]
u_1D_vals = u_1D_vals[start_interval:end_interval, :]

x_1D_vals = x_1D_vals.reshape(-1)[:]
u_1D_vals = u_1D_vals.reshape(-1)[:]

#%% ############## Train the Regression model ##############

polyf = np.polyfit(x_1D_vals, u_1D_vals, 4)
new_polynomial = np.poly1d(polyf)

print(polyf)

#%%############## Resampling ##############

prediction_start_interval = start_interval

new_output = run_lorenz96_regression_truth(x_shrunken[prediction_start_interval, :], polyf, Dt, x_shrunken.shape[0]-prediction_start_interval, burn_in, skip)
new_x = new_output[0]

############## Predict the trajectory for 3 * error_doubling_time ##############
prediction_time = 0.9

new_output_pred = run_lorenz96_regression_truth(x_shrunken[end_interval, :], polyf, Dt, int(prediction_time/Dt), burn_in, skip)
new_x_pred = new_output_pred[0]

apse = APSE(x_shrunken[end_interval:end_interval+int(prediction_time/Dt), :], new_x_pred[:, :])
print("Testing APSE: " + str(apse))

#%% ############### PDF ##############
from new_gaussian_kde import ave_kl_div

x_range = np.linspace(-7.5, 12.5, 1000)

kl_vals = ave_kl_div(x_shrunken[prediction_start_interval:, :], new_x, x_range)

print("Average Kullback-Leibler Divergence: " + str(np.mean(kl_vals)))
