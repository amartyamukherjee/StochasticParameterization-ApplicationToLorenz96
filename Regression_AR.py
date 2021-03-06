#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:09:39 2020

@author: marty
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from RungeKuttaUtils import run_lorenz96_regression_truth_noise, APSE

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

#%% ############### ARMA ##############
from sklearn.linear_model import LinearRegression

errors = u_shrunken - new_polynomial(x_shrunken)
error0 = errors[:-1,:]
error1 = errors[1:,:]

reg = LinearRegression().fit(error0,error1)

phi = reg.coef_[0,0]
print("Phi: ", phi)

residuals = error1 - phi*error0
sigma = np.std(residuals)
print("Sigma: ",sigma)

sigma_e = sigma / np.sqrt(1-phi**2)
print("Sigma_e: ",sigma_e)

plt.plot(error0,error1, "ro")
plt.plot([np.min(error0),np.max(error0)],[phi*np.min(error0), phi*np.max(error0)], "b--")
plt.show()

#%% ############## Predict the trajectory for 3 * error_doubling_time ##############

prediction_start_interval = start_interval

prediction_time = 0.9

new_output_pred, _, _, errors = run_lorenz96_regression_truth_noise(x_shrunken[end_interval, :], polyf, Dt, int(prediction_time/Dt), burn_in, skip, phi, sigma, np.zeros(K), seed=541)
new_x_pred = new_output_pred

apse = APSE(x_shrunken[end_interval:end_interval+int(prediction_time/Dt), :], new_x_pred[:, :])
print("Testing APSE: " + str(apse))

new_output = run_lorenz96_regression_truth_noise(x_shrunken[prediction_start_interval, :], polyf, Dt, x_shrunken.shape[0]-prediction_start_interval, burn_in, skip, phi, sigma, residuals[end_interval, :], seed=541)
new_x = new_output[0]

#%% ############### PDF ##############
from new_gaussian_kde import ave_kl_div

x_range = np.linspace(-7.5, 12.5, 1000)

kl_val = ave_kl_div(x_shrunken[prediction_start_interval:, :], new_x, x_range)

print("Average Kullback-Leibler Divergence: " + str(kl_val))
