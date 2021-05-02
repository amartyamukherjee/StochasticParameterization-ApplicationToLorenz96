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

# These are our constants
K = 40  # Number of variables
J = 10
total_number_of_vars = K * J + K
h = 1
c = 10
b = 10
Fx = 10  # Forcing
Fy = 6

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
polynomial_regression = np.poly1d(polyf)

a = 2
b = 4
c = 100

def hampel_weight(residual):
  if np.abs(residual) <= a:
    return 1
  if np.abs(residual) <= b:
    return a/np.abs(residual)
  if np.abs(residual) <= c:
    return a*((c-np.abs(residual))/(c-b))/np.abs(residual)
  return 0


for i in range(10):
  new_model = np.polyfit(x_1D_vals, u_1D_vals, 4, w=np.vectorize(hampel_weight)(u_1D_vals-polynomial_regression(x_1D_vals)))
  print("Polynomial regression model at iteration "+str(i+1)+": ", new_model)
  polynomial_regression = np.poly1d(new_model)

#%% ############## Resampling ##############

prediction_start_interval = start_interval

new_output = run_lorenz96_regression_truth(x_shrunken[prediction_start_interval, :], new_model, Dt, x_shrunken.shape[0]-prediction_start_interval, burn_in, skip)
new_x = new_output[0]

apse = APSE(x_shrunken[prediction_start_interval:end_interval, :], new_x[:end_interval-prediction_start_interval, :])

prediction_time = 0.9

new_output_pred = run_lorenz96_regression_truth(x_shrunken[end_interval, :], new_model, Dt, int(prediction_time/Dt), burn_in, skip)
new_x_pred = new_output_pred[0]

apse = APSE(x_shrunken[end_interval:end_interval+int(prediction_time/Dt), :], new_x_pred[:, :])
print("Testing APSE: " + str(apse))

#%% ############### PDF ##############
from new_gaussian_kde import ave_kl_div

x_range = np.linspace(-7.5, 12.5, 1000)

kl_vals = ave_kl_div(x_shrunken[prediction_start_interval:, :], new_x, x_range)

print("Average Kullback-Leibler Divergence: " + str(np.mean(kl_vals)))