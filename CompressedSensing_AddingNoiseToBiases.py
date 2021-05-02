#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 00:29:57 2021

@author: marty
"""

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Lasso

from RungeKuttaUtils import run_lorenz96_matrix_truth, APSE
from CompressedSensingUtils import CompressedSensing

cur_seed = 411

# These are our constants
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
features = ["X_"+str(i) for i in range(1, K+1)] + ["X_"+str(i)+"^2" for i in range(1, K+1)] + ["X_"+str(i)+"^3" for i in range(1, K+1)] + ["X_"+str(i)+"^4" for i in range(1, K+1)]

#%% Train the SindY model

lasso_optimizer = Lasso(1e-3)
start_interval = int(500/Dt)
end_interval = int(1000/Dt)

cs_model = CompressedSensing(x_shrunken,u_shrunken,shrunken_t,features,Dt)
cs_model.RunCSAlgorithm(start_interval,end_interval,lasso_optimizer)
cs_model.findBiases()
cs_model.averageCoefficients(True)

#%% Resampling
from new_gaussian_kde import ave_kl_div

biasSeed1, biasSeed2 = 297, 441

cs_model.addNoiseToBiases(biasNoiseSeed=biasSeed1)
xi = cs_model.xi
biases_averaged = cs_model.biases_averaged + cs_model.bias_noise

prediction_time = 0.9
new_output_pred = run_lorenz96_matrix_truth(x_shrunken[end_interval, :], xi, Dt, int(prediction_time/Dt), burn_in, skip, bias=biases_averaged)
new_x_pred = new_output_pred[0]
apse = APSE(x_shrunken[end_interval:end_interval+int(prediction_time/Dt), :], new_x_pred[:, :])
print("Testing APSE: " + str(apse))

#%% PDF
cs_model.addNoiseToBiases(biasNoiseSeed=biasSeed2)
cs_model.printModel()

xi = cs_model.xi
biases_averaged = cs_model.biases_averaged + cs_model.bias_noise
prediction_start_interval = start_interval
new_output = run_lorenz96_matrix_truth(x_shrunken[prediction_start_interval, :], xi, Dt, x_shrunken.shape[0]-prediction_start_interval, burn_in, skip, bias=biases_averaged)
new_x = new_output[0]

x_range = np.linspace(-7.5, 12.5, 1000)
kl_val = ave_kl_div(x_shrunken[prediction_start_interval:, :], new_x, x_range)

print("Average Kullback-Leibler Divergence: " + str(np.round(kl_val, 6)))
