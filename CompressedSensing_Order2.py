#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 6 22:15:20 2020

@author: marty
"""

import pysindy as ps
import numpy as np
from scipy.io import loadmat
from pysindy.feature_library import CustomLibrary

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

functions = [
             lambda x: x,     #f0
]
feature_library = CustomLibrary(library_functions=functions)

Output = loadmat("Lorenz-96-data_"+str(cur_seed)+".mat")
x_shrunken = Output["x"]
u_shrunken = Output["u"]
shrunken_t = Output["t"][0,:]
features = ["X_"+str(i) for i in range(1, K+1)]
for i in range(K):
    for j in range(i,K):
        if i==j:
            features.append(features[i]+"^2")
        else:
            features.append(features[i]+features[j])


from sklearn.linear_model import Lasso
from pysindy.differentiation import BaseDifferentiation


class ReturnUnresolvedValues(BaseDifferentiation):
    def __init__(self, order=2, drop_endpoints=False, data=None):
        if order != 2:
          raise NotImplementedError

        self.order = order
        self.data = data
        self.drop_endpoints = drop_endpoints

    def _differentiate(self, x, t):
        return self.data

normalize_u_vals = u_shrunken.copy()

quad_size = K
for i in range(K):
    for j in range(i,K):
        quad_size += 1

normalize_x_vals =  np.zeros((x_shrunken.shape[0], quad_size))
normalize_x_vals[:,:K] = x_shrunken
index = K
for i in range(K):
    for j in range(i,K):
        normalize_x_vals[:,index] =  x_shrunken[:,i]*x_shrunken[:,j]
        index = index + 1

norm_u = np.linalg.norm(normalize_u_vals)
norm_x = np.linalg.norm(normalize_x_vals, axis=0)

normalize_u_vals = normalize_u_vals/norm_u
normalize_x_vals = normalize_x_vals/norm_x

#%%

start_interval = int(500/Dt)
end_interval = int(1000/Dt)

lasso_optimizer = Lasso(1e-9)

xi = np.zeros((K,quad_size))

print("Training model...")
for i in range(K):
    ruv = ReturnUnresolvedValues(data=normalize_u_vals[start_interval:end_interval, i:i+1])
    model = ps.SINDy(feature_library=feature_library, feature_names=features, optimizer=lasso_optimizer, differentiation_method=ruv)
    model.fit(normalize_x_vals[start_interval:end_interval, :], t=Dt)
    xi[i:i+1,:] = model.coefficients()

xi_backup = xi.copy()

#%%

xi = xi_backup.copy()

# Filter out small coefficients
for i in range(xi.shape[0]):
    for j in range(xi.shape[1]):
        if np.abs(xi[i,j]) < 0.008:
            xi[i,j] = 0

# Use this function to print the model
def print_model_from_xi(xi, features):
    n = len(features)
    for i in range(K):
        a = "U_"+str(i+1)+" = "
        plus_index = 0
        for j in range(n):
            if xi[i,j] != 0:
                if plus_index == 0:
                    a = a+str(np.round(xi[i,j], 6))+str(features[j])
                    plus_index = plus_index + 1
                else:
                    a = a + " + "+str(np.round(xi[i,j], 6))+str(features[j])
        print(a)

# Run this to print the model
print("Model:")
print_model_from_xi(xi*norm_u/norm_x, features)

