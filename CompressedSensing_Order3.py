#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:43:25 2020

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
features = ["X_"+str(i) for i in range(1, K+1)] + ["X_"+str(i)+"^2" for i in range(1, K+1)] + ["X_"+str(i)+"^3" for i in range(1, K+1)] + ["F"+str(i) for i in range(100)]


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
norm_u = np.linalg.norm(normalize_u_vals)
normalize_u_vals = normalize_u_vals/norm_u

#%%

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

start_interval = int(500/Dt)
end_interval = int(1000/Dt)

lasso_optimizer = Lasso(1e-9)

quad_size = 3*K+100

print("Training model...")
for i in range(100):
    xi = np.zeros((K,quad_size))
    normalize_x_vals =  np.zeros((x_shrunken.shape[0], quad_size))
    normalize_x_vals[:,:K] = x_shrunken
    normalize_x_vals[:,K:2*K] = x_shrunken**2
    normalize_x_vals[:,2*K:3*K] = x_shrunken**3
    for j in range(100):
        a = np.random.randint(0,K)
        b = np.random.randint(0,K)
        c = np.random.randint(0,K)
        # print("j: "+str(j)+", a = "+str(a)+", b = "+str(b)+", c = "+str(c))
        # print(x_shrunken[:,a]*x_shrunken[:,b]*x_shrunken[:,c])
        normalize_x_vals[:,3*K+j] = x_shrunken[:,a]*x_shrunken[:,b]*x_shrunken[:,c]
    norm_x = np.linalg.norm(normalize_x_vals, axis=0)
    normalize_x_vals = normalize_x_vals/norm_x
    for j in range(K):
        indices = [j] + [j+K] + [j+2*K] + list(range(3*K,3*K+100))
        ruv = ReturnUnresolvedValues(data=normalize_u_vals[start_interval:end_interval, j:j+1])
        model = ps.SINDy(feature_library=feature_library, feature_names=[features[i] for i in indices], optimizer=lasso_optimizer, differentiation_method=ruv)
        model.fit(normalize_x_vals[start_interval:end_interval, indices], t=Dt)
        xi[j:j+1,indices] = model.coefficients()
    # Filter out small coefficients
    for k in range(xi.shape[0]):
        for j in range(2*K,xi.shape[1]):
            if np.abs(xi[k,j]) < 0.005 or np.abs(xi[k,j]) > 1000:
                xi[k,j] = 0
    # Run this to print the model
    print("Model "+str(i+1)+":")
    print_model_from_xi(xi*norm_u/norm_x, features)
    print("")


