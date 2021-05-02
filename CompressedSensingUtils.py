#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:37:12 2021

@author: marty
"""

import numpy as np
import pysindy as ps
from pysindy.differentiation import BaseDifferentiation
from pysindy.feature_library import CustomLibrary
import matplotlib.pyplot as plt

class ReturnUnresolvedValues(BaseDifferentiation):
    def __init__(self, data, order=2, drop_endpoints=False):
        '''
        Takes unresolved variables to use for the compressed sensing algorithm

        Parameters
        ----------
        data : 2D ndarray
            The U values you intend to use.
        order : 
            DO NOT CHANGE. The default is 2.
        drop_endpoints : 
            DO NOT CHANGE. The default is False.

        Returns
        -------
        None.

        '''
        if order != 2:
          raise NotImplementedError

        self.order = order
        self.data = data
        self.drop_endpoints = drop_endpoints

    def _differentiate(self, x, t):
        return self.data

def create_quartic_array_2D(X):
    '''
    Takes in an array X and returns [X, X^2, X^3, X^4]

    Parameters
    ----------
    X : 2D ndarray

    Returns
    -------
    new_X : 2D ndarray

    '''
    K = 40
    index = 0
    new_X = np.zeros((X.shape[0], 4*K))
    for i in range(K):
        new_X[:,index] = X[:,i]
        index = index + 1
    for i in range(K):
        new_X[:,index] = X[:,i]**2
        index = index + 1
    for i in range(K):
        new_X[:,index] = X[:,i]**3
        index = index + 1
    for i in range(K):
        new_X[:,index] = X[:,i]**4
        index = index + 1
    return new_X

def print_model_from_xi(xi, features, biases):
    '''
    Use this function to print the model

    Parameters
    ----------
    xi : 2D ndarray
        Coefficients in compressed sensing model.
    features : 1D ndarray
        Labels for each column of the xi matrix.
    biases : 2D ndarray
        Bias in compressed sensing model.

    Returns
    -------
    None.

    '''
    K = 40
    
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
        a = a + " + " + str(np.round(biases[i], 6))
        print(a)


class CompressedSensing():
    
    def __init__(self, X, U, T, features, Dt, K=40, X_modify_fn=create_quartic_array_2D):
        self.raw_x = X
        self.x = X_modify_fn(X)
        self.u = U
        self.t = T
        self.features = features
        self.norm_x = np.linalg.norm(self.x, axis=0)
        self.norm_u = np.linalg.norm(self.u)
        self.K = K
        self.Dt = Dt
        self.biases = np.zeros(K)
        self.biases_averaged = np.zeros(K)
        self.bias_noise = np.zeros(K)
        self.xi = None
        self.xi_a = None
        self.start_interval = None
        self.end_interval = None
    
    def RunCSAlgorithm(self,start_interval,end_interval,lasso_optimizer,noise_seed=341,order=4):
        
        if order != 4:
            raise NotImplementedError
        
        self.start_interval = start_interval
        self.end_interval = end_interval
        
        functions = [lambda x: x]
        feature_library = CustomLibrary(library_functions=functions)
        
        print("Fitting raw model...")
        np.random.seed(noise_seed)
        error_vals = np.random.normal(0, 0.0001, (self.x.shape[0], self.x.shape[1]))
        self.xi = np.zeros((40, 160))
        for i in range(self.K):
            entries = [i,i+self.K,i+2*self.K,i+3*self.K]
            ruv = ReturnUnresolvedValues(data=self.u[start_interval:end_interval, i:i+1])
            model = ps.SINDy(feature_library=feature_library,
                             feature_names=[self.features[i] for i in entries],
                             optimizer=lasso_optimizer,
                             differentiation_method=ruv)
            model.fit(self.x[start_interval:end_interval, entries] + error_vals[start_interval:end_interval, entries], t=self.Dt)
            self.xi[i:i+1,entries] = model.coefficients()
    
        new_xi = self.xi.copy()
        new_xi = (new_xi*self.norm_x)/self.norm_u
        predictions = np.dot(new_xi[:,:], np.transpose(self.x[start_interval:end_interval, :]/self.norm_x[:]))
        print("RIP condition:", np.linalg.norm(predictions)/np.linalg.norm(new_xi))
    
    def averageCoefficients(self, modify=False):
        coeff_1 = np.mean([self.xi[i,i] for i in range(self.K)])
        coeff_2 = np.mean([self.xi[i,i+self.K] for i in range(self.K)])
        coeff_3 = np.mean([self.xi[i,i+2*self.K] for i in range(self.K)])
        coeff_4 = np.mean([self.xi[i,i+3*self.K] for i in range(self.K)])
        
        self.xi_a = np.concatenate((coeff_1*np.identity(self.K), 
                                           coeff_2*np.identity(self.K), 
                                           coeff_3*np.identity(self.K), 
                                           coeff_4*np.identity(self.K)), axis=1)
        
        if modify:
            self.xi = np.concatenate((coeff_1*np.identity(self.K), 
                                               coeff_2*np.identity(self.K), 
                                               coeff_3*np.identity(self.K), 
                                               coeff_4*np.identity(self.K)), axis=1)
    
    def findBiases(self):
        temp_u_data = np.transpose(self.u[self.start_interval:self.end_interval,:])
        temp_x_data = np.transpose(self.x[self.start_interval:self.end_interval,:])
        a = np.dot(self.xi, temp_x_data)
        biases = np.zeros(self.K)
        for i in range(self.K):
            biases[i] = np.mean(temp_u_data[i,:]-a[i,:])
        self.biases = biases
        self.biases_averaged = np.mean(biases)*np.ones(self.K)
    
    def addNoiseToBiases(self, sigma=0.07, biasNoiseSeed=411):
        np.random.seed(biasNoiseSeed)
        self.bias_noise = np.random.normal(0,0.007,self.K)
    
    def plotUnresolvedPredictions(self):
        for i in range(self.K):
            plt.plot(self.t, self.u[i,:], "darkorange", linewidth=2)
            a = np.dot(self.xi, np.transpose(self.x))
            plt.plot(self.t, a[i,:] + self.biases[i])
            plt.title("U_"+str(i+1))
            plt.show()

    def printModel(self, averaged=True):
        print("Model:")
        if averaged:
            print_model_from_xi(self.xi, self.features, self.biases_averaged+self.bias_noise)
        else:
            print_model_from_xi(self.xi, self.features, self.biases+self.bias_noise)





