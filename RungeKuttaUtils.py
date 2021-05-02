#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:54:01 2021

@author: marty
"""
import numpy as np
from numba import jit
from sklearn.metrics import mean_squared_error

@jit(nopython=True, cache=True)
def l96_regression_truth_step(X, p):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.
    Args:
        X (1D ndarray): Values of X variables at the current time step
        p (Function): Regression model
    Returns:
        dXdt (1D ndarray): Array of X increments
    """
    # These are our constants
    K = 40  # Number of variables
    Fx = 10  # Forcing
    
    dXdt = np.zeros(K)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + Fx + p[0]*X[k]**4 + p[1]*X[k]**3 + p[2]*X[k]**2 + p[3]*X[k] + p[4]
    return dXdt
def run_lorenz96_regression_truth(x_initial, p, time_step, num_steps, burn_in, skip):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.
    Args:
        x_initial (1D ndarray): Initial X values.
        p (Function): Regression model
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival
    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        times (number of timesteps): Each time step
        steps(int): Number of steps
    """
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    x[:] = x_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            print(n, n * time_step)
        k1_dxdt[:] = l96_regression_truth_step(x, p)
        k2_dxdt[:] = l96_regression_truth_step(x + k1_dxdt * time_step / 2, p)
        k3_dxdt[:] = l96_regression_truth_step(x + k2_dxdt * time_step / 2, p)
        k4_dxdt[:] = l96_regression_truth_step(x + k3_dxdt * time_step, p)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            i += 1
    return x_out, times, steps
def run_lorenz96_regression_truth_noise(x_initial, p, time_step, num_steps, burn_in, skip, phi=0, sigma=0, res_0=0, seed=0):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.
    Args:
        x_initial (1D ndarray): Initial X values.
        p (Function): Regression model
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival
        phi (float): Phi value in ARIMA model
        sigma (float): Sigma value in ARIMA model
        res_0 (1D ndarray): Initial residual
    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        times (number of timesteps): Each time step
        steps(int): Number of steps
    """
    np.random.seed(seed)
    K = 40  # Number of variables
    
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    x[:] = x_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    i = 0
    errors = np.zeros((num_steps,K))
    error = res_0
    errors[0,:] = error
    if burn_in == 0:
        x_out[i] = x
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            print(n, n * time_step)
        k1_dxdt[:] = l96_regression_truth_step(x, p)
        k2_dxdt[:] = l96_regression_truth_step(x + k1_dxdt * time_step / 2, p)
        k3_dxdt[:] = l96_regression_truth_step(x + k2_dxdt * time_step / 2, p)
        k4_dxdt[:] = l96_regression_truth_step(x + k3_dxdt * time_step, p)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt + error) / 6 * time_step
        error = phi*error + np.random.normal(0,sigma,K)
        errors[n,:] = error
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            i += 1
    return x_out, times, steps, errors

@jit(nopython=True, cache=True)
def create_quartic_array_1D(X):
    """
    Takes in an array X and returns [X, X^2, X^3, X^4]

    Parameters
    ----------
    X : 1D ndarray

    Returns
    -------
    new_X : 1D ndarray

    """
    K = 40
    new_X = np.zeros(4*K)
    new_X[0:K] = X
    new_X[K:2*K] = X**2
    new_X[2*K:3*K] = X**3
    new_X[3*K:4*K] = X**4
    return new_X

@jit(nopython=True, cache=True)
def l96_matrix_truth_step(X, xi, x_norm=1, u_norm=1, bias=0):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.
    Args:
        X (1D ndarray): Values of X variables at the current time step
        xi (2D ndarray): Coefficients in compressed sensing model
        x_norm (float): Norm of x values
        u_norm (float): Norm of u values
        bias (1D ndarray): Bias in compressed sensing model
    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    # These are our constants
    K = 40  # Number of variables
    Fx = 10  # Forcing
    
    dXdt = np.zeros(K)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + Fx

    U_vals = create_quartic_array_1D(X)/x_norm
    dXdt = dXdt + (np.dot(xi, np.transpose(U_vals)) + np.transpose(bias))*u_norm
    return dXdt
def run_lorenz96_matrix_truth(x_initial, xi, time_step, num_steps, burn_in, skip, x_norm=1, u_norm=1, bias=np.zeros(40), printTime=True):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.
    Args:
        x_initial (1D ndarray): Initial X values.
        xi (2D ndarray): Coefficients in compressed sensing model
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival
        x_norm (float): Norm of x values
        u_norm (float): Norm of u values
        bias (1D ndarray): Bias in compressed sensing model
    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        times (number of timesteps): Each time step
        steps(int): Number of steps
    """
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    x[:] = x_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            if printTime:
                print(n, n * time_step)
        k1_dxdt[:] = l96_matrix_truth_step(x, xi, x_norm, u_norm, bias)
        k2_dxdt[:] = l96_matrix_truth_step(x + k1_dxdt * time_step / 2, xi, x_norm, u_norm, bias)
        k3_dxdt[:] = l96_matrix_truth_step(x + k2_dxdt * time_step / 2, xi, x_norm, u_norm, bias)
        k4_dxdt[:] = l96_matrix_truth_step(x + k3_dxdt * time_step, xi, x_norm, u_norm, bias)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            i += 1
    return x_out, times, steps
def run_lorenz96_matrix_truth_noise(x_initial, xi, time_step, num_steps, burn_in, skip, x_norm=1, u_norm=1, bias=np.zeros(40), phi=0, sigma=0, res_0=0, noiseSeed=0):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.
    Args:
        x_initial (1D ndarray): Initial X values.
        xi (2D ndarray): Coefficients in compressed sensing model
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival
        x_norm (float): Norm of x values
        u_norm (float): Norm of u values
        bias (1D ndarray): Bias in compressed sensing model
        phi (float): Phi value in ARIMA model
        sigma (float): Sigma value in ARIMA model
        res_0 (1D ndarray): Initial residual
    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        times (number of timesteps): Each time step
        steps(int): Number of steps
    """
    # These are our constants
    K = 40  # Number of variables
    
    np.random.seed(noiseSeed)
    
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    x[:] = x_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    i = 0
    errors = np.zeros((num_steps,K))
    error = res_0
    errors[0,:] = error
    if burn_in == 0:
        x_out[i] = x
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            print(n, n * time_step)
        k1_dxdt[:] = l96_matrix_truth_step(x, xi, x_norm, u_norm, bias)
        k2_dxdt[:] = l96_matrix_truth_step(x + k1_dxdt * time_step / 2, xi, x_norm, u_norm, bias)
        k3_dxdt[:] = l96_matrix_truth_step(x + k2_dxdt * time_step / 2, xi, x_norm, u_norm, bias)
        k4_dxdt[:] = l96_matrix_truth_step(x + k3_dxdt * time_step, xi, x_norm, u_norm, bias)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt + error) / 6 * time_step
        error = phi*error + np.random.normal(0,sigma,K)
        errors[n,:] = error
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            i += 1
    return x_out, times, steps, errors

def APSE(X1,X2):
    """
    Calculates the Average Prediction Square Error (APSE) between two time series

    Parameters
    ----------
    X1 : 2D ndrray
        Theoretical time series.
    X2 : 2D ndarray
        Predicted time series.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    K = 40
    total_mse = np.array([])
    for i in range(K):
          ss_res = mean_squared_error(X1[:, i], X2[:, i])
          total_mse = np.append(total_mse, ss_res)
    return np.mean(total_mse)
