#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 04:00:48 2020

@author: marty
"""

import numpy as np
from numba import jit
from scipy.io import savemat

cur_seed = 411

# These are our constants
K = 40  # Number of variables
J = 10
total_number_of_vars = K * J + K
h = 1
c = 10
b = 10
Fx = 10  # Forcing
Fy = 0

time_step = 0.001
num_steps = int(2000/time_step)
burn_in = 0
skip = 1

@jit(nopython=True, cache=True)
def l96_truth_step(X, Y):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.
    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio
    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    dXdt = np.zeros(K)
    dYdt = np.zeros(K*J)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + Fx - h*c/b * np.sum(Y[k * J: (k + 1) * J])
    for j in range(J * K):
        dYdt[j] = - c*b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j-1]) -  c * Y[j] + c/b * Fy + h*c/b * X[int(j / J)]
    return dXdt, dYdt
def run_lorenz96_truth(x_initial, y_initial, time_step, num_steps, burn_in, skip):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.
    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        f (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival
    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    archive_steps = (num_steps - burn_in) // skip
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    # Calculate total Y forcing over archive period using trapezoidal rule
    y_trap = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    y_trap[:] = y_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    i = 0
    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        if (n * time_step) % 1 == 0:
            print(n, n * time_step)
        k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y)
        k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                                y + k1_dydt * time_step / 2)
        k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                                y + k2_dydt * time_step / 2)
        k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                                y + k3_dydt * time_step)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = (y + y_trap) / skip
            i += 1
        elif n % skip == 1:
            y_trap[:] = y
        else:
            y_trap[:] += y
    return x_out, y_out, times, steps

@jit(nopython=True, cache=True)
def l96_modified_matrix_truth_step(X, p):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.
    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio
    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    dXdt = np.zeros(K)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + Fy + p[0]*X[k]**4 + p[1]*X[k]**3 + p[2]*X[k]**2 + p[3]*X[k] + p[4]
    return dXdt
def run_lorenz96_modified_matrix_truth(x_initial, p, time_step, num_steps, burn_in, skip):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.
    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        f (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival
    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
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
        k1_dxdt[:] = l96_modified_matrix_truth_step(x, p)
        k2_dxdt[:] = l96_modified_matrix_truth_step(x + k1_dxdt * time_step / 2, p)
        k3_dxdt[:] = l96_modified_matrix_truth_step(x + k2_dxdt * time_step / 2, p)
        k4_dxdt[:] = l96_modified_matrix_truth_step(x + k3_dxdt * time_step, p)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            i += 1
    return x_out, times, steps

np.random.seed(cur_seed)
X = np.random.normal(0, 1, K)
Y = np.random.normal(0, 1, K*J)

Output = run_lorenz96_truth(X, Y, time_step, num_steps, burn_in, skip)

X_out = Output[0]
Y_out = Output[1]
t = Output[2]
print(np.shape(X_out), np.shape(Y_out))

U_out = np.zeros(X_out.shape)
for i in range(K):
  for j in range(J):
    U_out[:,i] = U_out[:,i] + Y_out[:,i*J+j]
  U_out[:,i] = -h*c/b * U_out[:,i]

multiplier = 10
Dt = time_step*multiplier
steps = np.arange(0, t.shape[0], multiplier)
shrunken_t = t[steps]
x_shrunken = X_out[steps,:]
u_shrunken = U_out[steps,:]

savemat("Lorenz-96-data_"+str(cur_seed)+".mat", {"x": x_shrunken, "u": u_shrunken, "t": shrunken_t})












