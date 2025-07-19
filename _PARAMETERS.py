from tkinter.constants import FALSE
import numpy as np
from scipy.linalg import solve_discrete_are

with_attacks = True
SYSTEM_TYPE = 'linear'

# Create parameters

# 4-dimensional linear system

if SYSTEM_TYPE == 'linear':

    if with_attacks:
        A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    else:
        A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[1, 0], [0, 1], [0.2, 0], [0, 0.2]])

    Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    R = np.array([[1, 0], [0, 1]])
    rho =  0.8679197768583015

# chaotic map

if SYSTEM_TYPE == 'robotic_arm':


    # Linearized A, B, Q, R for the nonlinear robotic arm dynamics (1D)
    # Linearize around x = 0, u = 0 (assuming phi ≈ 0)
    # Dynamics: x_{t+1} = x_t + alpha * sin(x_t) + beta * u_t * exp(-|x_t|) + phi
    # At x=0, sin(0)=0, exp(-|0|)=1, so:
    # x_{t+1} ≈ x_t + beta * u_t

    beta = 0.2
    # State: x (scalar), Control: u (scalar)
    A = np.array([[1.5]])
    # Linearized B: beta (since d/d(u) of beta*u at u=0 is beta)
    B = np.array([[beta]])

    # Cost: x^2 + gamma * u^2, gamma = 0.1
    Q = np.array([[1.0]])
    R = np.array([[0.1]])