import numpy as np
from _PARAMETERS import *

# Dynamics functions

def robotic_arm_dynamics(x, u, phi):
    """Robotic arm angle dynamics."""
    alpha = 0.5
    beta = 0.2
    return float(x + alpha * np.sin(x) + beta * u * np.exp(-np.abs(x)) + phi)


def robotic_arm_cost(x, u):
    """Cost for robotic arm: penalize deviation and torque."""

    gamma = 0.1
    return float(x**2 + gamma * u**2)


def linear_dynamics(x, u, phi):
    """Linear dynamics."""
    return A @ x + B @ u + phi


def quadratic_cost(x, u):
    """Cost for robotic arm: penalize deviation and torque."""
    return x.T @ Q @ x + u.T @ R @ u
