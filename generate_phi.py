import numpy as np
# from fbm import FBM
import math
from _PARAMETERS import *

def tracking_coordinates(t):
    y_1 = 1/2 * math.cos(t/10) + math.cos(5 * t/10)
    y_2 = 1/2 * math.sin(t/10) + math.sin(5 * t/10)

    return y_1, y_2

def generate_phi(T = 200, N = 100, SYSTEM_TYPE = 'linear'):

    if SYSTEM_TYPE == 'robotic_arm':

        # Regenerate unknown parameters phi's for the robotic arm system
        # We'll use a random walk with bounded increments to simulate unknown, time-varying parameters

        # Parameters for random walk
        phi_min = -1
        phi_max = 1
        phi_t = np.zeros(T+N)
        phi_t[0] = np.random.uniform(phi_min, phi_max)  # Random initial value

        for t in range(1, T+N):
            # Random walk step with small noise
            step = np.random.normal(2, 0.5)
            phi_t[t] = phi_t[t-1] + step
            # Clip to ensure |phi_t| <= 0.02
            phi_t[t] = np.clip(phi_t[t], phi_min, phi_max)

    elif SYSTEM_TYPE == 'linear':
        # Initialize phi_t array

        phi_t = np.zeros((T+N, 4))
        for t in range(T+N):
            y_1, y_2 = tracking_coordinates(t)
            y_3, y_4 = tracking_coordinates(t + 1)

            # Ground-true predictions
            phi_t[t] = A @ np.array([y_1, y_2, 0, 0]) - np.array([y_3, y_4, 0, 0])

    return phi_t

