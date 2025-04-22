import numpy as np
# from fbm import FBM
import math
from _PARAMETERS import *

def tracking_coordinates(t):
    y_1 = 2 * math.cos(t/8.2) + math.cos(5 * t/8.2)
    y_2 = 2 * math.sin(t/8.2) + math.sin(5 * t/8.2)
    return y_1, y_2

def generate_phi(T = 100, N = 100, SYSTEM_TYPE = 'linear'):

    if SYSTEM_TYPE == 'robotic_arm':

        # Parameters
        A = 1.0  # Amplitude of seasonal component
        omega = 0.1  # Frequency of seasonal component
        rho = 0.8  # Autoregressive coefficient
        sigma_epsilon = 0.2  # Noise standard deviation

        # Time array
        t_values = np.arange(T+N)

        # Initialize phi_t array
        phi_t = np.zeros(T+N)
        phi_t[0] = 0.0  # Starting value

        # Generate phi_t
        for t in range(1, T+N):
            seasonal = A * np.sin(omega * t_values[t])
            autoregressive = rho * phi_t[t - 1]
            noise = np.random.normal(0, sigma_epsilon)
            phi_t[t] = seasonal + autoregressive + noise

    elif SYSTEM_TYPE == 'linear':
        # Initialize phi_t array

        phi_t = np.zeros((T+N, 4))
        for t in range(T+N):
            y_1, y_2 = tracking_coordinates(t)
            y_3, y_4 = tracking_coordinates(t + 1)

            A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
            # Ground-true predictions
            phi_t[t] = A @ np.array([y_1, y_2, 0, 0]) - np.array([y_3, y_4, 0, 0])

        # omega = 10.0  # High frequency for rapid oscillation
        # amplitude = 1.0  # Amplitude of the oscillation
        # noise_level = 0.5  # Magnitude of random noise
        #
        # # Initialize phi_t array
        # phi_t = np.zeros((T + N, 4))
        #
        # # Generate phi_t with rapid variations
        # for t in range(T + N):
        #     # Time variable for oscillation
        #     time = t * 0.1  # Adjust time step if needed
        #     # High-frequency sine wave for each component (slightly different phases)
        #     sine_components = amplitude * np.array([
        #         np.sin(omega * time),
        #         np.sin(omega * time + 0.5),  # Phase shift for variety
        #         np.sin(omega * time + 1.0),
        #         np.sin(omega * time + 1.5)
        #     ])
        #     # Random noise for each component
        #     noise = noise_level * np.random.randn(4)
        #     # Combine oscillation and noise
        #     phi_t[t] = sine_components + noise


    # # Parameters
    # mu = 0.1  # Drift (trend)
    # sigma = 0.2  # Noise standard deviation
    #
    # # Generate time series
    # t_values = np.arange(T+N)
    # phi_t = np.zeros(T+N)
    # phi_t[0] = 0.0  # Starting value
    # for t in range(1, T+N):
    #     phi_t[t] = phi_t[t - 1] + mu + np.random.normal(0, sigma)

    # # Parameters
    # H = 0.7  # Hurst parameter for persistence
    #
    # # Generate fBM
    # f = FBM(n=T+N - 1, hurst=H, length=1, method='daviesharte')
    # phi_t = f.fbm()


    # t_values = np.arange(T + N)
    # phi_true_history = Amp * np.sin(omega * t_values) + np.random.normal(0, sigma_epsilon, T + N)

    return phi_t